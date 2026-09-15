"""K2 single-fusion math/precision/ownership tests, without task entry imports."""

from contextlib import ExitStack
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.kcdno.history import KernelHistoryCache, KernelHistoryReader, KernelHistoryWriter
from kernel_history_reference import history_reference, phi_reference, rms_reference, writer_reference


def setUpModule():
    global old_threads, old_tf32, old_precision
    old_threads = torch.get_num_threads()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f'K2 environment: torch={torch.__version__}, cuda={torch.version.cuda}, '
          f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}; TF32 off, compile off')


def tearDownModule():
    torch.set_num_threads(old_threads)
    torch.set_float32_matmul_precision(old_precision)
    torch.backends.cuda.matmul.allow_tf32 = old_tf32


def case(b=2, m=5, d=8, r=3, s=3, *, device='cpu', dtype=torch.float32, trained=True):
    reader = KernelHistoryReader(d, r).to(device)
    writers = [KernelHistoryWriter(d, r).to(device) for _ in range(s)]
    if trained:
        with torch.no_grad():
            reader.w.normal_(std=.3)
            reader.gamma.fill_(.35)
            for norm in [reader.query_norm, reader.depth_norm, *(w.key_norm for w in writers)]:
                norm.weight.uniform_(.6, 1.4)
    u = torch.randn(b, m, d, device=device, dtype=dtype).requires_grad_()
    history = [(torch.randn_like(u) + (index + 1) / 4).requires_grad_() for index in range(s)]
    return u, history, writers, reader


def leaves(u, history, writers, reader):
    return (u, *history, *(p for writer in writers for p in writer.parameters()), *reader.parameters())


def oracle(u, history, writers, reader, *, explicit=False):
    # Cast connected leaves to double here, not inside the independent oracle.
    return history_reference(u.double(), [t.double() for t in history],
        [{name:p.double() for name,p in writer.named_parameters()} for writer in writers],
        {name:p.double() for name,p in reader.named_parameters()}, explicit_tokens=explicit)


def production(u, history, writers, reader):
    caches = tuple(writer(t) for writer,t in zip(writers,history,strict=True))
    return reader(u, caches, return_weights=True)


class Operations(TorchDispatchMode):
    """Observe real operators, their shapes/dtypes, and autocast state."""
    def __init__(self, device_type):
        super().__init__()
        self.device_type = device_type
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if name.startswith(('aten.pow.', 'aten.mean.', 'aten.rsqrt.', 'aten.mul.', 'aten.add.',
                            'aten.sub.', 'aten.mm.', 'aten.bmm.', 'aten.elu.', 'aten.clamp_min.',
                            'aten._softmax.', 'aten.sum.', 'aten.div.')):
            tensors = [a for a in args if isinstance(a,torch.Tensor) and a.is_floating_point()]
            outputs = [result] if isinstance(result,torch.Tensor) else []
            self.records.append((name, [x.dtype for x in tensors+outputs],
                                 torch.is_autocast_enabled(self.device_type),
                                 [tuple(x.shape) for x in tensors+outputs]))
        return result


class KernelHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260916)

    def finite(self, tensors):
        for index,tensor in enumerate(tensors):
            self.assertIsNotNone(tensor,index)
            self.assertTrue(torch.isfinite(tensor).all().item(),index)

    def compare_output_gradients(self, actual, expected, variables, *, atol, rtol):
        ao,aa=actual
        eo,ea=expected.output,expected.alpha
        torch.testing.assert_close(ao.double(),eo,atol=atol,rtol=rtol)
        torch.testing.assert_close(aa.double(),ea,atol=atol,rtol=rtol)
        cot=torch.randn_like(eo);acot=torch.randn_like(ea)
        ga=torch.autograd.grad((ao.double()*cot).sum()+.03*(aa.double()*acot).sum(),variables,retain_graph=True)
        ge=torch.autograd.grad((eo*cot).sum()+.03*(ea*acot).sum(),variables,retain_graph=True)
        self.finite((*ga,*ge))
        for index,(a,e) in enumerate(zip(ga,ge,strict=True)):
            torch.testing.assert_close(a,e,atol=atol,rtol=rtol,msg=lambda msg:f'gradient {index}: {msg}')
        return max((a-e).abs().max().item() for a,e in zip(ga,ge,strict=True))

    def test_double_cached_equals_explicit_kernel_and_all_gradients(self):
        errors=[]
        for b in (1,2):
            for m,r,d in ((1,1,4),(5,3,8),(3,7,6)):
                for s in (1,3):
                    with self.subTest(b=b,m=m,r=r,s=s):
                        u,ts,ws,reader=case(b,m,d,r,s,dtype=torch.float64)
                        ws=[w.double() for w in ws];reader=reader.double()
                        a=oracle(u,ts,ws,reader)
                        e=oracle(u,ts,ws,reader,explicit=True)
                        self.assertEqual(a.output.dtype,torch.float64)
                        errors.append(self.compare_output_gradients((a.output,a.alpha),e,
                            leaves(u,ts,ws,reader),atol=2e-11,rtol=2e-10))
        print(f'double cached/explicit: 12 cases, max gradient abs={max(errors):.9g}')

    def test_independent_oracle_and_double_gradcheck_away_from_boundaries(self):
        u=torch.full((2,2,3),.8,dtype=torch.float64,requires_grad=True)
        t=torch.linspace(.4,1.2,12,dtype=torch.float64).reshape(2,2,3).requires_grad_()
        wq=torch.full((2,3),.3,dtype=torch.float64,requires_grad=True)
        wk=torch.full((2,3),.4,dtype=torch.float64,requires_grad=True)
        nq=torch.ones(3,dtype=torch.float64,requires_grad=True)
        nk=torch.ones_like(nq,requires_grad=True);nd=torch.ones_like(nq,requires_grad=True)
        w=torch.tensor([.2,-.1,.15],dtype=torch.float64,requires_grad=True)
        gamma=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
        values=(u,t,wq,wk,nq,nk,nd,w,gamma)
        self.assertGreater((rms_reference(u,nq)@wq.T).min().item(),.5)
        self.assertGreater((rms_reference(t,nk)@wk.T).min().item(),.5)
        def compute(*args, explicit=False):
            u,t,wq,wk,nq,nk,nd,w,gamma=args
            return history_reference(u,[t],[{'key_norm.weight':nk,'to_k.weight':wk}],
                {'query_norm.weight':nq,'to_q.weight':wq,'depth_norm.weight':nd,'w':w,'gamma':gamma},
                explicit_tokens=explicit).output
        with patch.object(KernelHistoryReader,'forward',side_effect=AssertionError('oracle calls reader')), \
             patch.object(KernelHistoryWriter,'forward',side_effect=AssertionError('oracle calls writer')), \
             patch.object(F,'scaled_dot_product_attention',side_effect=AssertionError('oracle calls SDPA')):
            for explicit in (False,True):
                self.assertTrue(torch.autograd.gradcheck(lambda *a:compute(*a,explicit=explicit),values,
                                                       eps=1e-6,atol=1e-5,rtol=1e-3))

    def test_production_fp32_against_double_oracle(self):
        errors=[]
        for b in (1,2):
            for m,r,d in ((1,1,4),(5,3,8),(3,7,6)):
                for s in (1,3):
                    for trained in (False,True):
                        with self.subTest(b=b,m=m,r=r,s=s,trained=trained):
                            u,ts,ws,reader=case(b,m,d,r,s,trained=trained)
                            errors.append(self.compare_output_gradients(production(u,ts,ws,reader),
                                oracle(u,ts,ws,reader,explicit=True),leaves(u,ts,ws,reader),atol=5e-6,rtol=1e-4))
        print(f'FP32/double: 24 cases, max gradient abs={max(errors):.9g}; atol5e-6 rtol1e-4')

    def test_empty_history_is_exact_identity_without_parameter_work(self):
        for b in (1,2):
            for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
                u,_,_,reader=case(b=b,s=0,dtype=dtype)
                expected=oracle(u,[],[],reader)
                self.assertEqual(expected.output.dtype,torch.float64)
                self.assertEqual(expected.alpha.shape,(b,5,1))
                torch.testing.assert_close(expected.output,u.double(),atol=0,rtol=0)
                with patch.object(reader.query_norm,'forward',side_effect=AssertionError('empty norm')), \
                     patch.object(F,'linear',side_effect=AssertionError('empty projection')):
                    out,alpha=reader(u,(),return_weights=True)
                    self.assertIs(out,u)
                    self.assertIs(reader(u,[]),u)
                self.assertEqual(alpha.shape,(b,5,1))
                torch.testing.assert_close(alpha,torch.ones_like(alpha),atol=0,rtol=0)
                out.sum().backward()
                torch.testing.assert_close(u.grad,torch.ones_like(u),atol=0,rtol=0)
                self.assertTrue(all(p.grad is None for p in reader.parameters()))

    def test_initialization_uniform_depth_raw_values_and_unconstrained_gate(self):
        with patch('torch.nn.init.xavier_uniform_',wraps=torch.nn.init.xavier_uniform_) as init:
            writer=KernelHistoryWriter(8,3);reader=KernelHistoryReader(8,3)
        self.assertEqual(init.call_count,2)
        for call in init.call_args_list: self.assertEqual(call.kwargs,{'gain':1.0})
        self.assertEqual(set(dict(writer.named_parameters())),{'key_norm.weight','to_k.weight'})
        self.assertEqual(set(dict(reader.named_parameters())),{'query_norm.weight','to_q.weight','depth_norm.weight','w','gamma'})
        self.assertIsNone(writer.to_k.bias);self.assertIsNone(reader.to_q.bias)
        self.assertEqual(reader.gamma.ndim,0)
        self.assertAlmostEqual(reader.gamma.item(),.1,places=7)
        for norm in (writer.key_norm,reader.query_norm,reader.depth_norm):
            torch.testing.assert_close(norm.weight,torch.ones_like(norm.weight),atol=0,rtol=0)
            self.assertEqual(set(dict(norm.named_parameters())),{'weight'})
        for projection in (writer.to_k,reader.to_q):
            self.assertLessEqual(projection.weight.abs().max().item(),math.sqrt(6/11))
        for s in (1,3):
            u,ts,ws,reader=case(s=s,trained=False)
            out,alpha=production(u,ts,ws,reader)
            expected=oracle(u,ts,ws,reader,explicit=True)
            torch.testing.assert_close(alpha,torch.full_like(alpha,1/(s+1)),atol=0,rtol=0)
            raw=expected.candidates
            formula=(.9+.1/(s+1))*u.double()+(.1/(s+1))*raw[:,:,1:].sum(dim=2)
            torch.testing.assert_close(out.double(),formula,atol=3e-7,rtol=3e-6)
            self.assertGreater((expected.token_weights[0]-expected.token_weights[0].mean(-1,keepdim=True)).abs().max().item(),1e-3)
            for gamma in (-.5,0.,1.5):
                with torch.no_grad(): reader.gamma.fill_(gamma)
                out,_=production(u,ts,ws,reader)
                formula=u.double()+gamma*(raw.mean(2)-u.double())
                torch.testing.assert_close(out.double(),formula,atol=2e-6,rtol=2e-5)

    def test_source_order_and_token_permutations(self):
        u,ts,ws,reader=case()
        cache=tuple(w(t) for w,t in zip(ws,ts,strict=True))
        out,alpha=reader(u,cache,return_weights=True)
        source_order=[2,0,1]
        other,weights=reader(u,tuple(cache[i] for i in source_order),return_weights=True)
        torch.testing.assert_close(other,out,atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(weights,alpha[:,:,[0,3,1,2]],atol=3e-7,rtol=3e-6)
        perm=torch.tensor([4,2,0,3,1])
        shuffled=tuple(w(t[:,perm]) for w,t in zip(ws,ts,strict=True))
        torch.testing.assert_close(reader(u,shuffled),out,atol=3e-7,rtol=3e-6)
        other,weights=reader(u[:,perm],cache,return_weights=True)
        torch.testing.assert_close(other,out[:,perm],atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(weights,alpha[:,perm],atol=3e-7,rtol=3e-6)
        self.assertGreater((alpha[0]-alpha[1]).abs().max().item(),1e-3)
        self.assertGreater((alpha[:,0]-alpha[:,1]).abs().max().item(),1e-3)

    def test_batch_isolation_values_and_cross_sample_gradients(self):
        u,ts,ws,reader=case()
        out,_=production(u,ts,ws,reader)
        singles=[]
        for b in range(2):
            singles.append(production(u[b:b+1],[t[b:b+1] for t in ts],ws,reader)[0])
        torch.testing.assert_close(out,torch.cat(singles),atol=3e-7,rtol=3e-6)
        gradients=torch.autograd.grad(out[0].sum(),(u,*ts))
        for grad in gradients:
            torch.testing.assert_close(grad[1],torch.zeros_like(grad[1]),atol=0,rtol=0)

    def test_isolated_reader_loss_reaches_earliest_writer_and_zero_scorer_norm(self):
        u,ts,ws,reader=case(trained=False)
        out,_=production(u,ts,ws,reader)
        (out*torch.randn_like(out)).sum().backward()  # no point-domain bypass exists here
        self.finite([p.grad for p in leaves(u,ts,ws,reader)])
        for tensor in (ts[0],ws[0].to_k.weight,reader.to_q.weight,reader.w,reader.gamma):
            self.assertGreater(tensor.grad.abs().max().item(),0)
        torch.testing.assert_close(reader.depth_norm.weight.grad,torch.zeros(8),atol=0,rtol=0)
        with torch.no_grad(): reader.w.add_(reader.w.grad,alpha=-.01)
        reader.zero_grad(set_to_none=True)
        production(u,ts,ws,reader)[0].square().sum().backward()
        self.assertGreater(reader.depth_norm.weight.grad.abs().max().item(),0)
        self.finite([reader.depth_norm.weight.grad])

    def test_one_source_write_can_feed_independent_readers_with_live_gradients(self):
        u,ts,ws,reader=case(s=1)
        second=KernelHistoryReader(8,3)
        v=torch.randn_like(u,requires_grad=True)
        with patch.object(ws[0],'forward',wraps=ws[0].forward) as write:
            cache=ws[0](ts[0])
            a=reader(u,(cache,));b=second(v,(cache,))
        self.assertEqual(write.call_count,1)
        self.assertIsNotNone(cache.memory.grad_fn)
        self.assertIsNotNone(cache.mass.grad_fn)
        refs=(oracle(u,ts,ws,reader),oracle(v,ts,ws,second))
        cot1=torch.randn_like(a);cot2=torch.randn_like(b)
        variables=(*leaves(u,ts,ws,reader),v,*second.parameters())
        grads=torch.autograd.grad((a*cot1).sum()+(b*cot2).sum(),variables,retain_graph=True)
        expected=torch.autograd.grad((refs[0].output*cot1).sum()+(refs[1].output*cot2).sum(),variables)
        self.finite(grads)
        for actual,ref in zip(grads,expected,strict=True):
            torch.testing.assert_close(actual,ref,atol=5e-6,rtol=1e-4)
        self.assertGreater(grads[1].abs().max().item(),0)  # earliest raw T

    def test_no_inplace_no_cache_attributes_and_parameters_independent(self):
        u,ts,ws,reader=case()
        another=KernelHistoryReader(8,3)
        params=[p for module in [*ws,reader,another] for p in module.parameters()]
        self.assertEqual(len({id(p) for p in params}),len(params))
        self.assertEqual(len({p.untyped_storage().data_ptr() for p in params}),len(params))
        before=[(t.detach().clone(),t._version) for t in [u,*ts]]
        caches=tuple(w(t) for w,t in zip(ws,ts,strict=True))
        cached=[(t.detach().clone(),t._version) for c in caches for t in c]
        keys=set(reader.state_dict())
        a=reader(u,caches)
        other=reader(u+2,caches)
        b=reader(u,caches)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertFalse(torch.equal(a,other))
        self.assertEqual(set(reader.state_dict()),keys)
        self.assertFalse(list(reader.buffers()))
        self.assertFalse(any(isinstance(v,(torch.Tensor,KernelHistoryCache)) for v in vars(reader).values()))
        for t,(old,version) in zip([u,*ts],before,strict=True):
            torch.testing.assert_close(t,old,atol=0,rtol=0);self.assertEqual(t._version,version)
        for t,(old,version) in zip([t for c in caches for t in c],cached,strict=True):
            torch.testing.assert_close(t,old,atol=0,rtol=0);self.assertEqual(t._version,version)
        with self.assertRaises(AttributeError): caches[0].mass=torch.ones_like(caches[0].mass)

    def test_one_query_no_rekey_no_token_matrix_and_single_source_softmax(self):
        u,ts,ws,reader=case(m=11,d=8,r=3,s=3)
        caches=tuple(w(t) for w,t in zip(ws,ts,strict=True))
        audit=Operations('cpu')
        with ExitStack() as stack:
            for writer in ws:
                stack.enter_context(patch.object(writer,'forward',side_effect=AssertionError('reader rekeys')))
            stack.enter_context(patch.object(F,'scaled_dot_product_attention',side_effect=AssertionError('kernel calls SDPA')))
            projection=stack.enter_context(patch.object(F,'linear',wraps=F.linear))
            with audit: reader(u,caches)
        self.assertEqual(projection.call_count,1)
        self.assertIs(projection.call_args.args[1],reader.to_q.weight)
        softmax=[record for record in audit.records if record[0].startswith('aten._softmax.')]
        self.assertEqual(len(softmax),1)
        self.assertEqual(softmax[0][3][-1],(2,11,4))
        for name,_,_,shapes in audit.records:
            if name.startswith(('aten.mm.','aten.bmm.')):
                self.assertNotEqual(shapes[-1][-2:],(11,11))
        self.assertEqual(caches[0].memory.shape,(2,3,8))
        self.assertEqual(caches[0].mass.shape,(2,3))

    def check_autocast(self,device,dtype,low_inputs):
        u,ts,ws,reader=case(device=device,dtype=dtype if low_inputs else torch.float32)
        audit=Operations(torch.device(device).type)
        with torch.autocast(torch.device(device).type,dtype=dtype):
            with audit: actual=production(u,ts,ws,reader)
            self.assertTrue(torch.is_autocast_enabled(torch.device(device).type))
        for name,dtypes,active,_ in audit.records:
            self.assertTrue(all(d==torch.float32 for d in dtypes),(name,dtypes))
            self.assertFalse(active,name)
        for prefix in ('aten.pow.','aten.mean.','aten.rsqrt.','aten.mm.','aten.bmm.','aten.elu.',
                       'aten.clamp_min.','aten._softmax.','aten.sum.','aten.div.'):
            self.assertTrue(any(row[0].startswith(prefix) for row in audit.records),prefix)
        self.assertEqual(actual[0].dtype,u.dtype);self.assertEqual(actual[1].dtype,torch.float32)
        expected=oracle(u,ts,ws,reader,explicit=True)
        tol=0.02 if low_inputs and dtype==torch.bfloat16 else (.003 if low_inputs else 5e-6)
        self.compare_output_gradients(actual,expected,leaves(u,ts,ws,reader),atol=tol,rtol=tol if low_inputs else 1e-4)
        print(f'autocast {device}/{dtype}/low_inputs={low_inputs}: observed FP32 ops, output+gradient parity passed')

    def test_cpu_autocast_fp32_operations(self):
        for low in (False,True): self.check_autocast('cpu',torch.bfloat16,low)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable: K2 GPU AMP not run')
    def test_gpu_autocast_fp32_operations(self):
        for dtype in (torch.float16,torch.bfloat16):
            if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for low in (False,True):
                with self.subTest(dtype=dtype,low=low): self.check_autocast('cuda',dtype,low)
        torch.cuda.synchronize()

    def test_dtype_casts_noncontiguous_and_half_parameter_storage(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            u,ts,ws,reader=case(dtype=dtype)
            u=u.transpose(0,1).contiguous().transpose(0,1).detach().requires_grad_()
            ts=[t.to(torch.float64 if i%2 else torch.float16).detach().requires_grad_() for i,t in enumerate(ts)]
            if dtype in (torch.float16,torch.bfloat16):
                reader=reader.to(dtype);ws=[w.to(dtype) for w in ws]
            actual=production(u,ts,ws,reader)
            self.assertEqual(actual[0].dtype,dtype)
            self.assertTrue(all(w(t).memory.dtype==torch.float32 for w,t in zip(ws,ts,strict=True)))
            tol=.03 if dtype==torch.bfloat16 else .005
            self.compare_output_gradients(actual,oracle(u,ts,ws,reader),leaves(u,ts,ws,reader),atol=tol,rtol=tol)

    def test_finite_extremes_clamp_and_denominator(self):
        for scale in (0.,1e-30,1e-12,1e-6,1.,1e4,1e8,1e15):
            with self.subTest(scale=scale):
                u,ts,ws,reader=case()
                u=(u.detach()*scale).requires_grad_();ts=[(t.detach()*scale).requires_grad_() for t in ts]
                caches=tuple(w(t) for w,t in zip(ws,ts,strict=True))
                out,alpha=reader(u,caches,return_weights=True)
                self.finite((out,alpha,*(t for cache in caches for t in cache)))
                gradients=torch.autograd.grad((out/max(1.,scale)).sum(),leaves(u,ts,ws,reader))
                self.finite(gradients)
                for cache in caches: self.assertTrue((cache.mass>0).all())
        # Force ELU rounding to zero before clamp. Small positive denominator
        # must still use SUM and the additive eps, without changing to mean.
        u=torch.ones(2,5,4,requires_grad=True);t=torch.full_like(u,2.,requires_grad=True)
        writer=KernelHistoryWriter(4,3);reader=KernelHistoryReader(4,3)
        with torch.no_grad(): writer.to_k.weight.fill_(-100);reader.to_q.weight.fill_(-100)
        cache=writer(t)
        torch.testing.assert_close(cache.mass,torch.full((2,3),5e-6),atol=1e-12,rtol=1e-6)
        torch.testing.assert_close(cache.memory,torch.full((2,3,4),1e-5),atol=1e-12,rtol=1e-6)
        out,alpha=reader(u,(cache,),return_weights=True)
        denominator=5*3*1e-12
        expected_read=2*denominator/(denominator+1e-6)
        expected=.95*u+.05*expected_read
        torch.testing.assert_close(out,expected,atol=1e-7,rtol=1e-6)
        self.finite((out,alpha,*torch.autograd.grad(out.sum(),leaves(u,[t],[writer],reader))))
        print('finite input amplitudes 0..1e15 passed; clamped kernel denominator=1.5e-11 + eps1e-6')

    def test_invalid_shapes_cache_contract_and_no_padding_api(self):
        for cls in (KernelHistoryWriter,KernelHistoryReader):
            for dim,r in ((0,3),(8,0),(True,3),(8,2.),(8,-1)):
                with self.assertRaises(ValueError): cls(dim,r)
        u,ts,ws,reader=case();writer=ws[0];cache=writer(ts[0])
        for bad in (None,torch.ones(2,5,8,dtype=torch.int64),torch.ones(5,8),torch.ones(2,5,7),torch.ones(0,5,8)):
            with self.assertRaises(ValueError): writer(bad)
            with self.assertRaises(ValueError): reader(bad,())
        for bad in (None,u,[ts[0]],[(cache.memory,cache.mass)]):
            with self.assertRaises(TypeError): reader(u,bad)
        for bad in (KernelHistoryCache(cache.memory[:1],cache.mass),
                    KernelHistoryCache(cache.memory,cache.mass[:,:2]),
                    KernelHistoryCache(cache.memory.double(),cache.mass),
                    KernelHistoryCache(cache.memory,cache.mass.half())):
            with self.assertRaises(ValueError): reader(u,(bad,))
        with self.assertRaises(TypeError): writer(ts[0],mask=torch.ones(2,5))
        with self.assertRaises(TypeError): reader(u,(cache,),mask=torch.ones(2,5))


if __name__=='__main__': unittest.main()
