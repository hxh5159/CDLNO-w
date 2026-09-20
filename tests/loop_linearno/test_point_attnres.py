import json
import math
import os
from pathlib import Path
import unittest

import torch
from torch.func import functional_call
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno_loop.attnres import PointDepthAttnRes
from loop_linearno.point_attnres_oracle import point_attnres


def errors(actual, expected):
    a, b = actual.detach().double(), expected.detach().double()
    delta = (a-b).abs()
    relative = delta / b.abs().clamp_min(1e-12)
    return dict(max_abs=delta.max().item(), mean_abs=delta.mean().item(),
                max_relative=relative.max().item(), mean_relative=relative.mean().item(),
                relative_floor=1e-12)


class PointAttnResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        if path := os.environ.get('LOOP_LL2_ATTNRES_REPORT'):
            Path(path).write_text(json.dumps(dict(torch=str(torch.__version__),
                scope='independent point AttnRes only',rows=cls.rows),indent=2)+'\n')

    def test_parameter_inventory_independence_initialization_and_rng(self):
        before=torch.get_rng_state().clone()
        a,b=PointDepthAttnRes(7),PointDepthAttnRes(7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual({k:list(v.shape) for k,v in a.named_parameters()},
                         {'query':[7],'norm_scale':[7]})
        self.assertEqual(sum(p.numel() for p in a.parameters()),14)
        self.assertEqual(list(a.buffers()),[])
        self.assertEqual(list(a.children()),[])
        self.assertIsNot(a.query,b.query);self.assertIsNot(a.norm_scale,b.norm_scale)
        from cdlno.linearno.attention import initialize_release_weights
        a.apply(initialize_release_weights)
        self.assertTrue(torch.equal(a.query,torch.zeros(7)))
        self.assertTrue(torch.equal(a.norm_scale,torch.ones(7)))

    def test_zero_query_uniform_raw_values_and_expected_first_gradients(self):
        a=PointDepthAttnRes(2).double()
        sources=[torch.tensor([[[3.,4.],[8.,1.]]],dtype=torch.float64,requires_grad=True),
                 torch.tensor([[[0.,2.],[1.,7.]]],dtype=torch.float64,requires_grad=True)]
        output=a(sources);weights=a.source_weights(sources)
        torch.testing.assert_close(output,(sources[0]+sources[1])/2,atol=0,rtol=0)
        torch.testing.assert_close(weights,torch.full((2,1,2),.5,dtype=torch.float64),atol=0,rtol=0)
        output.square().sum().backward()
        self.assertGreater(a.query.grad.abs().sum().item(),0)
        self.assertTrue(torch.equal(a.norm_scale.grad,torch.zeros(2,dtype=torch.float64)))
        for value in sources:
            torch.testing.assert_close(value.grad,output.detach(),atol=0,rtol=0)

    def test_nonzero_query_independent_scalar_hand_calculation(self):
        a=PointDepthAttnRes(2).double()
        with torch.no_grad():
            a.query.copy_(torch.tensor([.7,-.3],dtype=torch.float64))
            a.norm_scale.copy_(torch.tensor([1.2,.8],dtype=torch.float64))
        values=[(3.,4.),(0.,2.)]
        scores=[(.7*1.2*x-.3*.8*y)/math.sqrt((x*x+y*y)/2+1e-6) for x,y in values]
        p=math.exp(scores[0])/(math.exp(scores[0])+math.exp(scores[1]))
        sources=[torch.tensor([[v]],dtype=torch.float64) for v in values]
        expected=torch.tensor([[[3*p,4*p+2*(1-p)]]],dtype=torch.float64)
        torch.testing.assert_close(a(sources),expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(a.source_weights(sources)[:,0,0],
            torch.tensor([p,1-p],dtype=torch.float64),atol=1e-12,rtol=1e-12)

    def check_oracle(self,device,dtype,S,N):
        tol=(1e-12,1e-10) if dtype==torch.float64 else ((1e-5,1e-4) if device=='cuda' else (1e-6,1e-5))
        gen=torch.Generator().manual_seed(902+S+N)
        sources=[torch.randn(2,N,5,generator=gen,dtype=dtype).to(device).requires_grad_() for _ in range(S)]
        refs=[x.detach().clone().requires_grad_() for x in sources]
        a=PointDepthAttnRes(5).to(device=device,dtype=dtype)
        with torch.no_grad():
            a.query.copy_(torch.randn(5,generator=gen,dtype=dtype)*.4)
            a.norm_scale.copy_(torch.randn(5,generator=gen,dtype=dtype)*.2+1)
        q=a.query.detach().clone().requires_grad_();g=a.norm_scale.detach().clone().requires_grad_()
        expected, ew=point_attnres(refs,q,g)
        actual=a(sources);aw=a.source_weights(sources).permute(1,2,0)
        row=dict(device=device,dtype=str(dtype),B=2,N=N,H=5,S=S,atol=tol[0],rtol=tol[1],errors={})
        def check(x,y,label):
            torch.testing.assert_close(x,y,atol=tol[0],rtol=tol[1],msg=label)
            self.assertTrue(torch.isfinite(x).all())
            row['errors'][label]=errors(x,y)
        check(actual,expected,'output');check(aw,ew,'weights')
        check(aw.sum(-1),torch.ones((2,N),device=device,dtype=dtype),'source_weight_sum')
        cotangent=torch.randn(actual.shape,generator=gen,dtype=dtype).to(device)
        params=[*sources,a.query,a.norm_scale];refparams=[*refs,q,g]
        grads=torch.autograd.grad((actual*cotangent).sum(),params,allow_unused=True)
        refgrads=torch.autograd.grad((expected*cotangent).sum(),refparams,allow_unused=True)
        for i,(x,y) in enumerate(zip(grads,refgrads)):
            x=torch.zeros_like(params[i]) if x is None else x
            y=torch.zeros_like(refparams[i]) if y is None else y
            check(x,y,'gradient_'+str(i))
            if S>1:self.assertGreater(x.abs().sum().item(),0)
        # One explicit SGD update verifies query/scale VJPs as parameters too.
        for name,value,refvalue,grad,refgrad in zip(('query','norm_scale'),(a.query,a.norm_scale),(q,g),grads[-2:],refgrads[-2:]):
            grad=torch.zeros_like(value) if grad is None else grad
            refgrad=torch.zeros_like(refvalue) if refgrad is None else refgrad
            check(value-.01*grad,refvalue-.01*refgrad,'step_'+name)
        self.rows.append(row)

    def test_cpu_float64_and_float32_oracle_all_inputs_parameters_and_step(self):
        for dtype in (torch.float64,torch.float32):
            for S,N in ((1,1),(2,7),(4,11)):
                with self.subTest(dtype=dtype,S=S,N=N):self.check_oracle('cpu',dtype,S,N)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_float32_oracle(self):
        old=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            for S,N in ((1,1),(2,7),(4,11)):self.check_oracle('cuda',torch.float32,S,N)
        finally:torch.backends.cuda.matmul.allow_tf32=old

    def test_source_batch_point_and_channel_permutations_and_isolation(self):
        gen=torch.Generator().manual_seed(28)
        a=PointDepthAttnRes(5).double()
        with torch.no_grad():
            a.query.copy_(torch.randn(5,generator=gen));a.norm_scale.copy_(torch.rand(5,generator=gen)+.5)
        sources=[torch.randn(2,7,5,generator=gen,dtype=torch.float64) for _ in range(3)]
        expected=a(sources);weights=a.source_weights(sources)
        order=[2,0,1]
        torch.testing.assert_close(a([sources[i] for i in order]),expected,atol=1e-12,rtol=1e-12)
        # Permuting S changes floating summation order in the softmax denominator.
        # Use the declared double threshold (observed max rounding 1.11e-16).
        torch.testing.assert_close(a.source_weights([sources[i] for i in order]),weights[order],atol=1e-12,rtol=1e-10)
        points=[6,3,1,4,0,5,2]
        torch.testing.assert_close(a([x[:,points] for x in sources]),expected[:,points],atol=0,rtol=0)
        torch.testing.assert_close(a([x.flip(0) for x in sources]),expected.flip(0),atol=0,rtol=0)
        channels=[2,0,4,3,1];permuted=PointDepthAttnRes(5).double()
        with torch.no_grad():
            permuted.query.copy_(a.query[channels]);permuted.norm_scale.copy_(a.norm_scale[channels])
        torch.testing.assert_close(permuted([x[...,channels] for x in sources]),expected[...,channels],atol=1e-12,rtol=1e-12)
        changed=[x.clone() for x in sources];changed[1][0,2]+=5
        output=a(changed)
        torch.testing.assert_close(output[1],expected[1],atol=0,rtol=0)
        torch.testing.assert_close(output[0,[0,1,3,4,5,6]],expected[0,[0,1,3,4,5,6]],atol=0,rtol=0)
        self.assertFalse(torch.equal(output[0,2],expected[0,2]))

    def test_rms_suppresses_key_magnitude_but_never_normalizes_raw_values(self):
        a=PointDepthAttnRes(2).double()
        with torch.no_grad():a.query.copy_(torch.tensor([.2,.3]))
        sources=[torch.tensor([[[3.,4.]]],dtype=torch.float64),torch.tensor([[[4.,-3.]]],dtype=torch.float64)]
        large=[50*sources[0],sources[1]]
        w=a.source_weights(large)
        torch.testing.assert_close(w,a.source_weights(sources),atol=1e-8,rtol=1e-7)
        expected=w[0].unsqueeze(-1)*large[0]+w[1].unsqueeze(-1)*large[1]
        torch.testing.assert_close(a(large),expected,atol=0,rtol=0)
        self.assertGreater(a(large).norm().item(),10*a(sources).norm().item())

    def test_double_finite_difference_gradcheck(self):
        a=PointDepthAttnRes(3).double();gen=torch.Generator().manual_seed(19)
        sources=[torch.randn(2,2,3,generator=gen,dtype=torch.float64,requires_grad=True) for _ in range(2)]
        q=torch.randn(3,generator=gen,dtype=torch.float64,requires_grad=True)
        scale=torch.randn(3,generator=gen,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y,q,g:functional_call(a,{'query':q,'norm_scale':g},([x,y],)),
            (*sources,q,scale),eps=1e-6,atol=1e-6,rtol=1e-4))

    def test_invalid_inputs_have_explicit_shape_dtype_device_errors(self):
        for hidden in (0,-1,True,2.,'3'):
            with self.assertRaises(ValueError):PointDepthAttnRes(hidden)
        a=PointDepthAttnRes(5);x=torch.randn(2,7,5)
        for sources in ([],(),None,'x',x,{},[1],[torch.empty(0,7,5)],[torch.empty(2,0,5)],
                        [torch.empty(2,7,4)],[x,torch.empty(1,7,5)],[x,torch.empty(2,8,5)],[x,torch.empty(2,7,4)]):
            with self.subTest(type=str(type(sources))),self.assertRaises(ValueError):a(sources)
        for sources in ([x.to(torch.int64)],[x,x.double()]):
            with self.assertRaisesRegex(TypeError,'dtype'):a(sources)
        with self.assertRaisesRegex(ValueError,'device'):a([x,torch.empty_like(x,device='meta')])
        with self.assertRaisesRegex(ValueError,'device'):PointDepthAttnRes(5).to('meta')([x])
        with self.assertRaisesRegex(TypeError,'dtype'):a.double()([x])
        a=PointDepthAttnRes(5);a.norm_scale.data=a.norm_scale.data.double()
        with self.assertRaisesRegex(TypeError,'dtype'):a([x])

    def test_singleton_is_exact_identity_and_calls_keep_no_history(self):
        a=PointDepthAttnRes(5).double()
        for B,N in ((2,7),(1,3),(3,1)):
            x=torch.randn(B,N,5,dtype=torch.float64,requires_grad=True)
            self.assertIs(a([x]),x)
            grads=torch.autograd.grad(a([x]).sum(),(x,a.query,a.norm_scale),allow_unused=True)
            self.assertTrue(torch.equal(grads[0],torch.ones_like(x)))
            self.assertIsNone(grads[1]);self.assertIsNone(grads[2])
        before=set(vars(a))
        with self.assertRaises(ValueError):a([])
        first=a([torch.zeros(1,2,5,dtype=torch.float64),torch.ones(1,2,5,dtype=torch.float64)])
        second=a([torch.zeros(2,1,5,dtype=torch.float64),torch.ones(2,1,5,dtype=torch.float64)*4])
        torch.testing.assert_close(first,torch.full_like(first,.5),atol=0,rtol=0)
        torch.testing.assert_close(second,torch.full_like(second,2),atol=0,rtol=0)
        self.assertEqual(set(vars(a)),before);self.assertEqual(list(a.buffers()),[])

    def test_operation_trace_has_only_source_softmax_no_point_attention(self):
        calls=[]
        class Trace(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                if any(name in str(func) for name in ('mm.','bmm.','matmul','softmax')):
                    calls.append((str(func),tuple(args[0].shape),args[1] if len(args)>1 else None))
                return func(*args,**(kwargs or {}))
        a=PointDepthAttnRes(5)
        with Trace():out=a([torch.randn(2,17,5) for _ in range(3)])
        self.assertEqual(out.shape,(2,17,5))
        self.assertEqual(len(calls),1)
        self.assertIn('softmax',calls[0][0]);self.assertEqual(calls[0][1:],((3,2,17),0))

    def test_bfloat16_autocast_accumulates_fp32_and_returns_source_dtype(self):
        a=PointDepthAttnRes(5)
        with torch.no_grad():a.query.fill_(.25)
        values=[torch.randn(2,7,5).to(torch.bfloat16).requires_grad_() for _ in range(3)]
        expected,_=point_attnres([v.float() for v in values],a.query,a.norm_scale)
        with torch.autocast('cpu',dtype=torch.bfloat16):out=a(values)
        self.assertEqual(out.dtype,torch.bfloat16)
        torch.testing.assert_close(out,expected.to(torch.bfloat16),atol=0,rtol=0)
        out.float().square().sum().backward()
        self.assertTrue(torch.isfinite(a.query.grad).all())


if __name__=='__main__':unittest.main()
