import copy
import io
import unittest
import weakref
import torch
from torch.func import functional_call
from .primitive_oracles import adapter_indexed, adapter_einsum, adapter_merged_matrix
from .primitive_support import close, nonzero, params, values


def make(**kwargs):
    from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
    return BilateralQKLowRankAdapter(**dict(head_dim=3, latent_tokens=5, rank=2,
                                          alpha=3., feature_seed=123, **kwargs))


class AdapterTests(unittest.TestCase):
    def test_default_initialization_parameter_formula_and_per_position_ownership(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        for dh, M in ((3, 5), (26, 32), (16, 64)):
            a = BilateralQKLowRankAdapter(dh, M, feature_seed=100)
            b = BilateralQKLowRankAdapter(dh, M, feature_seed=101)
            self.assertEqual((a.rank, a.alpha, a.scale), (4, 4., 1.))
            self.assertEqual(set(a.state_dict()), {'A_q','B_q','A_k','B_k'})
            self.assertEqual(sum(p.numel() for p in a.parameters()), 2*4*(dh+M))
            self.assertEqual(len(list(a.buffers())), 0)
            for side in ('q','k'):
                A, B = getattr(a,'A_'+side), getattr(a,'B_'+side)
                self.assertEqual(tuple(A.shape), (4,dh)); self.assertEqual(tuple(B.shape), (M,4))
                self.assertGreater(A.count_nonzero().item(), 0); self.assertEqual(B.count_nonzero().item(), 0)
                self.assertLessEqual(A.abs().max().item(), dh**-.5)
                self.assertNotEqual(A.data_ptr(), getattr(b,'A_'+side).data_ptr())
                self.assertFalse(torch.equal(A, getattr(b,'A_'+side)))
            self.assertFalse(torch.equal(a.A_q,a.A_k))

    def test_forward_and_all_parameter_input_vjps_against_independent_oracles(self):
        for dtype in (torch.float64,torch.float32):
            module=nonzero(make(dtype=dtype))
            x=values((2,3,4,3),dtype=dtype).requires_grad_()
            actual=module(x)
            parameters=params(module)
            xref=x.detach().clone().requires_grad_()
            expected=tuple(adapter_einsum(xref,parameters['A_'+s],parameters['B_'+s],3.) for s in ('q','k'))
            for side, output, want in zip(('q','k'),actual,expected):
                close(output,want,label='adapter-forward-'+side,dtype=dtype)
                close(output,adapter_indexed(xref,parameters['A_'+side],parameters['B_'+side],3.),label='adapter-index-'+side,dtype=dtype)
                close(output,adapter_merged_matrix(xref,parameters['A_'+side],parameters['B_'+side],3.),label='adapter-merged-'+side,dtype=dtype)
            vectors=[values(o.shape,dtype=dtype,offset=.31+i) for i,o in enumerate(actual)]
            actual_grads=torch.autograd.grad(sum((o*v).sum() for o,v in zip(actual,vectors)),(x,*module.parameters()))
            reference_grads=torch.autograd.grad(sum((o*v).sum() for o,v in zip(expected,vectors)),(xref,*parameters.values()))
            for i,(ga,ge) in enumerate(zip(actual_grads,reference_grads)):
                close(ga,ge,label='adapter-vjp-'+str(i),dtype=dtype)
                self.assertTrue(torch.isfinite(ga).all());self.assertGreater(ga.abs().sum().item(),0)

    def test_finite_difference_all_inputs_and_parameters(self):
        module=nonzero(make(dtype=torch.float64))
        x=values((1,1,2,3)).requires_grad_();p=params(module);names=tuple(p)
        def f(x,*weights):return functional_call(module,dict(zip(names,weights)),(x,))
        self.assertTrue(torch.autograd.gradcheck(f,(x,*p.values()),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_zero_B_identity_and_two_step_gradient_startup(self):
        for dtype in (torch.float64,torch.float32):
            module=make(dtype=dtype);x=values((2,3,4,3),dtype=dtype).requires_grad_()
            base=[values((2,3,4,5),dtype=dtype,offset=i) for i in (1.,2.)]
            opt=torch.optim.AdamW(module.parameters(),lr=.01,weight_decay=0.)
            outputs=module(x)
            for b,u in zip(base,outputs):torch.testing.assert_close(b+u,b,atol=0,rtol=0)
            loss=sum(((b+u)-.15).square().mean() for b,u in zip(base,outputs));loss.backward()
            for side in ('q','k'):
                self.assertEqual(getattr(module,'A_'+side).grad.count_nonzero().item(),0)
                self.assertGreater(getattr(module,'B_'+side).grad.abs().sum().item(),0)
            self.assertEqual(x.grad.count_nonzero().item(),0)
            opt.step();opt.zero_grad(set_to_none=True);x.grad=None
            sum(((b+u)-.15).square().mean() for b,u in zip(base,module(x))).backward()
            for parameter in module.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(parameter.grad.abs().sum().item(),0)
            self.assertGreater(x.grad.abs().sum().item(),0)

    def test_head_sharing_batch_point_isolation_and_noncontiguous_layout(self):
        module=nonzero(make(dtype=torch.float64))
        x=values((2,7,3,3)).transpose(1,2);self.assertFalse(x.is_contiguous())
        q,k=module(x)
        self.assertEqual(q.shape,(2,3,7,5))
        for b in range(2):
            for h in range(3):
                for n in range(7):
                    qq,kk=module(x[b:b+1,h:h+1,n:n+1])
                    close(qq,q[b:b+1,h:h+1,n:n+1],label='adapter-isolation-q',dtype=x.dtype)
                    close(kk,k[b:b+1,h:h+1,n:n+1],label='adapter-isolation-k',dtype=x.dtype)
        same=x[:,:1].expand(-1,4,-1,-1);a,_=module(same)
        for h in range(4):
            close(a[:,h],a[:,0],label='adapter-identical-heads',dtype=x.dtype)
        changed=x.clone();changed[0,0,0]+=3
        out,_=module(changed);torch.testing.assert_close(out[1],q[1],atol=0,rtol=0)
        torch.testing.assert_close(out[0,1:],q[0,1:],atol=0,rtol=0)
        torch.testing.assert_close(out[0,0,1:],q[0,0,1:],atol=0,rtol=0)
        before=k.clone()
        with torch.no_grad():module.B_q.add_(1.)
        torch.testing.assert_close(module(x)[1],before,atol=0,rtol=0)

    def test_validation_types_shape_device_and_nonfinite_propagation(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        valid=dict(head_dim=3,latent_tokens=5,feature_seed=4)
        for field in ('head_dim','latent_tokens','rank','feature_seed'):
            for bad in (True,1.0,'2',-1,None):
                with self.subTest(field=field,bad=bad),self.assertRaises((TypeError,ValueError)):
                    BilateralQKLowRankAdapter(**{**valid,field:bad})
        for field in ('head_dim','latent_tokens','rank'):
            with self.assertRaises(ValueError):BilateralQKLowRankAdapter(**{**valid,field:0})
        for bad in (0,-1,True,'4',float('inf'),float('-inf'),float('nan')):
            with self.assertRaises((TypeError,ValueError)):BilateralQKLowRankAdapter(**valid,alpha=bad)
        with self.assertRaises(ValueError):BilateralQKLowRankAdapter(**{**valid,'feature_seed':2**63})
        for dtype in (torch.int64,torch.bool,torch.complex64):
            with self.assertRaises(TypeError):BilateralQKLowRankAdapter(**valid,dtype=dtype)
        module=nonzero(make())
        for bad in (None,[],torch.empty(1,2,3),torch.empty(1,2,3,4),torch.empty(0,2,3,3),
                    torch.empty(1,0,3,3),torch.empty(1,2,0,3)):
            with self.subTest(shape=getattr(bad,'shape',None)),self.assertRaises((TypeError,ValueError)):module(bad)
        for dtype in (torch.int64,torch.bool,torch.complex64,torch.float64):
            with self.assertRaises(TypeError):module(torch.ones(1,2,3,3,dtype=dtype))
        with self.assertRaisesRegex(ValueError,'device'):module(torch.ones(1,2,3,3,device='meta'))
        for bad in (float('nan'),float('inf'),float('-inf')):
            x=values((1,2,4,3),dtype=torch.float32);x[0,0,0,0]=bad
            self.assertTrue(any(not torch.isfinite(y).all() for y in module(x)))
        for output in module(values((1,2,4,3),dtype=torch.float32)):self.assertTrue(torch.isfinite(output).all())

    def test_strict_reload_no_cached_graph_or_forward_rng(self):
        module=nonzero(make(dtype=torch.float64));x=values((2,2,5,3)).requires_grad_()
        state=copy.deepcopy(module.state_dict());before=torch.get_rng_state().clone()
        output=module(x);sum(y.square().sum() for y in output).backward()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for name,p in module.state_dict().items():torch.testing.assert_close(p,state[name],atol=0,rtol=0)
        for _ in range(2):
            y=values((1,3,7,3),offset=.2).requires_grad_();sum(o.sum() for o in module(y)).backward()
            self.assertIsNotNone(y.grad)
        saved=io.BytesIO();torch.save(state,saved);saved.seek(0)
        restored=make(dtype=torch.float64);restored.load_state_dict(torch.load(saved,weights_only=True),strict=True)
        for a,b in zip(module(x),restored(x)):torch.testing.assert_close(a,b,atol=0,rtol=0)
        bad=dict(state);bad.pop('A_q')
        with self.assertRaises(RuntimeError):restored.load_state_dict(bad,strict=True)
        bad=dict(state,gate=torch.ones(1))
        with self.assertRaises(RuntimeError):restored.load_state_dict(bad,strict=True)
        self.assertFalse(any(isinstance(v,torch.Tensor) for k,v in vars(module).items() if not k.startswith('_')))
        temp=values((1,1,2,3)).requires_grad_();ref=weakref.ref(temp);out=module(temp)
        del temp,out
        self.assertIsNone(ref())
