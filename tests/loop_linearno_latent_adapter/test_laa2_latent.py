import copy
import io
import unittest
import weakref
import torch
from torch.func import functional_call
from .primitive_oracles import latent_expanded
from .primitive_support import close, nonzero, params, values


def make(hidden=6, inner_width=7, **kwargs):
    from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
    return build_latent_context_ffn(hidden,inner_width,feature_seed=123,**kwargs)


class LatentTests(unittest.TestCase):
    def test_exact_v2_class_keys_initialization_and_outputs(self):
        from cdlno.linearno_loop.v2.latent import LatentContextFFN
        for dtype in (torch.float64,torch.float32):
            actual=make(dtype=dtype)
            self.assertIs(type(actual),LatentContextFFN)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(123)
                old=LatentContextFFN(6,7);old.initialize_release_identity();old=old.to(dtype=dtype)
            self.assertEqual(set(actual.state_dict()),set(old.state_dict()))
            for key,p in old.state_dict().items():torch.testing.assert_close(actual.state_dict()[key],p,atol=0,rtol=0)
            x=values((2,3,5,2),dtype=dtype).requires_grad_()
            torch.testing.assert_close(actual(x),old(x),atol=0,rtol=0)
            torch.testing.assert_close(actual(x),x,atol=0,rtol=0)
            ga=torch.autograd.grad(actual(x).square().sum(),(x,*actual.parameters()))
            go=torch.autograd.grad(old(x).square().sum(),(x,*old.parameters()))
            for a,b in zip(ga,go):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_parameter_formula_norm_gelu_no_dropout_and_instance_independence(self):
        for H,Dz in ((6,7),(104,704),(208,776)):
            a,b=make(H,Dz),make(H,Dz)
            self.assertEqual(sum(p.numel() for p in a.parameters()),Dz*(2*H+1)+3*H)
            self.assertEqual(a.norm.eps,1e-5)
            self.assertIsInstance(a.activation,torch.nn.GELU)
            self.assertEqual(a.activation.approximate,'none')
            self.assertFalse(any(isinstance(m,torch.nn.Dropout) for m in a.modules()))
            self.assertFalse(list(a.buffers()))
            self.assertEqual(a.linear2.weight.count_nonzero().item(),0)
            self.assertEqual(a.linear2.bias.count_nonzero().item(),0)
            torch.testing.assert_close(a.norm.weight,torch.ones_like(a.norm.weight),atol=0,rtol=0)
            torch.testing.assert_close(a.norm.bias,torch.zeros_like(a.norm.bias),atol=0,rtol=0)
            for p,q in zip(a.parameters(),b.parameters()):self.assertNotEqual(p.data_ptr(),q.data_ptr())

    def test_expanded_oracle_forward_and_all_vjps(self):
        for dtype in (torch.float64,torch.float32):
            module=nonzero(make(dtype=dtype));p=params(module)
            # Noncontiguous head layout, distinct values across head/channel/token.
            x=values((2,5,3,2),dtype=dtype).transpose(1,2).requires_grad_()
            xr=x.detach().clone().requires_grad_()
            actual=module(x);expected=latent_expanded(xr,p)
            close(actual,expected,label='latent-forward',dtype=dtype)
            vector=values(actual.shape,dtype=dtype,offset=.7)
            ga=torch.autograd.grad((actual*vector).sum(),(x,*module.parameters()))
            ge=torch.autograd.grad((expected*vector).sum(),(xr,*p.values()))
            for i,(a,b) in enumerate(zip(ga,ge)):
                close(a,b,label='latent-vjp-'+str(i),dtype=dtype)
                self.assertTrue(torch.isfinite(a).all());self.assertGreater(a.abs().sum().item(),0)

    def test_finite_difference_all_inputs_and_parameters(self):
        module=nonzero(make(4,3,dtype=torch.float64));p=params(module);names=tuple(p)
        x=values((1,2,2,2)).requires_grad_()
        def f(x,*weights):return functional_call(module,dict(zip(names,weights)),(x,))
        self.assertTrue(torch.autograd.gradcheck(f,(x,*p.values()),eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_no_token_mixing_permutation_and_batch_isolation(self):
        module=nonzero(make(dtype=torch.float64));x=values((2,3,5,2))
        actual=module(x);order=torch.tensor([3,1,4,0,2])
        close(module(x[:,:,order]),actual[:,:,order],label='latent-token-permutation',dtype=x.dtype)
        for b in range(2):
            for token in range(5):
                close(module(x[b:b+1,:,token:token+1]),actual[b:b+1,:,token:token+1],label='latent-token-isolation',dtype=x.dtype)
        changed=x.clone();changed[0,:,0]+=values((3,2),offset=1.)
        after=module(changed)
        torch.testing.assert_close(after[1],actual[1],atol=0,rtol=0)
        torch.testing.assert_close(after[0,:,1:],actual[0,:,1:],atol=0,rtol=0)
        self.assertFalse(torch.equal(after[0,:,0],actual[0,:,0]))

    def test_zero_init_then_nonzero_gradients_after_one_adamw_step(self):
        for dtype in (torch.float64,torch.float32):
            module=make(dtype=dtype);x=values((2,3,5,2),dtype=dtype).requires_grad_()
            opt=torch.optim.AdamW(module.parameters(),lr=.01,weight_decay=0.)
            actual=module(x);torch.testing.assert_close(actual,x,atol=0,rtol=0)
            loss=(actual-.23).square().mean();loss.backward()
            close(x.grad,2*(x-.23)/x.numel(),label='latent-identity-input-gradient',dtype=dtype)
            for name,p in module.named_parameters():
                if name.startswith('linear2'):self.assertGreater(p.grad.abs().sum().item(),0)
                else:self.assertEqual(p.grad.count_nonzero().item(),0)
            opt.step();opt.zero_grad(set_to_none=True)
            (module(x)-.23).square().mean().backward()
            for p in module.parameters():
                self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)

    def test_invalid_shapes_dtype_nonfinite_and_exception_recovery(self):
        for H,Dz in ((0,2),(2,0),(True,2),(2,False),(2.0,2),(2,'3')):
            with self.assertRaises(ValueError):make(H,Dz)
        module=nonzero(make(dtype=torch.float64))
        for x in (None,torch.ones(1,3,6),torch.ones(1,3,2,3),torch.ones(0,3,2,2),torch.ones(1,3,0,2)):
            with self.assertRaises(ValueError):module(x)
        for dtype in (torch.int64,torch.complex64):
            with self.assertRaises(TypeError):module(torch.ones(1,3,2,2,dtype=dtype))
        for bad in (float('nan'),float('inf'),float('-inf')):
            x=values((2,3,5,2));x[0,0,0,0]=bad
            self.assertFalse(torch.isfinite(module(x)).all())
        self.assertTrue(torch.isfinite(module(values((2,3,5,2)))).all())

    def test_strict_reload_no_graph_history_or_rng_change(self):
        module=nonzero(make(dtype=torch.float64));state=copy.deepcopy(module.state_dict())
        before=torch.get_rng_state().clone();x=values((1,3,2,2)).requires_grad_();ref=weakref.ref(x)
        result=module(x);result.sum().backward();del result,x
        self.assertIsNone(ref());self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for _ in range(2):
            x=values((2,3,4,2)).requires_grad_();module(x).square().sum().backward();self.assertIsNotNone(x.grad)
        storage=io.BytesIO();torch.save(state,storage);storage.seek(0)
        new=make(dtype=torch.float64);new.load_state_dict(torch.load(storage,weights_only=True),strict=True)
        torch.testing.assert_close(module(x),new(x),atol=0,rtol=0)
        for key,p in module.state_dict().items():torch.testing.assert_close(p,state[key],atol=0,rtol=0)
        bad=dict(state);bad.pop('linear1.weight')
        with self.assertRaises(RuntimeError):new.load_state_dict(bad,strict=True)
