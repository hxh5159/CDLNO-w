import contextlib
import io
import random
import unittest
import numpy as np
import torch
from .primitive_oracles import adapter_einsum, latent_expanded
from .primitive_support import ERRORS, nonzero, params, values

GPU_ROWS=[]


def snapshot():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def assert_rng(test,a,b):
    test.assertEqual(a[0],b[0]);test.assertEqual(a[1][0],b[1][0]);test.assertTrue(np.array_equal(a[1][1],b[1][1]))
    test.assertEqual(a[1][2:],b[1][2:]);test.assertTrue(torch.equal(a[2],b[2]))
    test.assertEqual(len(a[3]),len(b[3]))
    for x,y in zip(a[3],b[3]):test.assertTrue(torch.equal(x,y))


class FeatureRNGTests(unittest.TestCase):
    def test_construct_and_reconstruct_do_not_advance_public_rng(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
        before=snapshot()
        a=BilateralQKLowRankAdapter(3,5,feature_seed=234)
        z=build_latent_context_ffn(6,7,feature_seed=345)
        aa=BilateralQKLowRankAdapter(3,5,feature_seed=234)
        zz=build_latent_context_ffn(6,7,feature_seed=345)
        assert_rng(self,before,snapshot())
        for first,second in ((a,aa),(z,zz)):
            for key,p in first.state_dict().items():torch.testing.assert_close(p,second.state_dict()[key],atol=0,rtol=0)
        other=build_latent_context_ffn(6,7,feature_seed=346)
        self.assertFalse(torch.equal(z.linear1.weight,other.linear1.weight))

    def test_all_four_feature_orders_preserve_following_backbone_draw(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
        before=snapshot();reference=None
        for latent,adapter in ((False,False),(True,False),(False,True),(True,True)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(72)
                first=torch.randn(7,9)
                if latent:build_latent_context_ffn(6,7,feature_seed=12)
                if adapter:BilateralQKLowRankAdapter(3,5,feature_seed=14)
                last=torch.randn(3,6)
            if reference is None:reference=(first,last)
            for a,b in zip((first,last),reference):torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert_rng(self,before,snapshot())

    def test_invalid_seed_dtype_and_failed_construction_preserve_rng(self):
        from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
        before=snapshot()
        for seed in (True,-1,2**63,1.0,None,'0'):
            with self.assertRaises((ValueError,TypeError)):build_latent_context_ffn(6,7,feature_seed=seed)
        for dtype in (torch.int64,torch.complex64):
            with self.assertRaises(TypeError):build_latent_context_ffn(6,7,feature_seed=1,dtype=dtype)
        with self.assertRaises(ValueError):build_latent_context_ffn(6,0,feature_seed=1)
        assert_rng(self,before,snapshot())


class CUDAPrimitiveTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_fp32_fp16_bf16_forward_backward_optimizer_strict_reload(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
        old_matmul=torch.backends.cuda.matmul.allow_tf32;old_cudnn=torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            for amp in (None,torch.float16,torch.bfloat16):
                if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    GPU_ROWS.append(dict(dtype=str(amp),status='SKIP',reason='BF16 unsupported'));continue
                for kind in ('adapter','latent'):
                    with self.subTest(amp=amp,kind=kind):
                        autocast=lambda:torch.autocast('cuda',dtype=amp) if amp else contextlib.nullcontext()
                        before=snapshot()
                        create=(lambda:BilateralQKLowRankAdapter(3,5,rank=2,alpha=3.,feature_seed=4,device='cuda')) if kind=='adapter' else (lambda:build_latent_context_ffn(6,7,feature_seed=5,device='cuda'))
                        module=create();assert_rng(self,before,snapshot())
                        # Zero-feature identity at the actual base-logit dtype.
                        x=values((2,2,7,3),dtype=torch.float32,device='cuda')
                        with autocast():
                            base=torch.nn.functional.linear(x,values((5,3),dtype=torch.float32,device='cuda'))
                            if kind=='adapter':
                                zero=module(x)
                                for delta in zero:
                                    self.assertEqual(delta.dtype,base.dtype)
                                    torch.testing.assert_close(base+delta,base,atol=0,rtol=0)
                            else:
                                context=x.to(amp or torch.float32)
                                torch.testing.assert_close(module(context),context,atol=0,rtol=0)
                        module=nonzero(module);x=x.to(amp or torch.float32).requires_grad_()
                        p={k:v.detach().double().requires_grad_() for k,v in module.named_parameters()}
                        xr=x.detach().double().requires_grad_()
                        reference=tuple(adapter_einsum(xr,p['A_'+s],p['B_'+s],3.) for s in ('q','k')) if kind=='adapter' else (latent_expanded(xr,p),)
                        with autocast():
                            result=module(x);result=result if isinstance(result,tuple) else (result,)
                            loss=sum((v.float()-.27).square().mean() for v in result)
                        tol=(4e-6,5e-5) if amp is None else (4e-3,3e-2) if amp==torch.float16 else (4e-2,8e-2)
                        maximum=0.
                        for actual,expected in zip(result,reference):
                            self.assertEqual(actual.dtype,amp or torch.float32)
                            maximum=max(maximum,float((actual.detach().double()-expected.detach()).abs().max()))
                            torch.testing.assert_close(actual.double(),expected,atol=tol[0],rtol=tol[1])
                        gradients=torch.autograd.grad(loss,(x,*module.parameters()))
                        ref_loss=sum((v-.27).square().mean() for v in reference)
                        ref_gradients=torch.autograd.grad(ref_loss,(xr,*p.values()))
                        grad_max=0.
                        for a,b in zip(gradients,ref_gradients):
                            self.assertTrue(torch.isfinite(a).all());self.assertGreater(a.abs().sum().item(),0)
                            grad_max=max(grad_max,float((a.detach().double()-b.detach()).abs().max()))
                            torch.testing.assert_close(a.double(),b,atol=tol[0],rtol=tol[1])
                        optimizer=torch.optim.AdamW(module.parameters(),lr=.001)
                        for parameter,gradient in zip(module.parameters(),gradients[1:]):parameter.grad=gradient
                        optimizer.step()
                        saved=io.BytesIO();torch.save(module.state_dict(),saved);saved.seek(0)
                        reloaded=create();reloaded.load_state_dict(torch.load(saved,weights_only=True,map_location='cuda'),strict=True)
                        with autocast():
                            a=module(x.detach());b=reloaded(x.detach())
                        if kind=='latent':a,b=(a,),(b,)
                        for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=0,rtol=0)
                        GPU_ROWS.append(dict(kind=kind,dtype=str(amp or torch.float32),status='PASS',forward_max_abs=maximum,
                                             vjp_max_abs=grad_max,atol=tol[0],rtol=tol[1],reload_max_abs=0.))
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cross_device_rejection_and_default_device_does_not_consume_cuda_rng(self):
        from cdlno.linearno_loop.v3.adapter import BilateralQKLowRankAdapter
        from cdlno.linearno_loop.v3.latent import build_latent_context_ffn
        module=BilateralQKLowRankAdapter(3,5,feature_seed=4)
        with self.assertRaisesRegex(ValueError,'device'):module(torch.ones(1,2,3,3,device='cuda'))
        before=snapshot()
        with torch.device('cuda'):
            a=BilateralQKLowRankAdapter(3,5,feature_seed=4)
            z=build_latent_context_ffn(6,7,feature_seed=5)
        self.assertEqual(a.A_q.device.type,'cpu');self.assertEqual(z.linear1.weight.device.type,'cpu')
        assert_rng(self,before,snapshot())
