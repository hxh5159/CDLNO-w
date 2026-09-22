import contextlib
import unittest

import torch

from cdlno.linearno.attention import LinearNOAttention
from .attention_oracle import attention_reference
from .attention_support import CUDA_ROWS, VARIANTS, clone_parameters, compare, fill, kwargs, make, sample


class AttentionAMPTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_six_variants_four_ablations_two_visits_three_precisions(self):
        old_mm=torch.backends.cuda.matmul.allow_tf32;old_conv=torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            for amp in (None,torch.float16,torch.bfloat16):
                if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    CUDA_ROWS.append(dict(dtype=str(amp),status='SKIP',reason='BF16 unsupported'));continue
                cast=lambda:torch.autocast('cuda',dtype=amp) if amp else contextlib.nullcontext()
                tol=(4e-6,5e-5) if amp is None else (4e-3,3e-2) if amp==torch.float16 else (4e-2,8e-2)
                for variant in VARIANTS:
                    for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                        for index in (0,1):
                            with self.subTest(dtype=amp,variant=variant,z=latent,a=adapter,r=index):
                                m=fill(make(variant,latent,adapter,device='cuda',dtype=torch.float32))
                                x=sample(torch.float32,'cuda').requires_grad_();xr=x.detach().double().requires_grad_()
                                p=clone_parameters(m,torch.float64)
                                expected,_=attention_reference(xr,p,variant=variant,heads=2,round_index=index,
                                    latent=latent,adapter=adapter,alpha=3.,grid=(2,3))
                                with cast():
                                    actual=m(x,round_index=index);loss=(actual.float()-.37).square().mean()
                                self.assertEqual(actual.dtype,amp or torch.float32)
                                compare(actual,expected,'cuda/'+variant,atol=tol[0],rtol=tol[1])
                                grads=torch.autograd.grad(loss,(x,*m.parameters()),allow_unused=True)
                                refs=torch.autograd.grad((expected-.37).square().mean(),(xr,*p.values()),allow_unused=True)
                                maximum=0.
                                for name,g,r in zip(('input',*p),grads,refs):
                                    self.assertEqual(g is None,r is None,name)
                                    if g is not None:
                                        self.assertTrue(torch.isfinite(g).all(),name)
                                        compare(g,r,'cuda/vjp/'+name,atol=tol[0],rtol=tol[1])
                                        maximum=max(maximum,float((g.double()-r).abs().max()))
                                opt=torch.optim.AdamW(m.parameters(),lr=.001)
                                for parameter,gradient in zip(m.parameters(),grads[1:]):parameter.grad=gradient
                                opt.step()
                                twin=make(variant,latent,adapter,device='cuda',dtype=torch.float32)
                                twin.load_state_dict(m.state_dict(),strict=True)
                                with cast():a,b=m(x.detach(),round_index=index),twin(x.detach(),round_index=index)
                                torch.testing.assert_close(a,b,atol=0,rtol=0)
                                CUDA_ROWS.append(dict(variant=variant,latent=latent,adapter=adapter,round_index=index,
                                    dtype=str(amp or torch.float32),status='PASS',forward_max_abs=float((actual.detach().double()-expected.detach()).abs().max()),
                                    vjp_max_abs=maximum,atol=tol[0],rtol=tol[1],reload_max_abs=0.))
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_mm;torch.backends.cudnn.allow_tf32=old_conv

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_zero_init_and_native_dropout_rng(self):
        for amp in (None,torch.float16,torch.bfloat16):
            if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported():continue
            cast=lambda:torch.autocast('cuda',dtype=amp) if amp else contextlib.nullcontext()
            for variant in VARIANTS:
                for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                    for index in (0,1):
                        with self.subTest(dtype=amp,variant=variant,z=latent,a=adapter,r=index):
                            m=make(variant,latent,adapter,device='cuda',dtype=torch.float32,dropout=.25)
                            native=LinearNOAttention(**kwargs(variant,.25)).cuda()
                            native.load_state_dict({k:v for k,v in m.state_dict().items()
                                if not k.startswith(('adapter.','latent_processor.'))},strict=True)
                            x=sample(torch.float32,'cuda').requires_grad_();xr=x.detach().clone().requires_grad_()
                            before=torch.cuda.get_rng_state()
                            with cast():a=native(x)
                            after=torch.cuda.get_rng_state();torch.cuda.set_rng_state(before)
                            with cast():b=m(xr,round_index=index)
                            self.assertTrue(torch.equal(after,torch.cuda.get_rng_state()))
                            torch.testing.assert_close(a,b,atol=0,rtol=0)
                            ga=torch.autograd.grad(a.float().square().sum(),(x,*native.parameters()),allow_unused=True)
                            gb=torch.autograd.grad(b.float().square().sum(),(xr,*[dict(m.named_parameters())[k] for k in dict(native.named_parameters())]),allow_unused=True)
                            for l,r in zip(ga,gb):
                                if l is None:self.assertIsNone(r)
                                elif not latent and not adapter:torch.testing.assert_close(l,r,atol=0,rtol=0)
                                else:torch.testing.assert_close(l,r,atol=4e-6 if amp is None else 4e-3,rtol=5e-5 if amp is None else .03)


if __name__=='__main__':unittest.main()
