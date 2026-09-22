import contextlib
import unittest
from unittest.mock import patch

import torch

from cdlno.linearno_loop.core import LinearNOLoopCore
from .core_support import GPU_ROWS,MODES,TASKS,Trace,configuration,excite,inputs,make


class AMPTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_modes_six_variants_fp32_fp16_bf16_raw_history_and_off_parity(self):
        for amp in (None,torch.float16,torch.bfloat16):
            if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported():continue
            cast=lambda:torch.autocast('cuda',dtype=amp) if amp else contextlib.nullcontext()
            for task in TASKS:
                for mode in MODES:
                    with self.subTest(amp=amp,task=task,mode=mode):
                        c=configuration(task,mode=mode);m=excite(make(c,dtype=torch.float32,device='cuda'))
                        self.assertIs(type(m)._rb_receive,LinearNOLoopCore._rb_receive)
                        raw_routes=[];original=LinearNOLoopCore._rb_receive
                        def receive(receiver,sources,anchor):
                            result=original(receiver,sources,anchor)
                            raw_routes.append((tuple(sources),anchor,result));return result
                        x=inputs(c,dtype=torch.float32,device='cuda')
                        with patch.object(type(m),'_rb_receive',staticmethod(receive)),Trace(m) as t,cast():y=m(x)
                        self.assertTrue(torch.isfinite(y).all())
                        if mode=='rb_attnres':
                            self.assertEqual(len(raw_routes),2*m.recurrent_core_blocks*m.loop_repeats+1)
                            for (sources,anchor,result),seen in zip(raw_routes,t.routes):
                                self.assertTrue(all(v.dtype==anchor.dtype for v in seen['sources']))
                                for raw,local in zip(sources,seen['sources']):
                                    torch.testing.assert_close(local,raw.to(anchor.dtype),atol=0,rtol=0)
                                    if raw.dtype==anchor.dtype:self.assertIs(raw,local)
                            C=m.recurrent_core_blocks
                            for r in range(m.loop_repeats):
                                terms=[z['u'] for z in t.steps[r*2*C:(r+1)*2*C]]
                                total=sum(terms[1:],terms[0])
                                summary=raw_routes[(r+1)*2*C][0][-1]
                                self.assertEqual(summary.dtype,terms[0].dtype)
                                torch.testing.assert_close(summary,total,atol=0,rtol=0)
                            first_gradient=torch.autograd.grad(y.float().square().mean(),t.steps[0]['u'],retain_graph=True)[0]
                            self.assertTrue(torch.isfinite(first_gradient).all());self.assertGreater(first_gradient.abs().max().item(),0)
                        y.float().square().mean().backward()
                        for name,p in m.named_parameters():
                            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
                        torch.optim.AdamW(m.parameters(),lr=.001).step()
                        twin=make(c,dtype=torch.float32,device='cuda');twin.load_configured_state_dict(m.state_dict(),saved_config=c)
                        with cast():a,b=m(x.detach()),twin(x.detach())
                        torch.testing.assert_close(a,b,atol=0,rtol=0)
                        # Exact old arithmetic, including AMP RB source cast and dropout sequence.
                        co=configuration(task,mode=mode,latent=False,adapter=False,dropout=.2)
                        off=excite(make(co,dtype=torch.float32,device='cuda'));v1=excite(make(co,v1=True,dtype=torch.float32,device='cuda'))
                        xx=inputs(co,dtype=torch.float32,device='cuda');xr=xx.detach().clone().requires_grad_()
                        rng=torch.cuda.get_rng_state()
                        with cast():a=off(xx)
                        end=torch.cuda.get_rng_state();torch.cuda.set_rng_state(rng)
                        with cast():b=v1(xr)
                        torch.testing.assert_close(a,b,atol=0,rtol=0);self.assertTrue(torch.equal(end,torch.cuda.get_rng_state()))
                        ga=torch.autograd.grad(a.float().square().mean(),(xx,*off.parameters()),allow_unused=True)
                        gb=torch.autograd.grad(b.float().square().mean(),(xr,*v1.parameters()),allow_unused=True)
                        for u,v in zip(ga,gb):
                            if u is None:self.assertIsNone(v)
                            else:torch.testing.assert_close(u,v,atol=0,rtol=0)
                        GPU_ROWS.append(dict(task=task,mode=mode,dtype=str(amp or torch.float32),status='PASS',reload_max_abs=0.,off_max_abs=0.,off_vjp_max_abs=0.,
                            raw_branch_dtypes=sorted({str(s['u'].dtype) for s in t.steps}),
                            receiver_dtypes=sorted({str(s.dtype) for r in t.routes for s in r['sources']})))
