"""R3 A-only equations, isolated VJPs, symmetry, timing and baseline parity."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from linearno.attnres_reference import reference
from linearno.attention_support import errors
from linearno.test_history_core import pair, inputs, call, VARIANTS
from model.LinearNO_History import AttnResModel
from cdlno.linearno_history.attnres import LatentSummaryAttnRes, SummaryRMSNorm
from cdlno.linearno_history.models import AirfRANSAttnResModel, ShapeNetAttnResModel


def fixture(dtype=torch.float64, sources=3):
    torch.manual_seed(30101)
    module = LatentSummaryAttnRes(4, 4, feature_seed=31011).to(dtype=dtype).eval()
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p)*.3)
        r = module.receivers[str(sources)]
        r.gamma.fill_(.7)
        r.w.copy_(torch.tensor([.2, -.4, .3, .1], dtype=dtype))
        r.norm.weight.copy_(torch.tensor([.7, 1.1, .8, 1.3], dtype=dtype))
    current = torch.randn(2, 3, 3, 4, dtype=dtype, requires_grad=True)
    history = tuple(torch.randn(2, 3, 2+i, 4, dtype=dtype, requires_grad=True) for i in range(sources))
    return module, current, history


class LatentAttnResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if path := os.environ.get('LINEARNO_R3_REPORT'):
            Path(path).write_text(json.dumps(dict(scope='R3 A-only CPU synthetic',
                torch=str(torch.__version__), rows=cls.rows), indent=2))

    def close(self, a, b, metrics, name, dtype):
        metrics[name] = errors(a, b)
        tol = (1e-12, 1e-10) if dtype == torch.float64 else (1e-6, 1e-5)
        torch.testing.assert_close(a, b, atol=tol[0], rtol=tol[1], msg=name)

    def test_independent_oracle_forward_all_gradients_optimizer_and_strict_state(self):
        for dtype in (torch.float64, torch.float32):
            for all_masked in (False, True):
                module, c, histories = fixture(dtype)
                mask = torch.ones(2,3,dtype=torch.bool) if all_masked else torch.tensor([[False,True,False],[True,False,False]])
                state = {k:v.detach().clone().requires_grad_() for k,v in module.state_dict().items()}
                cr = c.detach().clone().requires_grad_()
                hr = tuple(h.detach().clone().requires_grad_() for h in histories)
                trace = []
                result = module._evaluate(3,c,histories,_drop_mask=mask,observe=trace.append)
                expected, detail = reference(cr,hr,state,3,mask)
                metrics = {}
                for name,a,b in [('output',result,expected),('H',trace[0].H,detail['H']),('alpha',trace[0].alpha,detail['alpha'])]:
                    self.close(a,b,metrics,name,dtype)
                for i,a in enumerate(trace[0].attention):
                    self.close(a,detail['attention'][i],metrics,f'token/{i}',dtype)
                    torch.testing.assert_close(a.sum(-1),torch.ones_like(a[...,0]))
                torch.testing.assert_close(trace[0].alpha.sum(-1),torch.ones_like(trace[0].alpha[...,0]))
                self.assertEqual(trace[0].alpha.shape,(2,3,3,4))
                cotangent = torch.randn_like(result)
                (result*cotangent).sum().backward(); (expected*cotangent).sum().backward()
                for name,a,b in [('current',c,cr)]+[(f'history/{i}',a,b) for i,(a,b) in enumerate(zip(histories,hr))]+[(name,p,state[name]) for name,p in module.named_parameters()]:
                    if a.grad is None or b.grad is None:
                        self.assertIs(a.grad,b.grad,name)
                    else:
                        self.close(a.grad,b.grad,metrics,'grad/'+name,dtype)
                optim = torch.optim.AdamW(module.parameters(),lr=.001)
                oracle_optim = torch.optim.AdamW(state.values(),lr=.001)
                optim.step(); oracle_optim.step()
                for name,p in module.named_parameters():
                    self.close(p,state[name],metrics,'step/'+name,dtype)
                buffer=io.BytesIO();torch.save(module.state_dict(),buffer);buffer.seek(0)
                restored=LatentSummaryAttnRes(4,4,feature_seed=1).to(dtype=dtype).eval()
                restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
                torch.testing.assert_close(restored(3,c,histories),module(3,c,histories),atol=0,rtol=0)
                self.rows.append(dict(kind='independent_oracle',dtype=str(dtype),all_masked=all_masked,metrics=metrics))

    def test_permutations_and_sample_head_isolation(self):
        module,c,h=fixture()
        mask=torch.tensor([[False,True,False],[True,False,False]])
        base=module._evaluate(3,c,h,_drop_mask=mask)
        permuted=tuple(x[:,:,torch.randperm(x.shape[2]),:] for x in h)
        torch.testing.assert_close(module._evaluate(3,c,permuted,_drop_mask=mask),base,atol=1e-12,rtol=1e-10)
        order=[2,0,1]
        torch.testing.assert_close(module._evaluate(3,c,tuple(h[i] for i in order),_drop_mask=mask[:,order]),base,atol=1e-12,rtol=1e-10)
        order=[2,0,1]
        torch.testing.assert_close(module._evaluate(3,c[:,:,order],h,_drop_mask=mask),base[:,:,order],atol=1e-12,rtol=1e-10)
        for b in range(2):
            for head in range(3):
                isolated=module._evaluate(3,c[b:b+1,head:head+1],tuple(x[b:b+1,head:head+1] for x in h),_drop_mask=mask[b:b+1])
                torch.testing.assert_close(isolated,base[b:b+1,head:head+1],atol=1e-12,rtol=1e-10)
        changed=list(h); changed[0]=h[0].clone(); changed[0][0,1] += 5
        y=module._evaluate(3,c,tuple(changed),_drop_mask=mask)
        torch.testing.assert_close(y[1],base[1],atol=0,rtol=0)
        torch.testing.assert_close(y[0,0],base[0,0],atol=0,rtol=0)
        self.assertGreater((y[0,1]-base[0,1]).abs().max().item(),0)

    def test_depth_zero_singleton_null_and_all_mask_fallback(self):
        module,c,h=fixture(sources=1);module.train()
        r=module.receivers['1']
        with torch.no_grad():r.w.zero_()
        with patch('torch.rand',side_effect=AssertionError('unexpected RNG draw')):
            before=torch.get_rng_state().clone()
            self.assertIs(module(0,c,()),c)
            trace=[];y=module(1,c,h,observe=trace.append)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(trace[0].alpha,torch.full_like(trace[0].alpha,.5),atol=0,rtol=0)
        torch.testing.assert_close(trace[0].H,trace[0].aligned[0]*.5,atol=0,rtol=0)
        module,c,h=fixture();trace=[]
        y=module._evaluate(3,c,h,_drop_mask=torch.ones(2,3,dtype=torch.bool),observe=trace.append)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y,c,atol=0,rtol=0)
        self.assertEqual(torch.count_nonzero(trace[0].H).item(),0)
        torch.testing.assert_close(trace[0].alpha[...,-1],torch.ones_like(trace[0].alpha[...,-1]),atol=0,rtol=0)
        self.assertEqual(torch.count_nonzero(trace[0].alpha[...,:-1]).item(),0)

    def test_hand_calculation_excludes_current_and_keeps_unit_residual(self):
        module=LatentSummaryAttnRes(4,1,feature_seed=5).double().eval()
        with torch.no_grad():
            module.to_k.weight.fill_(1);module.to_v.weight.fill_(1)
            r=module.receivers['2'];r.to_q.weight.zero_();r.to_o.weight.fill_(1)
            r.gamma.fill_(.5)
        c=torch.tensor([[[[5.]]]],dtype=torch.float64)
        h=(torch.tensor([[[[2.],[4.]]]],dtype=torch.float64),
           torch.tensor([[[[6.],[8.],[7.]]]],dtype=torch.float64))
        trace=[];y=module(2,c,h,observe=trace.append)
        # Each history is averaged separately: R1=3, R2=7. Null=0.
        # w=0 => alpha=(1/3,1/3,1/3), H=10/3, y=5+.5*10/3.
        torch.testing.assert_close(trace[0].H,torch.full_like(c,10/3),atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(y,torch.full_like(c,20/3),atol=1e-14,rtol=1e-14)
        self.assertEqual(trace[0].alpha.shape[-1],3)
        self.rows.append(dict(kind='hand_calculation',R=[3,7],alpha=[1/3]*3,H=trace[0].H.item(),output=y.item()))

    def test_double_finite_difference_gradcheck(self):
        module=LatentSummaryAttnRes(4,2,feature_seed=24).double().eval()
        with torch.no_grad():
            r=module.receivers['1'];r.gamma.fill_(.8);r.w.copy_(torch.tensor([.3,-.2]))
            for p in (module.to_k.weight,module.to_v.weight,r.to_q.weight,r.to_o.weight):p.mul_(10)
        names=['to_k.weight','to_v.weight','receivers.1.to_q.weight','receivers.1.to_o.weight',
               'receivers.1.norm.weight','receivers.1.w','receivers.1.gamma']
        state=module.state_dict()
        c=torch.randn(1,1,2,2,dtype=torch.float64,requires_grad=True)
        h=torch.randn(1,1,3,2,dtype=torch.float64,requires_grad=True)
        weights=tuple(state[n].detach().clone().requires_grad_() for n in names)
        def evaluate(current,history,*values):
            parameters=dict(state);parameters.update(zip(names,values))
            return torch.func.functional_call(module,parameters,(1,current,(history,)),strict=True)
        self.assertTrue(torch.autograd.gradcheck(evaluate,(c,h,*weights),eps=1e-6,atol=1e-6,rtol=1e-4))

    def test_train_dropout_sampling_timing_broadcast_and_eval_rng(self):
        module,c,h=fixture();module.train()
        with torch.no_grad():module.receivers['3'].w.zero_()
        events=[]
        handles=[module.to_k.register_forward_hook(lambda *a:events.append('K')),
                 module.to_v.register_forward_hook(lambda *a:events.append('V')),
                 module.receivers['3'].norm.register_forward_hook(lambda *a:events.append('score'))]
        uniforms=torch.tensor([[.02,.5,.7],[.8,.05,.09]])
        def sample(*shape,**kwargs):
            self.assertEqual(shape,(2,3));self.assertEqual(events,['K','V','score']*3)
            return uniforms
        traces=[]
        with patch('torch.rand',side_effect=sample) as sampler:
            y=module(3,c,h,observe=traces.append);self.assertEqual(sampler.call_count,1)
        for handle in handles:handle.remove()
        t=traces[0]
        torch.testing.assert_close(t.drop_mask,uniforms<.1)
        for b in range(2):
            kept=[i for i in range(3) if not t.drop_mask[b,i]]
            expected=sum(t.aligned[i][b] for i in kept)/(len(kept)+1)
            torch.testing.assert_close(t.H[b],expected,atol=1e-12,rtol=1e-10)
            for i in range(3):
                expected_weight=0 if t.drop_mask[b,i] else 1/(len(kept)+1)
                torch.testing.assert_close(t.alpha[b,...,i],torch.full_like(t.alpha[b,...,i],expected_weight))
        # New draw each training call, but deterministic and RNG-free in eval.
        with patch('torch.rand',return_value=torch.ones(2,3)) as sampler:
            module(3,c,h);module(3,c,h);self.assertEqual(sampler.call_count,2)
        module.eval();before=torch.get_rng_state().clone()
        with patch('torch.rand',side_effect=AssertionError('eval sampled')):
            torch.testing.assert_close(module(3,c,h),module(3,c,h),atol=0,rtol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_isolated_branch_VJPs_zero_gate_then_zero_w_then_nonzero_w(self):
        module,c,h=fixture(); r=module.receivers['3']
        with torch.no_grad():r.w.zero_();r.gamma.zero_()
        y=module(3,c,h);cotangent=torch.randn_like(y)
        grads=torch.autograd.grad((y*cotangent).sum(),tuple(module.parameters()),allow_unused=True)
        for (name,p),g in zip(module.named_parameters(),grads):
            if name.startswith('receivers.') and not name.startswith('receivers.3.'):self.assertIsNone(g);continue
            self.assertTrue(torch.isfinite(g).all(),name)
            if name.endswith('gamma'):self.assertGreater(g.abs().sum().item(),0)
            else:self.assertEqual(g.abs().sum().item(),0,name)
        for nonzero_w in (False,True):
            with torch.no_grad():
                r.gamma.fill_(.7)
                if nonzero_w:r.w.copy_(torch.tensor([.3,-.2,.4,-.1]))
            trace=[];module(3,c,h,observe=trace.append)
            # Isolate the A increment: no main residual or backbone loss path.
            value=(r.gamma*trace[0].H*cotangent).sum()
            named=[(n,p) for n,p in module.named_parameters() if n.startswith('to_') or n.startswith('receivers.3.')]
            future=torch.randn_like(c,requires_grad=True)
            targets=[p for _,p in named]+list(h)+[future]
            grads=torch.autograd.grad(value,targets,allow_unused=True)
            norms={}
            for (name,_),grad in zip(named+[(f'raw/{i}',v) for i,v in enumerate(h)],grads[:-1]):
                self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
                norms[name]=grad.abs().sum().item()
                if not nonzero_w and name.endswith('norm.weight'):self.assertEqual(norms[name],0)
                else:self.assertGreater(norms[name],0,name)
            self.assertIsNone(grads[-1])
            self.rows.append(dict(kind='isolated_increment_VJP',nonzero_w=nonzero_w,gradient_l1=norms,future_grad=None))

    def test_parameter_ownership_linear_depth_scaling_and_fair_initialization(self):
        for depth in range(4,9):
            torch.manual_seed(4321);before=torch.get_rng_state().clone()
            a=LatentSummaryAttnRes(depth,4,feature_seed=8765)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            b=LatentSummaryAttnRes(depth,4,feature_seed=8765)
            torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=0,rtol=0)
            self.assertEqual(sum(p.numel() for p in a.parameters()),2*4**2+(depth-1)*(2*4**2+2*4+1))
            self.assertNotIn('0',a.receivers)
            self.assertEqual(list(a.named_buffers()),[])
            for r in a.receivers.values():
                self.assertEqual(r.gamma.shape,torch.Size([]));self.assertEqual(r.gamma.item(),0)
                self.assertEqual(torch.count_nonzero(r.w).item(),0)
                torch.testing.assert_close(r.norm.weight,torch.ones(4),atol=0,rtol=0)
            projections=[m for m in a.modules() if isinstance(m,nn.Linear)]
            self.assertEqual(len(projections),2+2*(depth-1))
            self.assertTrue(all(m.bias is None for m in projections))
            self.assertEqual(len({id(p) for p in a.parameters()}),len(list(a.parameters())))
            for field in ('to_q','to_o','norm','w','gamma'):
                self.assertEqual(len({id(getattr(r,field)) for r in a.receivers.values()}),depth-1)
            seen=[];seen_v=[]
            handle=a.to_k.register_forward_hook(lambda m,*args:seen.append(id(m.weight)))
            handle_v=a.to_v.register_forward_hook(lambda m,*args:seen_v.append(id(m.weight)))
            c=torch.randn(2,2,3,4)
            for i in range(1,depth):a(i,c,tuple(torch.randn_like(c) for _ in range(i)))
            handle.remove();handle_v.remove()
            self.assertEqual(set(seen),{id(a.to_k.weight)})
            self.assertEqual(set(seen_v),{id(a.to_v.weight)})
            self.assertEqual(len(seen),depth*(depth-1)//2)
            self.assertEqual(len(seen_v),len(seen))
        c=LatentSummaryAttnRes(4,4,feature_seed=8766)
        self.assertFalse(torch.equal(c.to_k.weight,b.to_k.weight))

    def test_six_variant_full_A_model_zero_gate_parity_and_checkpoint(self):
        for variant in VARIANTS:
            for depth in (4,8):
                baseline,_,kw=pair(variant,layers=depth)
                baseline.eval();rng=torch.get_rng_state().clone()
                cls=AirfRANSAttnResModel if variant=='airfrans' else ShapeNetAttnResModel if variant=='shapenet' else AttnResModel
                torch.manual_seed(21002);model=cls(**kw,feature_seed=713).eval()
                # Pure construction was first; feature initialization did not advance RNG.
                self.assertTrue(torch.equal(torch.get_rng_state(),rng))
                backbone={k:v for k,v in model.state_dict().items() if not k.startswith('latent_attnres.')}
                torch.testing.assert_close(backbone,baseline.state_dict(),atol=0,rtol=0)
                weights=model.state_dict();weights.update(baseline.state_dict());model.load_state_dict(weights,strict=True)
                va=inputs(variant,kw);vb={k:v.detach().clone().requires_grad_() for k,v in va.items()}
                blocks=[];handles=[b.register_forward_hook(lambda m,i,o:blocks.append(o)) for b in baseline.blocks]
                traces=[];a=call(baseline,variant,va);b=call(model,variant,vb,observe=traces.append)
                for handle in handles:handle.remove()
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                for output,trace in zip(blocks,traces):torch.testing.assert_close(output,trace.output,atol=0,rtol=0)
                target=torch.randn_like(a);la=(a-target).square().mean();lb=(b-target).square().mean()
                torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward()
                for k in va:
                    if va[k].grad is None:self.assertIsNone(vb[k].grad)
                    else:torch.testing.assert_close(va[k].grad,vb[k].grad,atol=0,rtol=0)
                params=dict(model.named_parameters())
                for name,p in baseline.named_parameters():
                    if p.grad is None:self.assertIsNone(params[name].grad)
                    else:torch.testing.assert_close(p.grad,params[name].grad,atol=0,rtol=0)
                for receiver in model.latent_attnres.receivers.values():
                    self.assertTrue(torch.isfinite(receiver.gamma.grad))
                torch.optim.AdamW(baseline.parameters(),lr=.001).step()
                torch.optim.AdamW(model.parameters(),lr=.001).step()
                for k,v in baseline.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
                restored=cls(**kw,feature_seed=714).eval()
                stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
                restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
                torch.testing.assert_close(call(restored,variant,vb),call(model,variant,vb),atol=0,rtol=0)
                self.rows.append(dict(kind='A_model_zero_gate',variant=variant,L=depth,max_abs=0,backbone_gradient_max_abs=0,strict_roundtrip_max_abs=0))

    def test_raw_not_fused_cache_and_no_cross_forward_or_future_sources(self):
        _,_,kw=pair('plain');model=AttnResModel(**kw,feature_seed=701).eval()
        with torch.no_grad():
            for r in model.latent_attnres.receivers.values():r.gamma.fill_(100)
        for B in (2,1,2):
            traces=[];y=call(model,'plain',inputs('plain',kw,B=B),observe=traces.append)
            self.assertEqual([len(t.history) for t in traces],[0,1,2,3])
            for t in traces:
                for i,raw in enumerate(t.history):
                    self.assertIs(raw,traces[i].factors.C_raw)
                    self.assertIsNotNone(raw.grad_fn)
                    if i:self.assertIsNot(raw,traces[i].C_tilde)
            self.assertGreater((traces[1].C_tilde-traces[1].factors.C_raw).abs().sum().item(),0)
            y.square().mean().backward();model.zero_grad(set_to_none=True)
        c=traces[-1].factors.C_raw
        with self.assertRaises(ValueError):model.latent_attnres(2,c,(c,c))
        with self.assertRaises(ValueError):model.latent_attnres(2,c,traces[-1].history)
        self.assertFalse(any('history' in n or 'cache' in n for n,_ in model.named_buffers()))
        def fail(t):
            if t.index==2:raise RuntimeError('injected failure')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            call(model,'plain',inputs('plain',kw),observe=fail)
        sizes=[];call(model,'plain',inputs('plain',kw),observe=lambda t:sizes.append(len(t.history)))
        self.assertEqual(sizes,[0,1,2,3])

    def test_reconstruction_original_Q_and_only_cross_depth_token_attention(self):
        from linearno.test_attention_structure import Trace
        _,_,kw=pair('conv_temp');model=AttnResModel(**kw,feature_seed=71).double().eval()
        with torch.no_grad():
            for r in model.latent_attnres.receivers.values():r.gamma.fill_(.5)
        projected=[];events=[]
        handles=[]
        for i,block in enumerate(model.blocks):
            handles.append(block.Attn.to_out.register_forward_pre_hook(lambda m,a:projected.append(a[0])))
            handles.append(block.mlp.register_forward_hook(lambda m,a,o,i=i:events.append(('FFN',i))))
        handles.append(model.blocks[-1].mlp2.register_forward_hook(lambda *a:events.append(('head',3))))
        traces=[];dispatch=Trace()
        with dispatch:call(model,'conv_temp',inputs('conv_temp',kw,dtype=torch.float64),observe=traces.append)
        for h in handles:h.remove()
        self.assertEqual(events,[('FFN',0),('FFN',1),('FFN',2),('FFN',3),('head',3)])
        for t,projection in zip(traces,projected):
            expected=(t.factors.Q @ t.C_tilde).transpose(1,2).reshape(2,15,12)
            torch.testing.assert_close(projection,expected,atol=1e-14,rtol=1e-12)
        softmax_shapes=[shape for shape,axis,result in dispatch.softmax]
        # Six old->current edges for L4, one independent token softmax per edge.
        cross=[shape for shape in softmax_shapes if shape[-2:]==(8,8)]
        self.assertEqual(len(cross),6)
        self.assertFalse(any(shape[-2:]==(15,15) for shape in softmax_shapes))
        self.rows.append(dict(kind='execution',L=4,cross_depth_token_softmax=len(cross),
                              same_depth_self_attention=0,FFN=4,final_head=1))

    def test_invalid_config_and_no_production_p0_or_K(self):
        for kwargs in [dict(n_layers=3,d_h=4,feature_seed=1),dict(n_layers=True,d_h=4,feature_seed=1),
                       dict(n_layers=4,d_h=0,feature_seed=1),dict(n_layers=4,d_h=4,feature_seed=True)]:
            with self.assertRaises(ValueError):LatentSummaryAttnRes(**kwargs)
        with self.assertRaises(TypeError):LatentSummaryAttnRes(4,4,feature_seed=1,dropout_p=0)
        with self.assertRaises(TypeError):AttnResModel(feature_seed=1,linearno_history_k_conditioning=True)
        module,c,h=fixture()
        for index,history in [(True,()),(4,h),(2,h),(3,list(h))]:
            with self.assertRaises(ValueError):module(index,c,history)
        with self.assertRaises(ValueError):module(3,c,tuple(x[:1] for x in h))
        allowed={'to_k.weight','to_v.weight'}
        for i in (1,2,3):allowed.update(f'receivers.{i}.{name}' for name in ('to_q.weight','to_o.weight','norm.weight','w','gamma'))
        self.assertEqual(set(module.state_dict()),allowed)
        self.assertTrue(all(type(m) in (LatentSummaryAttnRes,nn.ModuleDict,nn.Linear,SummaryRMSNorm,type(module.receivers['1'])) for m in module.modules()))
        broken=dict(module.state_dict());del broken['to_k.weight']
        with self.assertRaises(RuntimeError):module.load_state_dict(broken,strict=True)

    def test_A_classes_strict_load_in_three_fresh_workdirs(self):
        from linearno.test_history_core import ROOT, STANDARD
        worker='''
import importlib,sys,torch
from torch_geometric.data import Data
torch.set_num_threads(1)
p=torch.load(sys.argv[1],weights_only=True)
module,name=p['class_path'].rsplit('.',1)
model=getattr(importlib.import_module(module),name)(**p['kwargs'],feature_seed=6).eval()
model.load_state_dict(p['state'],strict=True)
v=p['inputs'];sizes=[]
if p['variant']=='airfrans': y=model(Data(**v),observe=lambda t:sizes.append(len(t.history)))
elif p['variant']=='shapenet': y=model((Data(**v),None),observe=lambda t:sizes.append(len(t.history)))
else: y=model(v['x'],v.get('fx'),v.get('T'),observe=lambda t:sizes.append(len(t.history)))
torch.testing.assert_close(y,p['output'],atol=0,rtol=0)
assert sizes==[0,1,2,3]
print('strict A-only roundtrip exact')
'''
        for variant,cwd,cls in [('temp',STANDARD,AttnResModel),
            ('airfrans',ROOT/'Airfoil-Design-AirfRANS',AirfRANSAttnResModel),
            ('shapenet',ROOT/'Car-Design-ShapeNetCar',ShapeNetAttnResModel)]:
            _,_,kw=pair(variant);model=cls(**kw,feature_seed=5).eval()
            with torch.no_grad():
                for r in model.latent_attnres.receivers.values():r.gamma.fill_(.5);r.w.fill_(.2)
                values={k:v.detach() for k,v in inputs(variant,kw).items()}
                output=call(model,variant,values)
            with tempfile.TemporaryDirectory(prefix='linearno-r3-checkpoint-') as tmp:
                checkpoint=Path(tmp)/'synthetic-state.pt'
                path=cls.__module__+'.'+cls.__name__
                torch.save(dict(variant=variant,class_path=path,kwargs=kw,state=model.state_dict(),
                                inputs=values,output=output),checkpoint)
                result=subprocess.run([sys.executable,'-B','-c',worker,str(checkpoint)],cwd=cwd,
                    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.rows.append(dict(kind='fresh_process_A_checkpoint',variant=variant,class_path=path,
                                      cwd=str(cwd),max_abs=0))


if __name__ == '__main__':unittest.main()
