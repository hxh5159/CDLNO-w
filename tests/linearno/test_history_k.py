"""R4 independent K-only mathematics, isolated gradients and task variants."""
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

from linearno.history_k_reference import reference, ln0 as reference_ln0
from linearno.attention_support import errors
from linearno.test_history_core import pair, inputs, call, VARIANTS, ROOT, STANDARD
from linearno.test_attention_structure import Trace
from model.LinearNO_History import HistoryKModel
from cdlno.linearno_history.models import AirfRANSHistoryKModel, ShapeNetHistoryKModel
from cdlno.linearno_history.history_k import HistoryConditionedK, ln0
from cdlno.linearno_history.core import attention_factors


class MatrixTrace(Trace):
    """Capture mm as well as bmm: broadcasting a shared 2D query can use mm."""
    def __init__(self):
        super().__init__();self.matrix=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=super().__torch_dispatch__(func,types,args,kwargs)
        if func in (torch.ops.aten.mm.default,torch.ops.aten.bmm.default):
            self.matrix.append((str(func),tuple(args[0].shape),tuple(args[1].shape),tuple(result.shape)))
        return result


def fixture(dtype=torch.float64):
    torch.manual_seed(40103)
    module=HistoryConditionedK(4,3,4,feature_seed=41213).to(dtype=dtype).eval()
    with torch.no_grad():
        for p in (module.uq.weight,module.uk.weight,module.uv.weight):p.copy_(torch.randn_like(p)*.3)
        module.raw_gates['3'].copy_(torch.tensor([.4,-.7,.2],dtype=dtype).reshape(1,3,1,1))
    Z=torch.randn(2,3,7,4,dtype=dtype,requires_grad=True)
    W=torch.randn(5,4,dtype=dtype,requires_grad=True)
    base=torch.randn(2,3,7,5,dtype=dtype,requires_grad=True)
    history=tuple(torch.randn(2,3,2+i,4,dtype=dtype,requires_grad=True) for i in range(3))
    return module,Z,base,W,history


def model_class(variant):
    return AirfRANSHistoryKModel if variant=='airfrans' else ShapeNetHistoryKModel if variant=='shapenet' else HistoryKModel


class HistoryKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1);cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if path:=os.environ.get('LINEARNO_R4_REPORT'):
            Path(path).write_text(json.dumps(dict(scope='R4 K-only CPU synthetic',torch=str(torch.__version__),rows=cls.rows),indent=2))

    def close(self,a,b,metrics,name,dtype):
        self.assertTrue(torch.isfinite(a).all(),name)
        metrics[name]=errors(a,b)
        tol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
        torch.testing.assert_close(a,b,atol=tol[0],rtol=tol[1],msg=name)

    def test_independent_oracle_all_gradients_step_and_strict_state(self):
        for dtype in (torch.float64,torch.float32):
            module,Z,base,W,history=fixture(dtype)
            state={k:v.detach().clone().requires_grad_() for k,v in module.state_dict().items()}
            zr,br,wr=[v.detach().clone().requires_grad_() for v in (Z,base,W)]
            hr=tuple(v.detach().clone().requires_grad_() for v in history)
            traces=[];actual=module(3,Z,base,W,history,observe=traces.append)
            expected,detail=reference(3,zr,br,wr,hr,state);metrics={}
            self.close(actual,expected,metrics,'combined_logits',dtype)
            for name,key in [('E','E'),('attention','A'),('G','G'),('uncentered','uncentered'),('delta','delta'),('eta','eta')]:
                self.close(getattr(traces[0],name),detail[key],metrics,name,dtype)
            cotangent=torch.randn_like(actual)
            (actual*cotangent).sum().backward();(expected*cotangent).sum().backward()
            pairs=[('Z',Z,zr),('base_logits',base,br),('base_weight',W,wr)]
            pairs += [(f'raw/{i}',v,r) for i,(v,r) in enumerate(zip(history,hr))]
            pairs += [(name,p,state[name]) for name,p in module.named_parameters()]
            for name,a,b in pairs:
                if a.grad is None or b.grad is None:self.assertIs(a.grad,b.grad,name)
                else:self.close(a.grad,b.grad,metrics,'grad/'+name,dtype)
            torch.optim.AdamW(module.parameters(),lr=.001).step()
            torch.optim.AdamW(state.values(),lr=.001).step()
            for name,p in module.named_parameters():self.close(p,state[name],metrics,'step/'+name,dtype)
            stream=io.BytesIO();torch.save(module.state_dict(),stream);stream.seek(0)
            restored=HistoryConditionedK(4,3,4,feature_seed=5).to(dtype=dtype).eval()
            restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
            torch.testing.assert_close(restored(3,Z,base,W,history),module(3,Z,base,W,history),atol=0,rtol=0)
            self.rows.append(dict(kind='oracle',dtype=str(dtype),metrics=metrics))

    def test_LN0_parameterless_population_variance_and_degenerate_rows(self):
        x=torch.tensor([[1.,2.,4.],[5.,5.,5.]],dtype=torch.float64,requires_grad=True)
        torch.testing.assert_close(ln0(x),reference_ln0(x),atol=1e-14,rtol=1e-12)
        torch.testing.assert_close(ln0(x)[1],torch.zeros(3,dtype=x.dtype),atol=0,rtol=0)
        y=ln0(torch.ones(2,1,dtype=torch.float64,requires_grad=True))
        self.assertTrue(torch.isfinite(y).all());self.assertEqual(y.abs().sum().item(),0)
        self.assertFalse(any(isinstance(m,nn.LayerNorm) for m in HistoryConditionedK(4,3,4,feature_seed=1).modules()))

    def test_point_constant_bias_counterexample_and_nonconstant_delta(self):
        logits=(torch.arange(30,dtype=torch.float64).reshape(1,2,5,3)%7)*.25
        bias=torch.tensor([2.,-4.,8.],dtype=torch.float64).reshape(1,1,1,3)
        # Dyadic values make even finite-precision shift invariance bit-exact here.
        torch.testing.assert_close(logits.softmax(-2),(logits+bias).softmax(-2),atol=0,rtol=0)
        module,Z,base,W,h=fixture();trace=[]
        result=module(3,Z,base,W,h,observe=trace.append);d=trace[0].delta
        self.assertGreater(d.var(dim=-2).max().item(),1e-6)
        self.assertGreater(d.var(dim=-1).max().item(),1e-8)
        torch.testing.assert_close(d.mean(-2),torch.zeros_like(d.mean(-2)),atol=1e-15,rtol=0)
        changed=list(h);changed[1]=h[1]+.4*torch.randn_like(h[1])
        other=module(3,Z,base,W,tuple(changed))
        difference=(result.softmax(-2)-other.softmax(-2)).abs().max().item()
        self.assertGreater(difference,1e-7)
        torch.testing.assert_close(result.softmax(-2).sum(-2),torch.ones_like(base[:,:,0]),atol=1e-14,rtol=1e-12)
        self.rows.append(dict(kind='point_constant_counterexample',wrong_bias_max_abs=0,
            mean_N_max_abs=d.mean(-2).abs().max().item(),history_change_K_max_abs=difference))

    def test_history_joint_token_permutation_and_sample_head_isolation(self):
        module,Z,base,W,h=fixture();t=[];y=module(3,Z,base,W,h,observe=t.append)
        bank=torch.cat(h,dim=-2);bank=bank[:,:,torch.randperm(bank.shape[-2])]
        permuted=tuple(bank.split([2,3,4],dim=-2));tr=[]
        torch.testing.assert_close(module(3,Z,base,W,permuted,observe=tr.append),y,atol=1e-14,rtol=1e-12)
        torch.testing.assert_close(tr[0].G,t[0].G,atol=1e-14,rtol=1e-12)
        for b in range(2):
            for head in range(3):
                small=HistoryConditionedK(4,1,4,feature_seed=3).double()
                sd=module.state_dict();sd={k:(v[:,head:head+1] if k.startswith('raw_gates.') else v) for k,v in sd.items()}
                small.load_state_dict(sd,strict=True)
                out=small(3,Z[b:b+1,head:head+1],base[b:b+1,head:head+1],W,tuple(x[b:b+1,head:head+1] for x in h))
                torch.testing.assert_close(out,y[b:b+1,head:head+1],atol=1e-14,rtol=1e-12)

    def test_gate_zero_then_isolated_increment_VJPs(self):
        module,Z,base,W,h=fixture()
        with torch.no_grad():module.raw_gates['3'].zero_()
        y=module(3,Z,base,W,h)
        torch.testing.assert_close(y,base,atol=0,rtol=0)
        cotangent=torch.randn_like(y)
        grads=torch.autograd.grad((y*cotangent).sum(),tuple(module.parameters()),allow_unused=True)
        for (name,p),g in zip(module.named_parameters(),grads):
            if name in ('raw_gates.1','raw_gates.2'):self.assertIsNone(g);continue
            self.assertTrue(torch.isfinite(g).all(),name)
            if name=='raw_gates.3':self.assertGreater(g.abs().sum().item(),0)
            else:self.assertEqual(g.abs().sum().item(),0,name)
        with torch.no_grad():module.raw_gates['3'].fill_(.6)
        t=[];module(3,Z,base,W,h,observe=t.append)
        # Independent old-raw leaves, base logits excluded: only the K branch.
        increment=t[0].eta*t[0].delta
        current_raw=torch.randn(2,3,5,4,dtype=Z.dtype,requires_grad=True)
        future=torch.randn_like(current_raw,requires_grad=True)
        named=[('Z',Z),('base_weight',W),('Uq',module.uq.weight),('Uk',module.uk.weight),('Uv',module.uv.weight)]+[(f'raw/{i}',v) for i,v in enumerate(h)]
        gradients=torch.autograd.grad((increment*cotangent).sum(),[v for _,v in named]+[current_raw,future,base],allow_unused=True)
        norms={}
        for (name,v),g in zip(named,gradients):
            self.assertIsNotNone(g,name);self.assertTrue(torch.isfinite(g).all(),name)
            norms[name]=g.abs().sum().item();self.assertGreater(norms[name],0,name)
        self.assertEqual(gradients[-3:],(None,None,None))
        self.rows.append(dict(kind='isolated_K_increment_VJP',gradient_l1=norms,current_C_grad=None,future_grad=None,base_logits_grad=None))

    def test_zero_layer_no_parameters_compute_or_RNG_and_global_parameter_ownership(self):
        for depth in range(4,9):
            before=torch.get_rng_state().clone();module=HistoryConditionedK(depth,3,4,feature_seed=1)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in module.parameters()),3*4**2+(depth-1)*3)
            self.assertEqual(set(module.state_dict()),{'uq.weight','uk.weight','uv.weight'}|{f'raw_gates.{i}' for i in range(1,depth)})
            self.assertEqual(list(module.named_buffers()),[])
            self.assertEqual(len({id(p) for p in module.raw_gates.values()}),depth-1)
            for p in module.raw_gates.values():self.assertEqual(tuple(p.shape),(1,3,1,1));self.assertEqual(p.abs().sum().item(),0)
            for m in (module.uq,module.uk,module.uv):self.assertIsNone(m.bias);self.assertGreater(m.weight.abs().sum().item(),0)
            Z=torch.randn(2,3,7,4);W=torch.randn(5,4);base=Z@W.T
            before=torch.get_rng_state().clone();dispatch=Trace()
            with patch.object(module.uq,'forward',side_effect=AssertionError('layer0 projected')):
                with dispatch:self.assertIs(module(0,Z,base,W,()),base)
            self.assertEqual(dispatch.bmms,[]);self.assertEqual(dispatch.softmax,[])
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            seen={name:[] for name in ('uq','uk','uv')};handles=[]
            for name in seen:
                handles.append(getattr(module,name).register_forward_hook(lambda m,a,o,name=name:seen[name].append(id(m.weight))))
            for i in range(1,depth):module(i,Z,base,W,tuple(torch.randn(2,3,5,4) for _ in range(i)))
            for handle in handles:handle.remove()
            for name,ids in seen.items():self.assertEqual(ids,[id(getattr(module,name).weight)]*(depth-1))

    def test_six_variants_temperature_order_dead_parameter_QV_and_rank(self):
        for variant in VARIANTS:
            _,model,kw=pair(variant);attn=model.double().blocks[1].Attn
            module=HistoryConditionedK(4,3,4,feature_seed=410).double().eval()
            with torch.no_grad():
                for p in (module.uq.weight,module.uk.weight,module.uv.weight):p.mul_(10)
                module.raw_gates['1'].fill_(.7)
            x=torch.randn(2,15,12,dtype=torch.float64);history=(torch.randn(2,3,8,4,dtype=torch.float64),)
            for temperature in (-3.,.5,8.):
                with torch.no_grad():
                    for name,p in attn.named_parameters():
                        if 'temperature' in name or 'tempreature' in name:p.fill_(temperature)
                base=attention_factors(attn,x)
                actual=attention_factors(attn,x,history_k=module,index=1,history=history)
                raw,detail=reference(1,base.Z,base.base_k_logits,attn.to_k.weight,history,module.state_dict())
                tau=max(.1,min(2.,temperature)) if variant=='shapenet' else max(.01,min(1.,temperature)) if variant in ('temp','conv_temp') else 1.
                torch.testing.assert_close(actual.K,(raw/tau).softmax(-2),atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(actual.Q,base.Q,atol=0,rtol=0)
                torch.testing.assert_close(actual.V,base.V,atol=0,rtol=0)
                torch.testing.assert_close(actual.base_k_logits,base.base_k_logits,atol=0,rtol=0)
                torch.testing.assert_close(actual.Q.sum(-1),torch.ones_like(actual.Q[...,0]))
                torch.testing.assert_close(actual.K.sum(-2),torch.ones_like(actual.K[:,:,0]))
                self.assertEqual(actual.K.shape[-1],attn.to_k.weight.shape[0])
                if variant=='airfrans':
                    self.assertIsNone(torch.autograd.grad(actual.K.square().sum(),attn.temperature,allow_unused=True)[0])

    def test_nonconv_point_permutation_and_execution_shapes(self):
        for variant in VARIANTS:
            _,model,kw=pair(variant);attn=model.double().blocks[1].Attn
            module=HistoryConditionedK(4,3,4,feature_seed=41).double().eval()
            with torch.no_grad():module.raw_gates['1'].fill_(.7)
            x=torch.randn(2,15,12,dtype=torch.float64);h=(torch.randn(2,3,8,4,dtype=torch.float64),)
            actual=attention_factors(attn,x,history_k=module,index=1,history=h)
            if not variant.startswith('conv'):
                order=torch.randperm(15);other=attention_factors(attn,x[:,order],history_k=module,index=1,history=h)
                torch.testing.assert_close(other.K,actual.K[:,:,order],atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(other.QC_raw,actual.QC_raw[:,:,order],atol=1e-12,rtol=1e-10)
        module,Z,base,W,h=fixture();trace=MatrixTrace();detail=[]
        with trace:module(3,Z,base,W,h,observe=detail.append)
        # PyTorch may flatten B/H/S into mm for E[ M,d ] @ K_hist.T;
        # the materialized interaction remains [B,H,M,S], observed by softmax.
        self.assertEqual(detail[0].attention.shape,(2,3,5,9))
        self.assertEqual(detail[0].delta.shape,(2,3,7,5))
        self.assertEqual([out[-2:] for a,b,out in trace.bmms][-2:],[(5,4),(7,5)])
        self.assertEqual([(s[-2:],a) for s,a,r in trace.softmax],[((5,9),-1)])
        self.assertFalse(any(out[-2:]==(7,7) for op,a,b,out in trace.matrix))
        self.rows.append(dict(kind='K_execution_shapes',N=7,M=5,S=9,d_h=4,matrix_ops=trace.matrix,softmax=[(s,a) for s,a,r in trace.softmax]))

    def test_six_full_models_zero_gate_eval_train_and_checkpoint(self):
        for variant in VARIANTS:
            for train,depth in [(False,4),(True,4),(False,8)]:
                base,_,kw=pair(variant,layers=depth,dropout=.2 if train else 0)
                rng=torch.get_rng_state().clone();torch.manual_seed(21002)
                model=model_class(variant)(**kw,feature_seed=713)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                state={k:v for k,v in model.state_dict().items() if not k.startswith('history_k.')}
                torch.testing.assert_close(state,base.state_dict(),atol=0,rtol=0)
                weights=model.state_dict();weights.update(base.state_dict());model.load_state_dict(weights,strict=True)
                self.assertFalse(any('attnres' in name for name,m in model.named_modules()))
                base.train(train);model.train(train)
                va=inputs(variant,kw);vb={k:v.detach().clone().requires_grad_() for k,v in va.items()}
                blocks=[];handles=[b.register_forward_hook(lambda m,a,o:blocks.append(o)) for b in base.blocks]
                before=torch.get_rng_state().clone();a=call(base,variant,va);after=torch.get_rng_state().clone()
                torch.set_rng_state(before);traces=[];b=call(model,variant,vb,observe=traces.append)
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                for handle in handles:handle.remove()
                torch.testing.assert_close(a,b,atol=0,rtol=0)
                for t,o in zip(traces,blocks):torch.testing.assert_close(t.output,o,atol=0,rtol=0)
                target=torch.randn_like(a);la=(a-target).square().mean();lb=(b-target).square().mean()
                torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward()
                for k in va:
                    if va[k].grad is None:self.assertIsNone(vb[k].grad)
                    else:torch.testing.assert_close(va[k].grad,vb[k].grad,atol=0,rtol=0)
                parameters=dict(model.named_parameters())
                for name,p in base.named_parameters():
                    if p.grad is None:self.assertIsNone(parameters[name].grad)
                    else:torch.testing.assert_close(p.grad,parameters[name].grad,atol=0,rtol=0)
                for gate in model.history_k.raw_gates.values():self.assertTrue(torch.isfinite(gate.grad).all())
                torch.optim.AdamW(base.parameters(),lr=.001).step();torch.optim.AdamW(model.parameters(),lr=.001).step()
                for k,v in base.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
                model.eval();restored=model_class(variant)(**kw,feature_seed=14).eval()
                stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
                restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
                torch.testing.assert_close(call(restored,variant,vb),call(model,variant,vb),atol=0,rtol=0)
                self.rows.append(dict(kind='full_K_zero_gate',variant=variant,L=depth,train=train,init_and_forward_rng_equal=True,forward_gradient_step_max_abs=0,strict_roundtrip_max_abs=0))

    def test_raw_history_sequence_reentry_exception_and_no_A(self):
        # Fail if any A construction/forward is accidentally reached.
        with (patch('cdlno.linearno_history.attnres.LatentSummaryAttnRes.__init__',side_effect=AssertionError('K constructed A')),
              patch('cdlno.linearno_history.attnres.LatentSummaryAttnRes.forward',side_effect=AssertionError('K called A'))):
            _,_,kw=pair('plain');model=HistoryKModel(**kw,feature_seed=43).eval()
            with torch.no_grad():
                for gate in model.history_k.raw_gates.values():gate.fill_(.5)
            before={name:set(vars(m)) for name,m in model.named_modules()}
            for B,N in [(2,15),(1,11),(2,7)]:
                trace=[];y=call(model,'plain',inputs('plain',kw,B=B,N=N),observe=trace.append)
                self.assertEqual([len(t.history) for t in trace],[0,1,2,3])
                for t in trace:
                    self.assertIs(t.C_tilde,t.factors.C_raw)
                    for j,raw in enumerate(t.history):self.assertIs(raw,trace[j].factors.C_raw);self.assertIsNotNone(raw.grad_fn)
                y.square().sum().backward();model.zero_grad(set_to_none=True)
            def fail(t):
                if t.index==2:raise RuntimeError('injected')
            values=inputs('plain',kw)
            with self.assertRaisesRegex(RuntimeError,'injected'):call(model,'plain',values,observe=fail)
            expected=call(model,'plain',values);sizes=[]
            def nested(t):
                sizes.append(len(t.history))
                if t.index==1:torch.testing.assert_close(call(model,'plain',values),expected,atol=0,rtol=0)
            torch.testing.assert_close(call(model,'plain',values,observe=nested),expected,atol=0,rtol=0)
            self.assertEqual(sizes,[0,1,2,3]);self.assertEqual(before,{n:set(vars(m)) for n,m in model.named_modules()})

    def test_actual_base_weight_queries_and_one_causal_update_per_block(self):
        _,_,kw=pair('plain');model=HistoryKModel(**kw,feature_seed=410).eval()
        with torch.no_grad():
            for gate in model.history_k.raw_gates.values():gate.fill_(.6)
        traces=[];events=[]
        def before_conditioning(module,args):
            index,Z,base,weight,history=args
            self.assertEqual(len(traces),index)  # current block/C_raw not yet completed
            self.assertIs(weight,model.blocks[index].Attn.to_k.weight)
            self.assertEqual(len(history),index)
            for j,raw in enumerate(history):self.assertIs(raw,traces[j].factors.C_raw)
            events.append(('K',index))
        def completed(t):traces.append(t);events.append(('block',t.index))
        handle=model.history_k.register_forward_pre_hook(before_conditioning)
        read=[];reader=model.history_k.uk.register_forward_hook(lambda m,a,o:read.append(a[0].shape[-2]))
        call(model,'plain',inputs('plain',kw),observe=completed)
        handle.remove();reader.remove()
        self.assertEqual(events,[(kind,i) for i in range(4) for kind in ('K','block')])
        self.assertEqual(read,[8,16,24])
        # Ordinary M semantics: use actual slot rows, including M>N, with no sqrt(N).
        for M in (1,32,64):
            module=HistoryConditionedK(4,3,4,feature_seed=1).double()
            with torch.no_grad():module.raw_gates['1'].fill_(.5)
            Z=torch.randn(2,3,7,4,dtype=torch.float64,requires_grad=True)
            W=torch.randn(M,4,dtype=torch.float64,requires_grad=True)
            h=(torch.randn(2,3,M,4,dtype=torch.float64,requires_grad=True),)
            trace=[];y=module(1,Z,Z@W.T,W,h,observe=trace.append)
            self.assertEqual(trace[0].G.shape,(2,3,M,4));self.assertEqual(y.shape,(2,3,7,M))
            y.square().mean().backward();self.assertTrue(torch.isfinite(Z.grad).all())
        self.rows.append(dict(kind='causal_single_update',events=events,bank_tokens=read,extra_actual_M=[1,32,64]))

    def test_double_finite_difference_and_invalid_configs(self):
        module,Z,base,W,h=fixture();module=HistoryConditionedK(4,1,3,feature_seed=23).double().eval()
        with torch.no_grad():module.raw_gates['1'].fill_(.4)
        Z=torch.randn(1,1,2,3,dtype=torch.float64,requires_grad=True)
        W=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
        raw=torch.randn(1,1,2,3,dtype=torch.float64,requires_grad=True)
        base=torch.zeros(1,1,2,2,dtype=torch.float64)
        names=['uq.weight','uk.weight','uv.weight','raw_gates.1'];state=module.state_dict()
        params=tuple(state[n].clone().requires_grad_() for n in names)
        def forward(z,weight,history,*values):
            s=dict(state);s.update(zip(names,values))
            return torch.func.functional_call(module,s,(1,z,base,weight,(history,)),strict=True)
        self.assertTrue(torch.autograd.gradcheck(forward,(Z,W,raw,*params),eps=1e-6,atol=1e-6,rtol=1e-4))
        for kw in [dict(n_layers=3,heads=1,d_h=3,feature_seed=1),dict(n_layers=4,heads=True,d_h=3,feature_seed=1),dict(n_layers=4,heads=1,d_h=0,feature_seed=1),dict(n_layers=4,heads=1,d_h=3,feature_seed=True)]:
            with self.assertRaises(ValueError):HistoryConditionedK(**kw)
        for i,history in [(0,(raw,)),(1,()),(1,[raw]),(True,()),(4,(raw,))]:
            with self.assertRaises(ValueError):module(i,Z,base,W,history)
        with self.assertRaises(ValueError):module(1,Z,base,W,(raw.float(),))
        broken=dict(state);del broken['uk.weight']
        with self.assertRaises(RuntimeError):module.load_state_dict(broken,strict=True)
        _,no_op,_=pair('plain')
        with self.assertRaisesRegex(TypeError,'K operator'):
            no_op.blocks(torch.randn(2,15,12),latent_attnres=object(),history_k=object())

    def test_fresh_process_K_only_import_and_strict_checkpoint(self):
        worker='''
import importlib,sys,torch
from torch_geometric.data import Data
torch.set_num_threads(1)
p=torch.load(sys.argv[1],weights_only=True)
module,name=p['class_path'].rsplit('.',1)
model=getattr(importlib.import_module(module),name)(**p['kwargs'],feature_seed=4).eval()
assert 'cdlno.linearno_history.attnres' not in sys.modules
model.load_state_dict(p['state'],strict=True)
v=p['inputs'];sizes=[]
if p['variant']=='airfrans':y=model(Data(**v),observe=lambda t:sizes.append(len(t.history)))
elif p['variant']=='shapenet':y=model((Data(**v),None),observe=lambda t:sizes.append(len(t.history)))
else:y=model(v['x'],v.get('fx'),v.get('T'),observe=lambda t:sizes.append(len(t.history)))
assert 'cdlno.linearno_history.attnres' not in sys.modules
torch.testing.assert_close(y,p['output'],atol=0,rtol=0)
assert sizes==[0,1,2,3]
print('K-only: no A import; strict checkpoint exact')
'''
        for variant,cwd in [('temp',STANDARD),('airfrans',ROOT/'Airfoil-Design-AirfRANS'),('shapenet',ROOT/'Car-Design-ShapeNetCar')]:
            _,_,kw=pair(variant);cls=model_class(variant);model=cls(**kw,feature_seed=43).eval()
            with torch.no_grad():
                for gate in model.history_k.raw_gates.values():gate.fill_(.6)
                vals={k:v.detach() for k,v in inputs(variant,kw).items()};out=call(model,variant,vals)
            with tempfile.TemporaryDirectory(prefix='linearno-r4-checkpoint-') as tmp:
                p=Path(tmp)/'state.pt';path=cls.__module__+'.'+cls.__name__
                torch.save(dict(class_path=path,kwargs=kw,state=model.state_dict(),variant=variant,inputs=vals,output=out),p)
                proc=subprocess.run([sys.executable,'-B','-c',worker,str(p)],cwd=cwd,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                self.rows.append(dict(kind='K_fresh_process',class_path=path,variant=variant,no_A_import=True,max_abs=0))


if __name__=='__main__':unittest.main()
