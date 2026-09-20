import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.config import resolve_config
from loop_linearno.rb_oracle import rb_reference
from loop_linearno.rb_support import rb_config,excite_routers,Trace
from loop_linearno.sr_support import config,inputs,TOPOLOGIES
from loop_linearno.test_point_attnres import errors


class RBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=[];cls.structure=[];cls.optimizer_rows=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL4_REPORT'):
            Path(path).write_text(json.dumps(dict(rows=cls.rows,structure=cls.structure,optimizer_rows=cls.optimizer_rows),indent=2)+'\n')

    def test_train_dropout_adamw_against_native_unroll(self):
        for variant in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
            task='airfrans' if variant=='airfrans' else 'car' if variant=='shapenet' else 'darcy'
            for preset in TOPOLOGIES[:2]:
                for dtype in (torch.float64,torch.float32):
                    with self.subTest(variant=variant,preset=preset,dtype=dtype):
                        c=rb_config(task,preset,variant=variant if task=='darcy' else None);r=c['request']
                        c=resolve_config(task,options=r['options'],profile_overrides={**r['profile_overrides'],'model.dropout':.2})
                        core=build_from_config(c).loop.to(dtype);excite_routers(core);ref=copy.deepcopy(core)
                        gen=torch.Generator().manual_seed(193)
                        x=torch.randn(2,15,8,generator=gen,dtype=dtype,requires_grad=True)
                        rx=x.detach().clone().requires_grad_()
                        start=torch.get_rng_state().clone();out=core(x);end=torch.get_rng_state().clone()
                        torch.set_rng_state(start);z=rx
                        for b in ref.prefix:z=b.block(z)
                        completed=[z]
                        for r in range(ref.loop_repeats):
                            partial=None
                            for j in range(2*ref.recurrent_core_blocks):
                                z=ref.rb_receivers[r][j](tuple(completed) if partial is None else (*completed,partial))
                                b=ref.core[j//2].block
                                u=b.Attn(b.ln_1(z)) if j%2==0 else b.mlp(b.ln_2(z))
                                partial=u if partial is None else partial+u
                            completed.append(partial)
                        z=ref.rb_output(completed)
                        for b in ref.suffix:z=b.block(z)
                        self.assertTrue(torch.equal(end,torch.get_rng_state()))
                        torch.testing.assert_close(out,z,atol=0,rtol=0)
                        target=torch.randn(out.shape,generator=gen,dtype=dtype)
                        (out-target).square().mean().backward();(z-target).square().mean().backward()
                        torch.testing.assert_close(x.grad,rx.grad,atol=0,rtol=0)
                        rp=dict(ref.named_parameters())
                        for name,p in core.named_parameters():
                            if p.grad is None:self.assertIsNone(rp[name].grad)
                            else:torch.testing.assert_close(p.grad,rp[name].grad,atol=0,rtol=0)
                        opt=torch.optim.AdamW(core.parameters(),lr=.001);ropt=torch.optim.AdamW(ref.parameters(),lr=.001)
                        opt.step();ropt.step()
                        for name,p in core.named_parameters():
                            torch.testing.assert_close(p,rp[name],atol=0,rtol=0)
                            for k,v in opt.state[p].items():torch.testing.assert_close(v,ropt.state[rp[name]][k],atol=0,rtol=0)
                        self.optimizer_rows.append(dict(variant=variant,preset=preset,dtype=str(dtype),dropout=.2,
                            atol=0,rtol=0,forward_gradient_weight_adamw_state_max_error=0,exact_rng=True))

    def test_independent_full_oracle_sources_weights_raw_partials_gradients(self):
        for variant in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
            task='airfrans' if variant=='airfrans' else 'car' if variant=='shapenet' else 'darcy'
            for preset in TOPOLOGIES:
                for dtype in (torch.float64,torch.float32):
                    with self.subTest(variant=variant,preset=preset,dtype=dtype):
                        core=build_from_config(rb_config(task,preset,variant=variant if task=='darcy' else None)).loop.to(dtype)
                        excite_routers(core);gen=torch.Generator().manual_seed(143)
                        x=torch.randn(2,15,8,generator=gen,dtype=dtype,requires_grad=True)
                        rx=x.detach().clone().requires_grad_()
                        state={k:v.detach().clone().requires_grad_() for k,v in core.named_parameters()}
                        with Trace(core) as trace:out=core(x)
                        reference,rt=rb_reference(rx,state,P=core.prefix_blocks,C=core.recurrent_core_blocks,R=core.loop_repeats,
                            S=core.suffix_blocks,variant=variant,heads=2,H=3,W=5)
                        atol,rtol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
                        row=dict(variant=variant,topology=preset,dtype=str(dtype),atol=atol,rtol=rtol,errors={},unused=[])
                        self.rows.append(row)
                        def check(a,b,name):
                            row['errors'][name]=errors(a,b)
                            torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=name)
                            self.assertTrue(torch.isfinite(a).all())
                        for j,expected in enumerate(rt['steps']):
                            actual=trace.routes[j];self.assertEqual(len(actual['sources']),len(expected['sources']))
                            for k,(a,b) in enumerate(zip(actual['sources'],expected['sources'])):check(a,b,f'step{j}.source{k}')
                            check(actual['weights'],expected['weights'],f'step{j}.weights')
                            check(actual['h'],expected['h'],f'step{j}.h')
                            self.assertIs(actual['h'],trace.norm_inputs[j])
                            check(trace.raw[j],expected['u'],f'step{j}.raw')
                            check(trace.partial_after(j),expected['partial'],f'step{j}.partial')
                        for k,(a,b) in enumerate(zip(trace.routes[-1]['sources'],rt['completed'])):check(a,b,f'completed{k}')
                        check(trace.routes[-1]['h'],rt['final_route']['h'],'core_final')
                        check(trace.routes[-1]['weights'],rt['final_route']['weights'],'output_weights')
                        check(out,reference,'final')
                        target=torch.randn(out.shape,generator=gen,dtype=dtype)
                        loss=(out-target).square().mean();ref_loss=(reference-target).square().mean()
                        check(loss,ref_loss,'loss');loss.backward();ref_loss.backward();check(x.grad,rx.grad,'input_gradient')
                        for name,p in core.named_parameters():
                            q=state[name]
                            if name.startswith('rb_receivers.0.0.'):
                                self.assertIsNone(p.grad)
                                self.assertTrue(q.grad is None or torch.equal(q.grad,torch.zeros_like(q)))
                                row['unused'].append(name)
                            elif name.endswith('Attn.temperature'):
                                self.assertEqual(variant,'airfrans');self.assertIsNone(p.grad);self.assertIsNone(q.grad);row['unused'].append(name)
                            else:
                                check(p.grad,q.grad,'gradient.'+name)
                                if name.startswith(('rb_receivers.','rb_output.')):self.assertGreater(p.grad.abs().sum().item(),0)
                        # Linear update sensitivity keeps independent FP32
                        # reduction roundoff separate from AdamW eps effects.
                        torch.optim.SGD(core.parameters(),lr=.001).step();torch.optim.SGD(state.values(),lr=.001).step()
                        for name,p in core.named_parameters():check(p,state[name],'step.'+name)

    def test_receiver_inventory_source_schedule_raw_only_and_call_order(self):
        schedules={'p1_c3_r2_s1':[1,2,2,2,2,2,2,3,3,3,3,3,3],
                   'p2_c2_r2_s2':[1,2,2,2,2,3,3,3,3],
                   'custom':[1,2,2,2,2,3,3,3,3,4,4,4,4]}
        for preset,expected in schedules.items():
            core=build_from_config(rb_config(preset=preset)).loop.double()
            receivers=[*(v for row in core.rb_receivers for v in row),core.rb_output]
            self.assertEqual(len(receivers),2*core.recurrent_core_blocks*core.loop_repeats+1)
            params=[p for m in receivers for p in m.parameters()]
            self.assertEqual(sum(p.numel() for p in params),2*8*len(receivers))
            self.assertEqual(len(params),len({id(p) for p in params}))
            for m in receivers:
                self.assertEqual(set(m.state_dict()),{'query','norm_scale'})
                self.assertTrue(torch.equal(m.query,torch.zeros_like(m.query)))
                self.assertTrue(torch.equal(m.norm_scale,torch.ones_like(m.norm_scale)))
            with Trace(core) as t:core(torch.randn(2,15,8,dtype=torch.float64))
            self.assertEqual([len(r['sources']) for r in t.routes],expected)
            self.assertIs(t.routes[0]['h'],t.routes[0]['sources'][0])
            for row in t.routes:
                for value in row['sources']:self.assertEqual(value.shape,(2,15,8))
                torch.testing.assert_close(row['weights'],torch.full_like(row['weights'],1/len(row['sources'])),atol=0,rtol=0)
                torch.testing.assert_close(row['h'],torch.stack(row['sources']).mean(0),atol=1e-12,rtol=1e-10)
            J=2*core.recurrent_core_blocks
            for r in range(core.loop_repeats):
                acc=None
                for j in range(J):
                    raw=t.raw[r*J+j];acc=raw if acc is None else acc+raw
                    torch.testing.assert_close(t.partial_after(r*J+j),acc,atol=0,rtol=0)
                self.assertIs(t.routes[(r+1)*J]['sources'][r+1],t.partial_after(r*J+J-1))
            events=[]
            for r in range(core.loop_repeats):
                for j in range(J):events.extend([f'route.{r}.{j}',f'{j//2}.{"operator" if j%2==0 else "mlp"}.ln',f'{j//2}.{"operator" if j%2==0 else "mlp"}.raw'])
            events.append('route.out');self.assertEqual(t.events,events);self.assertEqual(t.head,['ln_3','mlp2'])
            for i in range(core.recurrent_core_blocks):
                for factor in ('to_q','to_k','to_v'):
                    vals=[v for k,f,v in t.qkv if (k,f)==(i,factor)]
                    self.assertEqual(len(vals),core.loop_repeats);self.assertFalse(torch.equal(vals[0],vals[1]))
            self.structure.append(dict(preset=preset,sources=expected,receivers=len(receivers),router_parameters=sum(p.numel() for p in params)))

    def test_common_backbone_initialization_values_rng_and_unique_router_state(self):
        for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'):
            for preset in TOPOLOGIES[:2]:
                c=rb_config(task,preset);path,name=c['model_spec']['class_path'].rsplit('.',1)
                cls=getattr(importlib.import_module(path),name);kw=c['model_spec']['constructor_kwargs']
                torch.manual_seed(82);sr=cls(**{**kw,'residual_mode':'sr_1_over_r'});rng=torch.get_rng_state().clone()
                torch.manual_seed(82);rb=cls(**kw)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                common={k:v for k,v in rb.state_dict().items() if '.rb_' not in k}
                self.assertEqual(set(common),set(sr.state_dict()))
                for k,v in common.items():torch.testing.assert_close(v,sr.state_dict()[k],atol=0,rtol=0)
                self.assertEqual(sum(p.numel() for p in rb.parameters())-sum(p.numel() for p in sr.parameters()),
                                 2*8*(2*rb.loop.recurrent_core_blocks*rb.loop.loop_repeats+1))
                self.assertFalse(any('round' in key for key in common))
                self.assertEqual(len(list(rb.parameters())),len({id(p) for p in rb.parameters()}))

    def test_branch_vjp_point_dependent_weights_and_no_future_source(self):
        core=build_from_config(rb_config('elasticity')).loop.double();excite_routers(core)
        x=torch.randn(2,11,8,dtype=torch.float64,requires_grad=True)
        with Trace(core) as t:core(x)
        self.assertGreater((t.routes[-1]['weights'][:,0]-t.routes[-1]['weights'][:,1]).abs().max().item(),1e-8)
        J=2*core.recurrent_core_blocks
        old=t.routes[J]['sources'][1]
        grads=torch.autograd.grad(t.raw[J].square().sum(),(old,x,core.core[0].block.Attn.to_v.weight),retain_graph=True)
        for g in grads:self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)
        for i,row in enumerate(t.routes[:-1]):
            r,j=divmod(i,J)
            self.assertEqual(len(row['sources']),r+1+(j>0))
            for value in row['sources']:
                self.assertTrue(value.requires_grad)
                self.assertFalse(any(value is future for future in t.raw[i:]))

    def test_no_point_or_slot_self_attention_and_no_native_core_forward(self):
        core=build_from_config(rb_config('elasticity')).loop
        class Shapes(TorchDispatchMode):
            def __init__(self):self.mats=[]
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                result=func(*args,**(kwargs or {}))
                if str(func)=='aten.bmm.default':self.mats.append((tuple(args[0].shape),tuple(args[1].shape),tuple(result.shape)))
                return result
        with Shapes() as shapes:
            from contextlib import ExitStack
            with ExitStack() as stack:
                for b in core.core:
                    stack.enter_context(patch.object(b,'forward',side_effect=AssertionError('core ordinary residual called')))
                    stack.enter_context(patch.object(b.block,'forward',side_effect=AssertionError('core full block called')))
                core(torch.randn(2,11,8))
        # Legal KTV is [M,d_h], here M=d_h=4: distinguish using operands!
        self.assertEqual(len(shapes.mats),2*core.executed_depth)
        for a,b,y in shapes.mats:
            self.assertIn((a[-2:],b[-2:]),[((4,11),(11,4)),((11,4),(4,4))])
            self.assertNotEqual(y[-2:],(11,11))

    def test_history_is_forward_local_exception_batch_point_permutation_and_time(self):
        c=rb_config('elasticity');m=build_from_config(c).double();excite_routers(m.loop)
        x=torch.randn(2,11,2,dtype=torch.float64,requires_grad=True);known={n:set(vars(v)) for n,v in m.named_modules()}
        with Trace(m.loop) as t:first=m(x,None)
        perm=torch.tensor([4,2,1,8,5,0,3,10,6,9,7])
        torch.testing.assert_close(m(x[:,perm],None),first[:,perm],atol=1e-12,rtol=1e-10)
        for i in range(2):torch.testing.assert_close(m(x[i:i+1],None),first[i:i+1],atol=1e-12,rtol=1e-10)
        def fail(*a):raise RuntimeError('injected after populated history')
        h=m.loop.rb_receivers[1][1].register_forward_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError,'populated'):m(x,None)
        h.remove();fresh=torch.randn(1,7,2,dtype=torch.float64,requires_grad=True)
        with Trace(m.loop) as later:out=m(fresh,None)
        self.assertEqual(later.routes[0]['sources'][0].shape,(1,7,8))
        self.assertEqual(len(later.routes[0]['sources']),1)
        self.assertIsNone(torch.autograd.grad(out.sum(),x,allow_unused=True)[0])
        torch.testing.assert_close(m(x,None),first,atol=0,rtol=0)
        for n,v in m.named_modules():
            self.assertEqual(set(vars(v)),known[n]);self.assertFalse(any(isinstance(a,torch.Tensor) for a in vars(v).values()))
        for task in ('ns','plasticity','airfrans','car'):
            c=rb_config(task);a=build_from_config(c);b=build_from_config(c);args=inputs(c)
            a(*args)
            with Trace(a.loop) as t:y=a(*args)
            self.assertEqual(len(t.routes[0]['sources']),1)
            torch.testing.assert_close(y,b(*args),atol=0,rtol=0)
            if task=='plasticity':self.assertFalse(torch.equal(y,a(args[0],args[1],args[2]+1)))

    def test_strict_reload_three_wrappers_and_illegal_structure(self):
        for task in ('darcy','airfrans','car'):
            c=rb_config(task);m=build_from_config(c).eval();excite_routers(m.loop)
            expected=m(*inputs(c)).detach()
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory);(p/'config.json').write_text(json.dumps(c));torch.save(m.state_dict(),p/'weights.pt')
                code='''import json,sys,torch
from pathlib import Path
from cdlno.linearno_loop.construction import build_from_config
from loop_linearno.sr_support import inputs
p=Path(sys.argv[1]);c=json.loads((p/'config.json').read_text());m=build_from_config(c).eval()
m.load_state_dict(torch.load(p/'weights.pt',weights_only=True),strict=True)
torch.save(m(*inputs(c)).detach(),p/'result.pt')
'''
                root=Path(__file__).resolve().parents[2]
                proc=subprocess.run([sys.executable,'-B','-c',code,directory],cwd=directory,capture_output=True,text=True,
                    env={**os.environ,'PYTHONPATH':str(root/'tests')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1'})
                self.assertEqual(proc.returncode,0,proc.stderr)
                torch.testing.assert_close(torch.load(p/'result.pt',weights_only=True),expected,atol=0,rtol=0)
            sr=build_from_config(config(task))
            with self.assertRaises(RuntimeError):sr.load_state_dict(m.state_dict(),strict=True)
            with self.assertRaises(RuntimeError):m.load_state_dict(sr.state_dict(),strict=True)
        core=build_from_config(rb_config()).loop
        core.rb_receivers[1][0]=core.rb_receivers[0][0]
        with self.assertRaisesRegex(ValueError,'independent'):core(torch.zeros(2,15,8))
        core=build_from_config(rb_config()).loop;core.residual_mode='sr_1_over_r'
        with self.assertRaisesRegex(ValueError,'SR must not'):core(torch.zeros(2,15,8))

    def test_single_round_single_block_raw_hand_calculation(self):
        # R1 does NOT turn RB into an ordinary residual. With zero queries,
        # b1=2a+3*((a+2a)/2)=6.5a, final=(a+b1)/2=3.75a.
        from torch import nn
        from cdlno.linearno_loop.core import LinearNOLoopCore
        from loop_linearno.test_block_body import make_block
        class Multiply(nn.Module):
            def __init__(self,k):
                super().__init__();self.k=k;self.dim=8;self.heads=2;self.dim_head=4;self.rank=4;self.variant='plain'
            def forward(self,x):return self.k*x
        def factory(*,last_layer):
            b=make_block('plain',last_layer);b.ln_1=nn.Identity();b.ln_2=nn.Identity()
            b.Attn=Multiply(0 if last_layer else 2);b.mlp=Multiply(0 if last_layer else 3)
            if last_layer:b.ln_3=nn.Identity();b.mlp2=nn.Identity()
            return b
        core=LinearNOLoopCore(prefix_blocks=0,recurrent_core_blocks=1,loop_repeats=1,suffix_blocks=1,
                              residual_mode='rb_attnres',block_factory=factory)
        x=torch.arange(16,dtype=torch.float32).reshape(1,2,8)
        torch.testing.assert_close(core(x),3.75*x,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in core.parameters()),6*8)


if __name__=='__main__':unittest.main()
