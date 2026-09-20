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
from torch import nn

from cdlno.linearno_loop.construction import build_from_config, common_initial_state
from cdlno.linearno_loop.core import LinearNOLoopCore
from linearno_loop.config import resolve_config
from linearno_loop.contracts import seal, RESIDUAL_MODES
from loop_linearno.sr_support import config, pure_constructor, pure_kwargs, to_pure_state, inputs, TOPOLOGIES
from loop_linearno.sr_oracle import core_reference
from loop_linearno.test_point_attnres import errors


class SRCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL3_PARITY_REPORT'):
            Path(path).write_text(json.dumps(dict(torch=str(torch.__version__),rows=cls.rows),indent=2)+'\n')

    def test_presets_custom_module_identity_visit_order_qkv_and_single_head(self):
        expected={
            'p1_c3_r2_s1':['prefix.0','core.0','core.1','core.2','core.0','core.1','core.2','suffix.0'],
            'p2_c2_r2_s2':['prefix.0','prefix.1','core.0','core.1','core.0','core.1','suffix.0','suffix.1'],
            'custom':['core.0','core.1','core.0','core.1','core.0','core.1','suffix.0']}
        for preset in TOPOLOGIES:
            model=build_from_config(config(preset=preset));core=model.loop
            order=[];values={};identities={};heads=[];handles=[];events=[]
            for name,block in [(f'{g}.{i}',b) for g in ('prefix','core','suffix') for i,b in enumerate(getattr(core,g))]:
                def visit(mod,args,out,name=name):
                    order.append(name);identities.setdefault(name,[]).append((id(mod),id(mod.block),*(id(p) for p in mod.parameters())))
                handles.append(block.register_forward_hook(visit))
                for factor in ('to_q','to_k','to_v'):
                    def trace(mod,args,out,name=name,factor=factor):
                        values.setdefault(name+'.'+factor,[]).append(out.detach().clone())
                    handles.append(getattr(block.block.Attn,factor).register_forward_hook(trace))
            handles.append(core.suffix[-1].block.ln_3.register_forward_hook(lambda *a:heads.append('ln_3')))
            handles.append(core.suffix[-1].block.mlp2.register_forward_hook(lambda *a:heads.append('mlp2')))
            original=torch.einsum
            def einsum(equation,*args,**kwargs):
                events.append(equation)
                return original(equation,*args,**kwargs)
            try:
                with patch('torch.einsum',side_effect=einsum):core(torch.randn(2,15,8))
            finally:
                for h in handles:h.remove()
            self.assertEqual(order,expected[preset]);self.assertEqual(heads,['ln_3','mlp2'])
            self.assertEqual(core.executed_depth,len(order))
            self.assertEqual(core.unique_depth,len(set(order)))
            self.assertEqual(events.count('bhnm,bhnd->bhmd'),len(order))
            self.assertEqual(events.count('bhnm,bhmd->bhnd'),len(order))
            for name,ids in identities.items():
                self.assertTrue(all(x==ids[0] for x in ids))
                self.assertEqual(len(ids),core.loop_repeats if name.startswith('core.') else 1)
            for name,series in values.items():
                if name.startswith('core.'):
                    self.assertEqual(len(series),core.loop_repeats)
                    self.assertFalse(torch.equal(series[0],series[1]),name+' must recompute from updated input')
            parameters=list(model.named_parameters(remove_duplicate=False))
            self.assertEqual(len({id(p) for _,p in parameters}),len(parameters))
            self.assertFalse(any('round' in k or k.startswith('blocks.') for k in model.state_dict()))
            self.assertTrue(all(not b.block.last_layer and not hasattr(b.block,'mlp2') for b in core.core))

    def test_independent_core_oracle_all_six_variants_three_topologies_two_dtypes(self):
        for variant in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
            task='airfrans' if variant=='airfrans' else 'car' if variant=='shapenet' else 'darcy'
            for preset in TOPOLOGIES:
                for dtype in (torch.float64,torch.float32):
                    with self.subTest(variant=variant,preset=preset,dtype=dtype):
                        c=config(task,preset,variant=variant if task=='darcy' else None)
                        model=build_from_config(c).to(dtype=dtype);core=model.loop
                        gen=torch.Generator().manual_seed(44)
                        x=torch.randn(2,15,8,generator=gen,dtype=dtype,requires_grad=True)
                        rx=x.detach().clone().requires_grad_()
                        state={k:v.detach().clone().requires_grad_() for k,v in core.named_parameters()}
                        seen=[];handles=[b.register_forward_hook(lambda m,a,y:seen.append(y)) for b in (*core.prefix,*core.core,*core.suffix)]
                        try:actual=core(x)
                        finally:
                            for h in handles:h.remove()
                        expected,steps=core_reference(rx,state,P=core.prefix_blocks,C=core.recurrent_core_blocks,
                            R=core.loop_repeats,S=core.suffix_blocks,variant=variant,heads=2,H=3,W=5)
                        atol,rtol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
                        row=dict(variant=variant,topology=preset,dtype=str(dtype),device='cpu',atol=atol,rtol=rtol,errors={},unused=[])
                        def check(a,b,name):
                            torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=name)
                            self.assertTrue(torch.isfinite(a).all())
                            row['errors'][name]=errors(a,b)
                        check(actual,expected,'final')
                        for i,(a,b) in enumerate(zip(seen,steps)):check(a,b,'visit_'+str(i))
                        self.assertEqual(len(seen),core.executed_depth)
                        target=torch.randn(actual.shape,generator=gen,dtype=dtype)
                        loss=(actual-target).square().mean();rloss=(expected-target).square().mean()
                        check(loss,rloss,'loss');loss.backward();rloss.backward();check(x.grad,rx.grad,'input_gradient')
                        for name,param in core.named_parameters():
                            if name.endswith('Attn.temperature'):
                                self.assertEqual(variant,'airfrans');self.assertIsNone(param.grad);self.assertIsNone(state[name].grad);row['unused'].append(name)
                            else:check(param.grad,state[name].grad,'gradient.'+name)
                        # Independent FP32 LN/erf reductions can perturb ~1e-9
                        # gradients: AdamW's g/(abs(g)+eps) amplifies this. Keep
                        # the declared tolerances; use linear SGD for this
                        # FP32 mathematical oracle, AdamW in FP64 and a separate
                        # same-kernel, independently unrolled reference below.
                        optimizer=torch.optim.AdamW if dtype==torch.float64 else torch.optim.SGD
                        row['optimizer']=optimizer.__name__
                        opt=optimizer(core.parameters(),lr=.001);refopt=optimizer(state.values(),lr=.001)
                        opt.step();refopt.step()
                        for name,param in core.named_parameters():check(param,state[name],'step.'+name)
                        self.rows.append(row)

    def test_native_unrolled_adamw_step_and_train_dropout_rng(self):
        for variant in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
            task='airfrans' if variant=='airfrans' else 'car' if variant=='shapenet' else 'darcy'
            for preset in TOPOLOGIES:
                for dtype in (torch.float64,torch.float32):
                    with self.subTest(variant=variant,preset=preset,dtype=dtype):
                        c=config(task,preset,variant=variant if task=='darcy' else None)
                        r=c['request']
                        c=resolve_config(task,options=r['options'],profile_overrides={**r['profile_overrides'],'model.dropout':.2})
                        core=build_from_config(c).loop.to(dtype=dtype).train()
                        # Independent loop control over the original native
                        # submodules, with shared reference blocks across rounds.
                        refs={f'{g}.{i}.block':copy.deepcopy(b.block) for g in ('prefix','core','suffix')
                              for i,b in enumerate(getattr(core,g))}
                        reference_params={f'{key}.{k}':v for key,b in refs.items() for k,v in b.named_parameters()}
                        gen=torch.Generator().manual_seed(321)
                        x=torch.randn(2,15,8,generator=gen,dtype=dtype,requires_grad=True)
                        rx=x.detach().clone().requires_grad_()
                        rng=torch.get_rng_state().clone();actual=core(x);after=torch.get_rng_state().clone()
                        torch.set_rng_state(rng);expected=rx
                        for i in range(core.prefix_blocks):expected=refs[f'prefix.{i}.block'](expected)
                        for _ in range(core.loop_repeats):
                            for i in range(core.recurrent_core_blocks):
                                b=refs[f'core.{i}.block']
                                expected=expected+(1/core.loop_repeats)*b.Attn(b.ln_1(expected))
                                expected=expected+(1/core.loop_repeats)*b.mlp(b.ln_2(expected))
                        for i in range(core.suffix_blocks):expected=refs[f'suffix.{i}.block'](expected)
                        self.assertTrue(torch.equal(after,torch.get_rng_state()))
                        row=dict(reference='independent native-submodule unroll',variant=variant,topology=preset,
                                 dtype=str(dtype),device='cpu',atol=0.,rtol=0.,optimizer='AdamW',dropout=.2,errors={})
                        def exact(a,b,name):
                            torch.testing.assert_close(a,b,atol=0,rtol=0,msg=name)
                            row['errors'][name]=errors(a,b)
                        exact(actual,expected,'final')
                        target=torch.randn(actual.shape,generator=gen,dtype=dtype)
                        loss=(actual-target).square().mean();ref_loss=(expected-target).square().mean()
                        exact(loss,ref_loss,'loss');loss.backward();ref_loss.backward();exact(x.grad,rx.grad,'input_gradient')
                        for name,param in core.named_parameters():
                            if param.grad is None:self.assertIsNone(reference_params[name].grad)
                            else:exact(param.grad,reference_params[name].grad,'gradient.'+name)
                        optimizer=torch.optim.AdamW(core.parameters(),lr=.001)
                        ref_optimizer=torch.optim.AdamW(reference_params.values(),lr=.001)
                        optimizer.step();ref_optimizer.step()
                        for name,param in core.named_parameters():
                            exact(param,reference_params[name],'step.'+name)
                            for k,v in optimizer.state[param].items():
                                exact(v,ref_optimizer.state[reference_params[name]][k],'adamw_state.'+name+'.'+k)
                        self.rows.append(row)

    def test_single_apply_unique_initialization_matches_native_U_model_and_rng(self):
        # This catches constructing eight and discarding modules, reinitializing
        # per round, or creating placeholder before the native whole-tree apply.
        for task in ('darcy','elasticity','ns','plasticity','airfrans','car'):
            for preset in TOPOLOGIES:
                c=config(task,preset);cls=pure_constructor(c)
                kw=c['model_spec']['constructor_kwargs'];path,name=c['model_spec']['class_path'].rsplit('.',1)
                target=getattr(importlib.import_module(path),name)
                torch.manual_seed(19);pure=cls(**pure_kwargs(c));rng=torch.get_rng_state().clone()
                torch.manual_seed(19);loop=target(**kw)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()),(task,preset))
                s=c['loop_spec'];mapped=to_pure_state(loop.state_dict(),s['prefix_blocks'],s['recurrent_core_blocks'],s['suffix_blocks'])
                self.assertEqual(set(mapped),set(pure.state_dict()))
                for k,v in mapped.items():torch.testing.assert_close(v,pure.state_dict()[k],atol=0,rtol=0,msg=k)

    def test_common_initial_state_three_mode_configs_and_unsupported_execution(self):
        for task in ('darcy','plasticity','airfrans','car'):
            c=config(task);r=c['request'];states=[]
            before=torch.get_rng_state().clone()
            for mode in RESIDUAL_MODES:
                request=resolve_config(task,options={**r['options'],'residual_mode':mode},profile_overrides=r['profile_overrides'])
                states.append(common_initial_state(request))
                self.assertEqual(build_from_config(request).loop.residual_mode,mode)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for state in states[1:]:
                self.assertEqual(set(state),set(states[0]))
                for k,v in state.items():torch.testing.assert_close(v,states[0][k],atol=0,rtol=0)
            self.assertFalse(any('query' in k or 'norm_scale' in k for k in states[0]))

    def test_r1_full_wrapper_equivalence_inputs_time_and_positions(self):
        for task in ('darcy','elasticity','airfoil','pipe','ns','plasticity','airfrans','car'):
            c=config(task,'custom',R=1);model=build_from_config(c);pure=pure_constructor(c)(**pure_kwargs(c))
            s=c['loop_spec'];pure.load_state_dict(to_pure_state(model.state_dict(),s['prefix_blocks'],s['recurrent_core_blocks'],s['suffix_blocks']),strict=True)
            args=inputs(c);out=model(*args);reference=pure(*args)
            torch.testing.assert_close(out,reference,atol=0,rtol=0)
            self.assertIs(type(model).forward,type(pure).forward)
            if task=='plasticity':
                x,fx,T=args;self.assertFalse(torch.equal(out,model(x,fx,T+1)))
                out.square().sum().backward()
                self.assertGreater(model.time_fc[0].weight.grad.abs().sum().item(),0)
            for name,value in model.named_buffers():
                self.assertTrue(name in ('pos','reference'))
                self.assertNotIn(name,model.state_dict())

    def test_hand_computed_two_branches_and_unscaled_suffix(self):
        from loop_linearno.test_block_body import make_block
        class Multiply(nn.Module):
            def __init__(self,k):
                super().__init__();self.k=k
                self.dim=8;self.heads=2;self.dim_head=4;self.rank=4;self.variant='plain'
            def forward(self,x):return self.k*x
        def factory(*,last_layer):
            block=make_block('plain',last_layer)
            block.ln_1=nn.Identity();block.ln_2=nn.Identity();block.Attn=Multiply(2);block.mlp=Multiply(3)
            if last_layer:block.ln_3=nn.Identity();block.mlp2=nn.Identity()
            return block
        core=LinearNOLoopCore(prefix_blocks=0,recurrent_core_blocks=1,loop_repeats=2,suffix_blocks=1,
                             residual_mode='sr_1_over_r',block_factory=factory)
        x=torch.tensor([[[1.,-2.]]]);torch.testing.assert_close(core(x),300*x,atol=0,rtol=0)

    def test_second_round_branch_vjp_reaches_first_graph_and_same_parameter(self):
        c=config('elasticity','custom');model=build_from_config(c).double();core=model.loop
        outputs=[];raw=[]
        h=core.core[0].register_forward_hook(lambda m,a,y:outputs.append(y))
        a=core.core[0].block.Attn.register_forward_hook(lambda m,x,y:raw.append(y))
        try:core(torch.randn(2,9,8,dtype=torch.float64,requires_grad=True))
        finally:h.remove();a.remove()
        param=core.core[0].block.Attn.to_v.weight
        grad_old,grad_shared=torch.autograd.grad(raw[1].square().sum(),(outputs[0],param))
        for grad in (grad_old,grad_shared):
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)

    def test_no_forward_tensor_cache_and_exception_then_new_batch(self):
        model=build_from_config(config('elasticity'));known={name:set(vars(m)) for name,m in model.named_modules()}
        args=inputs(config('elasticity'));first=model(*args)
        with self.assertRaises(ValueError):model(torch.randn(2,3,4),None)
        second=model(torch.randn(1,6,2),None)
        self.assertEqual(second.shape,(1,6,1))
        torch.testing.assert_close(model(*args),first,atol=0,rtol=0)
        for name,m in model.named_modules():
            self.assertEqual(set(vars(m)),known[name])
            self.assertFalse(any(isinstance(v,torch.Tensor) for v in vars(m).values()))

    def test_invalid_config_model_structure_and_strict_fresh_process_roundtrip(self):
        c=config();before=torch.get_rng_state().clone()
        for field,value in (('executed_depth',9),('unique_depth',8),('resolved_rank',64)):
            bad=copy.deepcopy(c);bad['loop_spec'][field]=value
            with self.assertRaises(ValueError):build_from_config(seal(bad,'config_hash'))
        model=build_from_config(c);kw=dict(c['model_spec']['constructor_kwargs'])
        for field in ('recurrent_core_blocks','loop_repeats','suffix_blocks'):
            for value in (0,True,'2'):
                with self.assertRaises(ValueError):type(model)(**{**kw,field:value})
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaises(TypeError):type(model)(**{**kw,'executed_depth':8})
        for task in ('darcy','airfrans','car'):
            c=config(task);model=build_from_config(c).eval();args=inputs(c)
            expected=model(*args).detach()
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory);(p/'config.json').write_text(json.dumps(c))
                torch.save(model.state_dict(),p/'weights.pt')
                code='''import json,sys,torch
from pathlib import Path
from cdlno.linearno_loop.construction import build_from_config
from loop_linearno.sr_support import inputs
p=Path(sys.argv[1]);c=json.loads((p/'config.json').read_text())
m=build_from_config(c).eval();m.load_state_dict(torch.load(p/'weights.pt',weights_only=True),strict=True)
torch.save(m(*inputs(c)).detach(),p/'output.pt')
'''
                root=Path(__file__).resolve().parents[2]
                proc=subprocess.run([sys.executable,'-B','-c',code,directory],cwd=directory,
                    env={**os.environ,'PYTHONPATH':str(root/'tests')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                torch.testing.assert_close(torch.load(p/'output.pt',weights_only=True),expected,atol=0,rtol=0)
            state=model.state_dict();state.pop(next(iter(state)))
            with self.assertRaises(RuntimeError):model.load_state_dict(state,strict=True)

    def test_shared_core_gradients_sum_across_unrolled_independent_copies(self):
        c=config('elasticity','custom',R=2);model=build_from_config(c).double();core=model.loop
        x=torch.randn(2,7,8,dtype=torch.float64,requires_grad=True);rx=x.detach().clone().requires_grad_()
        copies=[[copy.deepcopy(b) for b in core.core] for _ in range(core.loop_repeats)]
        suffix=copy.deepcopy(core.suffix)
        out=core(x);y=rx
        for blocks in copies:
            for b in blocks:y=b(y,scale=.5)
        for i,b in enumerate(suffix):y=b(y,finalize=i==len(suffix)-1)
        torch.testing.assert_close(out,y,atol=0,rtol=0)
        out.square().sum().backward();y.square().sum().backward()
        for i,b in enumerate(core.core):
            for name,param in b.named_parameters():
                expected=sum(dict(row[i].named_parameters())[name].grad for row in copies)
                torch.testing.assert_close(param.grad,expected,atol=1e-12,rtol=1e-10)


if __name__=='__main__':unittest.main()
