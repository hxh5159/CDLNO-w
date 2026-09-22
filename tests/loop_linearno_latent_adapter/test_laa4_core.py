import copy
import gc
import io
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

from cdlno.linearno_loop.core import LinearNOLoopCore
from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
from linearno_loop.v3.config import resolve_config
from .core_oracle import full_reference
from .core_support import MODES,TASKS,ROWS,PROFILE_ROWS,Trace,close,configuration,excite,factory,inputs,make
from .test_laa2_rng_amp import assert_rng,snapshot


class CoreTests(unittest.TestCase):
    def test_formal_profiles_d12_d20_actual_widths_and_shapenet_rank(self):
        from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR
        from linearno_loop.v3.costs import analytic_cost
        for task in ('ns','car'):
            for profile in ('matched_v1','efficient_v1'):
                for depth in (12,20):
                    for mode in MODES:
                        c=resolve_config(task,options=dict(architecture=ARCHITECTURE_SELECTOR,cost_profile=profile,
                            executed_depth=depth,residual_mode=mode))
                        m=make(c,dtype=torch.float32);x=inputs(c,dtype=torch.float32,batch=1);y=m(x)
                        y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
                        self.assertEqual(m.executed_depth,depth)
                        cost=analytic_cost(c,batch=1,points=6)
                        self.assertEqual(sum(p.numel() for p in m.parameters()),
                            cost['total_parameters']-cost['parameter_groups']['stem']-cost['parameter_groups']['time'])
                        s=c['loop_spec'];PROFILE_ROWS.append(dict(task=task,cost_profile=profile,mode=mode,
                            executed_depth=depth,unique_depth=m.unique_depth,H=s['hidden_width'],heads=s['heads'],
                            M=s['actual_M'],Dz=s['latent_width'],core_parameters=sum(p.numel() for p in m.parameters()),
                            analytic_error=0,status='PASS',scope='core_only_B1_N6_no_stem_or_task_wrapper'))

    def test_constructor_rng_matches_v1_and_post_apply_feature_installation(self):
        from cdlno.linearno.attention import initialize_release_weights
        from cdlno.linearno_loop.v3.core import LinearNOLoopCoreV3
        for task in TASKS:
            for mode in MODES:
                c=configuration(task,mode=mode);s=c['loop_spec'];states=[];trees=[]
                for v1 in (True,False):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(381)
                        m=(LinearNOLoopCore(**{k:s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks','residual_mode')},block_factory=factory(c))
                           if v1 else LinearNOLoopCoreV3(config=c,block_factory=factory(c)))
                        allocations=torch.get_rng_state().clone();m.apply(initialize_release_weights)
                        initialized=torch.get_rng_state().clone()
                        if not v1:m.install_features()
                        self.assertTrue(torch.equal(initialized,torch.get_rng_state()))
                        states.append((allocations,initialized,torch.rand(s['hidden_width'])))
                        trees.append(m)
                for a,b in zip(*states):torch.testing.assert_close(a,b,atol=0,rtol=0)
                for k,p in trees[0].state_dict().items():torch.testing.assert_close(p,trees[1].state_dict()[k],atol=0,rtol=0)

    def test_first_round_feature_gradients_and_nonzero_ablation_effects(self):
        for mode in MODES:
            c=configuration(mode=mode);m=excite(make(c));x=inputs(c)
            with Trace(m) as t:y=m(x)
            first_loss=t.steps[2*m.recurrent_core_blocks-1]['u'].square().mean()
            adapter_parameters=[p for n,p in m.named_parameters() if '.adapter.' in n]
            latent_parameters=[p for n,p in m.named_parameters() if '.latent_processor.' in n]
            grads=torch.autograd.grad(first_loss,(*adapter_parameters,*latent_parameters),allow_unused=True,retain_graph=True)
            self.assertTrue(all(g is None for g in grads[:len(adapter_parameters)]))
            for g in grads[len(adapter_parameters):]:self.assertIsNotNone(g);self.assertGreater(g.abs().max().item(),0)
            for feature in ('adapter','latent_processor'):
                other=make(c);other.load_configured_state_dict(m.state_dict(),saved_config=c)
                with torch.no_grad():
                    for p in other.core:
                        a=p.block.Attn
                        if feature=='adapter':a.adapter.B_q.zero_();a.adapter.B_k.zero_()
                        else:a.latent_processor.linear2.weight.zero_();a.latent_processor.linear2.bias.zero_()
                self.assertGreater((other(x)-y).abs().max().item(),1e-10)

    def test_oracle_six_variants_three_modes_presets_and_custom_all_gradients(self):
        for task in TASKS:
            for mode in MODES:
                for preset in ('p1_c3_r2_s1','p2_c2_r2_s2','custom'):
                    for dtype in (torch.float64,torch.float32):
                        with self.subTest(task=task,mode=mode,preset=preset,dtype=dtype):
                            changes=dict(prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1) if preset=='custom' else {}
                            c=configuration(task,preset,mode,adapter=preset!='custom',**changes)
                            m=excite(make(c,dtype=dtype));x=inputs(c,dtype=dtype);xr=x.detach().clone().requires_grad_()
                            state={k:p.detach().clone().requires_grad_() for k,p in m.named_parameters()}
                            y,ref=full_reference(xr,state,c['loop_spec']);errs={}
                            with Trace(m) as t:actual=m(x)
                            check=lambda a,b,k:close(a,b,k,dtype,errs)
                            check(actual,y,'output');self.assertEqual(len(t.steps),len(ref['steps']))
                            for j,(a,b) in enumerate(zip(t.steps,ref['steps'])):
                                check(a['h'],b['h'],f'{j}.h');check(a['u'],b['u'],f'{j}.raw')
                                if b['partial'] is not None:
                                    start=b['round']*2*m.recurrent_core_blocks
                                    partial=sum([s['u'] for s in t.steps[start+1:j+1]],t.steps[start]['u'])
                                    check(partial,b['partial'],f'{j}.partial')
                            self.assertEqual(len(t.routes),len(ref['routes']))
                            for j,(a,b) in enumerate(zip(t.routes,ref['routes'])):
                                self.assertEqual(a['key'],b['key']);self.assertEqual(len(a['sources']),len(b['sources']))
                                check(a['weights'],b['weights'],f'route{j}.weights');check(a['h'],b['h'],f'route{j}.h')
                                for n,(aa,bb) in enumerate(zip(a['sources'],b['sources'])):check(aa,bb,f'route{j}.source{n}')
                            if mode=='lb_attnres_1_over_r':
                                visits=[v for v in t.visits if v[0]=='core'];C=m.recurrent_core_blocks
                                for r,row in enumerate(ref['rounds']):
                                    entry=visits[r*C][2];end=visits[(r+1)*C-1][3]
                                    check(entry,row['entry'],f'round{r}.H');check(end,row['end'],f'round{r}.Y')
                                    torch.testing.assert_close(t.routes[r]['sources'][-1],end-entry,atol=0,rtol=0)
                            probe=torch.cos(torch.arange(actual.numel(),dtype=dtype)).reshape_as(actual)
                            ga=torch.autograd.grad((actual*probe).mean(),(x,*m.parameters()),allow_unused=True)
                            gb=torch.autograd.grad((y*probe).mean(),(xr,*state.values()),allow_unused=True)
                            for name,a,b in zip(('input',*state),ga,gb):
                                self.assertEqual(a is None,b is None,name)
                                if a is not None:self.assertTrue(torch.isfinite(a).all());check(a,b,'grad/'+name)
                            ROWS.append(dict(task=task,mode=mode,preset=preset,dtype=str(dtype),errors=errs,status='PASS'))

    def test_feature_off_v1_exact_output_gradients_adamw_and_dropout_rng(self):
        for task in TASKS:
            for mode in MODES:
                for dtype in (torch.float64,torch.float32):
                    c=configuration(task,mode=mode,latent=False,adapter=False,dropout=.2)
                    a=excite(make(c,dtype=dtype));b=excite(make(c,v1=True,dtype=dtype))
                    self.assertEqual(set(a.state_dict()),set(b.state_dict()))
                    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],atol=0,rtol=0)
                    x=inputs(c,dtype=dtype);xr=x.detach().clone().requires_grad_()
                    rng=torch.get_rng_state();y=a(x);after=torch.get_rng_state();torch.set_rng_state(rng);z=b(xr)
                    self.assertTrue(torch.equal(after,torch.get_rng_state()));torch.testing.assert_close(y,z,atol=0,rtol=0)
                    y.square().mean().backward();z.square().mean().backward()
                    torch.testing.assert_close(x.grad,xr.grad,atol=0,rtol=0)
                    for p,q in zip(a.parameters(),b.parameters()):
                        if p.grad is None:self.assertIsNone(q.grad)
                        else:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
                    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001);oa.step();ob.step()
                    for p,q in zip(a.parameters(),b.parameters()):
                        torch.testing.assert_close(p,q,atol=0,rtol=0)
                        for name,value in oa.state[p].items():torch.testing.assert_close(value,ob.state[q][name],atol=0,rtol=0)

    def test_zero_features_all_ablations_same_backbone_across_modes(self):
        for task in TASKS:
            common=None
            for mode in MODES:
                c0=configuration(task,mode=mode,latent=False,adapter=False,dropout=.2);off=make(c0)
                backbone={k:p for k,p in off.state_dict().items() if not k.startswith(('rb_','lb_'))}
                if common is None:common=backbone
                for name,p in backbone.items():torch.testing.assert_close(p,common[name],atol=0,rtol=0)
                for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                    c=configuration(task,mode=mode,latent=latent,adapter=adapter,dropout=.2);m=make(c)
                    for key,p in off.state_dict().items():torch.testing.assert_close(p,m.state_dict()[key],atol=0,rtol=0)
                    rng=torch.get_rng_state();y=off(inputs(c));after=torch.get_rng_state();torch.set_rng_state(rng);z=m(inputs(c))
                    torch.testing.assert_close(y,z,atol=0,rtol=0);self.assertTrue(torch.equal(after,torch.get_rng_state()))
                    for i,physical in enumerate(m.core):
                        keys=physical.block.Attn.state_dict()
                        self.assertEqual(any(k.startswith('adapter.') for k in keys),adapter)
                        self.assertEqual(any(k.startswith('latent_processor.') for k in keys),latent)

    def test_module_identity_visits_features_and_single_head(self):
        for mode in MODES:
            for preset in ('p1_c3_r2_s1','p2_c2_r2_s2','custom','d12','d20'):
                custom=dict(prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1) if preset=='custom' else {}
                c=configuration(preset=preset,mode=mode,adapter=preset!='custom',**custom);records=[];m=make(c,records=records)
                U,D=m.unique_depth,m.executed_depth
                self.assertEqual(len(records),U);self.assertEqual(len({r[0] for r in records}),U)
                blocks=[*m.prefix,*m.core,*m.suffix]
                for p,(bid,aid,old_ids) in zip(blocks,records):
                    self.assertEqual(id(p.block),bid);self.assertEqual(id(p.block.Attn),aid)
                    current=dict(p.block.named_parameters())
                    for key,identity in old_ids.items():self.assertEqual(id(current[key]),identity)
                params=list(m.named_parameters(remove_duplicate=False))
                self.assertEqual(len(params),len({id(p) for _,p in params}))
                hooks=[];calls=[];visits=[];qkv={};heads=[]
                for group in ('prefix','core','suffix'):
                    for i,p in enumerate(getattr(m,group)):
                        hooks.append(p.block.Attn.register_forward_pre_hook(lambda mod,args,kw,g=group,i=i:visits.append((g,i,kw.get('round_index'),id(mod))),with_kwargs=True))
                        for q in ('to_q','to_k','to_v'):
                            hooks.append(getattr(p.block.Attn,q).register_forward_hook(lambda mod,args,y,g=group,i=i,q=q:qkv.setdefault((g,i,q),[]).append(y.detach().clone())))
                        if group!='core':self.assertNotIsInstance(p.block.Attn,V3LinearNOAttention)
                        else:
                            for feature in ('adapter','latent_processor'):
                                if hasattr(p.block.Attn,feature):
                                    hooks.append(getattr(p.block.Attn,feature).register_forward_hook(lambda mod,args,y,i=i,f=feature:calls.append((i,f,id(mod)))))
                for name in ('ln_3','mlp2'):hooks.append(getattr(m.suffix[-1].block,name).register_forward_hook(lambda *args,name=name:heads.append(name)))
                equations=[];original=torch.einsum
                def einsum(eq,*args,**kw):equations.append(eq);return original(eq,*args,**kw)
                try:
                    with patch('torch.einsum',side_effect=einsum):m(inputs(c))
                finally:
                    for h in hooks:h.remove()
                expected=[('prefix',i,None) for i in range(m.prefix_blocks)]+[('core',i,r) for r in range(m.loop_repeats) for i in range(m.recurrent_core_blocks)]+[('suffix',i,None) for i in range(m.suffix_blocks)]
                self.assertEqual([v[:3] for v in visits],expected);self.assertEqual(len(visits),D);self.assertEqual(heads,['ln_3','mlp2'])
                for eq in ('bhnm,bhnd->bhmd','bhnm,bhmd->bhnd'):self.assertEqual(equations.count(eq),D)
                for i in range(m.recurrent_core_blocks):
                    self.assertEqual(len({v[3] for v in visits if v[:2]==('core',i)}),1)
                    for q in ('to_q','to_k','to_v'):
                        series=qkv[('core',i,q)];self.assertEqual(len(series),m.loop_repeats);self.assertFalse(torch.equal(series[0],series[1]))
                    self.assertEqual(sum(i==j and f=='latent_processor' for j,f,_ in calls),m.loop_repeats)
                    self.assertEqual(sum(i==j and f=='adapter' for j,f,_ in calls),int(preset!='custom'))
                self.assertFalse(any('round' in k or 'core_ffns' in k for k in m.state_dict()))

    def test_gradients_round_sharing_features_and_failure_history(self):
        for mode in MODES:
            c=configuration(mode=mode);m=excite(make(c));x=inputs(c)
            with Trace(m) as t:y=m(x)
            for i,p in enumerate(m.core):
                first=t.steps[2*i]['u'];second=t.steps[2*m.recurrent_core_blocks+2*i]['u']
                g=torch.autograd.grad(second.square().sum(),(first,p.block.Attn.to_q.weight,p.block.ln_2.weight),retain_graph=True)
                for v in g:self.assertTrue(torch.isfinite(v).all());self.assertGreater(v.abs().max().item(),0)
            y.square().sum().backward()
            for name,p in m.named_parameters():
                if '.adapter.' in name or '.latent_processor.' in name:
                    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().max().item(),0,name)
            # A failure after local sources exist must not persist any graph.
            handle=m.suffix[0].register_forward_pre_hook(lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic failure')))
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):m(inputs(c))
            handle.remove();del t,y,x
            fresh=inputs(c,batch=1);ref=weakref.ref(fresh);m(fresh).sum().backward();del fresh;gc.collect();self.assertIsNone(ref())
            self.assertEqual(list(m.buffers()),[])
            for module in m.modules():self.assertFalse(any(isinstance(v,torch.Tensor) for v in vars(module).values()))

    def test_strict_reload_metadata_conflicts_before_weight_application(self):
        from cdlno.linearno_loop.v3.core import LinearNOLoopCoreV3
        for mode in MODES:
            for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                c=configuration(mode=mode,latent=latent,adapter=adapter);m=excite(make(c));other=make(c)
                stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);state=torch.load(stream,weights_only=True)
                other.load_configured_state_dict(state,saved_config=c)
                torch.testing.assert_close(m(inputs(c)),other(inputs(c)),atol=0,rtol=0)
                options=c['request']['options']
                changes=[{'residual_mode':v} for v in MODES if v!=mode]+[{'latent_enabled':not latent},
                    {'adapter_mode':'none' if adapter else 'bilateral_qk_lowrank_second_visit'},
                    {'hidden_width':8},{'heads':3},{'actual_M':7},{'latent_width':9},
                    {'adapter_rank':3},{'adapter_alpha':6.}, {'topology_preset':'p1_c3_r2_s1'}]
                candidates=[configuration(mode=mode,latent=latent,adapter=adapter,**change) for change in changes]
                candidates.append(configuration('ns',mode=mode,latent=latent,adapter=adapter))
                if not adapter:
                    candidates.append(configuration(mode=mode,latent=latent,adapter=False,preset='custom',
                        prefix_blocks=2,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=2))
                for saved in candidates:
                    with patch.object(nn.Module,'load_state_dict',side_effect=AssertionError('weights applied early')):
                        with self.assertRaises(ValueError):other.load_configured_state_dict(state,saved_config=saved)
                for key in ('family','architecture_extension','schema_version','checkpoint_version'):
                    bad=copy.deepcopy(c);bad[key]='wrong'
                    with self.assertRaises(ValueError):other.load_configured_state_dict(state,saved_config=bad)
                for bad in ({k:v for k,v in state.items() if k!=next(iter(state))},{**state,'round2.copy':torch.ones(1)}):
                    before={k:v.clone() for k,v in other.state_dict().items()}
                    with self.assertRaises((ValueError,RuntimeError)):other.load_configured_state_dict(bad,saved_config=c)
                    for k,v in before.items():torch.testing.assert_close(v,other.state_dict()[k],atol=0,rtol=0)
                bad=dict(state);key=next(iter(bad));bad[key]=torch.ones(1)
                with self.assertRaisesRegex(ValueError,'shape'):other.load_configured_state_dict(bad,saved_config=c)
                with self.assertRaises(ValueError):other.load_configured_state_dict(state,saved_config=c,strict=False)
        c=configuration();calls=[]
        for key,value in [('executed_depth',99),('suffix_blocks',0),('recurrent_core_blocks',0),('loop_repeats',0)]:
            forged=copy.deepcopy(c);forged['loop_spec'][key]=value
            with self.assertRaises(ValueError):LinearNOLoopCoreV3(config=forged,block_factory=lambda **kw:calls.append(kw))
        self.assertEqual(calls,[])

    def test_structure_rejects_aliasing_or_mismatch_and_install_is_rng_isolated(self):
        from cdlno.linearno_loop.v3.core import LinearNOLoopCoreV3
        from cdlno.linearno.attention import initialize_release_weights
        c=configuration();m=LinearNOLoopCoreV3(config=c,block_factory=factory(c));m.apply(initialize_release_weights)
        with self.assertRaisesRegex(ValueError,'install'):m(inputs(c,dtype=torch.float32))
        before=snapshot();m.install_features();assert_rng(self,before,snapshot())
        with self.assertRaisesRegex(RuntimeError,'already'):m.install_features()
        m.core[1].block.ln_2=m.core[0].block.ln_2
        with self.assertRaisesRegex(ValueError,'distinct'):m.validate_structure()
        def wrong(*,last_layer):
            b=factory(c)(last_layer=last_layer);b.Attn.rank+=1;return b
        with self.assertRaises(ValueError):LinearNOLoopCoreV3(config=c,block_factory=wrong)
        m=make(c);m.loop_repeats=3
        with self.assertRaises(ValueError):m(inputs(c))


if __name__=='__main__':unittest.main()
