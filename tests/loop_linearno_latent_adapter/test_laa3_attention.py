import ast
import gc
import inspect
import io
import textwrap
import unittest
import weakref
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno.attention import LinearNOAttention, initialize_release_weights
from .attention_oracle import attention_reference
from .attention_support import VARIANTS, clone_parameters, compare, fill, kwargs, make, sample
from .test_laa2_rng_amp import snapshot, assert_rng


class ContractionTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.bmms=[]; self.softmaxes=[]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result=func(*args, **(kwargs or {}))
        if func==torch.ops.aten.bmm.default:
            self.bmms.append((tuple(args[0].shape),tuple(args[1].shape),tuple(result.shape)))
        if func==torch.ops.aten._softmax.default:
            self.softmaxes.append((tuple(args[0].shape),args[1]))
        return result


class AttentionTests(unittest.TestCase):
    def test_inherited_constructor_and_active_base_ast_exact(self):
        from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
        self.assertIs(V3LinearNOAttention.__init__,LinearNOAttention.__init__)
        native=ast.parse(textwrap.dedent(inspect.getsource(LinearNOAttention.forward))).body[0]
        active=ast.parse(textwrap.dedent(inspect.getsource(V3LinearNOAttention.forward))).body[0]
        first=next(i for i,node in enumerate(active.body) if ast.dump(node)==ast.dump(native.body[0]))
        projected=[node for node in active.body[first:] if not (isinstance(node,ast.If)
                   and isinstance(node.test,ast.Name) and node.test.id in ('use_adapter','use_latent'))]
        self.assertEqual(ast.dump(ast.Module(body=projected,type_ignores=[])),
                         ast.dump(ast.Module(body=native.body,type_ignores=[])))

    def test_distinct_positions_own_features_and_processor_contract_failure_is_local(self):
        first=make('plain',True,True);second=make('plain',True,True)
        for l,r in zip(first.parameters(),second.parameters()):
            self.assertIsNot(l,r);self.assertNotEqual(l.data_ptr(),r.data_ptr())
        self.assertIsNot(first.adapter,second.adapter)
        self.assertIsNot(first.latent_processor,second.latent_processor)
        expected=first(sample(),round_index=1)
        for bad in (lambda c:c[:,:,:-1,:],lambda c:c.float()):
            with patch.object(first.latent_processor,'forward',side_effect=bad):
                with self.assertRaisesRegex(ValueError,'preserve'):first(sample(),round_index=1)
        torch.testing.assert_close(first(sample(),round_index=1),expected,atol=0,rtol=0)

    def test_six_variants_four_ablations_two_visits_oracle_and_all_vjps(self):
        for dtype in (torch.float64,torch.float32):
            for variant in VARIANTS:
                for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                    for round_index in (0,1):
                        with self.subTest(dtype=dtype,variant=variant,z=latent,a=adapter,round=round_index):
                            m=fill(make(variant,latent,adapter,dtype=dtype)); x=sample(dtype).requires_grad_()
                            weights=clone_parameters(m); xr=x.detach().clone().requires_grad_()
                            expected,trace=attention_reference(xr,weights,variant=variant,heads=2,round_index=round_index,
                                latent=latent,adapter=adapter,alpha=3.,grid=(2,3))
                            seen=[]; original=torch.einsum
                            def observe(eq,*args):
                                result=original(eq,*args);seen.append((eq,args,result));return result
                            with patch('torch.einsum',side_effect=observe):actual=m(x,round_index=round_index)
                            label=f'{variant}/z{latent}a{adapter}/r{round_index}'
                            compare(actual,expected,label)
                            self.assertEqual([s[0] for s in seen],['bhnm,bhnd->bhmd','bhnm,bhmd->bhnd'])
                            for a,b,name in ((seen[0][1][0],trace['keys'],'keys'),(seen[0][1][1],trace['values'],'values'),
                                              (seen[0][2],trace['raw_context'],'KtV'),(seen[1][1][0],trace['queries'],'queries'),
                                              (seen[1][1][1],trace['context'],'processed'),(seen[1][2],trace['readout'],'QC')):
                                compare(a,b,label+'/'+name)
                            probe=torch.cos(torch.arange(actual.numel(),dtype=dtype)).reshape_as(actual)
                            grads=torch.autograd.grad((actual*probe).sum(),(x,*m.parameters()),allow_unused=True)
                            refs=torch.autograd.grad((expected*probe).sum(),(xr,*weights.values()),allow_unused=True)
                            for name,g,r in zip(('input',*weights),grads,refs):
                                self.assertEqual(g is None,r is None,name)
                                if g is not None:
                                    self.assertTrue(torch.isfinite(g).all(),name)
                                    compare(g,r,label+'/vjp/'+name)
                                    self.assertGreater(g.abs().max().item(),0,name)

    def test_off_delegates_native_bitwise_output_gradients_and_dropout_rng(self):
        from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
        for dtype in (torch.float64,torch.float32):
            for variant in VARIANTS:
                with self.subTest(dtype=dtype,variant=variant):
                    native=LinearNOAttention(**kwargs(variant,.35)).to(dtype=dtype)
                    m=make(variant,dtype=dtype,dropout=.35);native.load_state_dict(m.state_dict(),strict=True)
                    self.assertEqual(set(native.state_dict()),set(m.state_dict()))
                    x=sample(dtype).requires_grad_();xr=x.detach().clone().requires_grad_()
                    before=torch.get_rng_state();a=native(x);post=torch.get_rng_state()
                    ga=torch.autograd.grad(a.square().sum(),(x,*native.parameters()),allow_unused=True)
                    torch.set_rng_state(before)
                    with patch.object(LinearNOAttention,'forward',autospec=True,side_effect=LinearNOAttention.forward) as forward:
                        b=m(xr,round_index=1)
                    self.assertEqual(forward.call_count,1)
                    self.assertTrue(torch.equal(post,torch.get_rng_state()))
                    gb=torch.autograd.grad(b.square().sum(),(xr,*m.parameters()),allow_unused=True)
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                    for l,r in zip(ga,gb):
                        if l is None:self.assertIsNone(r)
                        else:torch.testing.assert_close(l,r,atol=0,rtol=0)
                    self.assertFalse(any(k.startswith(('adapter.','latent_processor.')) for k in m.state_dict()))

    def test_zero_init_ablations_preserve_output_common_gradients_and_rng(self):
        for variant in VARIANTS:
            for latent,adapter in ((False,True),(True,False),(True,True)):
                for index in (0,1):
                    with self.subTest(variant=variant,z=latent,a=adapter,r=index):
                        off=make(variant,dropout=.3);on=make(variant,latent,adapter,dropout=.3)
                        common=dict(off.named_parameters())
                        for name,p in common.items():torch.testing.assert_close(p,on.state_dict()[name],atol=0,rtol=0)
                        x=sample().requires_grad_();xr=x.detach().clone().requires_grad_()
                        rng=torch.get_rng_state();a=off(x,round_index=index);post=torch.get_rng_state()
                        ga=torch.autograd.grad(a.square().sum(),(x,*common.values()),allow_unused=True)
                        torch.set_rng_state(rng);b=on(xr,round_index=index)
                        self.assertTrue(torch.equal(post,torch.get_rng_state()))
                        torch.testing.assert_close(a,b,atol=0,rtol=0)
                        gb=torch.autograd.grad(b.square().sum(),(xr,*[dict(on.named_parameters())[k] for k in common]),allow_unused=True)
                        for name,g,r in zip(('input',*common),ga,gb):
                            if g is None:self.assertIsNone(r)
                            else:compare(g,r,f'zero/{variant}/{index}/{name}')

    def test_first_visit_never_reads_adapter_and_default_forward_is_native(self):
        for latent in (False,True):
            m=fill(make('temp',latent,True));x=sample()
            reference=make('temp',latent,False)
            reference.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('adapter.')},strict=True)
            with patch.object(m.adapter,'forward',side_effect=AssertionError('adapter accessed')):
                torch.testing.assert_close(m(x,round_index=0),reference(x,round_index=0),atol=0,rtol=0)
                # Native calls used outside recurrent visits must ignore installed features.
                native=LinearNOAttention(**kwargs('temp')).double()
                native.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith(('adapter.','latent_processor.'))},strict=True)
                with patch.object(m.latent_processor,'forward',side_effect=AssertionError('latent accessed')) if latent else patch.dict({},{}):
                    torch.testing.assert_close(m(x),native(x),atol=0,rtol=0)
            m(x.requires_grad_(),round_index=0).sum().backward()
            self.assertTrue(all(p.grad is None for p in m.adapter.parameters()))

    def test_nonzero_each_feature_changes_only_intended_visit_and_temperatures(self):
        for variant in VARIANTS:
            base=fill(make(variant,True,True));x=sample()
            for target in ('adapter.B_q','adapter.B_k','latent_processor.linear2.weight'):
                m=make(variant,True,True);m.load_state_dict(base.state_dict(),strict=True)
                with torch.no_grad():dict(m.named_parameters())[target].zero_()
                for index in (0,1):
                    a,b=base(x,round_index=index),m(x,round_index=index)
                    if target.startswith('adapter') and index==0:torch.testing.assert_close(a,b,atol=0,rtol=0)
                    else:self.assertGreater((a-b).abs().max().item(),1e-9,(variant,target,index))
        for variant in ('temp','conv_temp','shapenet'):
            m=fill(make(variant,True,True));x=sample()
            for temperatures in ((.31,.67),(-.2,3.)):
                prefix='tempreature' if variant=='shapenet' else 'temperature'
                with torch.no_grad():
                    getattr(m,prefix+'_q').fill_(temperatures[0]);getattr(m,prefix+'_k').fill_(temperatures[1])
                weights=clone_parameters(m)
                right,_=attention_reference(x,weights,variant=variant,heads=2,round_index=1,adapter=True,latent=True,alpha=3.,grid=(2,3))
                wrong,_=attention_reference(x,weights,variant=variant,heads=2,round_index=1,adapter=True,latent=True,alpha=3.,grid=(2,3),wrong_temperature_order=True)
                compare(m(x,round_index=1),right,'temperature/'+variant)
                self.assertGreater((right-wrong).abs().max().item(),1e-7)

    def test_module_hooks_and_contractions_two_visits_no_quadratic_attention(self):
        for variant in VARIANTS:
            for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                m=make(variant,latent,adapter);calls=[];handles=[];layouts=[]
                for name,mod in m.named_modules():
                    if name in ('in_project_x','to_q','to_k','to_v','to_out','adapter','latent_processor','dropout','softmax'):
                        handles.append(mod.register_forward_hook(lambda mod,args,out,name=name:calls.append((name,id(mod)))))
                handles.append(m.to_q.register_forward_pre_hook(lambda mod,args:layouts.append(args[0].is_contiguous())))
                try:
                    for index in (0,1):
                        start=len(calls)
                        with ContractionTrace() as trace:m(sample()+index*.2,round_index=index)
                        names=[n for n,_ in calls[start:]]
                        for name in ('in_project_x','to_q','to_k','to_v','to_out'):self.assertEqual(names.count(name),1)
                        self.assertEqual(names.count('adapter'),int(adapter and index==1))
                        self.assertEqual(names.count('latent_processor'),int(latent))
                        self.assertNotIn('dropout',names);self.assertNotIn('softmax',names)
                        self.assertEqual(trace.bmms,[((4,5,6),(4,6,3),(4,5,3)),((4,6,5),(4,5,3),(4,6,3))])
                        self.assertEqual(trace.softmaxes,[((2,2,6,5),-1),((2,2,6,5),-2)])
                    if latent:self.assertEqual(len({i for n,i in calls if n=='latent_processor'}),1)
                    if variant=='airfrans':self.assertTrue(all(layouts))
                finally:
                    for h in handles:h.remove()

    def test_constructor_install_isolation_and_atomic_validation(self):
        from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
        public=None;following=None
        for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(810)
                m=V3LinearNOAttention(**kwargs('temp'));m.apply(initialize_release_weights)
                old=snapshot();before={k:v.clone() for k,v in m.state_dict().items()}
                m.install_features(latent_enabled=latent,latent_width=7,
                    adapter_mode='bilateral_qk_lowrank_second_visit' if adapter else 'none',
                    latent_seed=8,adapter_seed=9)
                assert_rng(self,old,snapshot())
                after=torch.randn(9)
                if public is None:public,following=before,after
                for k,v in public.items():torch.testing.assert_close(m.state_dict()[k],v,atol=0,rtol=0)
                torch.testing.assert_close(after,following,atol=0,rtol=0)
                with self.assertRaisesRegex(RuntimeError,'already'):m.install_features()
        for changes in ({'latent_enabled':1},{'latent_enabled':True,'latent_width':0},
                        {'adapter_mode':'q_only'},{'adapter_mode':'bilateral_qk_lowrank_second_visit','adapter_seed':-1}):
            m=V3LinearNOAttention(**kwargs('plain'));old=snapshot();keys=set(m.state_dict())
            with self.assertRaises((ValueError,TypeError)):m.install_features(**changes)
            assert_rng(self,old,snapshot());self.assertEqual(set(m.state_dict()),keys)
            self.assertNotIn('adapter',m._modules);self.assertNotIn('latent_processor',m._modules)

    def test_shapenet_actual_M_independent_and_legacy_attributes(self):
        from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
        for rank in (5,32):
            m=V3LinearNOAttention(208,heads=8,dim_head=26,rank=rank,variant='shapenet')
            m.install_features(latent_enabled=True,latent_width=11,
                adapter_mode='bilateral_qk_lowrank_second_visit',latent_seed=1,adapter_seed=2)
            self.assertEqual(m.to_q.out_features,rank);self.assertEqual(m.to_k.out_features,rank)
            self.assertIn('tempreature_q',m.state_dict());self.assertNotIn('temperature_q',m.state_dict())
            x=torch.ones(1,7,208,requires_grad=True)
            m(x,round_index=1).square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
        air=make('airfrans',True,True);x=sample().requires_grad_()
        a=air(x,round_index=1)
        with torch.no_grad():air.temperature.fill_(123.)
        b=air(x,round_index=1);torch.testing.assert_close(a,b,atol=0,rtol=0)
        b.sum().backward();self.assertIsNone(air.temperature.grad)
        self.assertEqual(air.scale,3**-.5);self.assertEqual(air.softmax.dim,-1)

    def test_strict_roundtrip_and_invalid_input_no_retained_activations(self):
        for variant in VARIANTS:
            for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                m=fill(make(variant,latent,adapter));before={k:v.clone() for k,v in m.state_dict().items()}
                stream=io.BytesIO();torch.save(before,stream);stream.seek(0)
                twin=make(variant,latent,adapter);twin.load_state_dict(torch.load(stream,weights_only=True),strict=True)
                torch.testing.assert_close(m(sample(),round_index=1),twin(sample(),round_index=1),atol=0,rtol=0)
                bad=dict(before);bad.pop(next(iter(bad)))
                with self.assertRaises(RuntimeError):twin.load_state_dict(bad,strict=True)
                bad=dict(before);bad['round2.to_q.weight']=bad['to_q.weight']
                with self.assertRaises(RuntimeError):twin.load_state_dict(bad,strict=True)
                for x in (torch.ones(2,6),torch.ones(1,0,6),torch.ones(1,6,7),torch.ones(1,6,6,dtype=torch.int64)):
                    with self.assertRaises((ValueError,TypeError)):m(x,round_index=1)
                if variant in ('conv','conv_temp'):
                    with self.assertRaisesRegex(ValueError,'H.*W'):m(torch.ones(2,5,6,dtype=torch.float64),round_index=1)
                for index in (-1,True,1.5):
                    with self.assertRaises(ValueError):m(sample(),round_index=index)
                if adapter:
                    with self.assertRaisesRegex(ValueError,'round'):m(sample(),round_index=2)
                else:torch.testing.assert_close(m(sample(),round_index=2),m(sample(),round_index=0),atol=0,rtol=0)
                x=sample(batch=1).requires_grad_();ref=weakref.ref(x)
                y=m(x,round_index=0);y.sum().backward();del x,y;gc.collect();self.assertIsNone(ref())
                m(sample().requires_grad_(),round_index=1).sum().backward()
                self.assertEqual(list(m.buffers()),[])
                for mod in m.modules():
                    self.assertFalse(any(isinstance(value,torch.Tensor) for key,value in vars(mod).items() if key not in ('_parameters','_buffers')))
                for k,v in before.items():torch.testing.assert_close(m.state_dict()[k],v,atol=0,rtol=0)

    def test_noncontiguous_inputs_and_no_cross_batch_effect(self):
        for variant in VARIANTS:
            m=fill(make(variant,True,True));x=sample().transpose(0,1).contiguous().transpose(0,1)
            self.assertFalse(x.is_contiguous())
            a=m(x,round_index=1);compare(a,m(x.contiguous(),round_index=1),'noncontiguous/'+variant)
            modified=x.clone();modified[0].add_(.5)
            b=m(modified,round_index=1);torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
