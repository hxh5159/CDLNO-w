"""Phase9 synthetic benchmark structure, cost and measurement contracts."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import torch

from cdlno.cdpa import CDPA
from cdlno.modules import ConvFFN, IPOTBridge, LRSAFeatureReadout, LRSAFrontBlock, PersistentLatentBlock
from tools.cdlno_perf.models import Case, TASKS, build, synthetic_inputs, ROOT
from tools.cdlno_perf.costs import audit
from tools.cdlno_perf.measure import (benchmark, compare, contexts, output_gradients,
                                    precision_settings, quantiles, backend_probe)


class PerformanceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small(self, task='elasticity', **changes):
        kwargs = dict(B=2, d=16, h=4, M=4)
        kwargs.update(grid=(5, 7)) if task == 'airfoil' else kwargs.update(N=35)
        kwargs.update(changes)
        return Case.preset(task, **kwargs)

    def checked(self, case, name, chunk=0):
        model, actual, _ = build(case, name, chunk=chunk)
        args, target = synthetic_inputs(actual)
        result = audit(model, args, target, contexts(torch.device('cpu')))
        self.assertTrue(result['finite_output'])
        self.assertFalse(result['parameters']['nonfinite_grad_names'])
        return model, result

    def test_matched_composition_head_stem_independence_and_manual_forward(self):
        case = self.small('airfoil')
        matched, _, _ = build(case, 'lrsa_matched')
        off, _, _ = build(case, 'cdlno_off')
        torch.testing.assert_close(matched.preprocess[0].weight, off.preprocess[0].weight, rtol=0, atol=0)
        torch.testing.assert_close(matched.core.output.weight, off.core.readout.output.weight, rtol=0, atol=0)
        torch.testing.assert_close(matched.core.output_norm.weight, off.core.readout.output_norm.weight, rtol=0, atol=0)
        self.assertEqual(matched.core.output_norm.eps, 1e-6)
        modules = list(matched.core.modules())
        self.assertEqual(sum(isinstance(m, LRSAFrontBlock) for m in modules), 8)
        self.assertEqual(sum(isinstance(m, ConvFFN) for m in modules), 8)
        self.assertFalse(any(isinstance(m, (IPOTBridge, CDPA, PersistentLatentBlock, LRSAFeatureReadout)) for m in modules))
        weights = list(matched.named_parameters(remove_duplicate=False))
        self.assertEqual(len(weights), len({id(p) for _, p in weights}))
        self.assertEqual(len(weights), len({p.untyped_storage().data_ptr() for _, p in weights}))
        args, target = synthetic_inputs(case)
        h = matched.preprocess(args[0]) + matched.placeholder[None, None]
        for block in matched.core.blocks:
            h, _ = block(h)
        expected = matched.core.output(matched.core.output_norm(h))
        torch.testing.assert_close(matched(*args), expected, rtol=0, atol=0)
        (matched(*args) - target).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in matched.parameters()))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in matched.modules()))

    def test_live_counts_formulas_and_chunk_invariant_matrix_cost(self):
        case = self.small('airfoil')
        b,n,d,m,l,f = case.B,case.N,case.d,case.M,case.L,case.F
        for name, sources, locations in [('cdlno_off',0,0),('cdlno_entry',2,1),('cdlno_every_block',27,6)]:
            previous = None
            for chunk in (0,1,2,99):
                model, a = self.checked(case, name, chunk)
                c = a['counts']; costs = a['matrix_macs_by_component']
                self.assertEqual([c[k] for k in ('down_bridge','up_readout','latent_sa','structured_convffn')], [3,3,8,3])
                self.assertEqual(c['cdpa_logical_sources'], sources)
                self.assertEqual(c['cdpa_locations'], locations)
                expected_calls = sum(1 if chunk == 0 else (s+chunk-1)//chunk for s in a['history_sizes'])
                self.assertEqual(c['history_sdpa_calls'], expected_calls)
                self.assertEqual(c['all_sdpa_calls'], 14 + expected_calls)
                self.assertTrue(all(row['k'][-2] == m for row in a['sdpa'] if '.cdpa_at.' in row['path']))
                attention = costs.get('cdpa_qk_av',0) + costs['non_cdpa_sdpa_qk_av']
                self.assertEqual(attention, 2*b*d*(2*(f+1)*n*m + (l+sources)*m*m))
                self.assertEqual(costs.get('cdpa',0), b*m*d*d*(locations+3*sources))
                self.assertEqual(costs['front_two_latent_ffns'], f*4*case.ratio*b*m*d*d)
                self.assertEqual(costs['rear_geglu'], (l-f)*3*case.ratio*b*m*d*d)
                self.assertEqual(costs['point_ffn_dense_conv'], 3*9*b*n*d*d)
                if previous is not None:
                    self.assertEqual(a['matrix_macs'],previous)
                previous=a['matrix_macs']
                self.assertFalse(a['parameters']['missing_grad_names'])
                self.assertFalse(any(mod._forward_hooks or mod._forward_pre_hooks for mod in model.modules()))
        _, a = self.checked(case,'lrsa_matched')
        self.assertEqual([a['counts'][k] for k in ('down_bridge','up_readout','latent_sa','structured_convffn')], [8,8,8,8])
        self.assertEqual(a['matrix_macs_by_component']['non_cdpa_sdpa_qk_av'],2*b*d*(2*l*n*m+l*m*m))

    def test_projection_ffn_and_dense_conv_all_counted_against_closed_form(self):
        case=self.small('airfoil')
        b,n,d,m,l,f,r=case.B,case.N,case.d,case.M,case.L,case.F,case.ratio
        stem=b*n*(2*2*d+2*d*d)
        head=b*n*d
        # One front: down 2N+M; SA 4M; up 2N+2M; two latent FFNs 4rM; point 2rN+9N.
        front=b*d*d*(4*n+7*m+4*r*m+2*r*n+9*n)
        bridge_readout=b*d*d*(4*n+4*m+2*r*n+9*n)
        rear=(l-f)*b*m*d*d*(4+3*r)
        _, matched=self.checked(case,'lrsa_matched')
        self.assertEqual(matched['matrix_macs'],stem+head+l*front+2*b*d*(2*l*n*m+l*m*m))
        for name,s,a in [('cdlno_off',0,0),('cdlno_entry',2,1),('cdlno_every_block',27,6)]:
            _, result=self.checked(case,name)
            expected=stem+head+f*front+bridge_readout+rear+b*m*d*d*(a+3*s)+2*b*d*(2*(f+1)*n*m+(l+s)*m*m)
            self.assertEqual(result['matrix_macs'],expected)

    def test_front_modes_live_counts_whole_model_costs_and_actual_parameters(self):
        case = self.small('airfoil')
        b,n,d,m,l,f,r = case.B,case.N,case.d,case.M,case.L,case.F,case.ratio
        for name,sources,locations in [('cdlno_off',0,0),('cdlno_entry',f,1),('cdlno_every_block',27,6)]:
            full_model, full = self.checked(case,name)
            full_parameters = sum(p.numel() for p in full_model.parameters())
            for mode in ('full','no_sa','identity'):
                removed_sa = int(mode!='full')
                removed_ffn = int(mode=='identity')
                # Independent formula includes removed SA projections/QK+AV,
                # both FFN linears, biases/pre-norm and head Q/K scales.
                removed_macs = f*b*m*(removed_sa*(4*d*d+2*m*d)+removed_ffn*4*r*d*d)
                removed_params = f*(removed_sa*(4*d*d+2*d+2*d//case.h)
                                    + removed_ffn*(4*r*d*d+2*(r+2)*d))
                for chunk in (0,1,2):
                    model,a = self.checked(replace(case,front_latent_mode=mode),name,chunk)
                    c,costs = a['counts'],a['matrix_macs_by_component']
                    with self.subTest(cdpa=name,mode=mode,chunk=chunk):
                        self.assertEqual([c[k] for k in ('front_sa','front_latent_ffn','rear_sa','rear_ffn')],
                                         [f if mode=='full' else 0,2*f if mode!='identity' else 0,l-f,l-f])
                        sa = l if mode=='full' else l-f
                        self.assertEqual(c['latent_sa'],sa)
                        self.assertEqual([c[k] for k in ('down_bridge','up_readout','structured_convffn')],[f+1]*3)
                        self.assertEqual(c['cdpa_logical_sources'],sources)
                        calls=sum(1 if chunk==0 else (s+chunk-1)//chunk for s in a['history_sizes'])
                        self.assertEqual(c['history_sdpa_calls'],calls)
                        self.assertEqual(c['all_sdpa_calls'],2*(f+1)+sa+calls)
                        self.assertEqual(costs.get('cdpa',0), b*m*d*d*(locations+3*sources))
                        self.assertEqual(costs.get('front_two_latent_ffns',0),0 if mode=='identity' else f*4*r*b*m*d*d)
                        self.assertEqual(costs['rear_geglu'],(l-f)*3*r*b*m*d*d)
                        self.assertEqual(costs['point_ffn_dense_conv'],(f+1)*9*b*n*d*d)
                        self.assertEqual(a['matrix_macs'],full['matrix_macs']-removed_macs)
                        self.assertEqual(a['matrix_flops_2_per_mac'],2*a['matrix_macs'])
                        self.assertEqual(a['parameters']['registered'],full_parameters-removed_params)
                        self.assertEqual(a['parameters']['requires_grad'],sum(p.numel() for p in model.parameters()))
                        self.assertEqual(sum(a['parameters']['by_component'].values()),a['parameters']['registered'])
                        self.assertFalse(a['parameters']['missing_grad_names'])
                        expected=sum(a['linear_conv_macs_by_module'].values())+costs['non_cdpa_sdpa_qk_av']+costs.get('cdpa_qk_av',0)
                        self.assertEqual(a['matrix_macs'],expected)
                        if mode!='full': self.assertFalse(any('.latent_sa.' in k for k in model.state_dict()))
                        if mode=='identity': self.assertFalse(any('latent_ffn_' in k for k in model.state_dict()))

    def test_matched_lrsa_and_transolver_are_full_even_for_ablation_request(self):
        case=self.small('airfoil')
        for name in ('lrsa_matched','transolver'):
            baseline,_,_=build(case,name)
            for mode in ('full','no_sa','identity'):
                model,actual,_=build(replace(case,front_latent_mode=mode),name)
                self.assertEqual(actual.front_latent_mode,'full')
                torch.testing.assert_close(model.state_dict(),baseline.state_dict(),atol=0,rtol=0)
                if name=='lrsa_matched':
                    self.assertEqual(model.config.front_latent_mode,'full')
                    self.assertTrue(all(block.front_latent_mode=='full' for block in model.core.blocks))
                    args,target=synthetic_inputs(actual)
                    a=audit(model,args,target,contexts(torch.device('cpu')))
                    self.assertEqual([a['counts'][k] for k in ('front_sa','front_latent_ffn','rear_sa','rear_ffn')],[case.L,2*case.L,0,0])

    def test_transolver_actual_per_head_projection_and_unused_parameters(self):
        for task in ('elasticity','airfoil'):
            case=self.small(task)
            _,result=self.checked(case,'transolver')
            b,n,d,h,m,l,r=case.B,case.N,case.d,case.h,case.M,case.L,case.ratio
            stem=b*n*(4*d+2*d*d); head=b*n*d
            per_block=b*n*((18 if case.grid else 2)*d*d+d*m+d*d+2*r*d*d)+3*b*m*d*d//h
            expected=stem+head+l*per_block+l*(2*b*n*m*d+2*b*m*m*d)
            self.assertEqual(result['matrix_macs'],expected)
            self.assertEqual(result['counts']['all_sdpa_calls'],0)
            self.assertEqual(result['counts']['transolver_physics_attention'],8)
        case=Case.preset('airfrans',N=9,d=8,h=2,M=4,L=3,F=1)
        _,result=self.checked(case,'transolver')
        self.assertTrue(result['parameters']['missing_grad_names'])
        self.assertTrue(all('mlp_new' in n for n in result['parameters']['missing_grad_names']))

    def test_chunk_same_weights_output_all_gradients_and_memory_ledgers(self):
        case=self.small()
        ref=None; raw_costs=[]
        for chunk in (0,1,2):
            model,_,_=build(case,'cdlno_every_block',chunk=chunk)
            args,target=synthetic_inputs(case)
            actual=output_gradients(model,args,target,contexts(torch.device('cpu')))
            if ref is None: ref=actual
            else: self.assertTrue(compare(ref,actual,'fp32')['passed'])
            a=audit(model,args,target,contexts(torch.device('cpu')))
            self.assertEqual(a['storage']['h_f_payload_bytes'],case.B*case.N*case.d*4)
            self.assertEqual(a['storage']['largest_history_list_payload_bytes'],7*case.B*case.M*case.d*4)
            self.assertGreater(a['storage']['saved_activation_unique_backing_bytes'],0)
            self.assertTrue(any('stack' in k and 'cdpa_at' in k for k in a['temporary_materializations']))
            self.assertTrue(all(row['dtype']=='torch.float32' for row in a['depth_work_estimate']))
            raw_costs.append(a['matrix_macs'])
        self.assertEqual(len(set(raw_costs)),1)

    def test_zero_front_extended_depth_and_invalid_inputs(self):
        for l,f in ((1,0),(8,0),(12,2),(16,6),(10,8)):
            case=self.small(L=l,F=f,d=8,h=2,M=3,N=7,B=1)
            for mode in ('full','no_sa','identity'):
                for name in ('cdlno_off','cdlno_entry','cdlno_every_block'):
                    _,a=self.checked(replace(case,front_latent_mode=mode),name)
                    s=0 if name=='cdlno_off' else f if name=='cdlno_entry' else (l-f)*f+(l-f)*(l-f-1)//2
                    self.assertEqual(a['counts']['cdpa_logical_sources'],s)
                    self.assertEqual(a['counts']['latent_sa'],l if mode=='full' else l-f)
                    self.assertEqual(a['counts']['front_latent_ffn'],2*f if mode!='identity' else 0)
        for values in ({'L':0},{'F':8},{'F':-1},{'d':15},{'M':0},{'B':0}):
            with self.assertRaises(ValueError): self.small(**values)
        with self.assertRaises(ValueError): Case.preset('airfrans',B=2)
        with self.assertRaises(ValueError): Case.preset('ns',grid=(5,7))
        with self.assertRaises(ValueError): Case.preset('pipe',comparison='task',M=32)
        with self.assertRaisesRegex(ValueError,'front_latent_mode'): self.small(front_latent_mode='off')

    def test_task_preset_comparison_matches_current_sources(self):
        filenames=dict(darcy='Darcy',elasticity='Elas',airfoil='Airfoil',pipe='Pipe',ns='NS',plasticity='Plasticity')
        old_names=dict(darcy='Darcy',elasticity='Elas',airfoil='Airfoil',pipe='Pipe',ns='NS',plasticity='Plas')
        standard=ROOT/'PDE-Solving-StandardBenchmark'
        for task in TASKS:
            case=Case.preset(task,'task')
            new=case.for_model('cdlno_entry'); old=case.for_model('transolver')
            if task in filenames:
                paths=list((standard/'scripts').glob('Transolver*.sh'))
                path=next(p for p in paths if p.stem.lower()==('transolver_'+old_names[task]).lower())
                words=shlex.split(path.read_text().replace('\\\n',' '))
                get=lambda flag:int(words[words.index(flag)+1])
                self.assertEqual((old.d,old.h,old.M,old.L,old.B),(get('--n-hidden'),get('--n-heads'),get('--slice_num'),get('--n-layers'),get('--batch-size')))
                preset=json.loads((standard/'configs/CDLNO'/f'{task}.json').read_text())['model']
                self.assertEqual((new.d,new.h,new.M,new.L,new.F),(preset['n_hidden'],preset['n_heads'],preset['slice_num'],preset['n_layers'],preset['front_blocks']))
            else:
                project='Car-Design-ShapeNetCar' if task=='shapenet-car' else 'Airfoil-Design-AirfRANS'
                tree=ast.parse((ROOT/project/'main.py').read_text())
                call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('Model','Transolver') and any(k.arg=='unified_pos' for k in n.keywords))
                kw={k.arg:ast.literal_eval(k.value) for k in call.keywords}
                self.assertEqual((old.d,old.h,old.M,old.L),(kw['n_hidden'],kw['n_head'],kw['slice_num'],kw['n_layers']))

    def test_all_eight_task_wrapper_and_original_baseline_synthetic_calls(self):
        for task in TASKS:
            kw=dict(d=8,h=2,M=3,L=2,F=1,B=1)
            if task in ('darcy','airfoil','pipe'): kw['grid']=(5,7)
            elif task not in ('ns','plasticity'): kw['N']=11
            case=Case.preset(task,**kw)
            for name in ('transolver','lrsa_matched','cdlno_entry'):
                model,actual,source=build(case,name)
                args,target=synthetic_inputs(case)
                out=model(*args)
                self.assertEqual(out.shape,target.shape)
                (out-target).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(any(n.startswith('exp_') for n in sys.modules))

    def test_initialized_optimizer_timing_quantiles_and_profiler_gap(self):
        case=self.small(L=2,F=1,d=8,h=2,M=3,N=7,B=1)
        model,_,_=build(case,'cdlno_entry')
        args,target=synthetic_inputs(case);ctx=contexts(torch.device('cpu'),backend='math')
        p=backend_probe(model,args,ctx,torch.device('cpu'))
        self.assertTrue(p['observed_sdpa_ops'])
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
        r=benchmark(model,args,target,ctx,torch.device('cpu'),warmup=1,iterations=3)
        self.assertEqual(r['optimizer']['successful_steps_per_active_parameter'],[4])
        self.assertGreater(r['optimizer']['state_tensor_bytes'],0)
        for key in ('forward','train_step'):
            self.assertIsNone(r[key]['peak_allocated_bytes'])
            self.assertGreaterEqual(r[key]['p90_ms'],r[key]['median_ms'])
        self.assertEqual(quantiles([1,2,3,4,5])['p90_ms'],4.6)

    def test_fresh_cli_no_overwrite_and_python310_syntax(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json'
            cmd=[sys.executable,'-B',str(ROOT/'tools/cdlno_benchmark.py'),'--task','airfoil','--grid','5','7',
                 '--B','2','--d','8','--h','2','--M','3','--audit-only','--output',str(path)]
            result=subprocess.run(cmd,cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            content=path.read_bytes(); data=json.loads(content)
            self.assertFalse(data['errors']);self.assertEqual(len(data['results']),11)
            self.assertTrue(all(r['status']=='passed' for r in data['results']))
            result=subprocess.run(cmd,cwd='/tmp',capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(path.read_bytes(),content)
            ablated=Path(folder)/'identity.json'
            result=subprocess.run(cmd[:-1]+[str(ablated),'--front-latent-mode','identity'],cwd='/tmp',text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            changed=json.loads(ablated.read_text())
            for row in changed['results']:
                self.assertEqual(row['config']['front_latent_mode'],'identity' if row['model'].startswith('cdlno_') else 'full')
                if row['model'].startswith('cdlno_'):
                    self.assertEqual(row['cost']['counts']['front_latent_ffn'],0)
        for file in [ROOT/'tools/cdlno_benchmark.py',*(ROOT/'tools/cdlno_perf').glob('*.py')]:
            ast.parse(file.read_text(),feature_version=(3,10))

    def test_precision_global_settings_restored(self):
        old=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
        with precision_settings(False):
            self.assertEqual(torch.get_float32_matmul_precision(),'highest')
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        new=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
        self.assertEqual(old,new)


if __name__=='__main__': unittest.main()
