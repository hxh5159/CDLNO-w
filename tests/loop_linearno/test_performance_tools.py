"""LL9 independent accounting and observational isolation checks."""
import copy
import json
import unittest
from unittest.mock import patch
import torch
from cdlno.linearno_loop.construction import build_from_config
from tools.linearno_loop_support import TASKS,PRESETS,MODES,configuration,inputs,point_count
from tools.linearno_loop_accounting import analytic,audit,measured_parameters
from tools.linearno_loop_diagnostics import LoopDiagnostics


class PerformanceToolsTests(unittest.TestCase):
    def test_independent_formula_against_actual_aten_all_tasks_modes_presets(self):
        for task in TASKS:
            for preset in PRESETS:
                for mode in MODES:
                    with self.subTest(task=task,preset=preset,mode=mode):
                        c=configuration(task,preset,mode,small=True,multiplier=1)
                        m=build_from_config(c);a=inputs(c);B=1 if task in ('airfrans','car') else 2;N=point_count(c,False)
                        e=analytic(c,B=B,N=N);v=audit(m,a,B=B,N=N)
                        self.assertEqual(e['parameter_parts'],measured_parameters(m))
                        self.assertEqual(e['executed_matrix_macs'],v['matrix_macs'])
                        self.assertEqual(e['router_contraction_mac_equivalents'],v['router_contraction_mac_equivalents'])
                        self.assertFalse(v['forbidden_attention'])
                        # M=dh=4 is a legitimate square context, never latent self-attention.
                        self.assertTrue(any(x['kind']=='K^T V' and x['output'][-2:]==[4,4] for x in v['contractions']))
                        self.assertEqual(len(v['contractions']),2*e['executed_depth'])
                        self.assertTrue(all(not s._forward_hooks and not s._forward_pre_hooks for s in m.modules()))

    def test_profiler_rejects_real_quadratic_attention(self):
        c=configuration('elasticity',PRESETS[0],MODES[0],small=True);m=build_from_config(c)
        target=m.loop.core[0].block.Attn;original=target.forward
        def bad(x):return original(x)+torch.bmm(torch.bmm(x,x.transpose(1,2)),x)*0
        with patch.object(target,'forward',bad):v=audit(m,inputs(c),B=2,N=35)
        self.assertTrue(v['forbidden_attention'])
        self.assertTrue(any(r.get('output',[])[-2:]==[35,35] for r in v['forbidden_attention']))

    def test_disabled_diagnostics_no_hooks_tensor_sync_or_rng(self):
        c=configuration('darcy',PRESETS[0],MODES[1],small=True);m=build_from_config(c)
        before=torch.get_rng_state().clone()
        with patch('tools.linearno_loop_diagnostics.stats',side_effect=AssertionError('disabled synchronized')):
            with LoopDiagnostics(m) as diag:
                self.assertTrue(all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules()))
                m(*inputs(c));self.assertIsNone(diag.gradients())
            self.assertEqual(diag.records,[])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_enabled_diagnostics_exact_forward_gradients_rng_and_sources(self):
        for mode in MODES:
            c=configuration('plasticity',PRESETS[0],mode,small=True);a=inputs(c)
            m=build_from_config(c);other=copy.deepcopy(m);before=torch.get_rng_state().clone()
            expected=m(*a);expected.square().mean().backward();rng=torch.get_rng_state().clone()
            torch.set_rng_state(before)
            with LoopDiagnostics(other,enabled=True) as d:
                actual=other(*a);actual.square().mean().backward();grad=d.gradients()
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for (name,p),(n,q) in zip(m.named_parameters(),other.named_parameters()):
                self.assertEqual(name,n)
                if p.grad is None:self.assertIsNone(q.grad)
                else:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
            record=d.records[0];self.assertEqual(len(record['visits']),6);self.assertEqual(len(record['routing']),12)
            self.assertEqual([r['sources'] for r in record['routers']],analytic(c)['router_sources'])
            self.assertEqual(len(grad),3);self.assertTrue(all(r['nan']==r['inf']==0 for r in grad))
            self.assertEqual([(r['round'],r['physical_core_index']) for r in record['visits']],[(r,i) for r in range(2) for i in range(3)])
            json.dumps(record,allow_nan=False) # no tensors/graphs retained
            self.assertFalse(d.pending);self.assertIsNone(d.previous_update);self.assertIsNone(d.partial)
            self.assertTrue(all(not x._forward_hooks and not x._forward_pre_hooks for x in other.modules()))

    def test_exception_and_changed_batch_no_retained_forward_state(self):
        c=configuration('elasticity',PRESETS[1],MODES[1],small=True);m=build_from_config(c).eval()
        a=inputs(c)
        with LoopDiagnostics(m,enabled=True) as d:
            with patch.object(m.loop.core[1].block.Attn,'forward',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):m(*a)
            self.assertFalse(d.records[-1]['completed']);self.assertFalse(d.pending);self.assertIsNone(d.partial)
            expected=m(*a);smaller=(a[0][:1,:11],None,None)
            m(*smaller);actual=m(*a)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertEqual([len(r['visits']) for r in d.records[1:]],[4,4,4])
            for r in d.records[1:]:self.assertEqual(r['visits'][0]['logical_visit'],0)

    def test_cuda_failure_exit_code_and_diagnostic_timing_guard(self):
        import tempfile
        from pathlib import Path
        from tools.linearno_loop_performance import main
        with tempfile.TemporaryDirectory() as directory:
            with patch('sys.argv',['performance','cuda','--output',str(Path(directory)/'result.json')]), \
                 patch('torch.set_num_threads'),patch('torch.set_num_interop_threads'), \
                 patch('tools.linearno_loop_performance.cuda_smoke',return_value=False):
                with self.assertRaises(SystemExit) as caught:main()
                self.assertEqual(caught.exception.code,1)
            with patch('sys.argv',['performance','cpu','--diagnostics','--output',str(Path(directory)/'result.json')]):
                with self.assertRaises(SystemExit) as caught:main()
                self.assertEqual(caught.exception.code,2)

    def test_custom_rank_controls_and_mode_router_costs(self):
        from linearno_loop.config import resolve_config
        for mode in MODES:
            c=resolve_config('car',options=dict(topology_preset='custom',prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1,residual_mode=mode,rank_multiplier=1))
            m=build_from_config(c);cost=analytic(c)
            self.assertEqual(cost['parameter_parts'],measured_parameters(m))
            self.assertEqual((cost['unique_depth'],cost['executed_depth']),(3,7))
            count=0 if mode==MODES[0] else 13 if mode==MODES[1] else 3
            self.assertEqual(cost['parameter_parts']['router'],2*cost['hidden']*count)

if __name__=='__main__':unittest.main()
