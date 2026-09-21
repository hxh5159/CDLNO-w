import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from cdlno.linearno_loop.v2.construction import build_from_config
from cdlno.linearno_loop.v2_projection import EXPECTED,project
from linearno_loop.config import resolve_config as resolve_v1
from linearno_loop.v2.config import run_directory_id as v2_run_id
from tran_evaluate.linearno_loop import launch,recording
from tran_evaluate.linearno_loop.ffn_matrix import matrix as command_matrix
from tools.linearno_loop_accounting import analytic,audit,measured_parameters
from tools.linearno_loop_ffn_diagnostics import V2LoopDiagnostics
from tools.linearno_loop_ffn_support import (
    CORE_FFN_MODES,PRESETS,RESIDUALS,TASKS,configuration,inputs,point_count,
)


ROOT=Path(__file__).resolve().parents[2]


class LF7DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_command_matrix_and_v1_v2_run_isolation(self):
        value=command_matrix()
        self.assertEqual(len(value['primary']),8*2*3*2*3)
        self.assertEqual(len(value['rank_x2_controls']),8*2)
        self.assertEqual(len({row['run_id'] for row in value['primary']}),len(value['primary']))
        for row in value['primary']:
            self.assertIn('--then-eval',row['train_then_eval'])
            self.assertIn('--linearno-loop-core-ffn-mode',row['train_then_eval'])
            self.assertNotIn('--linearno-loop-rank-multiplier',row['train_then_eval'])
            self.assertIn(row['run_id'],row['resume_then_eval']);self.assertIn(row['run_id'],row['eval'])
        for row in value['rank_x2_controls']:
            self.assertIn('--linearno-loop-rank-multiplier 2',row['train_then_eval'])
        for task in TASKS:
            runs=[]
            for mode in CORE_FFN_MODES:
                config=configuration(task,PRESETS[0],RESIDUALS[0],mode,seed=3)
                runs.append(v2_run_id(config));self.assertIn('__v2__',runs[-1]);self.assertIn(mode,runs[-1])
                self.assertEqual(config['loop_spec']['resolved_rank'],config['loop_spec']['base_rank'])
            self.assertNotEqual(*runs)
            old=resolve_v1(task,options={'topology_preset':PRESETS[0],'residual_mode':RESIDUALS[0]},
                           profile_overrides={'runtime.seed':3})
            self.assertEqual(old['loop_spec']['resolved_rank'],2*old['loop_spec']['base_rank'])
            self.assertNotIn('core_ffn_mode',old['loop_spec'])

    def test_real_launcher_train_dry_run_all_tasks_and_new_modes(self):
        environment={'CUDA_VISIBLE_DEVICES':''}
        paths=[]
        for task in TASKS:
            for mode in CORE_FFN_MODES:
                tokens=['--linearno-loop','1','--linearno-loop-topology','p2_c2_r2_s2',
                    '--linearno-loop-residual-mode','lb_attnres_1_over_r',
                    '--linearno-loop-core-ffn-mode',mode,'--seed','2','--gpu','0']
                value=launch.plan(task,'train',tokens,environment)
                self.assertEqual(value['config']['loop_spec']['core_ffn_mode'],mode)
                self.assertEqual(value['config']['loop_spec']['resolved_rank'],
                                 value['config']['loop_spec']['base_rank'])
                self.assertIn(mode,value['run']);self.assertFalse(Path(value['run']).exists())
                paths.append(value['run'])
        self.assertEqual(len(paths),len(set(paths)))

    def test_v2_recording_schedule_partitions_and_observation_neutrality(self):
        with tempfile.TemporaryDirectory(prefix='lf7-observer-') as temporary:
            common=[]
            for mode in CORE_FFN_MODES:
                config=configuration('darcy',PRESETS[0],'rb_attnres',mode,small=True)
                model=build_from_config(config);plain=copy.deepcopy(model);args=inputs(config)
                directory=Path(temporary)/mode;directory.mkdir()
                namespace=SimpleNamespace(_linearno_loop_config=config,linearno_run_dir=directory,
                    seed=17,linearno_task='darcy',eval=False,resume=False)
                before=torch.get_rng_state().clone();recording.observe(namespace,model)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                observed=model(*args);expected=plain(*args)
                torch.testing.assert_close(observed,expected,atol=0,rtol=0)
                saved=json.loads((directory/recording.MANIFEST).read_text());member=saved['members']['member_000']
                self.assertEqual(saved['schema_version'],2);self.assertEqual(saved['core_ffn_mode'],mode)
                self.assertEqual(member['actual_call_schedule'],saved['expected_call_schedule'])
                self.assertEqual(sum(member['parameter_parts'].values()),member['parameters'])
                self.assertEqual(member['point_ffn_instances'],8)
                self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))
                common.append(member['public_backbone_initial_sha256'])
            self.assertEqual(common[0],common[1])

    def test_full_profile_parameter_matrix_and_reduced_actual_macs(self):
        constructors=0
        for task in TASKS:
            for preset in PRESETS:
                for residual in RESIDUALS:
                    for mode in CORE_FFN_MODES:
                        config=configuration(task,preset,residual,mode)
                        model=build_from_config(config);expected=analytic(config)
                        self.assertEqual(expected['parameter_parts'],measured_parameters(model))
                        clone=build_from_config(config);clone.load_state_dict(model.state_dict(),strict=True)
                        constructors+=1;del model,clone
        self.assertEqual(constructors,96)
        for task in ('ns','elasticity','plasticity','darcy','airfrans','car'):
            config=configuration(task,PRESETS[0],'rb_attnres','round_specific_latent',small=True)
            model=build_from_config(config);args=inputs(config)
            B=1 if task in ('airfrans','car') else 2;N=point_count(config,False)
            expected=analytic(config,B=B,N=N);actual=audit(model,args,B=B,N=N)
            self.assertEqual(actual['matrix_macs'],expected['executed_matrix_macs'])
            self.assertEqual(actual['router_contraction_mac_equivalents'],expected['router_contraction_mac_equivalents'])
            self.assertFalse(actual['forbidden_attention'])

    def test_diagnostics_default_off_and_enabled_are_observational(self):
        config=configuration('darcy',PRESETS[0],'sr_1_over_r','round_specific_latent',small=True)
        first=build_from_config(config);second=copy.deepcopy(first);args=inputs(config)
        before=torch.get_rng_state().clone()
        with V2LoopDiagnostics(first) as disabled:
            expected=first(*args);expected.square().mean().backward()
            self.assertIsNone(disabled.gradients());self.assertEqual(disabled.records,[])
            self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in first.modules()))
        after=torch.get_rng_state().clone();torch.set_rng_state(before)
        with V2LoopDiagnostics(second,enabled=True) as enabled:
            actual=second(*args);actual.square().mean().backward();gradients=enabled.gradients()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0);self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for (_,left),(_,right) in zip(first.named_parameters(),second.named_parameters()):
            if left.grad is None:self.assertIsNone(right.grad)
            else:torch.testing.assert_close(left.grad,right.grad,atol=0,rtol=0)
        record=enabled.records[0];self.assertEqual(len(record['operators']),6)
        self.assertEqual(len(record['point_ffns']),6);self.assertEqual(len(record['latent']),6)
        self.assertEqual(len(gradients),12);json.dumps(record,allow_nan=False)
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in second.modules()))

    def test_v1_projection_is_exact_and_mutation_sensitive(self):
        for relative,expected in EXPECTED.items():
            text=(ROOT/relative).read_text();self.assertEqual(__import__('hashlib').sha256(text.encode()).hexdigest(),expected)
            self.assertNotEqual(project(relative,text),text)
            with self.assertRaisesRegex(ValueError,'unrecognized v2 dispatch source mutation'):
                project(relative,text+'\n# mutation probe\n')


if __name__=='__main__':unittest.main()
