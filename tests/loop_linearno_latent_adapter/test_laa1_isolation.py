import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from linearno_loop.v3.config import resolve_config,run_directory_id
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR,RESIDUAL_MODES
from .support import config

ROOT=Path(__file__).resolve().parents[2]

class IsolationTests(unittest.TestCase):
    def test_152_old_config_bytes_and_default_rank_unchanged(self):
        from cdlno.linearno.profiles import resolve_config as pure
        from linearno_loop.config import resolve_config as v1
        from linearno_loop.v2.config import resolve_config as v2
        from linearno_loop.contracts import canonical_json,PRESETS
        path=ROOT/'docs/loop_linearno_latent_adapter_audit/laa1/legacy-config-baseline.json'
        rows=json.loads(path.read_text());self.assertEqual(len(rows),152)
        for row in rows:
            family=row['family'];old=row['config']
            if family=='pure':new=pure(row['task'])
            else:
                r=old['request'];new=(v1 if family=='v1' else v2)(row['task'],r['profile'],options=r['options'],profile_overrides=r['profile_overrides'])
            self.assertEqual(canonical_json(new),canonical_json(old))
        self.assertEqual(set(PRESETS),{'p1_c3_r2_s1','p2_c2_r2_s2'})
        opts=dict(topology_preset='p2_c2_r2_s2',residual_mode='sr_1_over_r')
        self.assertEqual(v1('darcy',options=opts)['loop_spec']['resolved_rank'],128)
        self.assertEqual(v2('darcy',options={**opts,'core_ffn_mode':'round_specific'})['loop_spec']['resolved_rank'],64)
        for api,extra in [(v1,{}),(v2,{'core_ffn_mode':'round_specific'})]:
            with self.assertRaises(ValueError):api('car',options={**opts,**extra,'linearno_rank':31})
    def test_fresh_process_import_guard_and_python_rng(self):
        code='''
import builtins,random,sys,json
before=random.getstate();old_import=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy') or name.startswith(('cdlno.linearno_loop','tran_evaluate')) or 'exp_' in name:
  raise AssertionError('forbidden import '+name)
 return old_import(name,*args,**kwargs)
builtins.__import__=guarded
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.costs import analytic_cost
from linearno_loop.v3.matrix import custom_cases
from loop_linearno_latent_adapter.support import metadata
from linearno_loop.v3.schema import restore_config
c=resolve_config('car',options={'architecture':'operator_latent_adapter_v3'})
restore_config(metadata(c));analytic_cost(c);custom_cases()
assert before==random.getstate()
assert not any(n=='torch' or n.startswith('torch.') or n.startswith('cdlno.linearno_loop') for n in sys.modules)
print('stdlib schema import guard and global RNG PASS')
'''
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'tests')+':'+str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_preloaded_numpy_torch_rng_zero_effect_and_no_model_import(self):
        code='''
import torch,numpy as np,random
p=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone()
from loop_linearno_latent_adapter.support import config,metadata
from linearno_loop.v3.schema import restore_config
from linearno_loop.v3.costs import analytic_cost
for task in ('car','airfrans','darcy'):
 c=config(task);restore_config(metadata(c));analytic_cost(c)
assert p==random.getstate()
b=np.random.get_state();assert n[0]==b[0] and np.array_equal(n[1],b[1]) and n[2:]==b[2:]
assert torch.equal(t,torch.get_rng_state())
print('preloaded Python/NumPy/Torch CPU RNG unchanged; no model constructed')
'''
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'tests')+':'+str(ROOT),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_paired_init_seeds_independent_of_feature_and_residual(self):
        for task in ('darcy','car','ns'):
            baseline=None;ids=set()
            for mode in RESIDUAL_MODES:
                for latent in (False,True):
                    for adapter in ('none','bilateral_qk_lowrank_second_visit'):
                        c=config(task,residual_mode=mode,latent_enabled=latent,adapter_mode=adapter)
                        fair=c['fair_comparison']
                        if baseline is None:baseline=fair
                        self.assertEqual(fair,baseline)
                        ids.add(run_directory_id(c))
            self.assertEqual(len(ids),12)
            for seed in (1,2):
                c=resolve_config(task,options={'architecture':ARCHITECTURE_SELECTOR},profile_overrides={'runtime.seed':seed})
                self.assertNotEqual(c['fair_comparison']['backbone_pair_id'],baseline['backbone_pair_id'])
                self.assertNotEqual(c['fair_comparison']['dataloader_generators'],baseline['dataloader_generators'])
    def test_hash_stable_order_independent_and_run_id_bounded(self):
        opts=dict(architecture=ARCHITECTURE_SELECTOR,cost_profile='efficient_v1',executed_depth=60,residual_mode='lb_attnres_1_over_r')
        a=resolve_config('plasticity',options=opts)
        b=resolve_config('plasticity',options=dict(reversed(list(opts.items()))))
        self.assertEqual(a['config_hash'],b['config_hash']);self.assertEqual(run_directory_id(a),run_directory_id(b))
        self.assertLess(len(run_directory_id(a).encode()),256)
        self.assertIn(a['config_hash'],run_directory_id(a))
