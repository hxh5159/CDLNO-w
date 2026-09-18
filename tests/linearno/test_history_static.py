"""R6 native Standard task routing and real-shape synthetic epoch acceptance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from linearno.static_worker import ROOT, PROJECT, SHAPES
from linearno.test_static_integration import args_for
from model_dict import get_model
from cdlno.linearno.standard_entry import model_kwargs, provenance
from cdlno.linearno_history import checkpoint as ck
from cdlno.linearno_history.provenance import baseline_source, ROUTING_REPLACEMENTS
from cdlno.training_state import _same

SIGS=('A0K0','A1K0','A0K1','A1K1')
TASKS=('airfoil','darcy','elasticity','pipe')


class HistoryStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.rows=[]
    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LINEARNO_R6_REPORT'):Path(path).write_text(json.dumps(cls.rows,indent=2))

    def test_native_config_factory_counts_and_no_legacy_RNG_change(self):
        for task in TASKS:
            for layers in range(4,9):
                a0=args_for(task,'--n-layers',layers)
                torch.manual_seed(45);pure=get_model(a0).Model(**model_kwargs(a0));state=torch.get_rng_state()
                for sig in SIGS:
                    args=args_for(task,'--n-layers',layers,'--linearno_latent_attnres',sig[1],
                                  '--linearno_history_k_conditioning',sig[3])
                    torch.manual_seed(45);model=get_model(args).Model(**model_kwargs(args))
                    self.assertTrue(torch.equal(state,torch.get_rng_state()))
                    for k,v in pure.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
                    count=sum(p.numel() for p in model.parameters())
                    self.rows.append(dict(kind='actual-task-parameter-count',task=task,depth=layers,signature=sig,parameters=count,
                        shape=dict(B='task batch',H=8,M=64,d_h=16,N=SHAPES[task][0]*SHAPES[task][1],
                          A_cross='[B,8,64,64] per real source' if sig[1]=='1' else None,
                          K_read='[B,8,64,l*64]' if sig[3]=='1' else None,
                          K_delta='[B,8,N,64]' if sig[3]=='1' else None)))
                    if sig=='A0K0':self.assertIs(type(model),type(pure));self.assertFalse(hasattr(args,'_linearno_history_config'))
        for task in ('unrecognized_task',):
            from linearno.static_worker import parser_for
            from cdlno_entry import parse_args
            with self.assertRaises(SystemExit):parse_args(parser_for('airfoil'),task,['--model','LinearNO_Structured_Mesh_2D','--linearno_latent_attnres','1'])

    def test_old_source_fingerprint_and_frozen_task_bodies(self):
        baseline=json.loads((ROOT/'docs/linearno_history_audit/r6/baseline.json').read_text())
        before=Path(baseline['snapshot'])/'source'
        for path in ROUTING_REPLACEMENTS:
            self.assertEqual(baseline_source(path,(ROOT/path).read_text()),(before/path).read_text(),path)
        for short in ('airfoil','darcy','elas','pipe','ns','plas'):
            path=f'PDE-Solving-StandardBenchmark/exp_{short}.py'
            self.assertEqual((ROOT/path).read_bytes(),(before/path).read_bytes())
        old=json.loads((ROOT/'docs/linearno_history_audit/r6/baseline-provenance.json').read_text())
        for key in ('source_sha256','normalized_patch_sha256'):self.assertEqual(provenance()[key],old[key])

    def test_sixteen_native_cycles_and_new_process_eval(self):
        artifacts=Path(os.environ.get('LINEARNO_R6_ARTIFACTS',tempfile.mkdtemp(prefix='history-r6-native-')))
        artifacts.mkdir(exist_ok=True,parents=True)
        orders={}
        initial_hashes={}
        for task in TASKS:
            for sig in SIGS:
                with self.subTest(task=task,signature=sig):
                    name=f'{task}__paper_table8_on_release_model__L4__{sig}__seed17'
                    continuous=artifacts/'continuous'/name;split=artifacts/'split'/name
                    reports={}
                    for action,directory,label in [('train',continuous,'continuous'),('interrupt',split,'interrupt'),('resume',split,'resume'),('eval',split,'eval')]:
                        report=artifacts/f'{name}-{label}.json';log=report.with_suffix('.log')
                        with log.open('w') as stream:
                            p=subprocess.run([sys.executable,'-B',str(ROOT/'tests/linearno/history_static_worker.py'),task,action,str(directory),str(report),sig],
                                cwd=PROJECT,env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
                                stdout=stream,stderr=subprocess.STDOUT,timeout=180)
                        self.assertEqual(p.returncode,0,log.read_text()[-3500:]);reports[label]=json.loads(report.read_text())
                    mc,pc=ck.inspect_checkpoint(continuous);ms,ps=ck.inspect_checkpoint(split)
                    self.assertTrue(_same(ck._read_pair(pc,mc),ck._read_pair(ps,ms)))
                    for field in ('optimizer','scheduler','rng','dataloader_generators','sampler_state'):
                        self.assertEqual(mc['resume_state'][field],ms['resume_state'][field])
                    self.assertEqual(reports['continuous']['batches'],reports['interrupt']['batches']+reports['resume']['batches'])
                    self.assertEqual(reports['continuous']['prediction_hash'],reports['eval']['prediction_hash'])
                    self.assertTrue(reports['eval']['immutable_metadata'])
                    initial=json.loads((continuous/'linearno_run_manifest.json').read_text())['public_backbone_initial_sha256']
                    if task in orders:
                        self.assertEqual(orders[task],reports['continuous']['batches'])
                        self.assertEqual(initial_hashes[task],initial)
                    else:orders[task]=reports['continuous']['batches'];initial_hashes[task]=initial
                    self.rows.append(dict(kind='native-cycle',task=task,signature=sig,depth=4,real_N=SHAPES[task][0]*SHAPES[task][1],
                        all_exact=True,synthetic=True,artifacts=str(split),normalizer_no_refit=True,status='PASS'))

    def test_launcher_preview_and_invalid_flags(self):
        for task in TASKS:
            for action in ('train','resume','eval'):
                script=ROOT/f'tran_evaluate/linearno_history/{task}.sh'
                p=subprocess.run(['bash',str(script),action,'--dry-run','--gpu','1'],text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--model LinearNO_',p.stdout)
        args_for('airfoil','--linearno_latent_attnres','1','--linearno_attnres_history_dropout_p','0')
        with self.assertRaises((ValueError,SystemExit)):
            args_for('airfoil','--linearno_latent_attnres','1','--linearno_history_k_conditioning','1','--linearno_attnres_history_dropout_p','0')
        with self.assertRaises(SystemExit):args_for('airfoil','--linearno_history_k_conditioning','true')

if __name__=='__main__':unittest.main()
