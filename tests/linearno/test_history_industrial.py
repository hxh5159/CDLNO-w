"""R8 industrial native four-mode cycles, fair ensembles and source preservation."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from cdlno.linearno.profiles import resolve_config
from cdlno.linearno_history.config import resolve_config as resolve_history
from cdlno.linearno_history.factory import build_fair_model
from cdlno.linearno_history.provenance import baseline_source,ROUTING_REPLACEMENTS
from cdlno.linearno_history.checkpoint import inspect_checkpoint,read_json

ROOT=Path(__file__).resolve().parents[2]
SIGS=('A0K0','A1K0','A0K1','A1K1')

class HistoryIndustrial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.rows=[]
        cls.artifacts=Path(os.environ.get('LINEARNO_R8_ARTIFACTS',tempfile.mkdtemp(prefix='history-r8-')))
        cls.artifacts.mkdir(parents=True,exist_ok=True)
    @classmethod
    def tearDownClass(cls):
        if p:=os.environ.get('LINEARNO_R8_REPORT'):Path(p).write_text(json.dumps(cls.rows,indent=2))

    def worker(self,task,sig,action,directory,label):
        short='air' if task=='airfrans' else 'car'
        report=self.artifacts/f'{task}-{sig}-{label}.json';log=report.with_suffix('.log')
        with log.open('w') as stream:
            p=subprocess.run([sys.executable,'-B',str(ROOT/f'tests/linearno/history_{short}_worker.py'),action,str(directory),str(report),sig],
                cwd=ROOT/('Airfoil-Design-AirfRANS' if task=='airfrans' else 'Car-Design-ShapeNetCar'),
                env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg'},
                stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        self.assertEqual(p.returncode,0,log.read_text()[-6500:]);return json.loads(report.read_text())

    def test_eight_native_cycles_four_modes_and_fair_data(self):
        for task in ('car','airfrans'):
            previous=None;initial=None
            for sig in SIGS:
                with self.subTest(task=task,signature=sig):
                    seed=901 if task=='airfrans' else 19
                    name=f'{task}__paper_table8_on_release_model__L4__{sig}__seed{seed}'
                    full=self.artifacts/'continuous'/name;split=self.artifacts/'split'/name
                    a=self.worker(task,sig,'train',full,'continuous')
                    b=self.worker(task,sig,'interrupt',split,'interrupt')
                    c=self.worker(task,sig,'resume',split,'resume')
                    d=self.worker(task,sig,'eval',split,'eval')
                    for key in ('state_hash','resume_hash','prediction'):self.assertEqual(a[key],c[key],key)
                    self.assertEqual(a['batches'],b['batches']+c['batches']);self.assertEqual(a['prediction'],d['prediction'])
                    self.assertTrue(d['metadata_immutable'])
                    h=json.loads((full/'linearno_run_manifest.json').read_text())['public_backbone_initial_sha256']
                    if previous is not None:self.assertEqual(previous,a['batches']);self.assertEqual(initial,h)
                    else:previous=a['batches'];initial=h
                    metadata,_=inspect_checkpoint(split/'member_000' if task=='airfrans' else split)
                    self.assertEqual(metadata['objective_spec']['kind'],'MSE')
                    self.assertEqual(metadata['family'],'linearno' if sig=='A0K0' else 'linearno_history')
                    self.assertEqual('innovation_spec' in metadata,sig!='A0K0')
                    self.rows.append(dict(kind='native_cycle',task=task,signature=sig,depth=4,real_PyG=True,synthetic=True,all_exact=True,artifacts=str(split),force_metric='NOT RUN: raw VTK absent',status='PASS'))

    def test_fair_two_member_ensembles_and_two_resume_boundaries(self):
        previous=None;initial=None
        for sig in SIGS:
            name=f'airfrans__paper_table8_on_release_model__L4__{sig}__seed901'
            full=self.artifacts/'ensemble'/name
            a=self.worker('airfrans',sig,'ensemble',full,'ensemble')
            self.assertEqual(a['members'],[dict(member_id=f'member_{i:03d}',order=i) for i in range(2)])
            self.assertTrue(a['distinct_members'])
            hashes=[json.loads((full/f'initial_member_{i:03d}.json').read_text())['backbone_sha256'] for i in range(2)]
            if previous is not None:self.assertEqual(previous,a['batches']);self.assertEqual(initial,hashes)
            else:previous=a['batches'];initial=hashes
            for action in ('ensemble_interrupt','ensemble_later_interrupt'):
                split=self.artifacts/action/name
                b=self.worker('airfrans',sig,action,split,action)
                c=self.worker('airfrans',sig,'resume',split,action+'-resume')
                d=self.worker('airfrans',sig,'eval',split,action+'-eval')
                for key in ('member_hashes','resume_hash'):self.assertEqual(a[key],c[key],key)
                self.assertEqual(a['batches'],b['batches']+c['batches']);self.assertEqual(a['prediction'],d['prediction'])
            self.rows.append(dict(kind='ensemble',signature=sig,members=2,boundaries=['between_members','inside_member_1'],exact=True,status='PASS'))

    def test_forty_presets_and_task_specific_math(self):
        for task in ('airfrans','car'):
            for depth in range(4,9):
                for sig in SIGS:
                    c=resolve_history(resolve_config(task,explicit={'model.layers':depth}),family='linearno_history',features=dict(linearno_latent_attnres=sig[1]=='1',linearno_history_k_conditioning=sig[3]=='1'))
                    model=build_fair_model(c)
                    self.assertEqual(model.blocks[0].Attn.rank,32)
                    keys=set(model.state_dict())
                    self.assertIn('blocks.0.Attn.temperature' if task=='airfrans' else 'blocks.0.Attn.tempreature_k',keys)
                    self.rows.append(dict(kind='preset_count',task=task,signature=sig,depth=depth,parameters=sum(p.numel() for p in model.parameters()),actual_M=32))

    def test_old_source_fingerprints_task_bodies_and_launchers(self):
        from cdlno.linearno.standard_entry import provenance
        from cdlno.linearno.air_entry import _provenance
        from cdlno.linearno.car_entry import task_provenance
        old=json.loads((ROOT/'docs/linearno_history_audit/r8/baseline-provenance.json').read_text())
        for task,fn in [('standard',provenance),('airfrans',_provenance),('car',task_provenance)]:
            now=fn()
            for key in ('source_sha256','normalized_patch_sha256'):self.assertEqual(now[key],old[task][key])
        before=Path(json.loads((ROOT/'docs/linearno_history_audit/r8/baseline.json').read_text())['snapshot'])/'source'
        for relative in ('Airfoil-Design-AirfRANS/train.py','cdlno/linearno/air_entry.py','cdlno/linearno/car_entry.py'):
            self.assertEqual(baseline_source(relative,(ROOT/relative).read_text()),(before/relative).read_text())
        for task in ('car','airfrans'):
            for action in ('train','resume','eval'):
                p=subprocess.run(['bash',str(ROOT/f'tran_evaluate/linearno_history/{task}.sh'),action,'--dry-run','--experiment-dir','test-run'],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
        # Whole old objective/eval/data helpers are reused as exact function objects.
        from cdlno.linearno import car_entry as pure_car, air_entry as pure_air
        from cdlno.linearno_history import car_entry as hist_car, air_entry as hist_air
        for name in ('evaluate','inspect_data','load_data','restore_coef_norm'):
            self.assertIs(getattr(pure_car,name),getattr(hist_car,name))
        for name in ('_load_dataset','_data_spec','_pressure_rL2_dataset','_restore_coef_norm'):
            self.assertIs(getattr(pure_air,name),getattr(hist_air,name))
        self.assertEqual(ast.dump(ast.parse(__import__('inspect').getsource(pure_car.CarRun.objective).strip())),ast.dump(ast.parse(__import__('inspect').getsource(hist_car.CarRun.objective).strip())))

    def test_production_metadata_conflicts_before_tensor_load(self):
        code = '''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1]+'/tests/linearno')
task=sys.argv[2];directory=sys.argv[3]
if task=='airfrans':
 from history_air_worker import parser_for,parse_args
 base=['--model','LinearNO','--experiment-dir',directory]
else:
 from history_car_worker import parser_for,parse_args
 base=['--cfd_model','LinearNO','--experiment-dir',directory]
for flags in (['--linearno_latent_attnres','0'],['--linearno_history_k_conditioning','0'],['--linearno-rank','99'],['--linearno-layers','5'],['--linearno-heads','4'],['--linearno-profile','official_release']):
 with patch('torch.load',side_effect=AssertionError('tensor load before metadata conflict')):
  try:parse_args(parser_for(True),evaluation=True,argv=base+flags)
  except (SystemExit,ValueError):pass
  else:raise AssertionError('wrong config accepted: '+str(flags))
print('six conflicts rejected before tensor load')
'''
        for task in ('car','airfrans'):
            seed=901 if task=='airfrans' else 19
            directory=self.artifacts/'split'/f'{task}__paper_table8_on_release_model__L4__A1K1__seed{seed}'
            p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),task,str(directory)],
                cwd=ROOT/('Airfoil-Design-AirfRANS' if task=='airfrans' else 'Car-Design-ShapeNetCar'),
                env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

if __name__=='__main__':unittest.main()
