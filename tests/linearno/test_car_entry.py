"""Car native integration including strict failure and preserved old AST."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

ROOT=Path(__file__).resolve().parents[2]

class CarEntry(unittest.TestCase):
    def test_native_fresh_process_resume_eval_and_metadata_failures(self):
        with tempfile.TemporaryDirectory(prefix='linearno-car-native-') as tmp:
            root=Path(tmp)
            def worker(action,directory):
                report=root/(directory.name+'-'+action+'.json')
                p=subprocess.run([sys.executable,'-B',str(ROOT/'tests/linearno/car_entry_worker.py'),action,str(directory),str(report)],
                    cwd=ROOT/'Car-Design-ShapeNetCar',env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=180)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                return json.loads(report.read_text())
            uninterrupted=worker('train',root/'full')
            first=worker('interrupt',root/'split');rest=worker('resume',root/'split');evaluation=worker('eval',root/'split')
            for field in ('state_hash','resume_hash','prediction','scheduler_step','scheduler_total'):
                self.assertEqual(uninterrupted[field],rest[field],field)
            self.assertEqual(uninterrupted['batches'],first['batches']+rest['batches'])
            self.assertEqual(evaluation['prediction'],rest['prediction'])
            self.assertTrue(evaluation['metadata_immutable'])
            self.assertEqual((rest['global_step'],rest['scheduler_step'],rest['scheduler_total']),(9,9,12))
            # Same real parser, separate process, all checks precede dataset loading.
            code="""
import sys
sys.path.insert(0,sys.argv[1]+'/tests/linearno')
from car_entry_worker import parser_for
from models.cdlno_run import parse_args
parse_args(parser_for(True),evaluation=True,argv=['--cfd_model','LinearNO','--experiment-dir',sys.argv[2],*sys.argv[3:]])
"""
            for flags in (['--linearno-profile','official_release'],['--linearno-rank','8'],['--fold_id','2'],['--linearno-hidden','16']):
                result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),str(root/'split'),*flags],cwd=ROOT/'Car-Design-ShapeNetCar',env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertIn('conflict',result.stderr)
            path=root/'split/architecture.json';obj=json.loads(path.read_text());obj['family']='Transolver';path.write_text(json.dumps(obj))
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),str(root/'split')],cwd=ROOT/'Car-Design-ShapeNetCar',env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('family',result.stderr)

    def test_official_loss_matches_original_and_missing_data_fail_fast(self):
        from cdlno.linearno.car_entry import CarRun,inspect_data
        from cdlno.linearno.profiles import resolve_config
        from types import SimpleNamespace
        run=object.__new__(CarRun);run.config=resolve_config('car','official_release',contract='car_l7')
        y=torch.tensor([[1.,2,3,4],[2,3,4,5],[3,4,5,6]])
        out=(y+torch.tensor([[1.,2,3,7],[2,3,4,9],[4,5,6,2.]])).requires_grad_()
        data=Data(y=y,surf=torch.tensor([False,True,True]))
        total,p,v=run.objective(out,data)
        self.assertAlmostEqual(p.item(),(81+4)/2)
        self.assertAlmostEqual(v.item(),(1+4+9+4+9+16+16+25+36)/9,places=5)
        self.assertEqual(total.item(),(v+.5*p).item());total.backward();self.assertTrue(torch.isfinite(out.grad).all())
        with tempfile.TemporaryDirectory(prefix='linearno-missing-car-') as temp:
            with self.assertRaisesRegex(FileNotFoundError,'param0'):
                inspect_data(SimpleNamespace(data_dir=temp,save_dir=temp,preprocessed=1,fold_id=0))

    def test_exact_legacy_source_projection(self):
        from linearno_entry_projection import strip_linearno
        baseline=json.loads((ROOT/'docs/linearno_audit/l7/integration/baseline.json').read_text())
        for name in ('main.py','main_evaluation.py','train.py','models/cdlno_run.py'):
            before=Path(baseline['snapshot'])/'source/Car-Design-ShapeNetCar'/name
            after=ROOT/'Car-Design-ShapeNetCar'/name
            self.assertEqual(ast.dump(strip_linearno(ast.parse(after.read_text()))),ast.dump(ast.parse(before.read_text())),name)

    def test_data_preflight_order_and_eval_uses_saved_normalizer(self):
        code = r'''
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from cdlno.linearno.car_entry import inspect_data,load_data
from cdlno.linearno.schema import normalizer_record
args=SimpleNamespace(data_dir='/SYNTHETIC_ONLY/raw',save_dir='/SYNTHETIC_ONLY/cache',
    preprocessed=1,fold_id=2,cfd_mesh=True,r=.2,val_iter=10,eval=1)
folds=[[f'param{i}/sample-b',f'param{i}/sample-a'] for i in range(9)]
with patch('pathlib.Path.is_dir',return_value=True),patch('dataset.load_dataset.get_samples',return_value=folds),patch('pathlib.Path.is_file',return_value=True),patch('cdlno.linearno.car_entry.sha256',return_value='a'*64):
    spec=inspect_data(args)
    assert spec['test_samples']==folds[2]
    assert spec['train_samples'][:2]==folds[0]
    args._linearno_metadata={'data_spec':dict(spec,checksums=dict(spec['checksums'],normalizer_fit_dataset='b'*64))}
    assert inspect_data(args)==spec
    with patch('pathlib.Path.is_file',new=lambda p: not str(p).endswith('surf.npy')):
        try:inspect_data(args)
        except FileNotFoundError as error:assert 'surf.npy' in str(error)
        else:raise AssertionError('incomplete preprocessed sample silently accepted')
coef=(np.zeros(7),np.ones(7),np.zeros(4),np.ones(4))
args._linearno_metadata['normalizer_spec']={'records':{'coef_norm':normalizer_record(
    dict(zip(('input_mean','input_std','output_mean','output_std'),coef)),fit_split='train',data_checksum='a'*64,algorithm='synthetic')}}
calls=[]
def get_data(root,names,**kwargs):
    assert names==folds[2] and 'norm' not in kwargs
    for a,b in zip(kwargs['coef_norm'],coef):np.testing.assert_array_equal(a,b)
    calls.append(names);return ['in-memory-placeholder-for-routing']*len(names)
with patch('dataset.dataset.get_datalist',side_effect=get_data),patch('dataset.dataset.GraphDataset',side_effect=lambda data,**kwargs:data):
    train,test,restored=load_data(args,spec)
    assert train is None and len(test)==2 and len(calls)==1
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT/'Car-Design-ShapeNetCar',
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
