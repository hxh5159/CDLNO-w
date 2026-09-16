"""Real PyG and original safe Car train/test functions, synthetic graphs only."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import torch
from test_shapenet_car import CAR, ROOT, entry_parser, HAS_PYG
from models.cdlno_run import parse_args
from models.KCDNO import Model
from cdlno.kcdno.industrial_entry import CarRun, car_model_kwargs
from cdlno.kcdno.metadata import KCDNOMetadataMismatch
from output_recording_projection import strip_recording
from test_kcdno_tasks import strip_new


def arguments(flags=(),evaluation=False):
    return parse_args(entry_parser(evaluation),evaluation=evaluation,argv=['--cfd_model','kcdno',*flags])
SMALL=['--n_hidden','8','--n_heads','2','--n_layers','2','--slice_num','3','--kernel-rank','5']


class CarKCDNO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_factory_defaults_and_frozen_sources(self):
        args=arguments();self.assertEqual((args.n_hidden,args.n_heads,args.slice_num,args.nb_epochs,args.batch_size),(256,8,64,200,1))
        a=arguments(['--profile','transolver_shape_match']);self.assertEqual(a.slice_num,32)
        before=json.loads((ROOT/'docs/kcdno_audit/k6/before.json').read_text());snap=Path(before['source_snapshot'])
        for name in ('main.py','main_evaluation.py'):
            self.assertEqual(ast.dump(strip_new(ast.parse((CAR/name).read_text()))),ast.dump(ast.parse((snap/'Car-Design-ShapeNetCar'/name).read_text())))
        from visualization_projection import strip_visualization
        from msar_entry_projection import strip_msar
        self.assertEqual(ast.dump(strip_visualization(strip_msar(ast.parse((CAR/'train.py').read_text())))),
                         ast.dump(ast.parse((snap/'Car-Design-ShapeNetCar/train.py').read_text())))
        with self.assertRaises(ValueError):arguments(['--batch_size','2'])
        with self.assertRaises(ValueError):arguments(['--front_latent_mode','no_sa'])

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_real_PyG_original_loss_step_and_checkpoints(self):
        from torch_geometric.data import Data,Batch
        import importlib.util
        spec=importlib.util.spec_from_file_location('_kcdno_car_train',CAR/'train.py');train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
        for mode in ('all','off'):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                path=Path(tmp)/'run';args=arguments([*SMALL,'--history-mode',mode,'--nb_epochs','3','--kcdno-run-dir',str(path)])
                model=Model(**car_model_kwargs(args));run=CarRun(args,device='cpu',model=model)
                graphs=[]
                for n in (11,19):
                    data=Data(x=torch.randn(n,7),y=torch.randn(n,4),surf=torch.arange(n)%2==0)
                    data=Batch.from_data_list([data]);geom=Data(x=torch.randn(5,3))
                    before=data.clone();model.eval();out=model((data,geom));self.assertEqual(out.shape,(n,4))
                    data.y+=100
                    torch.testing.assert_close(model((data,geom)),out,atol=0,rtol=0)
                    data.y=before.y
                    for key in before.to_dict():torch.testing.assert_close(data[key],before[key],atol=0,rtol=0)
                    graphs.append((data,geom))
                opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=20)
                # Real original velocity-all + surface-pressure weighted loss.
                train.train('cpu',model,graphs,opt,scheduler,reg=.5)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                train.test('cpu',model,graphs);model.eval();expected=model(graphs[0])
                with self.assertRaises(ValueError):model((Batch.from_data_list([graphs[0][0].to_data_list()[0],graphs[1][0].to_data_list()[0]]),None))
                torch.save(model,run.checkpoint)  # original whole-model protocol
                files={p.name:p.read_bytes() for p in (run.sidecar,path/'task.json')}
                ev=arguments(['--kcdno-run-dir',str(path)],True)
                self.assertEqual((ev.nb_epochs,ev.history_mode),(3,mode))
                loaded=CarRun(ev,device='cpu',evaluation=True).load().eval()
                torch.testing.assert_close(loaded(graphs[0]),expected,atol=0,rtol=0)
                for flag,value in (('--kernel-rank','7'),('--history-mode','off' if mode=='all' else 'all'),('--fold_id','1')):
                    with self.assertRaises(KCDNOMetadataMismatch):arguments(['--kcdno-run-dir',str(path),flag,value],True)
                self.assertEqual(files,{p.name:p.read_bytes() for p in (run.sidecar,path/'task.json')})
                torch.save(dict(x=graphs[0][0].x,expected=expected),Path(tmp)/'probe.pt')
                code='''
import sys, torch
from pathlib import Path
from types import SimpleNamespace
from test_kcdno_car import arguments
from cdlno.kcdno.industrial_entry import CarRun
torch.set_num_threads(1)
path=Path(sys.argv[1]);a=arguments(['--kcdno-run-dir',str(path/'run')],True)
m=CarRun(a,device='cpu',evaluation=True).load().eval()
probe=torch.load(path/'probe.pt',weights_only=True)
torch.testing.assert_close(m((SimpleNamespace(x=probe['x']),None)),probe['expected'],atol=0,rtol=0)
assert 'main' not in sys.modules and 'main_evaluation' not in sys.modules
'''
                result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=CAR,env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
