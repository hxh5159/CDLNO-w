"""Synthetic real PyG inputs and actual safe Air loss functions; no sampling."""
import ast
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
import yaml
from test_airfrans import AIR, ROOT, entry, parser, module, HAS_PYG
from test_kcdno_tasks import strip_new
from cdlno.kcdno.airfrans import AirfRANSModel
from cdlno.kcdno.air_entry import AirRun, model_kwargs
from cdlno.kcdno.metadata import KCDNOMetadataMismatch

SMALL=['--n_hidden','8','--n_heads','2','--n_layers','2','--slice_num','3','--kernel-rank','5']

def arguments(flags=(),evaluation=False):
    return entry.parse_args(parser(evaluation),evaluation=evaluation,argv=['--model','kcdno',*flags])

def hparams():return yaml.safe_load((AIR/'params.yaml').read_text())['kcdno']


class AirKCDNO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_defaults_and_exact_frozen_projection(self):
        a=arguments();self.assertEqual((a.n_layers,a.n_hidden,a.n_heads,a.slice_num,a.nb_epochs,a.batch_size),(8,256,8,64,398,1))
        a=arguments(['--profile','transolver_shape_match']);self.assertEqual(a.slice_num,32)
        snapshot=Path(json.loads((ROOT/'docs/kcdno_audit/k7/before.json').read_text())['source_snapshot'])/'Airfoil-Design-AirfRANS'
        for name in ('main.py','main_evaluation.py'):
            self.assertEqual(ast.dump(strip_new(ast.parse((AIR/name).read_text()))),ast.dump(ast.parse((snapshot/name).read_text())))
        from visualization_projection import strip_visualization
        from msar_entry_projection import strip_msar
        self.assertEqual(ast.dump(strip_visualization(strip_msar(ast.parse((AIR/'train.py').read_text())))),
                         ast.dump(ast.parse((snapshot/'train.py').read_text())))
        for name in ('utils/metrics.py','dataset/dataset.py'):
            self.assertEqual((AIR/name).read_bytes(),(snapshot/name).read_bytes())
        saved=yaml.safe_load((snapshot/'params.yaml').read_text());current=yaml.safe_load((AIR/'params.yaml').read_text())
        self.assertEqual({k:v for k,v in current.items() if k not in ('kcdno','lrsa_matched','msar_lno')},saved)
        self.assertEqual(current['kcdno'],saved['Transolver'])
        with self.assertRaises(ValueError):arguments(['--batch_size','2'])
        with self.assertRaises(ValueError):arguments(['--front_latent_mode','full'])

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_real_graph_mask_loss_step_and_list_load(self):
        from torch_geometric.data import Data,Batch
        from torch_geometric.loader import DataLoader
        train=module(AIR/'train.py','_kcdno_air_train')
        for mode in ('all','off'):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                path=Path(tmp)/'run'
                a=arguments([*SMALL,'--history-mode',mode,'--nmodel','2','--nb_epochs','3','--kcdno-run-dir',str(path)])
                run=AirRun(a,hparams(),device='cpu');models=[AirfRANSModel(**model_kwargs(a)) for _ in range(2)]
                graphs=[]
                for n in (11,19):
                    g=Batch.from_data_list([Data(x=torch.randn(n,7),pos=torch.randn(n,2),y=torch.randn(n,4),surf=torch.arange(n)%2==0,edge_index=torch.tensor([[0,1],[1,0]]))])
                    old=g.clone();m=models[0].eval();y=m(g);self.assertEqual(y.shape,(n,4));g.y+=17
                    torch.testing.assert_close(m(g),y,atol=0,rtol=0);g.y=old.y
                    for key in old.to_dict():torch.testing.assert_close(g[key],old[key],atol=0,rtol=0)
                    # Original sampled-ptr contract, original point order retained.
                    part=g.clone();idx=torch.tensor([5,1,7]);part.x=g.x[idx];part.pos=g.pos[idx];part.batch=g.batch[idx]
                    self.assertEqual(m(part).shape,(3,4))
                    graphs.append(old)
                with self.assertRaises(ValueError):models[0](Batch.from_data_list([g.to_data_list()[0] for g in graphs]))
                optimizer=torch.optim.Adam(models[0].parameters(),lr=.001)
                scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,total_steps=20)
                train.train('cpu',models[0],graphs,optimizer,scheduler,criterion='MSE_weighted',reg=1.)
                train.test('cpu',models[0],graphs,criterion='MSE')
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in models[0].parameters()))
                for m in models:m.eval()
                torch.save(models,run.checkpoint)
                for i,m in enumerate(models):
                    Path(run.member_dir(i)).mkdir();torch.save(m,Path(run.member_dir(i))/'model')
                before={f.name:f.read_bytes() for f in (run.sidecar,path/'task.json')}
                ev=arguments(['--kcdno-run-dir',str(path)],True)
                self.assertEqual((ev.nmodel,ev.nb_epochs,ev.history_mode),(2,3,mode))
                er=AirRun(ev,hparams(),device='cpu',evaluation=True);loaded=er.load()
                for i,m in enumerate(models):
                    torch.testing.assert_close(loaded[i](graphs[0]),m(graphs[0]),atol=0,rtol=0)
                    torch.testing.assert_close(er.load(member=i)(graphs[0]),m(graphs[0]),atol=0,rtol=0)
                for flag,value in (('--kernel-rank','6'),('--history-mode','off' if mode=='all' else 'all'),('--nb_epochs','4'),('--task','aoa'),('--nmodel','1')):
                    with self.assertRaises(KCDNOMetadataMismatch):arguments(['--kcdno-run-dir',str(path),flag,value],True)
                self.assertEqual(before,{f.name:f.read_bytes() for f in (run.sidecar,path/'task.json')})
                torch.save(dict(x=graphs[0].x,pos=graphs[0].pos,expected=models[0](graphs[0])),Path(tmp)/'probe.pt')
                code='''
import sys,torch
from pathlib import Path
from types import SimpleNamespace
from test_kcdno_airfrans import arguments,hparams
from cdlno.kcdno.air_entry import AirRun
torch.set_num_threads(1)
p=Path(sys.argv[1]);a=arguments(['--kcdno-run-dir',str(p/'run')],True)
r=AirRun(a,hparams(),device='cpu',evaluation=True)
d=torch.load(p/'probe.pt',weights_only=True);g=SimpleNamespace(x=d['x'],pos=d['pos'])
for m in (r.load()[0],r.load(member=0)):
 torch.testing.assert_close(m(g),d['expected'],atol=0,rtol=0)
assert 'main' not in sys.modules and 'main_evaluation' not in sys.modules
'''
                result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=AIR,env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
