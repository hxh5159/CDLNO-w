"""Finite eight-task coverage of the trainable matched-full control."""
import io
import importlib.util
import json
import tempfile
import os
import sys
import subprocess
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import torch
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.core import KCDNO
from cdlno.kcdno.matched import MatchedLRSA
from cdlno.kcdno.matched_config import MatchedLRSAConfig
from cdlno.kcdno.families import resolve_training, architecture_from_dict
from cdlno.kcdno.metadata import KCDNOMetadataMismatch
from test_kcdno_tasks import arguments,construct,sample,StandardRun,SMALL,TASKS,TEMPORAL


class MatchedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_formula_off_equivalence_full_lock_and_storage(self):
        from cdlno.modules import _SelfAttention
        for point in ('point_ffn','conv_ffn'):
            full=MatchedLRSA(MatchedLRSAConfig(L=2,d=8,h=2,M=3,point_module=point))
            off=KCDNO(KCDNOArchitectureConfig(L=2,d=8,h=2,M=3,history_mode='off',point_module=point))
            retained={k:v for k,v in full.state_dict().items() if '.latent_sa.' not in k and '.latent_norm_sa.' not in k}
            self.assertEqual(retained.keys(),off.state_dict().keys())
            off.load_state_dict(retained,strict=True)
            for block in full.blocks:
                block.latent_sa.attn.to_out.weight.data.zero_();block.latent_sa.attn.to_out.bias.data.zero_()
                self.assertEqual(block.front_latent_mode,'full')
            x=torch.randn(2,35,8);grid=(5,7) if point=='conv_ffn' else None
            torch.testing.assert_close(full(x,grid_shape=grid),off(x,grid_shape=grid),atol=0,rtol=0)
            self.assertEqual(sum(isinstance(m,_SelfAttention) for m in full.modules()),2)
            self.assertFalse(any('reader' in n or 'writer' in n for n,_ in full.named_parameters()))
            ptrs=[p.data_ptr() for p in full.parameters()];self.assertEqual(len(ptrs),len(set(ptrs)))
            for flag in ('front_latent_mode','kernel_rank','history_mode'):
                with self.assertRaises(ValueError):resolve_training('darcy',{'model':'lrsa_matched',flag:'full'})
        with self.assertRaises(ValueError):MatchedLRSAConfig(latent_processor='no_sa')
        with self.assertRaises(ValueError):architecture_from_dict({'family':'CDLNO'})

    def test_six_standard_forward_step_and_strict_checkpoint(self):
        for task in TASKS|TEMPORAL:
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                a=arguments(task,[*SMALL[:-2],'--model','lrsa_matched','--kcdno-run-dir',str(Path(tmp)/'run')])
                m=construct(task,a);self.assertIsInstance(m.core,MatchedLRSA)
                self.assertEqual(m.config.family,'lrsa_matched')
                opt=torch.optim.AdamW(m.parameters(),lr=.001);x,fx=sample(m)
                ins=(x,fx,torch.rand(2,1)) if task=='plasticity' else (x,fx)
                y=m(*ins);y.square().mean().backward();opt.step();m.eval();expected=m(*ins)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                r=StandardRun(a,m);r.save(m);before=r.sidecar.read_bytes()
                e=arguments(task,['--model','lrsa_matched','--eval','1','--kcdno-run-dir',str(r.directory)])
                loaded=construct(task,e);StandardRun(e,loaded).load(loaded);loaded.eval()
                torch.testing.assert_close(loaded(*ins),expected,atol=0,rtol=0)
                with self.assertRaises(KCDNOMetadataMismatch):arguments(task,['--eval','1','--kcdno-run-dir',str(r.directory)])
                self.assertEqual(before,r.sidecar.read_bytes())
                self.assertEqual(json.loads(before)['family'],'lrsa_matched')

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric") is not None, "real PyG unavailable")
    def test_industrial_matched_protocols(self):
        from torch_geometric.data import Data,Batch
        from test_kcdno_car import arguments as car_args,Model,CarRun,car_model_kwargs
        from test_kcdno_airfrans import arguments as air_args,AirfRANSModel,AirRun,model_kwargs,hparams
        for task in ('car','airfrans'):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                root=Path(tmp)/'run';flags=['--n_hidden','8','--n_heads','2','--n_layers','2','--slice_num','3','--kcdno-run-dir',str(root)]
                fn=car_args if task=='car' else air_args;sel=['--cfd_model' if task=='car' else '--model','lrsa_matched']
                a=fn([*flags,*sel]);model=Model(**car_model_kwargs(a)) if task=='car' else AirfRANSModel(**model_kwargs(a))
                self.assertIsInstance(model.core,MatchedLRSA)
                g=Batch.from_data_list([Data(x=torch.randn(11,7),pos=torch.randn(11,2),y=torch.randn(11,4),surf=torch.arange(11)%2==0)])
                inputs=((g,None),) if task=='car' else (g,)
                opt=torch.optim.Adam(model.parameters(),lr=.001);model(*inputs).square().mean().backward();opt.step();model.eval();expected=model(*inputs)
                if task=='car':
                    run=CarRun(a,device='cpu',model=model);torch.save(model,run.checkpoint)
                else:
                    run=AirRun(a,hparams(),device='cpu');torch.save([model],run.checkpoint)
                    Path(run.member_dir(0)).mkdir();torch.save(model,Path(run.member_dir(0))/'model')
                before=run.sidecar.read_bytes();e=fn([*sel,'--kcdno-run-dir',str(root)],True)
                loaded=CarRun(e,device='cpu',evaluation=True).load() if task=='car' else AirRun(e,hparams(),device='cpu',evaluation=True).load()[0]
                torch.testing.assert_close(loaded(*inputs),expected,atol=0,rtol=0)
                with self.assertRaises(KCDNOMetadataMismatch):fn(['--kcdno-run-dir',str(root)],True)
                self.assertEqual(before,run.sidecar.read_bytes())
                torch.save(dict(x=g.x,pos=g.pos,expected=expected),Path(tmp)/'probe.pt')
                code = """
import sys,torch
from pathlib import Path
from types import SimpleNamespace
from test_kcdno_car import arguments as ca,CarRun
from test_kcdno_airfrans import arguments as aa,AirRun,hparams
torch.set_num_threads(1)
p=Path(sys.argv[1]);task=sys.argv[2]
flags=['--cfd_model' if task=='car' else '--model','lrsa_matched','--kcdno-run-dir',str(p/'run')]
a=(ca if task=='car' else aa)(flags,True)
m=CarRun(a,device='cpu',evaluation=True).load() if task=='car' else AirRun(a,hparams(),device='cpu',evaluation=True).load()[0]
d=torch.load(p/'probe.pt',weights_only=True);g=SimpleNamespace(x=d['x'],pos=d['pos'])
y=m((g,None)) if task=='car' else m(g)
torch.testing.assert_close(y,d['expected'],atol=0,rtol=0)
"""
                from test_kcdno_tasks import ROOT
                cwd=ROOT/('Car-Design-ShapeNetCar' if task=='car' else 'Airfoil-Design-AirfRANS')
                r=subprocess.run([sys.executable,'-B','-c',code,tmp,task],cwd=cwd,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True,timeout=60)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_air_actual_argv_default_path(self):
        # Production main calls parse_args without argv; never import that main.
        from test_airfrans import entry,parser
        for family in ('kcdno','lrsa_matched'):
            with patch('sys.argv',['main.py','--model',family]):
                a=entry.parse_args(parser());self.assertEqual(a.kcdno_family,family)

if __name__=='__main__':unittest.main(verbosity=2)
