"""Final integration boundaries: actual CLI scripts, logging, legacy and pickle behavior."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from cdlno.kcdno.loading import validate_whole_model
from cdlno.kcdno.metadata import KCDNOMetadataMismatch
from cdlno.kcdno.profiles import profile_values
from cdlno.kcdno.standard import StandardModel
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.experiment import start,finish
from test_kcdno_tasks import ROOT,TASKS,TEMPORAL,arguments as standard_args,construct,StandardRun
from test_kcdno_car import arguments as car_args,Model as CarModel,car_model_kwargs,CarRun,HAS_PYG
from test_kcdno_airfrans import arguments as air_args,AirfRANSModel,AirRun,model_kwargs as air_kwargs,hparams


def args_for(task,flags=(),evaluation=False):
    if task=='car':return car_args(flags,evaluation)
    if task=='airfrans':return air_args(flags,evaluation)
    return standard_args(task,[*flags,'--eval',str(int(evaluation))])


def model_run(task,args,evaluation=False):
    if task=='car':
        model=None if evaluation else CarModel(**car_model_kwargs(args))
        run=CarRun(args,device='cpu',evaluation=evaluation,model=model)
    elif task=='airfrans':
        model=None if evaluation else AirfRANSModel(**air_kwargs(args))
        run=AirRun(args,hparams(),device='cpu',evaluation=evaluation)
    else:
        model=construct(task,args);run=StandardRun(args,model)
    return model,run


class DeliveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_legacy_parser_imports_without_shared_package(self):
        code='''
import ast,argparse,importlib.util,sys
from pathlib import Path
project=Path(sys.argv[1]);kind=sys.argv[2]
name='exp_darcy.py' if kind=='standard' else 'main.py'
tree=ast.parse((project/name).read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser') or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument')]
scope={'argparse':argparse};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<parser>','exec'),scope)
path=project/('models/cdlno_run.py' if kind=='car' else 'cdlno_entry.py')
spec=importlib.util.spec_from_file_location('original_entry_helper',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
a=mod.parse_args(scope['parser'],'darcy',argv=['--model','Transolver_Structured_Mesh_2D']) if kind=='standard' else mod.parse_args(scope['parser'],argv=['--cfd_model' if kind=='car' else '--model','Transolver'])
assert not any(k=='cdlno' or k.startswith('cdlno.') for k in sys.modules)
print('legacy parser independent:',kind)
'''
        for kind,project in (('standard','PDE-Solving-StandardBenchmark'),('car','Car-Design-ShapeNetCar'),('air','Airfoil-Design-AirfRANS')):
            r=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/project),kind],cwd=ROOT/project,capture_output=True,text=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        # New-package imports also occur only inside the new-model branch.
        for project,names in [('PDE-Solving-StandardBenchmark',[f'exp_{x[0]}.py' for x in (TASKS|TEMPORAL).values()]),('Car-Design-ShapeNetCar',['main.py','main_evaluation.py']),('Airfoil-Design-AirfRANS',['main.py','main_evaluation.py'])]:
            for name in names:
                tree=ast.parse((ROOT/project/name).read_text())
                self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module=='kcdno_entry' for n in tree.body))

    def test_temporal_off_model_optimizer_step(self):
        # Close the specific coverage gap: K5 all ran original time loops,
        # while off previously had forward/strict-checkpoint checks only.
        from test_kcdno_tasks import sample, SMALL
        for task in ('ns','plasticity'):
            a=standard_args(task,[*SMALL,'--history-mode','off'])
            model=construct(task,a);x,fx=sample(model)
            inputs=(x,fx,torch.tensor([[.2],[.8]])) if task=='plasticity' else (x,fx)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            before=model.output.weight.detach().clone()
            out=model(*inputs);out.square().mean().backward();optimizer.step()
            self.assertFalse(torch.equal(before,model.output.weight))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertFalse(any('reader' in n or 'writer' in n for n,_ in model.named_parameters()))

    def test_fixed_pickle_behavior_and_reference_rejected(self):
        config=KCDNOArchitectureConfig(L=2,d=8,h=2,M=3)
        good=AirfRANSModel(config=config)
        mutations=[lambda m:setattr(m.core.blocks[0].down,'heads',1),
                   lambda m:setattr(m.core.blocks[0].up_latent_norm,'eps',1e-3),
                   lambda m:m.reference.add_(.1),
                   lambda m:setattr(m.preprocess, '1', torch.nn.SiLU())]
        for mutate in mutations:
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises(KCDNOMetadataMismatch):validate_whole_model(bad,good)
        # A learned gamma is deliberately NOT compared with its initialization.
        learned=copy.deepcopy(good);learned.core.blocks[1].reader.gamma.data.fill_(.7)
        validate_whole_model(learned,good)
        with self.assertRaisesRegex(ValueError,'d>=2'):
            StandardModel(config=KCDNOArchitectureConfig(d=1,h=1,L=1,M=1,point_module='conv_ffn'),task_name='plasticity',H=101,W=31)

    def test_real_scripts_profiles_train_eval_and_early_records(self):
        count=0
        for task in (*TASKS,*TEMPORAL,'car','airfrans'):
            selection='--cfd_model' if task=='car' else '--model'
            for profile in ('kcdno_v1','transolver_shape_match'):
                for family in ('kcdno','lrsa_matched'):
                    a=args_for(task,[selection,family,'--profile',profile]);v=profile_values(task,profile)
                    self.assertEqual((a.n_hidden,a.n_heads,a.slice_num,a.n_layers),(v['d'],v['h'],v['M'],v['L']))
            for variant in ('all','off','lrsa_matched'):
                family='lrsa_matched' if variant=='lrsa_matched' else 'kcdno'
                with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    runpath=Path(tmp)/task
                    widths=['--n_hidden','8','--n_heads','2','--n_layers','2','--slice_num','3'] if task in ('car','airfrans') else ['--n-hidden','8','--n-heads','2','--n-layers','2','--slice_num','3']
                    flags=[selection,family,*widths,'--kcdno-run-dir',str(runpath)]
                    if family=='kcdno':flags+=['--history-mode',variant,'--kernel-rank','5']
                    script=ROOT/'tran_evaluate/kcdno'/f'{task}.sh'
                    preview=subprocess.run(['bash',str(script),'train',*flags,'--dry-run'],cwd=ROOT,text=True,capture_output=True,timeout=30)
                    self.assertEqual(preview.returncode,0,preview.stderr)
                    tokens=shlex.split(next(line.removeprefix('Command:') for line in preview.stdout.splitlines() if line.startswith('Command:')))[3:]
                    a=args_for(task,tokens)
                    record=start(a,task,hparams=hparams() if task=='airfrans' else None)
                    try:
                        pending=json.loads((runpath/'config.json').read_text())
                        self.assertEqual(pending['model'],family);self.assertIsNone(pending['parameters'])
                        m,r=model_run(task,a)
                        if task=='airfrans':record.attach_model(m,hparams=r.hparams,protocol=r.contract)
                        ready=json.loads((runpath/'config.json').read_text())
                        self.assertEqual(ready['parameters']['total'],sum(p.numel() for p in m.parameters()))
                        finish(a)
                    finally:
                        if not record.closed:record.finish(status='failed')
                    before={name:(runpath/name).read_bytes() for name in ('architecture.json','task.json','config.json')}
                    evpreview=subprocess.run(['bash',str(script),'eval',selection,family,'--kcdno-run-dir',str(runpath),'--dry-run'],cwd=ROOT,text=True,capture_output=True,timeout=30)
                    self.assertEqual(evpreview.returncode,0,evpreview.stderr)
                    evtokens=shlex.split(next(line.removeprefix('Command:') for line in evpreview.stdout.splitlines() if line.startswith('Command:')))[3:]
                    ev=args_for(task,evtokens,True)
                    er=start(ev,task,evaluation=True)
                    try:
                        loaded,run=model_run(task,ev,True)
                        self.assertEqual(ev.kcdno_architecture,a.kcdno_architecture)
                        er.record_metrics({'synthetic_cli_only':True});finish(ev)
                    finally:
                        if not er.closed:er.finish(status='failed')
                    self.assertEqual(before,{name:(runpath/name).read_bytes() for name in before})
                    count+=1
        self.assertEqual(count,24)
        print('24 real-script train/eval pairs + 32 profile/family defaults + early records passed; no dataset entry executed')

if __name__=='__main__':unittest.main(verbosity=2)
