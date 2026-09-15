"""Actual new task/factory/save paths, synthetic data only; never import exp."""
import argparse
import ast
import copy
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import torch
from torch.nn import functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'PDE-Solving-StandardBenchmark'
sys.path.insert(0, str(PROJECT))
from cdlno_entry import parse_args
from model_dict import get_model
from cdlno.kcdno.entry import model_kwargs, StandardRun
from cdlno.kcdno.metadata import KCDNOMetadataMismatch
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss

TASKS = dict(darcy=('darcy',85,85), elasticity=('elas',None,None),
             airfoil=('airfoil',221,51), pipe=('pipe',129,129))
TEMPORAL = dict(ns=('ns',64,64), plasticity=('plas',101,31))
SMALL = ['--n-hidden','8','--n-heads','2','--n-layers','2','--slice_num','3','--kernel-rank','5']


def tree(task):
    return ast.parse((PROJECT / f'exp_{(TASKS | TEMPORAL)[task][0]}.py').read_text())


def arguments(task, flags=()):
    nodes = [n for n in tree(task).body if
             (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser') or
             (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'parser.add_argument')]
    scope = {'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<parser-only>','exec'),scope)
    return parse_args(scope['parser'],task,['--model','kcdno',*flags])


def construct(task, args, real=False):
    height,width = (TASKS | TEMPORAL)[task][1:] if real or task in TEMPORAL else (5,7)
    branch = next(n for n in ast.walk(tree(task)) if isinstance(n,ast.If)
                  and ast.unparse(n.test) in ("args.model == 'kcdno'", "args.model in ('kcdno', 'lrsa_matched')")
                  and any(isinstance(q,ast.Assign) and ast.unparse(q.targets[0])=='model' for q in n.body))
    call = branch.body[0].value.func.value  # only omit .cuda(), use real constructor
    return eval(compile(ast.Expression(body=call),'<actual-constructor>','eval'),
                dict(args=args,get_model=get_model,kcdno_model_kwargs=model_kwargs,s=height,h=height,s1=height,s2=width))


def sample(model,batch=2):
    n = model.H*model.W if model.structured else 972
    x = torch.randn(batch,n,2)
    fx = torch.randn(batch,n,model.fun_dim) if model.fun_dim else None
    return x,fx


def strip_new(tree):
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,n):
            return None if n.module=='kcdno_entry' else n
        def visit_If(self,n):
            if ast.unparse(n.test) in ("args.model == 'kcdno'", "args.model in ('kcdno', 'lrsa_matched')", "args.cfd_model == 'kcdno'", "args.cfd_model in ('kcdno', 'lrsa_matched')", "model == 'kcdno'", "model in ('kcdno', 'lrsa_matched')"):
                result=[]
                for item in n.orelse:
                    value=self.visit(item)
                    if isinstance(value,list):result.extend(value)
                    elif value is not None:result.append(value)
                return result
            return self.generic_visit(n)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))


class StaticKCDNO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_profiles_explicit_overrides_and_frozen_entries(self):
        before=json.loads((ROOT/'docs/kcdno_audit/k4/before.json').read_text())
        snapshot=Path(before['source_snapshot'])
        for task in TASKS:
            args=arguments(task)
            self.assertEqual((args.n_layers,args.n_hidden,args.epochs),(8,128,500))
            self.assertEqual(args.slice_num,32 if task=='pipe' else 64)
            matched=arguments(task,['--profile','transolver_shape_match'])
            self.assertEqual(matched.n_heads,8)
            self.assertEqual(matched.slice_num,64)
            explicit=arguments(task,[*SMALL,'--profile','transolver_shape_match'])
            self.assertEqual((explicit.n_layers,explicit.n_hidden,explicit.slice_num),(2,8,3))
            for flag in ('--front-blocks','--mlp_ratio','--cdpa-source-chunk-size'):
                with self.assertRaisesRegex(ValueError,'not applicable'):
                    arguments(task,[flag,'2'])
            relative=f'PDE-Solving-StandardBenchmark/exp_{TASKS[task][0]}.py'
            self.assertEqual(ast.dump(strip_new(tree(task))),ast.dump(ast.parse((snapshot/relative).read_text())))

    def test_train_eval_roundtrips_and_conflicts(self):
        for task in TASKS:
            for mode in ('all','off'):
                with self.subTest(task=task,mode=mode), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                    flags=[*SMALL,'--history-mode',mode,'--kcdno-run-dir',str(Path(tmp)/'run')]
                    args=arguments(task,flags);model=construct(task,args)
                    run=StandardRun(args,model);x,fx=sample(model)
                    self.assertEqual(hasattr(model,'placeholder'),not bool(model.fun_dim))
                    self.assertFalse(hasattr(model,'time_fc'))
                    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                    y=model(x,fx)
                    self.assertEqual(y.shape,(*x.shape[:2],1))
                    TestLoss(size_average=False)(y[...,0],torch.randn_like(y[...,0])+1).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    opt.step();model.eval();expected=model(x,fx);run.save(model)
                    files={p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')}
                    # No explicit d/h/M/r/history on eval: reconstruct saved resolved architecture.
                    ev=arguments(task,['--eval','1','--kcdno-run-dir',str(run.directory)])
                    loaded=construct(task,ev).eval();erun=StandardRun(ev,loaded);erun.load(loaded)
                    torch.testing.assert_close(loaded(x,fx),expected,atol=0,rtol=0)
                    with self.assertRaises(RuntimeError):erun.save(loaded)
                    with self.assertRaises(FileExistsError):StandardRun(args,model)
                    for flag,value in [('--history-mode','off' if mode=='all' else 'all'),('--kernel-rank','7'),('--n-heads','4'),('--ref','7')]:
                        with self.assertRaises((KCDNOMetadataMismatch,ValueError)):
                            arguments(task,['--eval','1','--kcdno-run-dir',str(run.directory),flag,value])
                    state=torch.load(run.checkpoint,weights_only=True);state.pop(next(iter(state)))
                    torch.save(state,run.checkpoint)
                    with self.assertRaises(RuntimeError):erun.load(loaded)
                    self.assertEqual(files,{p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')})

    def test_real_N_and_original_darcy_decode_loss(self):
        for task in TASKS:
            model=construct(task,arguments(task,SMALL),real=True);x,fx=sample(model,batch=1)
            self.assertEqual(x.shape[1],dict(darcy=7225,elasticity=972,airfoil=11271,pipe=16641)[task])
            TestLoss(size_average=False)(model(x,fx)[...,0],torch.randn(1,x.shape[1])+1).backward()
        task='darcy';module=tree(task)
        central=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='central_diff')
        loop=next(n for n in ast.walk(module) if isinstance(n,ast.For) and any(isinstance(q,ast.Expr) and ast.unparse(q.value)=='optimizer.zero_grad()' for q in n.body))
        start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='optimizer.zero_grad()')+1
        end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='loss.backward()')
        model=construct(task,arguments(task,SMALL),real=True);x,_=sample(model)
        coeff=torch.randn(2,7225,requires_grad=True);target=torch.randn(2,7225)+2
        xn=UnitTransformer(torch.randn(4,7225));yn=UnitTransformer(torch.randn(4,7225))
        scope=dict(model=model,x=x,fx=xn.encode(coeff),y=yn.encode(target),y_normalizer=yn,
                   myloss=TestLoss(size_average=False),de_x=TestLoss(size_average=False),de_y=TestLoss(size_average=False),
                   torch=torch,F=F,rearrange=rearrange,s=85,dx=1/85)
        exec(compile(ast.Module(body=[central,*loop.body[start:end]],type_ignores=[]),'<original-darcy-loss>','exec'),scope)
        torch.testing.assert_close(scope['y'],target)
        for name in ('l2loss','deriv_loss'):
            g=torch.autograd.grad(scope[name],coeff,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().max().item(),0)
        opt=torch.optim.AdamW(model.parameters(),lr=.001);scope['loss'].backward();opt.step()

    def test_scripts_preserve_profile_and_arguments(self):
        for task in TASKS:
            path=ROOT/'tran_evaluate/kcdno'/f'{task}.sh'
            subprocess.run(['bash','-n',str(path)],check=True)
            result=subprocess.run(['bash',str(path),'train','--profile','transolver_shape_match','--n-layers','2','--dry-run'],capture_output=True,text=True,check=True)
            self.assertIn('--profile transolver_shape_match',result.stdout)
            self.assertNotIn('--front-blocks',result.stdout)
            self.assertNotIn('--slice_num',result.stdout)
            self.assertNotIn('--n-hidden',result.stdout)
            cmd=shlex.split(next(line[9:] for line in result.stdout.splitlines() if line.startswith('Command: ')))
            args=arguments(task,cmd[3:])
            self.assertEqual((args.profile,args.n_layers,args.n_heads),('transolver_shape_match',2,8))

    def test_other_original_loss_normalizer_statements(self):
        for task in ('elasticity','airfoil','pipe'):
            module=tree(task)
            loop=next(n for n in ast.walk(module) if isinstance(n,ast.For) and any(isinstance(q,ast.Expr) and ast.unparse(q.value)=='optimizer.zero_grad()' for q in n.body))
            start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='optimizer.zero_grad()')+1
            end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='loss.backward()')
            model=construct(task,arguments(task,SMALL));x,_=sample(model)
            if task=='pipe':x=UnitTransformer(torch.randn(4,*x.shape[1:])).encode(x)
            target=torch.randn(*x.shape[:2])+2
            normalizer=UnitTransformer(torch.randn(4,x.shape[1]))
            y=target if task=='airfoil' else normalizer.encode(target)
            scope=dict(model=model,x=x,y=y,y_normalizer=normalizer,myloss=TestLoss(size_average=False))
            exec(compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'<original-task-loss>','exec'),scope)
            torch.testing.assert_close(scope['y'],target)
            opt=torch.optim.AdamW(model.parameters(),lr=.001);scope['loss'].backward();opt.step()

    def test_fresh_standard_working_directory_loading(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path=Path(tmp)/'run';args=arguments('pipe',[*SMALL,'--kcdno-run-dir',str(path)])
            model=construct('pipe',args).eval();run=StandardRun(args,model);run.save(model)
            x,fx=sample(model);torch.save(dict(x=x,y=model(x,fx)),Path(tmp)/'sample.pt')
            code='''
import sys, torch
from pathlib import Path
from test_kcdno_tasks import arguments, construct
from cdlno.kcdno.entry import StandardRun
torch.set_num_threads(1)
path=Path(sys.argv[1]);before=(path/'run/architecture.json').read_bytes()
args=arguments('pipe',['--eval','1','--kcdno-run-dir',str(path/'run')])
model=construct('pipe',args).eval();StandardRun(args,model).load(model)
data=torch.load(path/'sample.pt',weights_only=True)
torch.testing.assert_close(model(data['x'],None),data['y'],atol=0,rtol=0)
assert (path/'run/architecture.json').read_bytes()==before
assert not any(s.startswith('exp_') for s in sys.modules)
'''
            result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=PROJECT,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
