"""Synthetic NS/Plasticity contracts and frozen temporal-loop audit."""
from argparse import Namespace
from output_recording_projection import strip_recording
import ast, copy, io, json, os, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from unittest.mock import patch
from einops import rearrange

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'PDE-Solving-StandardBenchmark'
sys.path.insert(0,str(PROJECT))
from cdlno_entry import parse_args, model_kwargs, StaticRun
from model_dict import get_model
from cdlno.standard import TemporalStandardModel
from utils.testloss import TestLoss


def parser(task):
    filename = 'ns' if task == 'ns' else 'plas'
    src=ast.parse((PROJECT/f'exp_{filename}.py').read_text())
    nodes=[]
    for n in src.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='parser': nodes.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='parser': nodes.append(n)
    ns={'argparse':__import__('argparse')}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'<p>','exec'),ns); return ns['parser']

def args(task, extra=()): return parse_args(parser(task), task, ['--model','CDLNO',*extra])

def build(task, extra=(), real=False):
    a=args(task,extra)
    if task=='ns':
        kw=dict(space_dim=2,n_layers=a.n_layers,n_hidden=a.n_hidden,n_head=a.n_heads,Time_Input=False,mlp_ratio=a.mlp_ratio,fun_dim=10,out_dim=1,slice_num=a.slice_num,ref=a.ref,unified_pos=a.unified_pos,H=64,W=64,**model_kwargs(a))
    else:
        kw=dict(space_dim=2,n_layers=a.n_layers,n_hidden=a.n_hidden,n_head=a.n_heads,Time_Input=True,mlp_ratio=a.mlp_ratio,fun_dim=1,out_dim=4,slice_num=a.slice_num,ref=a.ref,unified_pos=a.unified_pos,H=101,W=31,**model_kwargs(a))
    return get_model(a).Model(**kw),a

def data(model,b=2):
    x=torch.randn(b,model.H*model.W,2,requires_grad=True)
    fx=torch.randn(b,model.H*model.W,model.fun_dim,requires_grad=True)
    return x,fx

class TemporalChecks(unittest.TestCase):
    def setUp(self): torch.manual_seed(77); torch.set_num_threads(1)
    def test_defaults_dimensions_and_time_registration(self):
        ns,a=build('ns'); pl,p=build('plasticity')
        self.assertEqual((a.n_hidden,a.n_layers,a.n_heads,a.slice_num,a.batch_size),(256,8,8,64,2))
        self.assertEqual((p.n_hidden,p.n_layers,p.n_heads,p.slice_num,p.batch_size),(128,8,8,64,8))
        self.assertEqual((ns.preprocess[0].in_features,ns.config.output_dim,ns.config.grid_shape),(74,1,(64,64)))
        self.assertEqual((pl.preprocess[0].in_features,pl.config.output_dim,pl.config.grid_shape),(3,4,(101,31)))
        self.assertIsNone(ns.time_fc); self.assertIsNotNone(pl.time_fc)
    def test_ns_original_no_clip_default_and_explicit_override(self):
        legacy=parse_args(parser('ns'),'ns',['--model','Transolver_Structured_Mesh_2D'])
        self.assertIsNone(legacy.max_grad_norm)
        self.assertEqual((legacy.mlp_ratio,legacy.slice_num),(1,32))
        default=args('ns')
        self.assertIsNone(default.max_grad_norm)
        self.assertEqual((default.mlp_ratio,default.slice_num),(2,64))
        source=ast.parse((PROJECT/'exp_ns.py').read_text())
        branch=next(n for n in ast.walk(source) if isinstance(n,ast.If)
                    and ast.unparse(n.test)=='args.max_grad_norm is not None')
        code=compile(ast.Module(body=[branch],type_ignores=[]),'<original-clip-branch>','exec')
        model=nn.Linear(1,1)
        for value,expected in ((default,0),(args('ns',['--max_grad_norm','0.1']),1)):
            with patch('torch.nn.utils.clip_grad_norm_') as clip:
                exec(code,dict(args=value,torch=torch,model=model))
                self.assertEqual(clip.call_count,expected)
                if expected: self.assertEqual(clip.call_args.args[1],.1)
    def test_ns_teacher_forcing_and_prediction_windows_batched(self):
        model,_=build('ns', ['--n-hidden','16','--n-heads','4','--slice_num','4'])
        x,initial=data(model); truth=torch.randn(2,model.H*model.W,10)
        # Each call is a fresh core; training feeds truth and evaluation feeds prediction.
        calls=[]
        original=model.core.forward
        model.core.forward=lambda h: (calls.append(h.detach().clone()), original(h))[1]
        preds=[]; train_loss=0; fx=initial
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        optimizer.zero_grad()
        for t in range(10):
            torch.testing.assert_close(fx,torch.cat((initial[...,t:],truth[...,:t]),-1))
            im=model(x,fx); preds.append(im)
            train_loss=train_loss+TestLoss(size_average=False)(im.reshape(2,-1),truth[...,t:t+1].reshape(2,-1))
            fx=torch.cat((fx[...,1:],truth[...,t:t+1]),-1)
        with patch.object(train_loss,'backward',wraps=train_loss.backward) as backward, patch.object(optimizer,'step',wraps=optimizer.step) as step:
            train_loss.backward(); optimizer.step()
            self.assertEqual(backward.call_count,1); self.assertEqual(step.call_count,1)
        self.assertEqual(len(calls),10); self.assertEqual(torch.cat(preds,-1).shape,(2,4096,10))
        torch.testing.assert_close(fx,truth)
        for p in model.parameters(): self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())
        model.zero_grad(set_to_none=True); model.eval(); pred=[]; fx2=initial.detach()
        with torch.no_grad():
            for t in range(10):
                expected=torch.cat((initial[...,t:],*pred),-1)
                torch.testing.assert_close(fx2,expected)
                im=model(x.detach(),fx2); pred.append(im); fx2=torch.cat((fx2[...,1:],im),-1)
        self.assertEqual(len(calls),20)
        self.assertEqual(torch.cat(pred,-1).shape,(2,4096,10))
        torch.testing.assert_close(fx2,torch.cat(pred,-1))
        self.assertFalse(torch.equal(calls[0],calls[-1]))
    def test_plasticity_time_changes_output_and_time_gradients(self):
        model,_=build('plasticity'); x,fx=data(model); t=torch.tensor([[.1],[.3]],requires_grad=True); t2=t.detach().clone().add_(.4).requires_grad_()
        y=model(x,fx,t); y2=model(x,fx,t2)
        self.assertEqual(y.shape,(2,3131,4)); self.assertGreater((y-y2).abs().max().item(),0)
        loss=TestLoss(size_average=False)(y.reshape(2,-1),torch.randn_like(y).reshape(2,-1)); loss.backward()
        self.assertIsNotNone(t.grad); self.assertTrue(torch.isfinite(t.grad).all()); self.assertGreater(t.grad.abs().max().item(),0)
        self.assertTrue(torch.isfinite(x.grad).all()); self.assertTrue(torch.isfinite(fx.grad).all())
        for name,p in model.time_fc.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().max().item(),0,name)
    def test_plasticity_twenty_independent_time_steps_and_loss(self):
        model,_=build('plasticity'); x,fx=data(model); times=torch.linspace(0,1,20).repeat(2,1); target=torch.randn(2,3131,4,20)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); count=0
        for t in range(20):
            optimizer.zero_grad(); out=model(x,fx,times[:,t:t+1]); loss=TestLoss(size_average=False)(out.reshape(2,-1),target[...,t].reshape(2,-1)); loss.backward(); optimizer.step(); count+=1
        self.assertEqual(count,20)
    def test_checkpoint_roundtrip_temporal(self):
        for task in ('ns','plasticity'):
            model,a=build(task, ['--n-hidden','8','--n-heads','2','--slice_num','4'])
            with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
                a.cdlno_run_dir=Path(d)/'run'; run=StaticRun(a,model); run.save(model)
                x,fx=data(model); t=None if task=='ns' else torch.tensor([[.3],[.2]])
                with torch.no_grad(): expected=model(x,fx,t)
                b=build(task,['--n-hidden','8','--n-heads','2','--slice_num','4','--eval','1','--cdlno-run-dir',str(a.cdlno_run_dir), '--cdpa-source-chunk-size','1'])[0]
                eval_args=args(task,['--n-hidden','8','--n-heads','2','--slice_num','4','--eval','1','--cdlno-run-dir',str(a.cdlno_run_dir),'--cdpa-source-chunk-size','1']); run2=StaticRun(eval_args,b); run2.load(b)
                torch.testing.assert_close(b(x,fx,t),expected,atol=1e-5,rtol=3e-4)
    def test_original_temporal_loop_ast_contract(self):
        for task in ('ns','plas'):
            source=(PROJECT/f'exp_{task}.py').read_text(); tree=ast.parse(source); main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');
            text=ast.unparse(main)
            self.assertIn('range(0, T, step)' if task=='ns' else 'range(T)', text)
            self.assertIn('optimizer.zero_grad()',text); self.assertIn('loss.backward()',text); self.assertIn('torch.cat',text)
            if task=='ns': self.assertIn('torch.cat((fx[..., step:], y)',text); self.assertIn('torch.cat((fx[..., step:], im)',text)
            else: self.assertIn('T=input_T',text); self.assertIn('optimizer.step()',text)
    def test_model_registry_excludes_unapproved_tasks(self):
        for task in ('ns','plasticity'): self.assertIsNotNone(get_model(Namespace(model='CDLNO',cdlno_task=task)))
        with self.assertRaises(ValueError): get_model(Namespace(model='CDLNO',cdlno_task='car'))

class StaticTemporalFreeze(unittest.TestCase):
    def test_complete_temporal_entries_project_to_baseline_ast(self):
        # Remove only authorized model/checkpoint/path branches. Everything
        # else, including the full nested time loops, must equal phase0 HEAD.
        class LegacyProjection(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                return None if node.module=='cdlno_entry' else node
            def visit_Assign(self,node):
                if isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('cdlno_run','results_dir'):
                    return None
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='parse_cdlno_args':
                    node.value=ast.parse('parser.parse_args()',mode='eval').body
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test) in ("args.model == 'CDLNO'",'cdlno_run is not None'):
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)
            def visit_Name(self,node):
                if node.id=='results_dir':
                    return ast.parse("'./results/' + save_name + '/'",mode='eval').body
                return node
        for task in ('ns','plas'):
            relative=f'PDE-Solving-StandardBenchmark/exp_{task}.py'
            original=subprocess.check_output(['git','show','75e0f67643806a81cd1d3f6adc88dd8c02416fe7:'+relative],cwd=ROOT,text=True)
            projected=LegacyProjection().visit(strip_recording(ast.parse((ROOT/relative).read_text())))
            self.assertEqual(ast.dump(projected),ast.dump(ast.parse(original)),task)

    def test_scripts_and_python_syntax(self):
        for p in sorted((PROJECT/'scripts').glob('CDLNO_NS*.sh'))+sorted((PROJECT/'scripts').glob('CDLNO_Plasticity*.sh')):
            subprocess.run(['bash','-n',str(p)],check=True)
        for p in [PROJECT/'exp_ns.py',PROJECT/'exp_plas.py',PROJECT/'cdlno_entry.py',ROOT/'cdlno/standard.py']:
            ast.parse(p.read_text(),feature_version=(3,10))

if __name__=='__main__': unittest.main()
