"""Original temporal statements on synthetic tensors, no entry imports."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest

import torch
from test_kcdno_tasks import ROOT, SMALL, TEMPORAL, arguments, construct, sample, tree, strip_new, TestLoss
from cdlno.kcdno.entry import StandardRun


def cpu_statements(nodes):
    class DeviceOnly(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' else node
    return compile(ast.fix_missing_locations(DeviceOnly().visit(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]))),'<original-temporal-statements>','exec')


class TemporalKCDNO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_defaults_and_complete_frozen_entries(self):
        before=json.loads((ROOT/'docs/kcdno_audit/k5/before.json').read_text())
        for task in TEMPORAL:
            args=arguments(task)
            self.assertEqual((args.n_layers,args.n_hidden,args.slice_num,args.batch_size),
                             (8,256,64,2) if task=='ns' else (8,128,64,8))
            self.assertEqual(args.epochs,500)
            if task=='ns':self.assertIsNone(args.max_grad_norm)
            rel=f'PDE-Solving-StandardBenchmark/exp_{TEMPORAL[task][0]}.py'
            self.assertEqual(ast.dump(strip_new(tree(task))),ast.dump(ast.parse((Path(before['source_snapshot'])/rel).read_text())))

    def test_original_NS_ten_step_windows_and_cache_isolation(self):
        model=construct('ns',arguments('ns',SMALL));x,initial=sample(model)
        self.assertEqual(model.preprocess[0].in_features,74)
        self.assertFalse(hasattr(model,'time_fc'));self.assertFalse(hasattr(model,'placeholder'))
        yy=torch.randn(2,4096,10)+1
        train=next(n for n in ast.walk(tree('ns')) if isinstance(n,ast.For) and ast.unparse(n.iter)=='train_loader')
        observed=[];caches=[]
        handle=model.register_forward_pre_hook(lambda m,a,k: observed.append(k['fx'].clone()),with_kwargs=True)
        writer=model.core.blocks[0].writer.register_forward_hook(lambda m,a,o:caches.append(o))
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=20)
        scope=dict(model=model,x=x,fx=initial.clone(),yy=yy,T=10,step=1,torch=torch,
                   optimizer=opt,scheduler=sched,myloss=TestLoss(size_average=False),
                   train_l2_step=0,train_l2_full=0,args=arguments('ns',SMALL))
        with patch.object(opt,'step',wraps=opt.step) as update,patch.object(sched,'step',wraps=sched.step) as schedule:
            exec(cpu_statements(train.body),scope)
            self.assertEqual((update.call_count,schedule.call_count),(1,1))
        self.assertEqual(len(observed),10);self.assertEqual(len({id(c) for c in caches}),10)
        for t in range(10):
            torch.testing.assert_close(observed[t],torch.cat((initial[...,t:],yy[...,:t]),-1),atol=0,rtol=0)
        observed.clear();caches.clear();model.eval()
        evaluate=next(n for n in ast.walk(tree('ns')) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(0, T, step)'
                      and 'im = model' in ast.unparse(n) and 'fx = torch.cat((fx[..., step:], im)' in ast.unparse(n))
        outputs=[];h=model.register_forward_hook(lambda m,a,o:outputs.append(o.clone()))
        scope.update(fx=initial.clone())
        with torch.no_grad():exec(cpu_statements([evaluate]),scope)
        for t in range(10):
            expected=initial[...,t:] if t==0 else torch.cat((initial[...,t:],*outputs[:t]),-1)
            torch.testing.assert_close(observed[t],expected,atol=0,rtol=0)
        self.assertEqual(scope['pred'].shape,(2,4096,10));self.assertEqual(len(caches),10)
        for item in (handle,writer,h):item.remove()

    def test_Plasticity_original_twenty_updates_and_time_gradient(self):
        args=arguments('plasticity',SMALL);model=construct('plasticity',args);x,fx=sample(model)
        self.assertEqual(model.preprocess[0].in_features,3);self.assertTrue(hasattr(model,'time_fc'))
        self.assertFalse(hasattr(model,'placeholder'))
        t=torch.tensor([[.1],[.8]],requires_grad=True)
        out=model(x,fx,t);self.assertEqual(out.shape,(2,3131,4))
        self.assertFalse(torch.equal(out,model(x,fx,t+.4)))
        grads=torch.autograd.grad(out.square().sum(),(t,*model.time_fc.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads))
        loop=next(n for n in ast.walk(tree('plasticity')) if isinstance(n,ast.For) and ast.unparse(n.iter)=='train_loader')
        opt=torch.optim.AdamW(model.parameters(),lr=.001);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=30)
        scope=dict(model=model,x=x,fx=fx,yy=torch.randn(2,3131,4,20)+1,
                   tim=torch.linspace(0,1,20)[None].repeat(2,1),T=20,torch=torch,args=args,
                   optimizer=opt,scheduler=sched,myloss=TestLoss(size_average=False),train_l2_step=0)
        with patch.object(opt,'step',wraps=opt.step) as update,patch.object(sched,'step',wraps=sched.step) as schedule:
            exec(cpu_statements(loop.body),scope)
            self.assertEqual(update.call_count,20)
            # Original scheduler steps once per batch, after its 20 time updates.
            self.assertEqual(schedule.call_count,1)
        self.assertEqual(scope['y'].shape,(2,3131,4,1))
        torch.testing.assert_close(scope['fx'],fx,atol=0,rtol=0)

    def test_temporal_roundtrips_all_off(self):
        for task in TEMPORAL:
            for mode in ('all','off'):
                with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    args=arguments(task,[*SMALL,'--history-mode',mode,'--kcdno-run-dir',str(Path(tmp)/'run')])
                    model=construct(task,args).eval();x,fx=sample(model)
                    t=torch.tensor([[.2],[.9]]) if task=='plasticity' else None
                    expected=model(x,fx,t);run=StandardRun(args,model);run.save(model)
                    side=run.sidecar.read_bytes()
                    ev=arguments(task,['--eval','1','--kcdno-run-dir',str(run.directory)])
                    loaded=construct(task,ev).eval();StandardRun(ev,loaded).load(loaded)
                    torch.testing.assert_close(loaded(x,fx,t),expected,atol=0,rtol=0)
                    self.assertEqual(run.sidecar.read_bytes(),side)


if __name__=='__main__':unittest.main(verbosity=2)
