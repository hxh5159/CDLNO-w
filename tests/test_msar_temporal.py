"""M6 real temporal loop fragments, synthetic tensors, no exp/data imports."""
import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from test_kcdno_tasks import ROOT, PROJECT, TEMPORAL, TestLoss, UnitTransformer
from test_kcdno_temporal import cpu_statements
from test_msar_static import SMALL
from cdlno_entry import parse_args
from model_dict import get_model
from cdlno.msar_lno.standard_entry import StandardRun, model_kwargs
from cdlno.msar_lno.temporal import TemporalModel
from cdlno.msar_lno.standard import StaticModel
from cdlno.msar_lno.config import MSARArchitectureConfig, MSARTrainingConfig
from cdlno.msar_lno.objective import (training_forward, training_objective,
                                     rollout_objective, ObjectiveMetrics)
from cdlno.experiment import start, finish
from msar_entry_projection import strip_msar

RECORDS = []


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)
    if os.environ.get('MSAR_M6_RESULTS'):
        Path(os.environ['MSAR_M6_RESULTS']).write_text(json.dumps(RECORDS, indent=2)+'\n')


def tree(task):
    return ast.parse((PROJECT / f'exp_{TEMPORAL[task][0]}.py').read_text())


def parser_for(task):
    nodes = [n for n in tree(task).body if
             (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser') or
             (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and ast.unparse(n.value.func) == 'parser.add_argument')]
    scope = dict(argparse=argparse)
    exec(compile(ast.Module(nodes, type_ignores=[]), '<real-temporal-parser>', 'exec'), scope)
    return scope['parser']


def arguments(task, flags=()):
    return parse_args(parser_for(task), task, ['--model', 'msar_lno', *flags])


def construct(task, args):
    _, h, w = TEMPORAL[task]
    branch = next(n for n in ast.walk(tree(task)) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "args.model == 'msar_lno'"
                  and any(isinstance(q, ast.Assign) and ast.unparse(q.targets[0]) == 'model' for q in n.body))
    call = branch.body[0].value.func.value  # Omit only the unconditional .cuda().
    return eval(compile(ast.Expression(call), '<real-temporal-constructor>', 'eval'),
                dict(args=args, get_model=get_model, msar_model_kwargs=model_kwargs, h=h, s1=h, s2=w))


def sample(task, b=2, steps=None, device='cpu'):
    n, f, c, times = (4096, 10, 1, 10) if task == 'ns' else (3131, 1, 4, 20)
    times = times if steps is None else steps
    x = torch.randn(b, n, 2, device=device)
    fx = torch.randn(b, n, f, device=device)
    if task == 'plasticity':
        # The accepted normalizer operates on fx, not on coordinates or target.
        fx = UnitTransformer(torch.randn(4, n, 1, device=device)).encode(fx)
    yy = torch.randn((b, n, times) if task == 'ns' else (b, n, c, times), device=device) + 2
    tim = torch.stack([torch.linspace(0, 1, times, device=device).roll(i) for i in range(b)])
    return x, fx, yy, tim


def train_batch(task, model, args, data=None):
    x, fx, yy, tim = sample(task, device=next(model.parameters()).device) if data is None else data
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                   epochs=args.epochs, steps_per_epoch=1)
    scope = dict(model=model.train(), args=args, x=x, fx=fx.clone(), yy=yy, tim=tim,
                 T=yy.shape[-1], step=1, torch=torch, optimizer=optimizer, scheduler=scheduler,
                 myloss=TestLoss(size_average=False), train_l2_step=0., train_l2_full=0.,
                 training_forward=training_forward, training_objective=training_objective,
                 rollout_objective=rollout_objective, msar_epoch=ObjectiveMetrics())
    loop = next(n for n in ast.walk(tree(task)) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'train_loader')
    with patch.object(optimizer, 'step', wraps=optimizer.step) as update, \
            patch.object(scheduler, 'step', wraps=scheduler.step) as schedule:
        exec(cpu_statements(loop.body), scope)
        assert (update.call_count, schedule.call_count) == ((1, 1) if task == 'ns' else (yy.shape[-1], 1))
    return scope


def evaluate(task, model, data):
    x, fx, yy, tim = data
    scope = dict(model=model.eval(), x=x, fx=fx.clone(), yy=yy, tim=tim, T=yy.shape[-1],
                 step=1, bsz=x.shape[0], torch=torch, myloss=TestLoss(size_average=False),
                 test_l2_step=0., test_l2_full=0.)
    loop = next(n for n in ast.walk(tree(task)) if isinstance(n, ast.For) and ast.unparse(n.iter) == 'test_loader')
    # All temporal forward/refill/metric statements; skip plotting, device transfer, IO.
    nodes = [n for n in loop.body if isinstance(n, ast.For) or
             (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('bsz', 'loss')) or
             (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id in ('test_l2_step', 'test_l2_full'))]
    with torch.no_grad():
        exec(cpu_statements(nodes), scope)
    return scope


class MSARTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91706)
        backend = sdpa_kernel(SDPBackend.MATH); backend.__enter__()
        self.addCleanup(backend.__exit__, None, None, None)

    def test_real_defaults_profiles_old_cli_and_frozen_time_regions(self):
        before = json.loads((ROOT/'docs/msar_lno_audit/m6/before.json').read_text())
        snapshot = Path(before['archive'])/'source'
        for task in TEMPORAL:
            args = arguments(task)
            self.assertEqual((args.epochs, args.batch_size, args.lr, args.max_grad_norm),
                             (500, 2, .001, None) if task == 'ns' else (500, 8, .001, .1))
            self.assertEqual(args.msar_architecture['d'], 96)
            self.assertEqual(args.msar_architecture['num_latents'], [512,256,128,64])
            self.assertEqual(arguments(task,['--profile','full']).msar_architecture['d'],192)
            custom = arguments(task,['--profile','full',*SMALL,'--coverage-mode','off','--epochs','3'])
            self.assertEqual(custom.msar_architecture['d'],8); self.assertEqual(custom.epochs,3)
            self.assertIn('coverage_off',custom.msar_run_dir.parts)
            base = parser_for(task).parse_args([]); actual = parse_args(parser_for(task),task,[])
            for k,v in vars(base).items(): self.assertEqual(getattr(actual,k),v)
            self.assertFalse(hasattr(actual,'msar_architecture'))
            with self.assertRaises(ValueError): model_kwargs(actual)
            for flag,value in (('--n-layers','2'),('--slice_num','2'),('--history-mode','off')):
                with self.assertRaisesRegex(ValueError,'not applicable'): arguments(task,[flag,value])
            path = f'PDE-Solving-StandardBenchmark/exp_{TEMPORAL[task][0]}.py'
            self.assertEqual(ast.dump(strip_msar(tree(task))),ast.dump(ast.parse((snapshot/path).read_text())))
            old = json.loads((PROJECT/f'configs/kcdno/{task}.json').read_text())
            new = json.loads((PROJECT/f'configs/msar_lno/{task}.json').read_text())
            self.assertEqual(new['training'],old['training']); self.assertEqual(new['adapter'],old['adapter'])

    def test_wrapper_lift_time_condition_and_shape_guards(self):
        from cdlno.standard import _grid
        for task in TEMPORAL:
            m = construct(task,arguments(task,SMALL))
            x,fx,_,tim = sample(task); t = tim[:,:1].clone().requires_grad_() if task == 'plasticity' else None
            seen=[]; h=m.core.register_forward_pre_hook(lambda mod,a:seen.append(a[0].detach().clone()))
            out=m(x,fx,t); h.remove()
            pos=(_grid(64,64)[:,None]-_grid(8,8)[None]).square().sum(-1).sqrt()[None].expand(2,-1,-1) if task=='ns' else x
            expected=m.preprocess(torch.cat((pos,fx),-1))
            if task=='plasticity':
                frequency=torch.exp(-torch.log(torch.tensor(10000.))*torch.arange(4)/4)
                embedding=t[:,:,None]*frequency[None,None]
                expected=expected+m.time_fc(torch.cat((embedding.cos(),embedding.sin()),-1)).expand(-1,3131,-1)
                gradients=torch.autograd.grad(out.square().sum(),(t,*m.time_fc.parameters()),retain_graph=True)
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().max()>0 for g in gradients))
                self.assertFalse(torch.equal(out,m(x,fx,t+.4)))
            torch.testing.assert_close(seen[0],expected,atol=0,rtol=0)
            self.assertEqual(out.shape,(2,4096,1) if task=='ns' else (2,3131,4))
            self.assertEqual(m.preprocess[0].in_features,74 if task=='ns' else 3)
            self.assertFalse(hasattr(m,'placeholder')); self.assertFalse(hasattr(m,'output'))
            self.assertFalse(any(isinstance(n,(nn.Conv1d,nn.Conv2d,nn.Conv3d)) for n in m.modules()))
            with self.assertRaisesRegex(ValueError,r'H\*W'):m(x[:,:-1],fx[:,:-1],t)
            with self.assertRaises(ValueError):m(x,fx[..., :0],t)
            with self.assertRaises(ValueError):m(x,fx,torch.ones(2,2))
            if task=='plasticity':
                with self.assertRaises(ValueError):m(x,fx)

    def test_actual_time_loops_loss_steps_eval_and_checkpoints(self):
        for task in TEMPORAL:
            for mode in ('floor','off'):
                with self.subTest(task=task,mode=mode),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    args=arguments(task,[*SMALL,'--coverage-mode',mode,'--coverage-kappa','1',
                                         '--msar-run-dir',str(Path(tmp)/'run')])
                    m=construct(task,args);run=StandardRun(args,m);data=sample(task)
                    observations=[];aux=[];downs=[[] for _ in range(4)];calls=[]
                    def observe(mod,a,k):
                        observations.append((a[1] if len(a)>1 else k['fx']).detach().clone())
                        calls.append(k.get('T'))
                    h=m.register_forward_pre_hook(observe,with_kwargs=True)
                    hh=m.core.register_forward_hook(lambda mod,a,out:aux.append(out))
                    handles=[down.register_forward_hook(lambda mod,a,out,i=i:downs[i].append(out)) for i,down in enumerate(m.core.downs)]
                    scope=train_batch(task,m,args,data)
                    count=10 if task=='ns' else 20
                    self.assertEqual(len(observations),count)
                    self.assertTrue(all(len(v)==count and len({id(q) for q in v})==count for v in downs))
                    self.assertEqual(scope['msar_epoch'].steps,1 if task=='ns' else 20)
                    for name,p in m.named_parameters():
                        self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                    x,initial,yy,tim=data
                    for t in range(count):
                        expected=torch.cat((initial[...,t:],yy[...,:t]),-1) if task=='ns' else initial
                        torch.testing.assert_close(observations[t],expected,atol=0,rtol=0)
                        if task=='plasticity':torch.testing.assert_close(calls[t],tim[:,t:t+1],atol=0,rtol=0)
                    if mode=='off':self.assertIs(scope['msar_loss'].total,scope['loss'])
                    else:
                        expected=torch.stack([v.coverage_mean for v in aux]).mean() if task=='ns' else aux[-1].coverage_mean
                        torch.testing.assert_close(scope['msar_loss'].coverage_raw,expected,atol=0,rtol=0)
                        self.assertGreater(expected.item(),0)
                        torch.testing.assert_close(scope['msar_loss'].total,scope['loss']+.01*expected,atol=0,rtol=0)
                    h.remove();hh.remove()
                    for h in handles:h.remove()
                    observations.clear();predictions=[]
                    h=m.register_forward_pre_hook(observe,with_kwargs=True)
                    hh=m.register_forward_hook(lambda mod,a,out:predictions.append(out.clone()))
                    with patch('cdlno.msar_lno.core.CoverageFloorLoss',side_effect=AssertionError('eval coverage')):
                        result=evaluate(task,m,data)
                    self.assertEqual(result['pred'].shape,yy.shape)
                    for t in range(count):
                        expected=torch.cat((initial[...,t:],*predictions[:t]),-1) if task=='ns' else initial
                        torch.testing.assert_close(observations[t],expected,atol=0,rtol=0)
                    h.remove();hh.remove()
                    torch.testing.assert_close(torch.as_tensor(result['test_l2_full']),
                        TestLoss(size_average=False)(result['pred'].reshape(2,-1),yy.reshape(2,-1)))
                    # Changing future evaluation labels cannot affect predictions/refill.
                    changed=evaluate(task,m,(x,initial,yy+37,tim))
                    torch.testing.assert_close(result['pred'],changed['pred'],atol=0,rtol=0)
                    expected=m(x,initial,T=tim[:,:1] if task=='plasticity' else None).detach();run.save(m)
                    files={p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')}
                    ev=arguments(task,['--eval','1','--msar-run-dir',str(run.directory),
                                      '--coverage-mode','off' if mode=='floor' else 'floor'])
                    loaded=construct(task,ev).eval();erun=StandardRun(ev,loaded);erun.load(loaded)
                    torch.testing.assert_close(loaded(x,initial,T=tim[:,:1] if task=='plasticity' else None),expected,atol=0,rtol=0)
                    self.assertEqual(files,{p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')})
                    RECORDS.append(dict(task=task,mode=mode,device='cpu',dtype='float32',backend='MATH',
                        spatial_N=x.shape[1],B=2,forwards=count,optimizer_steps=1 if task=='ns' else 20,
                        scheduler_steps=1,original_loss='TestLoss relative L2, unchanged temporal AST',
                        eval=True,strict_checkpoint=True,atol=0,rtol=0,coverage='temporal mean' if task=='ns' else 'per time update'))

    def test_rollout_mean_scale_graph_and_no_cached_state(self):
        m=construct('ns',arguments('ns',[*SMALL,'--coverage-kappa','1'])).train()
        x,fx,_,_=sample('ns',b=1)
        records=(training_forward(m,x,fx),training_forward(m,x,fx+1))
        pde=sum(record.prediction.square().mean() for record in records)
        loss=rollout_objective(pde,records); repeated=rollout_objective(pde,records*5)
        torch.testing.assert_close(loss.total,repeated.total,atol=1e-7,rtol=1e-6)
        for record in records:
            grad=torch.autograd.grad(loss.coverage_raw,record.auxiliary.coverage_mean,retain_graph=True)[0]
            torch.testing.assert_close(grad,torch.tensor(.5),atol=0,rtol=0)
        grads=torch.autograd.grad(loss.coverage_raw,(m.core.downs[0].latent_queries,m.preprocess[0].weight))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads))
        with self.assertRaises(ValueError):rollout_objective(pde,())
        with self.assertRaises(ValueError):rollout_objective(pde,(records[0],replace(records[1],training=MSARTrainingConfig(coverage_mode='off'))))
        keys={name:set(vars(mod)) for name,mod in m.named_modules()}
        m.eval()
        with torch.no_grad():
            a=m(x,fx);m(x,fx+2);again=m(x,fx)
        torch.testing.assert_close(a,again,atol=0,rtol=0)
        self.assertEqual(keys,{name:set(vars(mod)) for name,mod in m.named_modules()})
        # No prior output graph enters a later physical time call.
        first=fx.clone().requires_grad_();second=fx.clone().requires_grad_()
        m.train();unused=training_forward(m,x,first);next_call=training_forward(m,x,second)
        gradient=torch.autograd.grad(next_call.prediction.sum()+next_call.auxiliary.coverage_mean,first,allow_unused=True)[0]
        self.assertIsNone(gradient)

    def test_off_weight_zero_parity_and_explicit_eval_diagnostics(self):
        for task in TEMPORAL:
            m=construct(task,arguments(task,SMALL));x,fx,_,tim=sample(task,b=1)
            kw=dict(T=tim[:,:1]) if task=='plasticity' else {}
            on=training_forward(m,x,fx,**kw)
            for cfg in (replace(m.training_config,coverage_mode='off'),replace(m.training_config,coverage_weight=0)):
                with patch('cdlno.msar_lno.core.CoverageFloorLoss',side_effect=AssertionError('off coverage')):
                    off=training_forward(m,x,fx,training_config=cfg,**kw)
                self.assertIsNone(off.auxiliary)
                torch.testing.assert_close(off.prediction,on.prediction,atol=2e-6,rtol=1e-4)
                pde=off.prediction.square().mean()
                self.assertIs(rollout_objective(pde,(off,off)).total,pde)
            m.eval()
            with torch.no_grad(),patch('cdlno.msar_lno.core.CoverageFloorLoss',side_effect=AssertionError('eval coverage')):
                normal=m(x,fx,**kw)
                diagnostic=m(x,fx,return_aux=True,training_config=replace(m.training_config,diagnostics=True),**kw)
            torch.testing.assert_close(normal,diagnostic.prediction,atol=0,rtol=0)
            self.assertFalse(diagnostic.coverage_active);self.assertEqual(len(diagnostic.diagnostics['encoder']),4)

    def test_plasticity_original_collate_preserves_sample_time_label_pairing(self):
        function=next(n for n in tree('plasticity').body if isinstance(n,ast.FunctionDef) and n.name=='random_collate_fn')
        scope=dict(torch=torch);exec(cpu_statements([function]),scope)
        positions=torch.randn(2,3131,2);fx=torch.randn(2,3131,1)
        times=torch.arange(20.).repeat(2,1)
        labels=times[:,None,None,:].expand(-1,3131,4,-1).clone()
        result=scope['random_collate_fn'](list(zip(positions,times,fx,labels)))
        torch.testing.assert_close(result[0],positions,atol=0,rtol=0)
        torch.testing.assert_close(result[2],fx,atol=0,rtol=0)
        torch.testing.assert_close(result[3][:,0,0],result[1],atol=0,rtol=0)
        self.assertFalse(torch.equal(result[1][0],result[1][1]))
        for t in result[1]:torch.testing.assert_close(t.sort().values,times[0],atol=0,rtol=0)

    def test_formal_light_full_construct_and_real_N_forward(self):
        for task in TEMPORAL:
            for profile in ('light','full'):
                m=construct(task,arguments(task,['--profile',profile])).eval()
                x,fx,_,tim=sample(task,b=1)
                with torch.no_grad():y=m(x,fx,T=tim[:,:1] if task=='plasticity' else None)
                self.assertEqual(y.shape,(1,4096,1) if task=='ns' else (1,3131,4))
                self.assertEqual(m.config.num_latents[0],512 if profile=='light' else 1024)
                self.assertTrue(torch.isfinite(y).all())
                RECORDS.append(dict(task=task,profile=profile,N=x.shape[1],M1=m.config.num_latents[0],
                    parameters=sum(p.numel() for p in m.parameters()),device='cpu',forward='passed',backward='not_run'))

    def test_old_M5_same_weight_fixtures(self):
        for row in json.loads((ROOT/'docs/msar_lno_audit/m6/static-fixtures.json').read_text()):
            p=Path(row['path']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
            saved=torch.load(p,weights_only=True);a=saved['adapter'];grid=a['grid_shape'] or (None,None)
            m=StaticModel(config=MSARArchitectureConfig.from_dict(saved['config']),task_name=row['task'],
                training_config=MSARTrainingConfig.from_dict(saved['training']),H=grid[0],W=grid[1]).eval()
            m.load_state_dict(saved['state'],strict=True)
            with torch.no_grad():torch.testing.assert_close(m(saved['x'],saved['fx']),saved['output'],atol=0,rtol=0)
            self.assertEqual(m.adapter_architecture(),a)

    def test_temporal_conflicts_before_load_and_strict_state(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for task in TEMPORAL:
                args=arguments(task,[*SMALL,'--msar-run-dir',str(Path(tmp)/task)])
                m=construct(task,args);r=StandardRun(args,m);r.save(m)
                common=['--eval','1','--msar-run-dir',str(r.directory)]
                for flags in (['--d','16'],['--num-latents','9','5','3','2'],['--ref','4'],['--unified_pos',str(1-int(m.unified_pos))]):
                    with patch('torch.load',side_effect=AssertionError('must reject before weights')),self.assertRaises(ValueError):arguments(task,common+flags)
                with self.assertRaises(ValueError):arguments('plasticity' if task=='ns' else 'ns',common)
                side=r.directory/'task.json';contents=side.read_bytes();record=json.loads(contents)
                record['adapter']['Time_Input']=not m.Time_Input;side.write_text(json.dumps(record))
                ev=arguments(task,common);loaded=construct(task,ev)
                with patch('torch.load',side_effect=AssertionError('must reject first')),self.assertRaises(ValueError):StandardRun(ev,loaded)
                side.write_bytes(contents)
                er=StandardRun(ev,loaded);weights=torch.load(r.checkpoint,weights_only=True)
                weights.pop(next(iter(weights)));torch.save(weights,r.checkpoint)
                with self.assertRaises(RuntimeError):er.load(loaded)

    def test_shell_modes_fresh_cwd_loading_and_readonly_records(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for task in TEMPORAL:
                script=ROOT/f'tran_evaluate/msar_lno/{task}.sh'
                for flags in ([],['--coverage-mode','off','--coverage-weight','0'],['--profile','full']):
                    result=subprocess.run(['bash',str(script),'train',*flags,'--dry-run'],capture_output=True,text=True,check=True)
                    tokens=shlex.split(next(l[9:] for l in result.stdout.splitlines() if l.startswith('Command: ')))
                    a=arguments(task,tokens[3:]);self.assertEqual(a.msar_task,task)
                    self.assertEqual(a.msar_architecture['d'],192 if '--profile' in flags else 96)
                    self.assertNotIn('--slice_num',tokens)
                a=arguments(task,[*SMALL,'--coverage-mode','off','--msar-run-dir',str(Path(tmp)/task)])
                record=start(a,task)
                try:
                    self.assertEqual(json.loads((record.directory/'config.json').read_text())['parameter_count_status'],'pending_model_construction')
                    m=construct(task,a);r=StandardRun(a,m);data=sample(task,b=1,steps=2)
                    scope=train_batch(task,m,a,data);r.record_objective(1,scope['msar_epoch'],dict(train_step_loss=scope['train_l2_step']/2));r.save(m)
                    x,fx,_,tim=data;t=tim[:,:1] if task=='plasticity' else None
                    torch.save(dict(x=x,fx=fx,t=t,y=m.eval()(x,fx,t).detach()),Path(tmp)/(task+'.pt'))
                finally:finish(a)
                metric=json.loads((record.directory/'train_history.jsonl').read_text())['metrics']
                self.assertEqual(metric['coverage_mode'],'off');self.assertEqual(metric['coverage_raw'],0)
                frozen={name:(r.directory/name).read_bytes() for name in ('architecture.json','task.json','config.json','train_results.json')}
                for flags in ([],['--coverage-mode','floor'],['--profile','full']):
                    result=subprocess.run(['bash',str(script),'eval','--msar-run-dir',str(r.directory),*flags,'--dry-run'],capture_output=True,text=True,check=True)
                    tokens=shlex.split(next(l[9:] for l in result.stdout.splitlines() if l.startswith('Command: ')))
                    ev=arguments(task,tokens[3:]);self.assertEqual(ev.msar_architecture,m.config.to_dict())
                    er=start(ev,task,evaluation=True)
                    try:loaded=construct(task,ev);StandardRun(ev,loaded).load(loaded);er.record_metrics(dict(test_full_loss=1.))
                    finally:finish(ev)
                self.assertEqual(frozen,{name:(r.directory/name).read_bytes() for name in frozen})
                self.assertEqual(len(json.loads((r.directory/'eval_results.json').read_text())['evaluations']),3)
            code='''
import sys,torch
from pathlib import Path
from torch.nn.attention import SDPBackend,sdpa_kernel
from test_msar_temporal import arguments,construct,StandardRun
torch.set_num_threads(1)
with sdpa_kernel(SDPBackend.MATH):
 for task in ('ns','plasticity'):
  root=Path(sys.argv[1]);a=arguments(task,['--eval','1','--msar-run-dir',str(root/task)])
  m=construct(task,a).eval();StandardRun(a,m).load(m)
  v=torch.load(root/(task+'.pt'),weights_only=True)
  torch.testing.assert_close(m(v['x'],v['fx'],v['t']),v['y'],atol=0,rtol=0)
assert not any(n.startswith('exp_') for n in sys.modules)
'''
            result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=PROJECT,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_original_temporal_loss_steps(self):
        for task in TEMPORAL:
            a=arguments(task,[*SMALL,'--coverage-kappa','1']);m=construct(task,a).cuda()
            scope=train_batch(task,m,a,sample(task,b=1,device='cuda'))
            self.assertTrue(torch.isfinite(scope['msar_loss'].total))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            RECORDS.append(dict(task=task,device='cuda',dtype='float32',backend='MATH',
                N=4096 if task=='ns' else 3131,forwards=10 if task=='ns' else 20,original_loss_step='passed'))


if __name__=='__main__':unittest.main(verbosity=2)
