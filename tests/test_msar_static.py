"""M5 real static entries via safe AST fragments; never import exp/main or data."""
import argparse
import ast
from contextlib import redirect_stdout
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
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange

from test_static_standard import ROOT, PROJECT, parser_for, TestLoss, UnitTransformer
from cdlno_entry import parse_args
from model_dict import get_model
from cdlno.msar_lno.standard_entry import model_kwargs, StandardRun
from cdlno.msar_lno.standard import StaticModel
from cdlno.msar_lno.objective import training_forward, training_objective, ObjectiveMetrics
from cdlno.msar_lno.config import MSARArchitectureConfig, MSARTrainingConfig
from cdlno.msar_lno.profiles import resolve_profile
from cdlno.experiment import start, finish
from msar_entry_projection import strip_msar

TASKS={'darcy':('darcy',85,85,4),'elasticity':('elas',None,None,1),
       'airfoil':('airfoil',221,51,4),'pipe':('pipe',129,129,8)}
SMALL=['--d','8','--num-latents','7','5','3','2','--heads','2','2','4','4']
RECORDS=[]


def setUpModule():torch.set_num_threads(1)


def tearDownModule():
    if os.environ.get('MSAR_M5_RESULTS'):
        Path(os.environ['MSAR_M5_RESULTS']).write_text(json.dumps(RECORDS,indent=2)+'\n')


def source(task):return ast.parse((PROJECT/f'exp_{TASKS[task][0]}.py').read_text())


def arguments(task,flags=()):
    return parse_args(parser_for(task),task,['--model','msar_lno',*flags])


def construct(task,args,grid=(5,7),real=False):
    h,w=TASKS[task][1:3] if real else grid
    branch=next(n for n in ast.walk(source(task)) if isinstance(n,ast.If)
                and ast.unparse(n.test)=="args.model == 'msar_lno'"
                and any(isinstance(q,ast.Assign) and ast.unparse(q.targets[0])=='model' for q in n.body))
    call=branch.body[0].value.func.value # only omit .cuda()
    return eval(compile(ast.Expression(call),'<actual-msar-constructor>','eval'),
                dict(args=args,get_model=get_model,msar_model_kwargs=model_kwargs,s=h,s1=h,s2=w))


def sample(model,b=2,n=35,device='cpu'):
    if model.structured:n=model.H*model.W
    x=torch.randn(b,n,2,device=device)
    if model.task_name=='pipe':x=UnitTransformer(torch.randn(4,n,2,device=device)).encode(x)
    fx=torch.randn(b,n,device=device) if model.fun_dim else x.clone()
    if model.fun_dim:fx=UnitTransformer(torch.randn(4,n,device=device)).encode(fx)
    return x,fx


def train_nodes(task):
    tree=source(task)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='train_loader')
    nodes=[n for n in loop.body if not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and '.cuda()' in ast.unparse(n.value))]
    assert len(nodes)==len(loop.body)-1
    if task=='elasticity':
        epoch=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and loop in n.body)
        nodes += [n for n in epoch.body if ast.unparse(n)=='scheduler.step()']
    return nodes


def batch(task,model,args,device='cpu'):
    x,fx=sample(model,device=device);n=x.shape[1]
    normalizer=UnitTransformer(torch.randn(4,n,device=device))
    target=torch.randn(2,n,device=device)+2
    y=target if task=='airfoil' else normalizer.encode(target)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs) if task=='elasticity' else
               torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,epochs=args.epochs,steps_per_epoch=1))
    scope=dict(args=args,model=model.train(),x=x,fx=fx,y=y,y_normalizer=normalizer,
        torch=torch,F=F,rearrange=rearrange,myloss=TestLoss(size_average=False),
        de_x=TestLoss(size_average=False),de_y=TestLoss(size_average=False),s=model.H,dx=1/(model.H or 1),
        optimizer=optimizer,scheduler=scheduler,train_loss=0.,reg=0.,
        training_forward=training_forward,training_objective=training_objective,msar_epoch=ObjectiveMetrics())
    nodes=[]
    if task=='darcy':nodes.append(next(n for n in source(task).body if isinstance(n,ast.FunctionDef) and n.name=='central_diff'))
    nodes+=train_nodes(task)
    before=next(model.parameters()).detach().clone()
    with patch.object(optimizer,'step',wraps=optimizer.step) as step,patch.object(scheduler,'step',wraps=scheduler.step) as schedule:
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-msar-static-batch>','exec'),scope)
        assert step.call_count==schedule.call_count==1
    assert not torch.equal(before,next(model.parameters()))
    torch.testing.assert_close(scope['y'],target)
    return x,fx,target,normalizer,scope


class MSARStaticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91705)
        self.backend=sdpa_kernel(SDPBackend.MATH);self.backend.__enter__()
        self.addCleanup(self.backend.__exit__,None,None,None)

    def test_real_parsers_defaults_profiles_and_unsupported_tasks(self):
        for task in TASKS:
            args=arguments(task)
            self.assertEqual((args.epochs,args.batch_size,args.lr,args.max_grad_norm),(500,TASKS[task][3],.001,.1))
            self.assertEqual(args.msar_architecture['d'],96);self.assertEqual(args.msar_architecture['num_latents'],[512,256,128,64])
            self.assertEqual(args.msar_training['coverage_mode'],'floor')
            full=arguments(task,['--profile','full']);self.assertEqual(full.msar_architecture['d'],192)
            custom=arguments(task,['--profile','full',*SMALL,'--coverage-mode','off','--epochs','3'])
            self.assertEqual(custom.msar_architecture['d'],8);self.assertEqual(custom.epochs,3)
            self.assertIn('coverage_off',custom.msar_run_dir.parts)
            self.assertFalse(custom.msar_run_dir.exists())
            with self.assertRaises(ValueError):arguments(task,['--front-latent-mode','no_sa'])
            with self.assertRaises(ValueError):arguments(task,['--slice_num','32'])
            with self.assertRaises(ValueError):arguments(task,['--dropout','0'])
        from test_kcdno_tasks import arguments as kargs
        # M6 now authorizes the two temporal tasks; static defaults stay above.
        for task in ('ns', 'plasticity'):
            self.assertEqual(kargs(task,['--model','msar_lno']).msar_architecture['d'],96)

    def test_old_parser_defaults_and_no_msar_kwargs(self):
        for task in TASKS:
            base=parser_for(task).parse_args([]);actual=parse_args(parser_for(task),task,[])
            for key,value in vars(base).items():self.assertEqual(getattr(actual,key),value)
            self.assertFalse(hasattr(actual,'msar_architecture'))
            with self.assertRaises(ValueError):model_kwargs(actual)
            # The original parser rejects new options unless msar_lno is selected.
            with redirect_stdout(io.StringIO()),patch('sys.stderr',new=io.StringIO()),self.assertRaises(SystemExit):
                parse_args(parser_for(task),task,['--model','kcdno','--coverage-mode','off'])

    def test_experiment_json_objective_and_explicit_cli_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'PDE-Solving-StandardBenchmark/configs/msar_lno'
            dest.mkdir(parents=True)
            preset=json.loads((PROJECT/'configs/msar_lno/darcy.json').read_text())
            preset['objective'].update(coverage_mode='off',coverage_weight=0.)
            preset['profile']='full';(dest/'darcy.json').write_text(json.dumps(preset))
            with patch('cdlno.msar_lno.standard_entry.ROOT',Path(tmp)):
                args=arguments('darcy')
                self.assertEqual(args.msar_architecture['d'],192)
                self.assertEqual(args.msar_training['coverage_mode'],'off')
                args=arguments('darcy',['--profile','light','--coverage-mode','floor','--coverage-weight','.03'])
                self.assertEqual(args.msar_architecture['d'],96)
                self.assertEqual(args.msar_training['coverage_mode'],'floor')
                self.assertEqual(args.msar_training['coverage_weight'],.03)

    def test_input_lifting_conventions_and_no_convolution_or_extra_head(self):
        from cdlno.standard import _grid
        for task in TASKS:
            args=arguments(task,SMALL)
            model=(get_model(args).Model(H=5,W=7,**model_kwargs(args)) if task=='darcy' else construct(task,args))
            x,fx=sample(model)
            field=fx.unsqueeze(-1) if task=='darcy' else None
            saved={}
            handle=model.core.register_forward_pre_hook(lambda m,a:saved.update(e0=a[0].detach().clone()))
            result=model(x,field);handle.remove()
            if task=='darcy':
                pos=(_grid(5,7)[:,None]-_grid(8,8)[None]).square().sum(-1).sqrt()[None].expand(2,-1,-1)
                expected=model.preprocess(torch.cat((pos,field),-1))
                torch.testing.assert_close(model(x+7,field),result,atol=0,rtol=0)
            else:expected=model.preprocess(x)+model.placeholder[None,None]
            torch.testing.assert_close(saved['e0'],expected,atol=0,rtol=0)
            self.assertEqual(result.shape,(2,35,1))
            self.assertFalse(any(isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d)) for m in model.modules()))
            self.assertFalse(hasattr(model,'output'));self.assertFalse(hasattr(model,'time_fc'))
            self.assertEqual(hasattr(model,'placeholder'),task!='darcy')
            with self.assertRaises(ValueError):model(x,field,T=torch.ones(2,1))
            if model.structured:
                with self.assertRaisesRegex(ValueError,r'H\*W'):model(x[:,:-1],None if field is None else field[:,:-1])
            else:self.assertEqual(model(torch.randn(1,19,2),None).shape,(1,19,1))

    def test_four_tasks_real_loss_steps_eval_and_strict_checkpoint(self):
        for task in TASKS:
            for mode in ('floor','off'):
                with self.subTest(task=task,mode=mode),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    flags=[*SMALL,'--coverage-mode',mode,'--msar-run-dir',str(Path(tmp)/'run')]
                    args=arguments(task,flags)
                    # Original Darcy central_diff is square; wrapper separately tested 5x7.
                    grid=(7,7) if task=='darcy' else (5,7)
                    model=construct(task,args,grid=grid)
                    run=StandardRun(args,model)
                    x,fx,target,normalizer,scope=batch(task,model,args)
                    loss=scope['msar_loss']
                    torch.testing.assert_close(loss.total,scope['loss']+loss.coverage_weighted,atol=0,rtol=0)
                    if mode=='off':self.assertIs(loss.total,scope['loss'])
                    for name,p in model.named_parameters():
                        self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
                    if task=='darcy':torch.testing.assert_close(scope['loss'],scope['l2loss']+.1*scope['deriv_loss'])
                    model.eval();field=fx.unsqueeze(-1) if task=='darcy' else None
                    expected=model(x,field).detach();run.save(model)
                    # Execute original eval forward/decode/metric statements safely.
                    loop=next(n for n in ast.walk(source(task)) if isinstance(n,ast.For) and ast.unparse(n.iter)=='test_loader')
                    enodes=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('out','tl')]
                    scope.update(y=target)
                    with torch.no_grad():exec(compile(ast.Module(enodes,type_ignores=[]),'<real-static-eval>','exec'),scope)
                    prediction=expected.squeeze(-1) if task=='airfoil' else normalizer.decode(expected.squeeze(-1))
                    self.assertAlmostEqual(scope['tl'],TestLoss(size_average=False)(prediction,target).item(),places=6)
                    frozen={p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')}
                    ev=arguments(task,['--eval','1','--msar-run-dir',str(run.directory),'--coverage-mode','off' if mode=='floor' else 'floor'])
                    loaded=construct(task,ev,grid=grid).eval();erun=StandardRun(ev,loaded);erun.load(loaded)
                    torch.testing.assert_close(loaded(x,field),expected,atol=0,rtol=0)
                    self.assertEqual(frozen,{p.name:p.read_bytes() for p in (run.sidecar,run.directory/'task.json')})
                    with self.assertRaises(RuntimeError):erun.save(loaded)
                    RECORDS.append(dict(task=task,mode=mode,device='cpu',dtype='float32',shape=list(expected.shape),
                        original_loss='Darcy field+0.1 derivative L2' if task=='darcy' else 'original relative L2',
                        forward=True,backward=True,step=True,eval=True,strict_checkpoint=True,atol=0,rtol=0))

    def test_weight_zero_no_attention_request_prediction_parity_and_logging(self):
        args=arguments('airfoil',SMALL);on=construct('airfoil',args).train();off=construct('airfoil',arguments('airfoil',[*SMALL,'--coverage-weight','0'])).train()
        off.load_state_dict(on.state_dict(),strict=True);x=torch.randn(2,35,2)
        seen=[];handles=[d.register_forward_pre_hook(lambda m,a,k:seen.append(k),with_kwargs=True) for d in off.core.downs]
        a=training_forward(on,x,None);b=training_forward(off,x,None)
        for h in handles:h.remove()
        self.assertTrue(all(not kw.get('return_aux',False) for kw in seen));self.assertIsNone(b.auxiliary)
        torch.testing.assert_close(a.prediction,b.prediction,atol=2e-6,rtol=1e-4)
        pde=b.prediction.square().sum();loss=training_objective(pde,b);self.assertIs(loss.total,pde)
        stats=ObjectiveMetrics();stats.add(loss);self.assertFalse(stats.totals.requires_grad)
        values=stats.values();self.assertEqual(values['coverage_raw'],0);self.assertEqual(values['objective_total'],values['objective_pde'])

    def test_real_light_and_full_elasticity_expansion(self):
        # Full N972 is the actual point contract; weights/token counts are not shrunk.
        for profile,n in (('light',35),('full',972)):
            args=arguments('elasticity',['--profile',profile]);model=construct('elasticity',args).eval()
            with torch.no_grad():result=model(torch.randn(1,n,2),None)
            self.assertEqual(result.shape,(1,n,1));self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(model.config,resolve_profile(profile))
            self.assertEqual(model.core.downs[0].latent_queries.shape,(512 if profile=='light' else 1024,model.config.d))
            RECORDS.append(dict(profile=profile,task='elasticity',N=n,M1=model.config.num_latents[0],
                                parameters=sum(p.numel() for p in model.parameters()),device='cpu',forward='passed',backward='not_run'))

    def test_real_N_layout_reduced_dimensions(self):
        for task in TASKS:
            model=construct(task,arguments(task,SMALL),real=True).eval();x,fx=sample(model,b=1,n=972)
            with torch.no_grad():y=model(x,fx.unsqueeze(-1) if task=='darcy' else None)
            self.assertEqual(y.shape,(1,dict(darcy=7225,elasticity=972,airfoil=11271,pipe=16641)[task],1))
            RECORDS.append(dict(task=task,real_N=y.shape[1],d=8,grid='original',forward='passed',training='not_run'))

    def test_early_records_objective_history_and_repeated_eval_readonly(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=arguments('pipe',[*SMALL,'--coverage-mode','off','--msar-run-dir',str(Path(tmp)/'run')])
            record=start(args,'pipe')
            try:
                config=json.loads((record.directory/'config.json').read_text())
                self.assertEqual(config['parameter_count_status'],'pending_model_construction')
                self.assertEqual(config['model'],'msar_lno')
                model=construct('pipe',args);run=StandardRun(args,model)
                *_,scope=batch('pipe',model,args)
                run.record_objective(1,scope['msar_epoch'],dict(train_loss=scope['train_loss']/2,validation_relative_l2=1.))
                run.save(model)
            finally:finish(args)
            row=json.loads((record.directory/'train_history.jsonl').read_text());self.assertEqual(row['metrics']['coverage_mode'],'off')
            self.assertIn('objective_total',row['metrics']);self.assertEqual(row['metrics']['coverage_weighted'],0)
            frozen={n:(record.directory/n).read_bytes() for n in ('architecture.json','task.json','config.json','train_results.json')}
            for _ in range(2):
                ev=arguments('pipe',['--eval','1','--msar-run-dir',str(record.directory),'--profile','full'])
                r=start(ev,'pipe',evaluation=True)
                try:
                    m=construct('pipe',ev);erun=StandardRun(ev,m);erun.load(m);r.record_metrics({'relative_l2':.5})
                finally:finish(ev)
            self.assertEqual(frozen,{n:(record.directory/n).read_bytes() for n in frozen})
            self.assertEqual(len(json.loads((record.directory/'eval_results.json').read_text())['evaluations']),2)
            with self.assertRaises(FileExistsError):start(args,'pipe')

    def test_conflicts_before_loading_no_missing_field_repair_strict_state(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=arguments('darcy',[*SMALL,'--downsample','7','--msar-run-dir',str(Path(tmp)/'run')]);m=construct('darcy',args);run=StandardRun(args,m);run.save(m)
            common=['--eval','1','--msar-run-dir',str(run.directory)]
            self.assertEqual(arguments('darcy',common).downsample,7)
            for flags in (['--d','16'],['--num-latents','8','6','4','2'],['--ref','7'],['--downsample','5']):
                with patch('torch.load',side_effect=AssertionError('must reject first')),self.assertRaises(ValueError):arguments('darcy',common+flags)
            wrong=arguments('darcy',common);other=construct('darcy',wrong,grid=(7,5))
            with self.assertRaisesRegex(ValueError,'wrapper mismatch'):StandardRun(wrong,other)
            raw=json.loads((run.directory/'task.json').read_text());raw['arguments'].pop('ntrain')
            (run.directory/'task.json').write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'missing saved'):arguments('darcy',common)
            # Restore the exact field, then exercise the strict payload path.
            raw['arguments']['ntrain']=1000;(run.directory/'task.json').write_text(json.dumps(raw))
            ev=arguments('darcy',common);loaded=construct('darcy',ev);erun=StandardRun(ev,loaded)
            weights=torch.load(run.checkpoint,weights_only=True);weights.pop(next(iter(weights)));torch.save(weights,run.checkpoint)
            with self.assertRaises(RuntimeError):erun.load(loaded)

    def test_entire_old_entries_project_to_pre_M5_and_presets_match_training(self):
        before=json.loads((ROOT/'docs/msar_lno_audit/m5/before.json').read_text());snap=Path(before['archive'])/'source'
        for task in TASKS:
            path=f'PDE-Solving-StandardBenchmark/exp_{TASKS[task][0]}.py'
            self.assertEqual(ast.dump(strip_msar(source(task))),ast.dump(ast.parse((snap/path).read_text())))
            old=json.loads((PROJECT/'configs/kcdno'/f'{task}.json').read_text())
            new=json.loads((PROJECT/'configs/msar_lno'/f'{task}.json').read_text())
            self.assertEqual(new['training'],old['training']);self.assertEqual(new['adapter'],old['adapter'])
        for p in ('exp_ns.py','exp_plas.py'):
            self.assertEqual(ast.dump(strip_msar(ast.parse((PROJECT/p).read_text()))),
                             ast.dump(ast.parse((snap/'PDE-Solving-StandardBenchmark'/p).read_text())))

    def test_real_shell_options_all_profiles_and_last_override(self):
        for task in TASKS:
            script=ROOT/'tran_evaluate/msar_lno'/f'{task}.sh'
            subprocess.run(['bash','-n',str(script)],check=True)
            for flags in ([],['--coverage-mode','off','--coverage-weight','0'],['--profile','full']):
                result=subprocess.run(['bash',str(script),'train',*flags,'--dry-run'],capture_output=True,text=True,check=True)
                cmd=shlex.split(next(l[9:] for l in result.stdout.splitlines() if l.startswith('Command: ')))
                args=arguments(task,cmd[3:]);self.assertEqual(args.model,'msar_lno')
                self.assertNotIn('--slice_num',cmd);self.assertNotIn('--d',cmd)
                if flags and flags[0]=='--profile':self.assertEqual(args.msar_architecture['d'],192)
                if flags and flags[0]=='--coverage-mode':self.assertIn('coverage_off',args.msar_run_dir.parts)

    def test_all_eval_shell_modes_read_saved_structure_and_fresh_cwd_load(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for task in TASKS:
                args=arguments(task,[*SMALL,'--msar-run-dir',str(Path(tmp)/task)])
                model=construct(task,args).eval();run=StandardRun(args,model);run.save(model)
                for flags in ([],['--coverage-mode','off','--coverage-weight','0'],['--profile','full']):
                    result=subprocess.run(['bash',str(ROOT/'tran_evaluate/msar_lno'/f'{task}.sh'),'eval',
                        '--msar-run-dir',str(run.directory),*flags,'--dry-run'],capture_output=True,text=True,check=True)
                    cmd=shlex.split(next(l[9:] for l in result.stdout.splitlines() if l.startswith('Command: ')))
                    ev=arguments(task,cmd[3:])
                    self.assertEqual(ev.msar_architecture,model.config.to_dict())
                x,fx=sample(model);field=fx.unsqueeze(-1) if task=='darcy' else None
                torch.save(dict(x=x,fx=field,y=model(x,field).detach()),Path(tmp)/(task+'.pt'))
            code='''
import sys,torch
from pathlib import Path
from torch.nn.attention import SDPBackend,sdpa_kernel
from test_msar_static import arguments,construct
from cdlno.msar_lno.standard_entry import StandardRun
torch.set_num_threads(1)
backend=sdpa_kernel(SDPBackend.MATH);backend.__enter__()
root=Path(sys.argv[1])
for task in ('darcy','elasticity','airfoil','pipe'):
    args=arguments(task,['--eval','1','--msar-run-dir',str(root/task)])
    model=construct(task,args).eval();StandardRun(args,model).load(model)
    data=torch.load(root/(task+'.pt'),weights_only=True)
    torch.testing.assert_close(model(data['x'],data['fx']),data['y'],atol=0,rtol=0)
assert not any(name.startswith('exp_') for name in sys.modules)
backend.__exit__(None,None,None)
print('four static wrappers fresh-cwd strict load passed')
'''
            result=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=PROJECT,
                env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_visualization_eval_only_preserves_rng_training_and_weights(self):
        # Exercise the inherited observation hook with the actual new wrapper.
        # Capture rendering inputs rather than testing the established renderer again.
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            args=arguments('airfoil',[*SMALL,'--msar-run-dir',str(Path(tmp)/'run')]);r=start(args,'airfoil')
            try:
                model=construct('airfoil',args).train();StandardRun(args,model)
                x=torch.randn(2,35,2);target=torch.randn(2,35)
                dataset=torch.utils.data.TensorDataset(x,x,target)
                state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();seen=[]
                h=model.register_forward_pre_hook(lambda m,a:seen.append(m.training))
                with patch('cdlno.periodic_visualization.PeriodicFields._render') as render:
                    r.visualize(model,50,500,dataset=dataset,grid_shape=(5,7))
                    self.assertTrue(render.called)
                h.remove();self.assertTrue(seen);self.assertFalse(any(seen));self.assertTrue(model.training)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
            finally:finish(args)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_four_static_gpu_small_original_loss_steps(self):
        for task in TASKS:
            args=arguments(task,[*SMALL,'--coverage-kappa','1']);model=construct(task,args,grid=(7,7) if task=='darcy' else (5,7)).cuda()
            _,_,_,_,scope=batch(task,model,args,device='cuda')
            self.assertTrue(torch.isfinite(scope['msar_loss'].total))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            RECORDS.append(dict(task=task,device='cuda',dtype='float32',backend='MATH',original_loss_step='passed'))


if __name__=='__main__':unittest.main(verbosity=2)
