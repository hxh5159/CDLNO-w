"""M7 real single-graph PyG, original losses and trusted pickle; no main imports."""
import ast
import copy
from contextlib import redirect_stdout
from dataclasses import replace
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

import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
import yaml

from test_shapenet_car import CAR, ROOT, entry_parser, HAS_PYG
from models.cdlno_run import parse_args as car_parse
from test_airfrans import AIR, entry as air_entry, parser as air_parser, module
from msar_entry_projection import strip_msar
from cdlno.msar_lno.industrial import CarModel, AirfRANSModel
from cdlno.msar_lno.industrial_entry import CarRun, AirRun, model_kwargs, resolve_hparams
from cdlno.msar_lno.metadata import MSARMetadataMismatch
from cdlno.msar_lno.objective import training_forward, training_objective, ObjectiveMetrics
from cdlno.msar_lno.modules import CoverageFloorLoss, LearnedQueryDown

SMALL=['--d','8','--heads','2','2','4','4','--num-latents','7','5','3','2']
PROJECTS={'car':CAR,'airfrans':AIR}
CLASSES={'car':CarModel,'airfrans':AirfRANSModel}
CASES=[]


def tearDownModule():
    target=os.environ.get('MSAR_M7_RESULTS')
    if target:Path(target).write_text(json.dumps(CASES,indent=2)+'\n')


def arguments(task, flags=(), evaluation=False):
    if task=='car':
        return car_parse(entry_parser(evaluation), evaluation=evaluation, argv=['--cfd_model','msar_lno',*flags])
    return air_entry.parse_args(air_parser(evaluation),evaluation=evaluation,argv=['--model','msar_lno',*flags])


def run(args, *, model=None, evaluation=False, device='cpu'):
    if args.msar_task=='car':return CarRun(args,device=device,model=model,evaluation=evaluation)
    return AirRun(args,resolve_hparams(args,{}),device=device,evaluation=evaluation)


def graph(task,n=11):
    from torch_geometric.data import Data,Batch
    g=Data(x=torch.randn(n,7),pos=torch.randn(n,3 if task=='car' else 2),
           y=torch.randn(n,4),surf=torch.arange(n)%3==0,edge_index=torch.tensor([[0,1],[1,0]]))
    g=Batch.from_data_list([g])
    return (g,Data(x=torch.randn(5,3))) if task=='car' else g


def cfd(g):return g[0] if isinstance(g,tuple) else g


def pde(task,pred,data,reg=.7):
    errors=(pred-data.y).square()
    if task=='car':return errors[:,:3].mean()+reg*errors[data.surf,3].mean()
    return errors[~data.surf].mean()+reg*errors[data.surf].mean()


def trainer(task):return module(PROJECTS[task]/'train.py','_msar_'+task+'_train')


def save(run, models):
    if run.task=='car':torch.save(models[0],run.checkpoint)
    else:
        torch.save(models,run.checkpoint)
        for i,m in enumerate(models):
            Path(run.member_dir(i)).mkdir(exist_ok=True)
            torch.save(m,Path(run.member_dir(i))/'model')


class IndustrialMSAR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
    @classmethod
    def tearDownClass(cls):
        cls.math.__exit__(None,None,None);torch.set_num_threads(cls.threads)
    def setUp(self):torch.manual_seed(731)

    def test_real_parser_profiles_and_inapplicable_flags(self):
        for task in PROJECTS:
            a=arguments(task)
            self.assertEqual(a.msar_architecture['num_latents'],[512,256,128,64])
            self.assertEqual(a.msar_architecture['d'],96)
            self.assertEqual(a.nb_epochs,200 if task=='car' else 398)
            self.assertEqual((a.batch_size,a.lr),(1,.001))
            self.assertEqual(a.msar_training['coverage_mode'],'floor')
            a=arguments(task,['--profile','full','--d','16'])
            self.assertEqual(a.msar_architecture['num_latents'],[1024,512,256,128])
            self.assertEqual(a.msar_architecture['d'],16)
            a=arguments(task,['--coverage-mode','off','--coverage-weight','0'])
            self.assertIn('/msar_lno/light/coverage_off/',str(a.msar_run_dir))
            for flags in (['--batch_size','2'],['--front-blocks','2'],['--history-mode','off'],['--slice_num','64']):
                with self.assertRaises(ValueError):arguments(task,flags)
            with self.assertRaises(FileNotFoundError):arguments(task,['--msar-run-dir','/absent/msar/run'],True)

    def test_frozen_complete_entries_trainers_yaml_and_core(self):
        before=json.loads((ROOT/'docs/msar_lno_audit/m7/before.json').read_text())
        source=Path(before['archive'])/'source'
        for project in PROJECTS.values():
            for name in ('main.py','main_evaluation.py','train.py'):
                relative=project.relative_to(ROOT)/name
                self.assertEqual(ast.dump(strip_msar(ast.parse((project/name).read_text()))),
                                 ast.dump(ast.parse((source/relative).read_text())),str(relative))
        old=yaml.safe_load((source/'Airfoil-Design-AirfRANS/params.yaml').read_text())
        now=yaml.safe_load((AIR/'params.yaml').read_text());new=now.pop('msar_lno')
        self.assertEqual(now,old);self.assertEqual(new,old['kcdno'])
        for path in ['cdlno/msar_lno/core.py','cdlno/msar_lno/modules.py','cdlno/msar_lno/objective.py',
                     'cdlno/modules.py','cdlno/cdpa.py','Airfoil-Design-AirfRANS/utils/metrics.py',
                     'Car-Design-ShapeNetCar/utils/drag_coefficient.py']:
            self.assertEqual((ROOT/path).read_bytes(),(source/path).read_bytes())

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_real_graph_inputs_permutation_reference_and_batch_rejection(self):
        from torch_geometric.data import Batch
        for task in PROJECTS:
            a=arguments(task,SMALL);m=CLASSES[task](**model_kwargs(a)).eval()
            self.assertFalse(any(isinstance(layer,(nn.Conv1d,nn.Conv2d,nn.Conv3d)) for layer in m.modules()))
            for n in (11,23):
                g=graph(task,n);data=cfd(g);before=data.clone();out=m(g)
                self.assertEqual(out.shape,(n,4))
                order=torch.randperm(n);perm=data.clone()
                for field in ('x','pos','y','surf','batch'):perm[field]=data[field][order]
                permuted=(perm,g[1]) if task=='car' else perm
                torch.testing.assert_close(m(permuted),out[order],atol=2e-6,rtol=1e-4)
                data.y+=10;data.edge_index=torch.empty(2,0,dtype=torch.long)
                torch.testing.assert_close(m(g),out,atol=0,rtol=0)
                data.y=before.y;data.edge_index=before.edge_index
                for key,value in before.to_dict().items():torch.testing.assert_close(data[key],value,atol=0,rtol=0)
                multi=Batch.from_data_list([data.to_data_list()[0]]*2)
                with self.assertRaises(ValueError):m((multi,g[1]) if task=='car' else multi)
                if task=='airfrans':
                    captured=[]
                    hook=m.preprocess.register_forward_pre_hook(lambda _,inputs:captured.append(inputs[0].detach().clone()))
                    m(g);hook.remove()
                    distances=(data.pos[:,None]-m.reference[None]).square().sum(-1).sqrt()
                    torch.testing.assert_close(captured[0],torch.cat((data.x,distances),-1)[None],atol=0,rtol=0)
            self.assertEqual(len(list(m.parameters())),len({p.untyped_storage().data_ptr() for p in m.parameters()}))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_original_weighted_losses_steps_and_eval(self):
        for task in PROJECTS:
            for mode in ('off','floor'):
                a=arguments(task,[*SMALL,'--coverage-mode',mode,'--coverage-kappa','1'])
                m=CLASSES[task](**model_kwargs(a));reference=copy.deepcopy(m)
                data=graph(task);t=trainer(task)
                opt=torch.optim.Adam(m.parameters(),lr=.001)
                sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=20)
                refopt=torch.optim.Adam(reference.parameters(),lr=.001)
                refsched=torch.optim.lr_scheduler.OneCycleLR(refopt,max_lr=.001,total_steps=20)
                f=training_forward(reference,data);raw=pde(task,f.prediction,cfd(data))
                objective=training_objective(raw,f);objective.total.backward();refopt.step();refsched.step()
                metrics=ObjectiveMetrics()
                kwargs=dict(reg=.7,msar_metrics=metrics)
                if task=='airfrans':kwargs['criterion']='MSE_weighted'
                t.train('cpu',m,[data],opt,sched,**kwargs)
                self.assertEqual(sched.last_epoch,1)
                for key,value in reference.state_dict().items():torch.testing.assert_close(m.state_dict()[key],value,atol=0,rtol=0)
                expected=objective.log_values();got=metrics.values()
                for new,old in [('objective_pde','pde'),('coverage_raw','coverage_raw'),('coverage_weighted','coverage_weighted'),('objective_total','total')]:
                    self.assertAlmostEqual(got[new],expected[old].item(),places=6)
                if mode=='off':self.assertEqual(got['objective_pde'],got['objective_total'])
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                with patch.object(CoverageFloorLoss,'forward',side_effect=AssertionError('eval computed coverage')):
                    losses=t.test('cpu',m,[data]);self.assertTrue(np.isfinite(np.asarray(losses[0])).all())
                CASES.append(dict(task=task,mode=mode,device='cpu',real_PyG=True,original_loss=True,
                    optimizer_steps=1,eval=True,objective=got))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_coverage_off_weightzero_order_and_uniform_measure(self):
        for task in PROJECTS:
            m=CLASSES[task](**model_kwargs(arguments(task,SMALL))).train();g=graph(task)
            # A is requested only for four Down layers, no source_measure or label weights.
            requests=[];handles=[d.register_forward_pre_hook(lambda mod,args,kw:requests.append(kw.copy()),with_kwargs=True) for d in m.core.downs]
            off=replace(m.training_config,coverage_mode='off');zero=replace(m.training_config,coverage_weight=0)
            with patch.object(CoverageFloorLoss,'forward',side_effect=AssertionError('off computed coverage')):
                x=training_forward(m,g,training_config=off);z=training_forward(m,g,training_config=zero)
            self.assertTrue(all(not item.get('return_aux',False) for item in requests));requests.clear()
            recorded=[]
            original=CoverageFloorLoss.forward
            def capture(mod,attention,**kwargs):
                self.assertIsNone(kwargs.get('source_measure'));recorded.append(attention.shape)
                return original(mod,attention,**kwargs)
            with patch.object(CoverageFloorLoss,'forward',capture):y=training_forward(m,g)
            self.assertEqual(len(recorded),4)
            for h in handles:h.remove()
            torch.testing.assert_close(x.prediction,y.prediction,atol=2e-6,rtol=1e-4)
            torch.testing.assert_close(z.prediction,x.prediction,atol=0,rtol=0)
            self.assertIs(y.auxiliary.prediction,y.prediction)
            self.assertIsNone(y.auxiliary.diagnostics)
            with torch.no_grad():
                diagnostic=m(g,return_aux=True,training_config=replace(off,diagnostics=True))
            for group in diagnostic.diagnostics.values():
                for row in group:
                    for value in row.values():self.assertLessEqual(value.numel(),2);self.assertFalse(value.requires_grad)

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_checkpoint_whole_list_member_readfirst_and_fresh_cwd(self):
        for task in PROJECTS:
            for mode in ('floor','off'):
                with self.subTest(task=task,mode=mode),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    path=Path(tmp)/'run';flags=[*SMALL,'--coverage-mode',mode,'--nb_epochs','3','--msar-run-dir',str(path)]
                    if task=='airfrans':flags+=['--nmodel','2']
                    a=arguments(task,flags);models=[CLASSES[task](**model_kwargs(a)).eval() for _ in range(2 if task=='airfrans' else 1)]
                    r=run(a,model=models[0]);save(r,models);g=graph(task)
                    expected=models[0](g).detach();files={p.name:p.read_bytes() for p in (r.sidecar,path/'task.json')}
                    ev=arguments(task,['--msar-run-dir',str(path),'--coverage-mode','off' if mode=='floor' else 'floor'],True)
                    er=run(ev,evaluation=True);loaded=er.load();loaded=loaded[0] if task=='airfrans' else loaded
                    torch.testing.assert_close(loaded.eval()(g),expected,atol=0,rtol=0)
                    if task=='airfrans':
                        for i,m in enumerate(models):torch.testing.assert_close(er.load(member=i).eval()(g),m(g),atol=0,rtol=0)
                    for flags in (['--d','16'],['--num-latents','8','5','3','2'],['--nb_epochs','4']):
                        with patch('torch.load',side_effect=AssertionError('unpickle before validation')):
                            with self.assertRaises(MSARMetadataMismatch):arguments(task,['--msar-run-dir',str(path),*flags],True)
                    self.assertEqual(files,{p.name:p.read_bytes() for p in (r.sidecar,path/'task.json')})
                    torch.save(dict(x=cfd(g).x,pos=cfd(g).pos,expected=expected),Path(tmp)/'probe.pt')
                    code='''
import sys, torch
from pathlib import Path
from torch_geometric.data import Data
from torch.nn.attention import sdpa_kernel,SDPBackend
import ast, argparse
from cdlno.msar_lno.industrial_entry import CarRun,AirRun,model_kwargs,resolve_hparams
from models.MSAR_LNO import Model
p=Path(sys.argv[1]);task=sys.argv[2];torch.set_num_threads(1)
if task=='car':
 from models.cdlno_run import parse_args
else:
 from cdlno_entry import parse_args
nodes=[n for n in ast.parse(Path('main_evaluation.py').read_text()).body if
 (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser') or
 (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument')]
scope={'argparse':argparse};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-parser-only>','exec'),scope)
a=parse_args(scope['parser'],evaluation=True,argv=['--cfd_model' if task=='car' else '--model','msar_lno','--msar-run-dir',str(p/'run')])
r=CarRun(a,device='cpu',evaluation=True) if task=='car' else AirRun(a,resolve_hparams(a,{}),device='cpu',evaluation=True)
loaded=r.load();m=loaded[0] if task=='airfrans' else loaded
assert type(m) is Model and Model.__module__=='cdlno.msar_lno.industrial'
assert type(Model(**model_kwargs(a))) is type(m)
probe=torch.load(p/'probe.pt',weights_only=True);data=Data(x=probe['x'],pos=probe['pos'])
with sdpa_kernel(SDPBackend.MATH):
 torch.testing.assert_close(m.eval()((data,None) if task=='car' else data),probe['expected'],atol=0,rtol=0)
assert 'main' not in sys.modules and 'main_evaluation' not in sys.modules
'''
                    result=subprocess.run([sys.executable,'-B','-c',code,tmp,task],cwd=PROJECTS[task],env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tests')),capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    CASES.append(dict(task=task,mode=mode,strict_checkpoint=True,fresh_cwd=True,atol=0,rtol=0))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_checkpoint_corrupt_family_shape_behavior_and_shared_list(self):
        for task in PROJECTS:
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                p=Path(tmp)/'run';a=arguments(task,[*SMALL,'--msar-run-dir',str(p)])
                m=CLASSES[task](**model_kwargs(a));r=run(a,model=m);save(r,[m]);meta=r.sidecar.read_text()
                for family in ('kcdno',None):
                    val=json.loads(meta)
                    if family is None:del val['family']
                    else:val['family']=family
                    r.sidecar.write_text(json.dumps(val))
                    with patch('torch.load',side_effect=AssertionError('unpickle before metadata')):
                        with self.assertRaises(MSARMetadataMismatch):arguments(task,['--msar-run-dir',str(p)],True)
                r.sidecar.write_text(meta);ev=arguments(task,['--msar-run-dir',str(p)],True);er=run(ev,evaluation=True)
                for corruption in ('key','shape','heads','sharing','class','reference'):
                    if corruption=='reference' and task=='car':continue
                    bad=copy.deepcopy(m)
                    if corruption=='key':del bad.core.downs[0].to_k.weight
                    elif corruption=='shape':bad.core.output.weight=nn.Parameter(torch.zeros(5,8))
                    elif corruption=='heads':bad.core.downs[0].heads=4
                    elif corruption=='sharing':bad.core.encoders[0][1]=bad.core.encoders[0][0]
                    elif corruption=='class':bad=nn.Linear(8,4)
                    else:bad.reference.add_(1)
                    save(r,[bad])
                    with self.assertRaises(MSARMetadataMismatch):er.load()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            a=arguments('airfrans',[*SMALL,'--nmodel','2','--msar-run-dir',str(Path(tmp)/'run')]);r=run(a)
            m=AirfRANSModel(**model_kwargs(a));save(r,[m,m]);ev=arguments('airfrans',['--msar-run-dir',str(r.directory)],True)
            with self.assertRaises(MSARMetadataMismatch):run(ev,evaluation=True).load()

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_car_original_complete_synthetic_epoch_records(self):
        from cdlno.experiment import start,finish
        from torch_geometric.data import Data
        for mode in ('floor','off'):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                path=Path(tmp)/'run';a=arguments('car',[*SMALL,'--coverage-mode',mode,'--nb_epochs','1','--msar-run-dir',str(path)])
                record=start(a,'car',evaluation=False);m=CarModel(**model_kwargs(a));r=run(a,model=m)
                pairs=[]
                for n in (11,17):
                    g,geom=graph('car',n);pairs.append((g.to_data_list()[0],geom))
                hp=dict(lr=.001,batch_size=1,nb_epochs=1)
                trainer('car').main('cpu',pairs,pairs,m,hp,str(path),reg=.7,coef_norm=(np.zeros(7),np.ones(7),np.zeros(4),np.ones(4)),record=record)
                finish(a)
                rows=[json.loads(line) for line in (path/'train_history.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),1)
                metrics=rows[0]['metrics'];self.assertEqual(metrics['coverage_mode'],mode)
                self.assertAlmostEqual(metrics['objective_total'],metrics['objective_pde']+metrics['coverage_weighted'],places=6)
                saved={p.name:p.read_bytes() for p in (r.sidecar,path/'task.json',path/'config.json',path/'train_results.json')}
                for _ in range(2):
                    ev=arguments('car',['--msar-run-dir',str(path)],True);rr=start(ev,'car',evaluation=True);run(ev,evaluation=True).load();rr.record_metrics({'synthetic_eval':1});finish(ev)
                self.assertEqual(saved,{p.name:p.read_bytes() for p in (r.sidecar,path/'task.json',path/'config.json',path/'train_results.json')})
                CASES.append(dict(task='car',mode=mode,complete_synthetic_epoch=True,recorded=True,repeated_eval=True))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_air_original_epoch_record_fragment_and_eval_immutability(self):
        from cdlno.experiment import start,finish
        tree=ast.parse((AIR/'train.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        epoch=next(n for n in main.body if isinstance(n,ast.For))
        recording=next(n for n in epoch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='record is not None')
        # Epoch 1 of 398 does not schedule a field plot; the actual record and
        # visualize calls remain in the executed source fragment.
        for mode in ('floor','off'):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                path=Path(tmp)/'run';a=arguments('airfrans',[*SMALL,'--coverage-mode',mode,'--msar-run-dir',str(path)])
                record=start(a,'airfrans',evaluation=False);r=run(a)
                m=AirfRANSModel(**model_kwargs(a));record.attach_model(m,hparams=r.hparams,member=0,protocol=r.contract)
                g=graph('airfrans');t=trainer('airfrans');opt=torch.optim.Adam(m.parameters(),lr=.001)
                scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=20)
                metrics=ObjectiveMetrics();losses=t.train('cpu',m,[g],opt,scheduler,criterion='MSE_weighted',reg=.7,msar_metrics=metrics)
                val=t.test('cpu',m,[g]);train_loss,_,loss_surf_var,loss_vol_var,loss_surf,loss_vol=losses
                scope=dict(record=record,epoch=0,train_loss=train_loss,loss_surf=loss_surf,loss_vol=loss_vol,
                    loss_surf_var=loss_surf_var,loss_vol_var=loss_vol_var,criterion='MSE_weighted',reg=.7,
                    val_iter=10,val_loss=val[0],val_surf_var=val[2],val_vol_var=val[3],val_surf=val[4],val_vol=val[5],
                    msar_training=True,msar_values=metrics.values(),model=m,record_member=0,hparams=r.hparams,
                    val_dataset=[g],visualization_norm=None)
                exec(compile(ast.Module(body=[recording],type_ignores=[]),'<original-epoch-recording>','exec'),scope)
                save(r,[m.eval()]);finish(a)
                row=json.loads((path/'train_history.jsonl').read_text())
                self.assertEqual(row['metrics']['coverage_mode'],mode)
                self.assertEqual(row['metrics']['objective_pde'],metrics.values()['objective_pde'])
                saved={p.name:p.read_bytes() for p in (r.sidecar,path/'task.json',path/'config.json',path/'train_results.json')}
                for _ in range(2):
                    ev=arguments('airfrans',['--msar-run-dir',str(path),'--coverage-mode','off'],True)
                    rr=start(ev,'airfrans',evaluation=True);run(ev,evaluation=True).load();rr.record_metrics({'synthetic_masked_MSE':float(val[0])});finish(ev)
                self.assertEqual(saved,{p.name:p.read_bytes() for p in (r.sidecar,path/'task.json',path/'config.json',path/'train_results.json')})
                CASES.append(dict(task='airfrans',mode=mode,epoch_record_fragment=True,repeated_eval=True,
                    full_sampled_epoch=False,original_weighted_loss=True))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_air_original_sampled_epoch_when_extension_available(self):
        try:
            from torch_cluster import radius_graph
        except ImportError:self.skipTest('torch_cluster absent: original sampled epoch/graph rebuild not executed')
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            a=arguments('airfrans',SMALL);m=AirfRANSModel(**model_kwargs(a));graphs=[graph('airfrans').to_data_list()[0]]
            hp=resolve_hparams(a,{})|dict(nb_epochs=1,subsampling=7)
            trainer('airfrans').main('cpu',graphs,graphs,m,hp,tmp,criterion='MSE_weighted',reg=.7,name_mod='msar_lno',val_sample=True)

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_air_sampled_ptr_original_scatter_fragments(self):
        # Execute only safe original gather/scatter/boundary AST, not radius_graph
        # or VTK. This is NOT acceptance of the complete sampled evaluator.
        tree=ast.parse((AIR/'utils/metrics.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='Infer_test')
        loop=next(n for n in fn.body if isinstance(n,ast.While))
        gathers=[n for n in loop.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('data_sampled.pos','data_sampled.x','data_sampled.y','data_sampled.surf','data_sampled.batch')]
        scatters=[n for n in ast.walk(loop) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ('out[n][idx]','outs[n]','n_out[idx]')]
        boundary=next(n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(n, out)')
        m=AirfRANSModel(**model_kwargs(arguments('airfrans',SMALL))).eval();data=graph('airfrans',19)
        scope=dict(torch=torch,data=data,coef_norm=None,n=0,n_out=torch.zeros(19,1),outs=[torch.zeros(19,4)])
        expected=torch.zeros(19,4);counts=torch.zeros(19,1)
        for idx in (torch.tensor([5,1,7,3,12,18,0,6,10,14]),torch.tensor([2,4,8,9,11,13,15,16,17,5])):
            scope.update(data_sampled=data.clone(),idx=idx,out=[torch.zeros(19,4)])
            exec(compile(ast.Module(body=gathers,type_ignores=[]),'<original-gather>','exec'),scope)
            part=scope['data_sampled'];self.assertEqual(part.ptr.tolist(),[0,19])
            prediction=m(part).detach();self.assertEqual(prediction.shape,(len(idx),4))
            scope['o']=prediction
            exec(compile(ast.Module(body=scatters,type_ignores=[]),'<original-scatter>','exec'),scope)
            expected[idx]+=prediction;counts[idx]+=1
        expected/=counts;expected[data.surf,:2]=0;expected[data.surf,3]=0
        exec(compile(ast.Module(body=[boundary],type_ignores=[]),'<original-boundary>','exec'),scope)
        torch.testing.assert_close(scope['outs'][0],expected,atol=0,rtol=0)
        CASES.append(dict(task='airfrans',sampling_scatter='original AST fragments only',real_PyG=True,full_sampled_evaluator=False))

    @unittest.skipUnless(HAS_PYG,'real PyG unavailable')
    def test_profiles_real_light_full_small_N(self):
        for task in PROJECTS:
            for profile in ('light','full'):
                a=arguments(task,['--profile',profile]);m=CLASSES[task](**model_kwargs(a)).eval()
                with torch.no_grad():output=m(graph(task,11))
                self.assertEqual(output.shape,(11,4));self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(m.config.num_latents[0],11)
                CASES.append(dict(task=task,profile=profile,N=11,latent_expansion=True,device='cpu',
                    parameters=sum(p.numel() for p in m.parameters()),forward=True,backward=False))

    @unittest.skipUnless(HAS_PYG and torch.cuda.is_available(),'real PyG/CUDA unavailable')
    def test_cuda_real_graph_loss_steps(self):
        for task in PROJECTS:
            a=arguments(task,[*SMALL,'--coverage-kappa','1']);m=CLASSES[task](**model_kwargs(a)).cuda()
            g=graph(task);t=trainer(task);opt=torch.optim.Adam(m.parameters(),lr=.001)
            sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.001,total_steps=20);metrics=ObjectiveMetrics()
            kw=dict(msar_metrics=metrics,reg=.7)
            if task=='airfrans':kw['criterion']='MSE_weighted'
            t.train('cuda',m,[g],opt,sched,**kw);t.test('cuda',m,[g])
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            CASES.append(dict(task=task,device='cuda',dtype='float32',backend='MATH',real_PyG=True,original_loss=True,objective=metrics.values()))

    def test_scripts_real_parsers_train_and_eval(self):
        for task in PROJECTS:
            for profile,mode in [('light','floor'),('light','off'),('full','floor')]:
                with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                    path=Path(tmp)/'run';script=ROOT/'tran_evaluate/msar_lno'/f'{task}.sh'
                    flags=['--profile',profile,'--coverage-mode',mode,'--msar-run-dir',str(path)]
                    if mode=='off':flags+=['--coverage-weight','0']
                    result=subprocess.run(['bash',str(script),'train',*flags,'--dry-run'],cwd=tmp,capture_output=True,text=True,check=True)
                    command=shlex.split(next(line[9:] for line in result.stdout.splitlines() if line.startswith('Command: ')))
                    parse=car_parse if task=='car' else air_entry.parse_args
                    parser=entry_parser if task=='car' else air_parser
                    a=parse(parser(),argv=command[3:]);self.assertEqual(a.profile,profile)
                    # Save authentic full-profile metadata without fake model/checkpoint.
                    r=run(a)
                    result=subprocess.run(['bash',str(script),'eval','--msar-run-dir',str(path),'--dry-run'],cwd=tmp,capture_output=True,text=True,check=True)
                    command=shlex.split(next(line[9:] for line in result.stdout.splitlines() if line.startswith('Command: ')))
                    e=parse(parser(True),evaluation=True,argv=command[3:])
                    self.assertEqual(e.msar_architecture,a.msar_architecture)
                    self.assertEqual(e.msar_training,a.msar_training)


if __name__=='__main__':unittest.main(verbosity=2)
