"""Output lifecycle and observational regression, with no real dataset entry imports."""
import ast
from contextlib import redirect_stdout
import copy
import importlib.util
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
import test_front_task_modes as task_modes
from cdlno.experiment import start, finish, reserve_directory, default_directory
from output_recording_projection import strip_recording

ROOT = Path(__file__).resolve().parents[1]
BEFORE = Path(os.environ.get('CDLNO_OUTPUT_BASELINE', '/home/hwz/CDLNO-artifacts/output-before-kql2cnpe')) / 'source'


class ExperimentRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_task_modes_early_config_actual_counts_epochs_repeat_eval_checkpoint(self):
        for task in task_modes.TASKS:
            for mode in task_modes.MODES:
                with self.subTest(task=task, mode=mode), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                    path = Path(folder) / 'output' / task / 'timestamp'
                    flags = [*task_modes.SMALL, '--front-latent-mode', mode, *task_modes.run_option(task, path)]
                    args = task_modes.parse(task, flags)
                    record = start(args, task)
                    try:
                        self.assertTrue((path / 'train.log').exists())
                        pending = json.loads((path / 'config.json').read_text())
                        self.assertEqual(pending['parameter_count_status'], 'pending_model_construction')
                        self.assertIsNone(pending['parameters'])
                        self.assertEqual(pending['resolved_arguments']['front_latent_mode'], mode)
                        print('early config exists before constructing real model')
                        model = task_modes.build(task, args)
                        run = task_modes.make_run(task, args, model)
                        if task == 'airfrans':
                            record.attach_model(model, hparams=run.hparams, protocol=run.contract)
                        attached = json.loads((path / 'config.json').read_text())
                        self.assertEqual(attached['parameters']['total'], sum(p.numel() for p in model.parameters()))
                        rng = torch.get_rng_state().clone()
                        tensors = task_modes.inputs(task, model)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
                        record.record_training_setup(optimizer, scheduler)
                        # Purposefully labelled synthetic MSE, original loss tests are reused separately.
                        output = model(*tensors)
                        loss = output.square().mean()
                        loss.backward(); optimizer.step(); scheduler.step()
                        state_after_step = {k: v.clone() for k, v in model.state_dict().items()}
                        rng_after_step = torch.get_rng_state().clone()
                        record.record_epoch(1, {'synthetic_mse': loss, 'known_metric': 0.125})
                        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_step))
                        for k,v in state_after_step.items():
                            self.assertTrue(torch.equal(v, model.state_dict()[k]))
                        model.eval()
                        with torch.no_grad(): expected = model(*tensors)
                        task_modes.save(task, run, model)
                        finish(args)
                    finally:
                        if not record.closed: record.finish(status='failed')
                    saved_config, saved_sidecar = (path / 'config.json').read_bytes(), run.sidecar.read_bytes()
                    results = json.loads((path / 'train_results.json').read_text())
                    self.assertEqual(results['status'], 'completed')
                    self.assertEqual(results['final_recorded_metrics_by_member']['0']['known_metric'], .125)
                    self.assertEqual(len((path / 'train_history.jsonl').read_text().splitlines()), 1)
                    for attempt in range(2):
                        ev = task_modes.parse(task, [*flags, '--cdpa-source-chunk-size', '1'], True)
                        er = start(ev, task, evaluation=True)
                        try:
                            requested = task_modes.build(task, ev) if task in task_modes.STANDARD else None
                            loaded_run = task_modes.make_run(task, ev, requested, True)
                            loaded = task_modes.load(task, loaded_run, requested).eval()
                            with torch.no_grad(): actual = loaded(*tensors)
                            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=3e-4)
                            er.record_metrics(dict(synthetic_output_max_error=(actual-expected).abs().max(), attempt=attempt))
                            finish(ev)
                        finally:
                            if not er.closed: er.finish(status='failed')
                        self.assertEqual((path / 'config.json').read_bytes(), saved_config)
                        self.assertEqual(run.sidecar.read_bytes(), saved_sidecar)
                    index = json.loads((path / 'eval_results.json').read_text())['evaluations']
                    self.assertEqual(len(index), 2)
                    self.assertEqual({x['metrics']['attempt'] for x in index}, {0, 1})
                    self.assertTrue(all((path / x['result_file']).with_name('eval.log').exists() for x in index))
                    with self.assertRaises(FileExistsError): start(args, task)

    def test_exceptions_and_incomplete_exit_leave_early_records(self):
        for ending, status in [("raise FileNotFoundError('synthetic startup failure before data IO')", 'failed'),
                               ('raise KeyboardInterrupt()', 'interrupted'), ('', 'incomplete')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                code = '''
from argparse import Namespace
from cdlno.experiment import start
from pathlib import Path
import sys
args=Namespace(cdlno_run_dir=Path(sys.argv[1]), model='CDLNO', front_latent_mode='full')
start(args,'darcy')
print('startup log is preserved')
''' + ending
                path = Path(temp)/'run'
                p = subprocess.run([sys.executable,'-B','-c',code,str(path)], cwd=ROOT, capture_output=True,text=True)
                data = json.loads((path/'train_results.json').read_text())
                self.assertEqual(data['status'],status,p.stderr)
                self.assertEqual(data['phase'],'startup')
                self.assertEqual(json.loads((path/'config.json').read_text())['parameter_count_status'], 'pending_model_construction')
                log=(path/'train.log').read_text()
                self.assertIn('startup log is preserved',log)
                if ending: self.assertIn('Traceback',log)

    def test_same_process_only_reservation_and_unique_defaults(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CDLNO_RUNS_ROOT=temp), redirect_stdout(io.StringIO()):
            a=Namespace(cdlno_run_dir=None, model='CDLNO')
            record=start(a,'darcy')
            try:
                reserve_directory(a,a.cdlno_run_dir)
                with self.assertRaises(FileExistsError): reserve_directory(copy.copy(a),a.cdlno_run_dir)
                self.assertEqual(record.directory.parent,Path(temp)/'darcy')
                self.assertNotEqual(default_directory('darcy'),record.directory)
            finally: record.finish(status='incomplete')

    def test_entries_early_lifecycle_and_complete_observation_only_ast(self):
        if not BEFORE.is_dir():
            self.skipTest("pre-change source snapshot absent; set CDLNO_OUTPUT_BASELINE to the preserved artifact")
        paths = [Path('PDE-Solving-StandardBenchmark') / f'exp_{stem}.py'
                 for stem in ('darcy','elas','airfoil','pipe','ns','plas')]
        paths += [Path(project)/file for project in ('Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS')
                  for file in ('main.py','main_evaluation.py','train.py')]
        for relative in paths:
            with self.subTest(path=str(relative)):
                now=ast.parse((ROOT/relative).read_text(),feature_version=(3,10))
                old=ast.parse((BEFORE/relative).read_text())
                self.assertEqual(ast.dump(strip_recording(copy.deepcopy(now))),ast.dump(old))
                if relative.name=='train.py': continue
                calls=[n for n in ast.walk(now) if isinstance(n,ast.Call)]
                early=next(n for n in calls if ast.unparse(n.func)=='start_experiment')
                data_calls=[n for n in calls if ast.unparse(n.func) in ('scio.loadmat','np.load','Dataset',
                                    'load_train_val_fold','load_train_val_fold_file','MatReader')]
                self.assertTrue(data_calls)
                self.assertTrue(all(early.lineno < n.lineno for n in data_calls))
                self.assertTrue(any(ast.unparse(n.func)=='finish_experiment' for n in calls))
        self.assertFalse(any(k.startswith('exp_') for k in sys.modules))

    def test_all_launch_pairs_same_timestamp_explicit_override_and_no_writes(self):
        for task in task_modes.TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temp:
                env=dict(os.environ,CDLNO_RUNS_ROOT=str(Path(temp)/'output'))
                env.pop('CDLNO_RUN_TAG',None)
                for explicit in (False,True):
                    requested=Path(temp)/'explicit run with spaces'
                    options=task_modes.run_option(task,requested) if explicit else []
                    p=subprocess.run(['bash',str(ROOT/'tran_evaluate/train_eval.sh'),task,
                        '--front-blocks','3','--cdpa-mode','off',*options,'--dry-run'],env=env,
                        cwd=ROOT,capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stderr)
                    commands=[shlex.split(line.removeprefix('Command:')) for line in p.stdout.splitlines() if line.startswith('Command:')]
                    self.assertEqual(len(commands),2)
                    key='--cdlno-run-dir' if task in task_modes.STANDARD else '--run_dir'
                    resolved=[cmd[len(cmd)-1-cmd[::-1].index(key)+1] for cmd in commands]
                    self.assertEqual(resolved[0],resolved[1])
                    if explicit: self.assertEqual(resolved[0],str(requested))
                    else: self.assertEqual(Path(resolved[0]).parent,Path(temp)/'output'/task)
                    self.assertFalse(list(Path(temp).iterdir()))
                p=subprocess.run(['bash',str(ROOT/'tran_evaluate'/f'{task}.sh'),'eval','--dry-run'],
                                 env=env,cwd=ROOT,capture_output=True,text=True)
                self.assertNotEqual(p.returncode,0)
                self.assertIn('Evaluation needs',p.stderr)

    def test_industrial_real_epoch_observer_preserves_weights_rng_and_original_loss_values(self):
        if not task_modes.car.HAS_PYG:
            self.skipTest('real PyG unavailable')
        import random
        import numpy as np
        from torch_geometric.data import Data
        for task, project in [('car', 'Car-Design-ShapeNetCar'), ('airfrans', 'Airfoil-Design-AirfRANS')]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                spec=importlib.util.spec_from_file_location('output_safe_train_'+task, ROOT/project/'train.py')
                training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
                if task == 'airfrans':
                    if importlib.util.find_spec('torch_cluster') is None:
                        self.skipTest('AirfRANS complete sampled epoch not run: torch_cluster is absent; no dependency was installed')
                    try:
                        training.nng.radius_graph(torch.rand(4,2), r=1.)
                    except ImportError as exc:
                        self.skipTest('real radius_graph dependency unavailable: '+str(exc))
                a=task_modes.parse(task,[*task_modes.SMALL,*task_modes.run_option(task,Path(temp)/'recorded')])
                initial=task_modes.build(task,a)
                graph=Data(x=torch.randn(13,7),pos=torch.randn(13,2),y=torch.randn(13,4),surf=torch.arange(13)%2==0)
                dataset=[(graph,torch.randn(7,3))]*2 if task=='car' else [graph,graph.clone()]
                hparams=dict(lr=.001,batch_size=1,nb_epochs=2,subsampling=13,r=2.,max_neighbors=8)
                outputs=[];rngs=[]
                for observed in (False,True):
                    m=copy.deepcopy(initial)
                    random.seed(812);np.random.seed(812);torch.manual_seed(812)
                    record=start(a,task,hparams=hparams) if observed else None
                    if record: record.attach_model(m,hparams=hparams)
                    path=Path(temp)/('recorded' if observed else 'unobserved');path.mkdir(exist_ok=True)
                    returned=[];original_train=training.train
                    def spy(*args,**kwargs):
                        values=original_train(*args,**kwargs)
                        returned.append(values)
                        return values
                    try:
                        extra=dict(criterion='MSE_weighted',name_mod='CDLNO',val_sample=True) if task=='airfrans' else {}
                        with patch.object(training,'train',side_effect=spy):
                            trained=training.main('cpu',dataset,dataset,m,hparams,str(path),reg=.5,val_iter=10,
                                                  record=record,**extra)
                        outputs.append({k:v.clone() for k,v in trained.state_dict().items()})
                        rngs.append((torch.get_rng_state().clone(),random.getstate(),np.random.get_state()))
                        if record:
                            rows=[json.loads(row) for row in (path/'train_history.jsonl').read_text().splitlines()]
                            self.assertEqual(len(rows),2)
                            for row,values in zip(rows,returned):
                                metrics=row['metrics']
                                if task=='car':
                                    self.assertEqual(metrics['train_components']['pressure_mse'],float(values[0]))
                                    self.assertEqual(metrics['train_components']['velocity_mse'],float(values[1]))
                                else:
                                    self.assertEqual(metrics['train_surface_mse'],float(values[4]))
                                    self.assertEqual(metrics['train_volume_mse'],float(values[5]))
                                    self.assertEqual(metrics['train_loss'],float(.5*values[4]+values[5]))
                            finish(a)
                    finally:
                        if record and not record.closed:record.finish(status='failed')
                    if task=='airfrans':training.plt.close('all')
                for key in outputs[0]:self.assertTrue(torch.equal(outputs[0][key],outputs[1][key]),key)
                self.assertTrue(torch.equal(rngs[0][0],rngs[1][0]))
                self.assertEqual(rngs[0][1],rngs[1][1])
                np.testing.assert_equal(rngs[0][2],rngs[1][2])

    def test_air_weighted_loss_and_real_epoch_record_fragment_without_sampling(self):
        if not task_modes.car.HAS_PYG:self.skipTest('real PyG unavailable')
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        spec=importlib.util.spec_from_file_location('output_air_weighted',ROOT/'Airfoil-Design-AirfRANS/train.py')
        training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            a=task_modes.parse('airfrans',[*task_modes.SMALL,*task_modes.run_option('airfrans',Path(temp)/'run')])
            m=task_modes.build('airfrans',a);r=start(a,'airfrans');r.attach_model(m)
            try:
                graph=Data(x=torch.randn(13,7),pos=torch.randn(13,2),y=torch.randn(13,4),surf=torch.arange(13)%2==0)
                loader=DataLoader([graph],batch_size=1)
                reference=copy.deepcopy(m);pred=reference(graph);per=(pred-graph.y).square()
                expected=per[~graph.surf].mean()+.5*per[graph.surf].mean();expected.backward()
                optim=torch.optim.Adam(m.parameters(),lr=.001)
                sched=torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=.001,total_steps=4)
                values=training.train('cpu',m,loader,optim,sched,criterion='MSE_weighted',reg=.5)
                for p,q in zip(m.parameters(),reference.parameters()):
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
                scope=dict(record=r,record_member=0,epoch=0,train_loss=values[0],loss_surf_var=values[2],
                           loss_vol_var=values[3],loss_surf=values[4],loss_vol=values[5],criterion='MSE_weighted',
                           reg=.5,val_iter=None)
                tree=ast.parse((ROOT/'Airfoil-Design-AirfRANS/train.py').read_text())
                main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
                epoch=next(n for n in main.body if isinstance(n,ast.For))
                adjustment=next(n for n in epoch.body if isinstance(n,ast.If) and ast.unparse(n.test)=="criterion == 'MSE_weighted'")
                callback=next(n for n in epoch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='record is not None')
                exec(compile(ast.Module(body=[adjustment,callback],type_ignores=[]),'<actual-weighted-epoch-record-fragment>','exec'),scope)
                r.finish()
                row=json.loads((r.directory/'train_history.jsonl').read_text())
                self.assertAlmostEqual(row['metrics']['train_loss'],expected.item(),places=6)
                self.assertEqual(row['metrics']['criterion'],'MSE_weighted')
            finally:
                if not r.closed:r.finish(status='failed')

    def test_recording_lifecycle_in_three_original_fresh_workdirs(self):
        code = """
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(sys.argv[1])/'tests'))
import test_front_task_modes as modes
from cdlno.experiment import start, finish
from torch_geometric.data import Data
root=pathlib.Path(sys.argv[2]);task=sys.argv[3];torch.set_num_threads(1)
a=modes.parse(task,[*modes.SMALL,*modes.run_option(task,root/'run')]);rec=start(a,task)
m=modes.build(task,a);run=modes.make_run(task,a,m)
if task=='airfrans':rec.attach_model(m)
x=modes.inputs(task,m);m.eval();expected=m(*x).detach();modes.save(task,run,m);finish(a)
e=modes.parse(task,[*modes.SMALL,*modes.run_option(task,run.directory)],True);rec=start(e,task,evaluation=True)
requested=modes.build(task,e) if task in modes.STANDARD else None
loader=modes.make_run(task,e,requested,True);loaded=modes.load(task,loader,requested).eval()
torch.testing.assert_close(loaded(*x),expected,atol=0,rtol=0);rec.record_metrics(dict(exact=True));finish(e)
"""
        for project,task in [('PDE-Solving-StandardBenchmark','elasticity'),
                             ('Car-Design-ShapeNetCar','car'),('Airfoil-Design-AirfRANS','airfrans')]:
            with self.subTest(project=project),tempfile.TemporaryDirectory() as temp:
                result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),temp,task],
                    cwd=ROOT/project,env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_old_runs_without_config_keep_loadable(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path=Path(temp)/'runs/CDLNO/elasticity/old'
            a=task_modes.parse('elasticity',[*task_modes.SMALL,*task_modes.run_option('elasticity',path)])
            m=task_modes.build('elasticity',a);run=task_modes.make_run('elasticity',a,m);run.save(m)
            self.assertFalse((path/'config.json').exists())
            before=run.sidecar.read_bytes()
            ev=task_modes.parse('elasticity',[*task_modes.SMALL,*task_modes.run_option('elasticity',path)],True)
            r=start(ev,'elasticity',evaluation=True)
            try:
                load=task_modes.make_run('elasticity',ev,m,True);load.load(m);r.record_metrics(dict(synthetic=True));finish(ev)
            finally:
                if not r.closed:r.finish(status='failed')
            self.assertFalse((path/'config.json').exists())
            self.assertEqual(before,run.sidecar.read_bytes())


if __name__=='__main__': unittest.main()
