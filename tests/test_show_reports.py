"""Offline report tests using explicitly synthetic saved-result fixtures, no CFD data."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saved_result_report',ROOT/'tran_evaluate/show/_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def fixture(root,task,model='kcdno',seed=0):
    path=root/task/model/f'20260101_seed{seed}';path.mkdir(parents=True)
    args={'seed':seed,'weight':.5,'fold_id':0,'task':'full'}
    config=dict(task=task,model=model,resolved_arguments=args,parameters={'total':1234,'trainable':1234},architecture={'family':model,'history_mode':'all'})
    (path/'config.json').write_text(json.dumps(config))
    (path/'train_results.json').write_text(json.dumps({'status':'completed','metrics':{}}))
    history=[]
    for epoch in (1,10,50):
        if task=='car':m={'train_components':{'pressure_mse':.2/epoch,'velocity_mse':.4/epoch},'validation_pressure_mse':.3/epoch,'validation_velocity_mse':.5/epoch}
        elif task=='airfrans':
            m={'train_loss':.5/epoch,'train_surface_mse':.2/epoch,'train_volume_mse':.3/epoch,'train_surface_per_channel':[.2/epoch]*4,'train_volume_per_channel':[.3/epoch]*4,'criterion':'MSE_weighted'}
            if epoch in (1,50):m.update(validation_surface_mse=.25/epoch,validation_volume_mse=.35/epoch)
        elif task in ('ns','plasticity'):m={'train_step_loss':.3/epoch,'test_step_loss':.4/epoch,'test_full_loss':.45/epoch}
        else:m={'train_loss':.2/epoch,'validation_relative_l2':.25/epoch,'derivative_regularizer':.4/epoch}
        history.append({'epoch':epoch,'member':0,'metrics':m})
    (path/'train_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    for e,status in [('first','completed'),('second','failed')]:
        d=path/'evaluations'/e;d.mkdir(parents=True)
        metrics={'relative_l2':.12} if task not in ('ns','plasticity','car','airfrans') else {'test_full_loss':.15}
        if task=='car':metrics={'rho_d':.9,'c_d':.04,'relative_l2_pressure':.2,'relative_l2_velocity':.3,'rmse_pressure':.1,'rmse_velocity':.2}
        if task=='airfrans':
            score={'mean_score_vol':[[.1,.2,.3,.4]],'std_score_vol':[[.01]*4],
                   'mean_score_surf':[[.11,.21,.31,.41]],'mean_score_force':[[.04,.05]],'std_score_force':[[.001,.002]],'spearman_coef_mean':[[.91,.92]]}
            (d/'score.json').write_text(json.dumps(score));metrics={'score':score}
            np.save(d/'true_coefs.npy',np.array([[.1,.5],[.2,.7],[.3,.8]]))
            np.save(d/'pred_coefs_mean.npy',np.array([[[.11,.52]],[[.21,.69]],[[.28,.82]]]))
            np.save(d/'pred_coefs_std.npy',np.ones((3,1,2))*.003)
            surf=np.stack((np.c_[np.linspace(0,1,5),np.linspace(-1,1,5)],np.c_[np.linspace(0,1,5),np.linspace(0,.1,5)]))
            np.save(d/'true_surf_coefs_0.npy',surf);np.save(d/'surf_coefs_0.npy',surf[None]*1.05)
        (d/'results.json').write_text(json.dumps({'status':status,'metrics':metrics}))
    return path,config


def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def set_history(run,rows):
    (run/'train_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


class Reports(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='report fixture ');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.repo=self.root/'remote checkout'
        for project in report.PROJECTS:(self.repo/project).mkdir(parents=True)
        target=self.repo/'tran_evaluate/show';target.parent.mkdir();shutil.copytree(ROOT/'tran_evaluate/show',target,ignore=shutil.ignore_patterns('__pycache__'))
        self.args=report.parser().parse_args(['PDE-Solving-StandardBenchmark','--fields','none','--dpi','300'])

    def test_three_remote_launchers_actual_paths_and_empty_zip(self):
        for project in report.PROJECTS:
            script=self.repo/'tran_evaluate/show'/(project+'.sh')
            subprocess.run(['bash','-n',str(script)],check=True)
            env=dict(os.environ,CDLNO_REPO_ROOT='/wrong/old/checkout',CDLNO_PYTHON=sys.executable)
            env.pop('CDLNO_RUNS_ROOT',None)
            p=subprocess.run(['bash',str(script)],cwd=self.root,env=env,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            bundles=list((self.repo/project/'result_visualizations').glob('*.zip'));self.assertEqual(len(bundles),1)
            with zipfile.ZipFile(bundles[0]) as z:self.assertTrue(any(n.endswith('/index.html') for n in z.namelist()))
            self.assertNotIn('/wrong/old/checkout',p.stdout)
            self.assertIn('Runs: 0',p.stdout)

    def test_eight_tasks_read_only_metrics_epochs_eval_attempts(self):
        paths={t:fixture(self.repo/'output',t)[0] for tasks in report.PROJECTS.values() for t in tasks}
        before=hashes(self.repo/'output')
        with patch.object(report.Bundle,'save_figure',side_effect=lambda fig,*a,**k:report.plt.close(fig)),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            for project,tasks in report.PROJECTS.items():
                dest=self.root/('bundle_'+project)
                report.main([project,'--repo-root',str(self.repo),'--fields','none','--output-dir',str(dest)])
                manifest=json.loads((dest/'manifest.json').read_text());self.assertEqual({r['task'] for r in manifest['runs']},set(tasks))
                rows=list(csv.DictReader(io.StringIO((dest/'evaluation_metrics.csv').read_text(encoding='utf-8-sig'))))
                self.assertEqual({r['evaluation'] for r in rows},{'first','second'})
                self.assertEqual({r['status'] for r in rows},{'completed','failed'})
                self.assertTrue(all(r['seed']=='0' for r in rows))
                hist=list(csv.DictReader(io.StringIO((dest/'training_history.csv').read_text(encoding='utf-8-sig'))))
                self.assertEqual({r['epoch'] for r in hist},{'1','10','50'})
        self.assertEqual(hashes(self.repo/'output'),before)
        self.assertNotIn('torch',report.__dict__)

    def test_sparse_validation_and_member_no_blending(self):
        run,config=fixture(self.repo/'output','airfrans');out=self.root/'plots';out.mkdir()
        b=report.Bundle(self.args,out)
        info=dict(id='r',model='kcdno',seed=0,label='synthetic')
        seen=[]
        with patch.object(b,'curve',side_effect=lambda path,series,*a:seen.append((path,series))):b.training(run,out,'airfrans',info,config)
        surface=next(series for path,series in seen if path.name=='surface_mse')
        self.assertEqual(surface[0][1],[1,10,50]);self.assertEqual(surface[1][1],[1,50])
        self.assertEqual(surface[1][2],[.25,.005])

    def test_PDE_old_log_fallback_partial_jsonl_and_duplicates(self):
        run=self.root/'darcy'/'old';run.mkdir(parents=True)
        (run/'train.log').write_text('Epoch 0 Reg : 3.00000 Train loss : 1.00000\nrel_err:2.0\nEpoch 49 Reg : .30000 Train loss : .10000\nrel_err:.2\n')
        b=report.Bundle(self.args,self.root/'b');rows=b.history(run,'darcy')
        self.assertEqual([r['epoch'] for r in rows],[1,50]);self.assertEqual(rows[-1]['metrics']['validation_relative_l2'],.2)
        (run/'train_history.jsonl').write_text(json.dumps({'epoch':50,'metrics':{'train_loss':.123456789}})+'\n{"epoch":')
        rows=b.history(run,'darcy');self.assertEqual(rows[0]['metrics']['train_loss'],.123456789)
        self.assertTrue(any('incomplete' in w for w in b.warnings))

    def test_source_family_seed_filters_no_new_output_on_preview_or_conflict(self):
        a,_=fixture(self.repo/'output','darcy',seed=1);fixture(self.repo/'output','car')
        self.assertEqual(report.infer_task(a,{}, {},('car',)),'darcy')
        out=self.root/'preview'
        with redirect_stdout(io.StringIO()):report.main(['PDE-Solving-StandardBenchmark','--repo-root',str(self.repo),'--seed','0','--output-dir',str(out),'--dry-run'])
        self.assertFalse(out.exists())
        out.mkdir()
        with self.assertRaises(FileExistsError):report.main(['PDE-Solving-StandardBenchmark','--repo-root',str(self.repo),'--output-dir',str(out)])
        with self.assertRaises(ValueError):report.main(['PDE-Solving-StandardBenchmark','--repo-root',str(self.repo),'--run-dir',str(a),'--output-dir',str(a/'new_report')])

    def test_air_array_axes_safe_loading_and_no_reversed_coefficients(self):
        run,_=fixture(self.repo/'output','airfrans');out=self.root/'plots';out.mkdir();b=report.Bundle(self.args,out)
        with patch.object(b,'save_figure',side_effect=lambda fig,*a,**k:report.plt.close(fig)):
            b.air_coefficients(run/'evaluations/first',out,'synthetic')
        rows=list(csv.DictReader(io.StringIO((out/'coefficients.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual(rows[0]['channel'],'drag');self.assertEqual(float(rows[0]['reference']),.1)
        self.assertEqual(float(rows[3]['reference']),.5)
        p=self.root/'object.npy';np.save(p,np.array([{'not':'numeric'}],dtype=object))
        with self.assertRaises(ValueError):b.load_array(p)
        report.write_json(out/'nonfinite.json',{'value':float('nan')});self.assertEqual(json.loads((out/'nonfinite.json').read_text())['value'],'nan')

    def test_real_publication_exports_and_self_contained_archive(self):
        fixture(self.repo/'output','darcy');out=self.root/'final bundle'
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            report.main(['PDE-Solving-StandardBenchmark','--repo-root',str(self.repo),'--output-dir',str(out),'--fields','none','--dpi','300'])
        pdfs=list(out.rglob('*.pdf'));self.assertGreaterEqual(len(pdfs),3)
        data=pdfs[0].read_bytes();self.assertTrue(data.startswith(b'%PDF'));self.assertIn(b'/FontFile2',data)
        from PIL import Image
        with Image.open(next(out.rglob('*.png'))) as image:self.assertGreater(image.width,900)
        self.assertTrue(any(p.suffix=='.tex' for p in out.rglob('*')))
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith(('.pt','.pth')) for n in z.namelist()))
        self.assertIn('seed 0',(out/'index.html').read_text())

    def test_existing_fields_are_copied_with_metadata_not_regenerated(self):
        run,_=fixture(self.repo/'output','darcy');args=report.parser().parse_args(['PDE-Solving-StandardBenchmark'])
        for epoch in ('0050','0100'):
            d=run/f'visualizations/member_000/epoch_{epoch}/case_000/field';d.mkdir(parents=True)
            (d/'caption.txt').write_text('SYNTHETIC TEST PLACEHOLDER, not a CFD field')
            (d/'metadata.json').write_text('{}');np.savez(d/'fields.npz',truth=np.ones((2,1)))
        dest=self.root/'copied';b=report.Bundle(args,dest);b.fields(run,dest)
        self.assertTrue((dest/'field_snapshots/member_000/epoch_0100/case_000/field/caption.txt').exists())
        self.assertFalse(list(dest.rglob('*.npz')));self.assertFalse(list(dest.rglob('epoch_0050')))

    def test_final_validation_winner_per_PDE_task_six_panels_only_selected_assets(self):
        tasks=report.PROJECTS[self.args.project];winners={};runs={}
        for i,task in enumerate(tasks):
            for seed in range(3):
                run,_=fixture(self.repo/'output',task,seed=seed);runs[task,seed]=run
                key=report.VALIDATION_KEYS[task][0]
                final=.02 if seed==i%3 else .1+seed*.1
                # The seed with the best earlier loss deliberately loses on its final epoch.
                set_history(run,[{'epoch':1,'metrics':{key:.8 if seed==i%3 else .00001,'train_loss':.7}},
                                 {'epoch':500,'metrics':{key:final,'train_loss':.03,'train_full_loss':.04,'train_step_loss':.03}}])
                field=run/'visualizations/member_000/epoch_0500/case_000';field.mkdir(parents=True)
                (field/'caption.txt').write_text(f'Synthetic saved field placeholder for {task}, seed {seed}')
                # Selection must not depend on independent test scores or checkpoint content.
                (run/'evaluations/first/results.json').write_text(json.dumps({'status':'completed','metrics':
                            {key if task in ('ns','plasticity') else 'relative_l2':9 if seed==i%3 else .000001}}))
                (run/'model.pt').write_bytes(b'opaque test placeholder: must never be loaded or copied')
            winners[task]=i%3
        prior=self.repo/self.args.project/'result_visualizations/prior_report';prior.mkdir(parents=True)
        (prior/'index.html').write_text('EXISTING REPORT MUST REMAIN IDENTICAL')
        (prior.parent/'prior_report.zip').write_bytes(b'existing archive placeholder')
        before=hashes(self.repo);out=self.root/'selected';figures={}
        def capture(fig,path,caption):
            figures[path.name]={'titles':[a.get_title() for a in fig.axes],'caption':caption,
                                'lines':[[list(line.get_ydata()) for line in a.lines] for a in fig.axes]}
            report.plt.close(fig)
        with patch.object(report.Bundle,'save_figure',side_effect=capture),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            report.main([self.args.project,'--repo-root',str(self.repo),'--output-dir',str(out)])
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertEqual({r['task']:r['seed'] for r in manifest['runs']},winners)
        self.assertEqual(len(manifest['selection']),18);self.assertEqual(sum(r['selected'] for r in manifest['selection']),6)
        for name in ('pde_selected_training','pde_selected_evaluation'):
            self.assertEqual(len(figures[name]['titles']),6)
            for task,title in zip(tasks,figures[name]['titles']):self.assertIn(f'seed {winners[task]}',title)
        self.assertIn([.8,.02],figures['pde_selected_training']['lines'][0])
        for fname in ('runs.csv','training_history.csv','evaluation_metrics.csv'):
            rows=list(csv.DictReader(io.StringIO((out/fname).read_text(encoding='utf-8-sig'))))
            self.assertTrue(rows)
            for row in rows:self.assertEqual(int(row['seed']),winners[row['task']])
        for task,seed in winners.items():
            dirs=list((out/task).iterdir());self.assertEqual(len(dirs),1)
            self.assertIn(f'seed{seed}',dirs[0].name)
            self.assertIn(f'seed {seed}',next(dirs[0].rglob('caption.txt')).read_text())
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            self.assertIsNone(z.testzip())
            for task in tasks:
                names=[n for n in z.namelist() if '/'+task+'/' in n]
                self.assertTrue(names)
                self.assertFalse(any(f'seed{s}_' in n for n in names for s in range(3) if s!=winners[task]))
        self.assertEqual(before,hashes(self.repo))

    def test_reject_invalid_final_validation_and_unfinished_without_earlier_fallback(self):
        cases=[('completed',.05),('completed',float('nan')),('failed',.001),('running',.0001),
               ('completed',None),('interrupted',.00001),('incomplete',.000001)]
        candidates=[]
        for seed,(status,last) in enumerate(cases):
            run,cfg=fixture(self.repo/'output','darcy',seed=seed)
            (run/'train_results.json').write_text(json.dumps({'status':status}))
            set_history(run,[{'epoch':1,'metrics':{'validation_relative_l2':.0000001}},
                             {'epoch':500,'metrics':{'validation_relative_l2':last}}])
            candidates.append((run,'darcy',cfg,{}))
        b=report.Bundle(self.args,self.root/'b')
        with redirect_stderr(io.StringIO()):selected=b.select_runs(candidates)
        self.assertEqual([item[2]['resolved_arguments']['seed'] for item in selected],[0])
        self.assertEqual(b.selection[1]['validation_epochs'],{'0':500});self.assertFalse(b.selection[1]['eligible'])
        self.assertIn('nonfinite',b.selection[1]['reason'])
        self.assertFalse(b.selection[4]['eligible'])

    def test_corrupt_history_and_legacy_nonfinite_final_are_not_silently_selected(self):
        run,cfg=fixture(self.repo/'output','darcy')
        with (run/'train_history.jsonl').open('a') as stream:stream.write('{"epoch":500,"metrics":')
        with redirect_stderr(io.StringIO()):
            record=report.Bundle(self.args,self.root/'b').selection_record(run,'darcy',cfg,{})
        self.assertFalse(record['eligible']);self.assertIn('final validation cannot be confirmed',record['reason'])
        (run/'train_history.jsonl').unlink()
        (run/'train.log').write_text('Epoch 0 Train loss : .1\nrel_err:.001\nEpoch 499 Train loss : .1\nrel_err:nan\n')
        with redirect_stderr(io.StringIO()):
            record=report.Bundle(self.args,self.root/'c').selection_record(run,'darcy',cfg,{})
        self.assertFalse(record['eligible']);self.assertEqual(record['validation_epochs'],{'0':500})

    def test_car_selection_uses_correct_saved_components_and_weight(self):
        candidates=[]
        for seed,(vel,pressure) in enumerate(((.1,.4),(.3,.1),(.001,.001))):
            run,cfg=fixture(self.repo/'output','car',seed=seed)
            if seed==2:del cfg['resolved_arguments']['weight']
            set_history(run,[{'epoch':200,'metrics':{'validation_velocity_mse':vel,'validation_pressure_mse':pressure,
                                                   'upstream_validation_log_value':pressure+.5*vel}}])
            candidates.append((run,'car',cfg,{}))
        args=report.parser().parse_args(['Car-Design-ShapeNetCar']);b=report.Bundle(args,self.root/'b')
        with redirect_stderr(io.StringIO()):selected=b.select_runs(candidates)
        self.assertEqual(selected[0][2]['resolved_arguments']['seed'],0)
        self.assertAlmostEqual(b.selection[0]['final_validation_loss'],.3)
        self.assertIn('weight/reg',b.selection[2]['reason'])
        # Do not backtrack to an earlier complete validation if the last is missing a component.
        run,task,cfg,arch=candidates[0]
        set_history(run,[{'epoch':190,'metrics':{'validation_velocity_mse':.001,'validation_pressure_mse':.001}},
                         {'epoch':200,'metrics':{'validation_velocity_mse':.001}}])
        self.assertFalse(report.Bundle(args,self.root/'new').selection_record(run,task,cfg,arch)['eligible'])

    def test_air_sparse_final_validation_multiple_members_and_missing_member(self):
        candidates=[]
        for seed,volumes in enumerate(((.01,4.0),(.6,.6),(.001,))):
            run,cfg=fixture(self.repo/'output','airfrans',seed=seed)
            cfg['resolved_arguments']['nmodel']=2
            rows=[]
            for member,vol in enumerate(volumes):
                rows.extend([{'epoch':390,'member':member,'metrics':{'validation_volume_mse':vol,'validation_surface_mse':.2,
                                                                    'reg':.5,'criterion':'MSE_weighted','upstream_validation_log_value':99}},
                             {'epoch':398,'member':member,'metrics':{'train_loss':.03,'reg':.5,'criterion':'MSE_weighted'}}])
            set_history(run,rows);candidates.append((run,'airfrans',cfg,{}))
        args=report.parser().parse_args(['Airfoil-Design-AirfRANS']);b=report.Bundle(args,self.root/'b')
        with redirect_stderr(io.StringIO()):selected=b.select_runs(candidates)
        self.assertEqual(selected[0][2]['resolved_arguments']['seed'],1)
        self.assertAlmostEqual(b.selection[1]['final_validation_loss'],.7)
        self.assertEqual(b.selection[1]['validation_epochs'],{'0':390,'1':390})
        self.assertEqual(b.selection[1]['last_training_epochs'],{'0':398,'1':398})
        self.assertFalse(b.selection[2]['eligible']);self.assertIn('member 1',b.selection[2]['reason'])
        with self.assertRaisesRegex(ValueError,'disagree'):
            report.validation_value('airfrans',dict(validation_volume_mse=.1,validation_surface_mse=.1,reg=.1),candidates[0][2])

    def test_ties_legacy_missing_history_and_dry_run_selection(self):
        candidates=[]
        for seed in (2,1,0):
            run,cfg=fixture(self.repo/'output','darcy',seed=seed);candidates.append((run,'darcy',cfg,{}))
        legacy=candidates[-1][0];(legacy/'train_results.json').unlink()
        b=report.Bundle(self.args,self.root/'b')
        with redirect_stderr(io.StringIO()):selected=b.select_runs(candidates)
        self.assertEqual(selected[0][0],legacy);self.assertTrue(any('completion unverified' in w for w in b.warnings))
        self.assertEqual(sum(r['selected'] for r in b.selection),1)
        self.assertTrue(all('tie-break' in r['reason'] for r in b.selection if not r['selected']))
        # Seed/path tie is deterministic even if input discovery order changes.
        copy=legacy.with_name('000_copy');shutil.copytree(legacy,copy)
        with redirect_stderr(io.StringIO()):
            tied=report.Bundle(self.args,self.root/'c').select_runs(candidates+[(copy,'darcy',candidates[-1][2],{})])
        self.assertEqual(tied[0][0],copy)
        before=hashes(self.repo);out=self.root/'dry';stdout=io.StringIO()
        with redirect_stdout(stdout),redirect_stderr(io.StringIO()):
            report.main([self.args.project,'--repo-root',str(self.repo),'--output-dir',str(out),'--dry-run'])
        self.assertIn('SELECTED darcy seed=0',stdout.getvalue());self.assertIn('EXCLUDED darcy seed=2',stdout.getvalue())
        self.assertFalse(out.exists());self.assertFalse(out.with_suffix('.zip').exists());self.assertEqual(before,hashes(self.repo))
        (legacy/'train_history.jsonl').unlink()
        self.assertFalse(report.Bundle(self.args,self.root/'d').selection_record(legacy,'darcy',candidates[-1][2],{})['eligible'])


if __name__=='__main__':unittest.main(verbosity=2)
