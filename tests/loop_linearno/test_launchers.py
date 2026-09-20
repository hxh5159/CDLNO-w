"""LL8 real argv previews, paired initialization and RNG-neutral records."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch_geometric.data import Data

from tran_evaluate.linearno_loop import launch,recording
from linearno_loop.config import resolve_config
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
from cdlno.linearno_loop.construction import build_from_config

ROOT=Path(__file__).resolve().parents[2]


def small_config(task,preset,mode,seed):
    overrides={'model.hidden':8,'model.heads':2,'runtime.seed':seed}
    if task not in ('car','airfrans'):overrides.update({'model.H':3,'model.W':5})
    if task=='airfrans':overrides['model.ref']=3
    return resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode,linearno_rank=4),profile_overrides=overrides)


def example(task):
    if task in ('car','airfrans'):
        data=Data(x=torch.randn(15,7),pos=torch.randn(15,2 if task=='airfrans' else 3))
        return (data,) if task=='airfrans' else ((data,torch.randn(3,3)),)
    fx={'airfoil':None,'elasticity':None,'pipe':None,'darcy':1,'ns':10,'plasticity':1}[task]
    return (torch.randn(2,15,2),None if fx is None else torch.randn(2,15,fx),
            torch.randn(2,1) if task=='plasticity' else None)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_entry_bridge_delegates_native_argv_without_data_import(self):
        from tran_evaluate.linearno_loop import entry
        for task in launch.TASKS:
            for action in ('train','resume','eval'):
                filename=launch.ENTRIES.get(task) or ('main_evaluation.py' if action=='eval' else 'main.py')
                with patch.object(sys,'argv',['entry.py',task,action,filename,'--experiment-dir','/tmp/run with spaces']),\
                     patch.object(Path,'cwd',return_value=launch.project(task)),\
                     patch.object(recording,'install') as install,patch.object(entry.runpy,'run_path') as run:
                    previous=list(sys.path)
                    try:entry.main()
                    finally:sys.path[:]=previous
                    install.assert_called_once_with();run.assert_called_once_with(filename,run_name='__main__')
                    self.assertEqual(sys.argv,[filename,'--experiment-dir','/tmp/run with spaces'])
        with patch.object(sys,'argv',['entry.py','darcy','train','wrong.py']),self.assertRaises(ValueError):entry.main()

    def test_paired_three_seeds_all_tasks_backbone_and_generators(self):
        rows=[]
        with tempfile.TemporaryDirectory(prefix='loop-ll8-fair-') as tmp:
            for task in launch.TASKS:
                for preset in PRESETS:
                    for seed in (0,1,2):
                        common=None;generator_states=None;data_order=None
                        for mode in RESIDUAL_MODES:
                            cfg=small_config(task,preset,mode,seed);model=build_from_config(cfg)
                            directory=Path(tmp)/task/preset/str(seed)/mode;directory.mkdir(parents=True)
                            args=SimpleNamespace(_linearno_loop_config=cfg,linearno_run_dir=directory,
                                seed=seed,linearno_task=task,eval=False,resume=False)
                            before=torch.get_rng_state().clone();recording.observe(args,model)
                            self.assertTrue(torch.equal(before,torch.get_rng_state()))
                            manifest=json.loads((directory/recording.MANIFEST).read_text());member=manifest['members']['member_000']
                            h=member['public_backbone_initial_sha256']
                            generators={k:torch.Generator().manual_seed(v) for k,v in cfg['fair_comparison']['dataloader_generators'].items()}
                            order=[batch.tolist() for batch in torch.utils.data.DataLoader(torch.arange(23),batch_size=4,
                                shuffle=True,generator=generators['train'])]
                            gen={k:g.get_state().tolist() for k,g in generators.items()}
                            if common is not None:
                                self.assertEqual(h,common);self.assertEqual(gen,generator_states);self.assertEqual(order,data_order)
                            else:common=h;generator_states=gen;data_order=order
                            model.eval();out=model(*example(task));out.square().mean().backward()
                            observed=json.loads((directory/recording.MANIFEST).read_text())['members']['member_000']
                            self.assertEqual(observed['actual_call_schedule'],manifest['expected_call_schedule'])
                            self.assertEqual(observed['observation'],'first_forward_verified')
                            self.assertFalse(any(mod._forward_hooks or mod._forward_pre_hooks for mod in model.modules()))
                            rows.append(dict(task=task,preset=preset,mode=mode,seed=seed,backbone_hash=h,
                                data_order=order,unique_depth=manifest['unique_depth'],executed_depth=manifest['executed_depth'],
                                parameters=member['parameters'],router_parameters=member['router_parameters'],
                                actual_schedule=observed['actual_call_schedule']))
        if name:=os.environ.get('LOOP_LL8_FAIR_REPORT'):Path(name).write_text(json.dumps(rows,indent=2)+'\n')

    def test_observation_preserves_train_dropout_weights_rng_and_errors(self):
        with tempfile.TemporaryDirectory(prefix='loop-ll8-observer-') as tmp:
            for mode in RESIDUAL_MODES:
                cfg=small_config('darcy','p1_c3_r2_s1',mode,7)
                cfg=resolve_config('darcy',options=cfg['request']['options'],
                    profile_overrides={**cfg['request']['profile_overrides'],'model.dropout':.2})
                raw=build_from_config(cfg);seen=copy.deepcopy(raw)
                directory=Path(tmp)/mode;directory.mkdir()
                args=SimpleNamespace(_linearno_loop_config=cfg,linearno_run_dir=directory,seed=7,eval=False,resume=False)
                inputs=example('darcy');rng=torch.get_rng_state().clone()
                def step(model):
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
                    output=model(*inputs);output.square().mean().backward();optimizer.step()
                    return output,optimizer.state_dict(),torch.get_rng_state().clone()
                plain,opt,after=step(raw)
                torch.set_rng_state(rng);recording.observe(args,seen)
                with self.assertRaises(ValueError):seen(torch.randn(2,14,2),torch.randn(2,14,1))
                torch.set_rng_state(rng)
                observed,opt2,after2=step(seen)
                torch.testing.assert_close(plain,observed,atol=0,rtol=0)
                self.assertTrue(torch.equal(after,after2))
                for k,v in raw.state_dict().items():torch.testing.assert_close(v,seen.state_dict()[k],atol=0,rtol=0)
                for k,values in opt['state'].items():
                    for name,v in values.items():torch.testing.assert_close(v,opt2['state'][k][name],atol=0,rtol=0)
                frozen=(directory/recording.MANIFEST).read_bytes();args.resume=True
                recording.observe(args,build_from_config(cfg));self.assertEqual(frozen,(directory/recording.MANIFEST).read_bytes())

    def test_native_factory_record_installation_and_independent_air_members(self):
        from cdlno.linearno_loop import standard_entry,industrial_state
        originals=(standard_entry.model_module,industrial_state.construct)
        aliases={name:sys.modules[name].construct for name in ('cdlno.linearno_loop.air_entry','cdlno.linearno_loop.car_entry') if name in sys.modules}
        with tempfile.TemporaryDirectory(prefix='loop-ll8-installed-') as tmp:
            try:
                recording.install()
                for task in ('darcy','airfrans','car'):
                    cfg=small_config(task,'p1_c3_r2_s1','rb_attnres',17)
                    directory=Path(tmp)/task;directory.mkdir()
                    args=SimpleNamespace(_linearno_loop_config=cfg,linearno_run_dir=directory,seed=17,linearno_task=task,eval=False,resume=False)
                    if task=='darcy':m=standard_entry.model_module(args).Model(**cfg['model_spec']['constructor_kwargs'])
                    else:m=industrial_state.construct(args)
                    m(*example(task))
                    if task=='airfrans':
                        n=industrial_state.construct(args,1);n(*example(task))
                        self.assertIsNot(m.loop,n.loop)
                    saved=json.loads((directory/recording.MANIFEST).read_text())
                    self.assertEqual(len(saved['members']),2 if task=='airfrans' else 1)
                    self.assertTrue(all(r['actual_call_schedule'] for r in saved['members'].values()))
            finally:
                standard_entry.model_module,industrial_state.construct=originals
                for name,fn in aliases.items():sys.modules[name].construct=fn

    def test_gpu_mapping_shell_syntax_and_no_legacy_edits(self):
        for path in (ROOT/'tran_evaluate/linearno_loop').glob('*.sh'):
            self.assertEqual(subprocess.run(['bash','-n',str(path)]).returncode,0)
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        for task in ('airfoil','car','airfrans'):
            tokens=['--linearno-loop','1','--linearno-loop-topology','p2_c2_r2_s2',
                    '--linearno-loop-residual-mode','rb_attnres','--gpu','1']
            value=launch.plan(task,'train',tokens,{**env,'CUDA_VISIBLE_DEVICES':'4,7'})
            self.assertEqual(value['environment']['CUDA_VISIBLE_DEVICES'],'7')
            self.assertEqual(value['argv'][value['argv'].index('--gpu')+1],'7' if task=='airfoil' else '0')
            self.assertFalse(Path(value['run']).exists())
        with self.assertRaises(ValueError):launch.gpu_environment(['--gpu','2'],{'CUDA_VISIBLE_DEVICES':'4,7'})
        baseline=json.loads((ROOT/'docs/loop_linearno_audit/ll8/start-manifest.json').read_text())
        for row in baseline['files']:
            if row['path'].startswith(('cdlno/','linearno_loop/','tran_evaluate/linearno/','tran_evaluate/linearno_history/','monitor/')) or row['path']=='path.sh':
                from frozen_revisions import expected_hash
                from cdlno.linearno_loop.ll9r_projection import project, REPLACEMENTS
                raw = (ROOT/row['path']).read_bytes()
                if row['path'] in REPLACEMENTS:
                    raw = project(row['path'], raw.decode()).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                 expected_hash(row['path'],row['sha256']),row['path'])

    def test_shell_train_then_eval_spaces_and_failure_propagation(self):
        matrix=json.loads((ROOT/'docs/loop_linearno_audit/ll7/native-matrix.json').read_text())
        row=next(r for r in matrix['rows'] if r['task']=='darcy' and r['preset']=='p1_c3_r2_s1' and r['mode']=='sr_1_over_r')
        source=next((Path(row['artifact'])/'split').iterdir())
        config=json.loads((source/'architecture.json').read_text())['resolved_config']
        from linearno_loop.config import run_directory_id
        with tempfile.TemporaryDirectory(prefix='loop ll8 shell ') as tmp:
            temp=Path(tmp);fake=temp/'fake python';log=temp/'calls.jsonl'
            fake.write_text('#!'+sys.executable+'\n'+'''import json,os,sys,shutil
from pathlib import Path
args=sys.argv[1:]
if any(a.endswith('/launch.py') for a in args):os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON'],*args])
i=next(i for i,a in enumerate(args) if a.endswith('/entry.py'))
task,action,entry=args[i+1:i+4];tokens=args[i+4:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(dict(action=action,argv=tokens,visible=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\\n')
code=int(os.environ.get('TRAIN_EXIT','0') if action=='train' else os.environ.get('EVAL_EXIT','0'))
if action=='train' and code==0:
 run=tokens[tokens.index('--experiment-dir')+1];shutil.copytree(os.environ['SOURCE_RUN'],run)
sys.exit(code)
''');fake.chmod(0o755)
            flags=['--linearno-loop','1','--linearno-loop-topology','p1_c3_r2_s1',
                   '--linearno-loop-residual-mode','sr_1_over_r','--linearno-rank','4',
                   '--n-hidden','8','--n-heads','2','--epochs','3','--batch-size','2','--seed','17','--dropout','.1',
                   '--data_path',str(temp/'data with spaces'),'--gpu','1','--then-eval']
            for train_exit,eval_exit,expected_actions in ((17,0,['train']),(0,0,['train','eval']),(0,23,['train','eval'])):
                target=temp/str(train_exit)/str(eval_exit)/run_directory_id(config)
                env=dict(os.environ,CDLNO_PYTHON=str(fake),REAL_PYTHON=sys.executable,CALLS=str(log),SOURCE_RUN=str(source),
                         TRAIN_EXIT=str(train_exit),EVAL_EXIT=str(eval_exit),CUDA_VISIBLE_DEVICES='4,7',PYTHONDONTWRITEBYTECODE='1')
                before=len(log.read_text().splitlines()) if log.exists() else 0
                p=subprocess.run(['bash',str(ROOT/'tran_evaluate/linearno_loop/darcy.sh'),'train',*flags,'--experiment-dir',str(target)],env=env,cwd=temp,capture_output=True,text=True)
                self.assertEqual(p.returncode,train_exit or eval_exit,p.stdout+p.stderr)
                calls=[json.loads(l) for l in log.read_text().splitlines()[before:]]
                self.assertEqual([r['action'] for r in calls],expected_actions)
                for r in calls:
                    self.assertEqual(r['visible'],'7');self.assertIn(str(temp/'data with spaces'),r['argv']);self.assertIn(str(target),r['argv'])


if __name__=='__main__':unittest.main()
