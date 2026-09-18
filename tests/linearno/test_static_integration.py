"""L4 real entry contracts, strict native resume/eval, profiles and preservation."""
import argparse
import ast
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from linearno.static_worker import ROOT, PROJECT, parser_for, SHAPES, SHORT
from cdlno_entry import parse_args
from model_dict import get_model
from cdlno.linearno.standard_entry import model_kwargs, constructor_kwargs, _structural_check, verify_data
from cdlno.linearno.checkpoint import inspect_checkpoint, read_pair, strict_load
from cdlno.linearno.schema import read_metadata, unpack_state
from cdlno.linearno.profiles import validate_resolved, resolve_config
from cdlno.training_state import _same
from linearno_entry_projection import strip_linearno
from utils.testloss import TestLoss


def args_for(task, *tokens):
    key='LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D'
    return parse_args(parser_for(task),task,['--model',key,*map(str,tokens)])


class StaticIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        if path:=os.environ.get('LINEARNO_L4_INTEGRATION_REPORT'):
            Path(path).write_text(json.dumps(cls.rows,indent=2)+'\n')

    def test_profiles_factory_exact_config_override_and_legacy_isolation(self):
        for task in SHORT:
            for profile in ('paper_table8_on_release_model','official_release'):
                a=args_for(task,'--linearno-profile',profile)
                self.assertEqual((a.n_hidden,a.n_layers,a.n_heads,a.linearno_rank),(128,8,8,64))
                self.assertEqual(a.linearno_variant,'temp' if task=='elasticity' else 'conv_temp')
                self.assertEqual(a.batch_size,1 if task=='elasticity' else 4)
                self.assertEqual(a.mlp_ratio,1)
                self.assertEqual(get_model(a).Model.__module__,'model.LinearNO')
                self.assertNotIn('slice_num',model_kwargs(a))
                self.assertFalse(a.unified_pos)
                validate_resolved(a._linearno_config)
                explicit=args_for(task,'--n-hidden','64','--n-heads','4','--linearno-rank','32','--save_name','user_run')
                self.assertEqual((explicit.n_hidden,explicit.n_heads,explicit.linearno_rank,explicit.save_name),(64,4,32,'user_run'))
                self.assertEqual(explicit._linearno_config['field_sources']['model.hidden'],'cli_explicit')
                old=parse_args(parser_for(task),task,[])
                self.assertEqual((old.n_hidden,old.n_layers,old.slice_num),(64,3,32))
                self.assertFalse(hasattr(old,'linearno_family'))
                self.rows.append(dict(kind='resolved',task=task,profile=profile,model=model_kwargs(a),
                                     training=a._linearno_config['values']['training']))
        with self.assertRaises(SystemExit):args_for('darcy','--linearno-profile','official_release','--epochs','7')
        self.assertEqual(args_for('darcy','--epochs','7').epochs,7)
        for flags in (('--slice_num','32'),('--downsample','2'),('--linearno-rank','0')):
            with self.assertRaises(SystemExit):args_for('darcy',*flags)
        for task in ('car','airfrans'):
            with self.assertRaises(SystemExit):parse_args(parser_for('darcy'),task,['--model','LinearNO_Structured_Mesh_2D'])
        # No task entry may guess the historically nonexistent default factory key.
        for task in SHORT:
            with self.assertRaises(KeyError):get_model(parser_for(task).parse_args([]))

    def test_full_entry_AST_projects_to_pre_L4_all_legacy_paths(self):
        baseline=json.loads((ROOT/'docs/linearno_audit/l4/baseline.json').read_text())
        files=[f'PDE-Solving-StandardBenchmark/exp_{s}.py' for s in SHORT.values()]
        files+=['PDE-Solving-StandardBenchmark/model_dict.py','PDE-Solving-StandardBenchmark/cdlno_entry.py','cdlno/experiment.py']
        for path in files:
            self.assertEqual(ast.dump(strip_linearno(ast.parse((ROOT/path).read_text()))),
                             ast.dump(ast.parse((Path(baseline['snapshot'])/'source'/path).read_text())),path)
        # Data call, split, point order, decode and Darcy differential helper survive exactly.
        for path in ('PDE-Solving-StandardBenchmark/exp_ns.py','PDE-Solving-StandardBenchmark/exp_plas.py'):
            self.assertEqual(ast.dump(strip_linearno(ast.parse((ROOT/path).read_text()))),
                             ast.dump(ast.parse((Path(baseline['snapshot'])/'source'/path).read_text())))
        self.rows.append(dict(kind='complete_legacy_AST',files=files,exact=True,temporal_legacy_AST_equal=True))

    def test_losses_Darcy_decode_boundary_zero_padding_and_no_epsilon(self):
        from utils.normalizer import UnitTransformer
        from einops import rearrange
        import torch.nn.functional as F
        tree=ast.parse((PROJECT/'exp_darcy.py').read_text())
        diff=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='central_diff')
        scope=dict(torch=torch,F=F,rearrange=rearrange)
        exec(compile(ast.Module(body=[diff],type_ignores=[]),'<unchanged central_diff>','exec'),scope)
        B,H,W=2,85,85
        raw=torch.linspace(.1,2,B*H*W).reshape(B,H*W)
        norm=UnitTransformer(raw)
        normalized=torch.randn(B,H*W,requires_grad=True)
        truth=norm.decode(norm.encode(raw))
        out=norm.decode(normalized)
        loss=TestLoss(size_average=True)
        field=loss(out,truth)
        expected=(torch.linalg.vector_norm((out-truth).flatten(1),dim=1)/torch.linalg.vector_norm(truth.flatten(1),dim=1)).mean()
        torch.testing.assert_close(field,expected,atol=0,rtol=0)
        boundary=out.reshape(B,H,W).clone();boundary[:,0,:]=0;boundary[:,-1,:]=0;boundary[:,:,0]=0;boundary[:,:,-1]=0
        native=F.pad(out.reshape(B,1,H,W)[...,1:-1,1:-1].contiguous(),(1,1,1,1)).flatten(2).transpose(1,2)
        torch.testing.assert_close(native.squeeze(-1),boundary.flatten(1),atol=0,rtol=0)
        dx=1/85
        pg=scope['central_diff'](native,dx,H);tg=scope['central_diff'](truth.unsqueeze(-1),dx,H)
        # Independently slice zero-padded arrays, including boundary derivatives.
        zero=torch.zeros(B,H+2,W+2);zero[:,1:-1,1:-1]=boundary
        independent=((zero[:,1:-1,2:]-zero[:,1:-1,:-2])/(2*dx),
                     (zero[:,2:,1:-1]-zero[:,:-2,1:-1])/(2*dx))
        for a,b in zip(pg,independent):torch.testing.assert_close(a[...,0],b,atol=0,rtol=0)
        total=field+.1*(loss(pg[0],tg[0])+loss(pg[1],tg[1]));total.backward()
        self.assertTrue(torch.isfinite(normalized.grad).all())
        self.assertGreater(normalized.grad.abs().sum().item(),0)
        self.assertTrue(torch.isinf(loss(torch.ones(2,3),torch.zeros(2,3))))
        self.rows.append(dict(kind='Darcy_loss',shape=[2,7225],dx=dx,derivative_weight=.1,
                             zero_padded=True,crop_only_derivative_branch=True,decode_before_loss=True,batch_mean=True,epsilon=0))

    def test_actual_grid_order_Airfoil_swap_Pipe_square_and_Elasticity_none(self):
        for task in SHORT:
            H,W=SHAPES[task];N=H*W
            a=args_for(task,'--n-hidden','8','--n-heads','2','--n-layers','1','--linearno-rank','4')
            model=get_model(a).Model(**model_kwargs(a)).eval()
            tag=torch.arange(N).reshape(H,W)
            coordinates=torch.stack((tag//W,tag%W),dim=-1).float().reshape(1,N,2)
            if task=='elasticity':
                y=model(coordinates,None);self.assertEqual(tuple(y.shape),(1,972,1))
                continue
            seen=[]
            handle=model.blocks[0].Attn.in_project_x.register_forward_pre_hook(lambda m,i:seen.append(i[0].detach().clone()))
            features=torch.stack((tag.float(),tag.float()*3),dim=-1).repeat(1,1,4).reshape(1,N,8)
            model.blocks[0].Attn(features);handle.remove()
            torch.testing.assert_close(seen[0],features.transpose(1,2).reshape(1,8,H,W),atol=0,rtol=0)
            if task=='airfoil':
                self.assertEqual((H,W),(221,51))
                from model.LinearNO_Attention import LinearNO_Conv_temp
                wrong=LinearNO_Conv_temp(8,heads=2,dim_head=4,key_ratio=4,H=W,W=H)
                wrong.load_state_dict(model.blocks[0].Attn.state_dict(),strict=True)
                self.assertFalse(torch.allclose(wrong(features),model.blocks[0].Attn(features),atol=1e-7,rtol=1e-7))
                with self.assertRaisesRegex(ValueError,'grid H'):model_kwargs(a,H=W,W=H)
            if task=='pipe':self.assertEqual((H,W),(129,129))
        self.rows.append(dict(kind='actual_point_order',Airfoil=[221,51],Pipe=[129,129],Pipe_square=True,Elasticity_fx=None))

    def test_data_checksums_reject_changed_content_without_dataset_downloads(self):
        # Arbitrary bytes in a temporary folder only, not a named mock dataset.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fixture.bin';p.write_bytes(b'read-only checksum test')
            a=args_for('darcy');a.data_path=tmp
            a._linearno_config=copy.deepcopy(a._linearno_config)
            a._linearno_config['values']['data']['files']=['fixture.bin']
            verify_data(a);before=copy.deepcopy(a._linearno_data)
            a._linearno_metadata={'data_spec':before}
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksums'):verify_data(a)

    def test_native_four_task_train_checkpoint_resume_fresh_eval(self):
        artifact_root=Path(os.environ.get('LINEARNO_L4_ARTIFACT_ROOT',tempfile.mkdtemp(prefix='linearno-l4-test-')))
        artifact_root.mkdir(parents=True,exist_ok=True)
        for task in SHORT:
            with self.subTest(task=task):
                reports={}
                def worker(action,directory,label):
                    output=artifact_root/f'{task}-{label}.json';log=artifact_root/f'{task}-{label}.log'
                    command=[sys.executable,'-B',str(ROOT/'tests/linearno/static_worker.py'),task,action,str(directory),str(output)]
                    with log.open('w') as stream:
                        proc=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                            PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,timeout=180)
                    self.assertEqual(proc.returncode,0,log.read_text()[-5000:])
                    reports[label]=json.loads(output.read_text())
                continuous=artifact_root/f'{task}-continuous';split=artifact_root/f'{task}-split'
                worker('train',continuous,'continuous');worker('interrupt',split,'interrupt')
                worker('resume',split,'resume');worker('eval',split,'eval')
                cls=get_model(args_for(task)).Model
                mc,pc=inspect_checkpoint(continuous,'final',cls);ms,ps=inspect_checkpoint(split,'final',cls)
                _,wc=read_pair(pc,cls);_,ws=read_pair(ps,cls)
                self.assertTrue(_same(wc,ws),'continuous vs new-process resume weights')
                for key in ('optimizer','scheduler','rng','dataloader_generators','sampler_state'):
                    self.assertEqual(mc['resume_state'][key],ms['resume_state'][key],key)
                self.assertEqual(reports['continuous']['batches'],reports['interrupt']['batches']+reports['resume']['batches'])
                self.assertEqual(reports['continuous']['prediction_hash'],reports['eval']['prediction_hash'])
                self.assertEqual(reports['resume']['prediction_hash'],reports['eval']['prediction_hash'])
                self.assertEqual(ms['resume_state']['epoch'],3)
                self.assertEqual(ms['resume_state']['global_step'],6)
                plan=ms['data_spec']['scheduler'];self.assertEqual(plan['total_steps'],3 if task=='elasticity' else 6)
                # Wrong explicit architecture/profile rejected before Model allocation.
                @functools.wraps(cls.__init__)
                def forbidden_constructor(*args,**kwargs):
                    raise AssertionError('constructed before metadata conflict')
                for flags in (('--linearno-profile','official_release'),('--linearno-rank','9'),('--linearno-variant','plain')):
                    with patch.object(cls,'__init__',new=forbidden_constructor), self.assertRaises(SystemExit):
                        args_for(task,'--eval','1','--experiment-dir',split,*flags)
                model=cls(**ms['model_spec']['constructor_kwargs'])
                bad=copy.deepcopy(ws);bad.pop(next(iter(bad)))
                with self.assertRaises(ValueError):strict_load(model,bad)
                badmeta=copy.deepcopy(ms);badmeta['model_spec']['constructor_kwargs']['linearno_rank']=9
                with self.assertRaisesRegex(ValueError,'structural'):_structural_check(badmeta,task)
                # Read-only eval must not rewrite either training sidecar or config.
                self.assertTrue(reports['eval']['immutable_metadata'])
                self.rows.append(dict(kind='native_cycle',task=task,real_N=SHAPES[task][0]*SHAPES[task][1],
                    cwd=str(PROJECT),artifacts=str(split),synthetic=True,epochs=3,train_updates=6,
                    exact_weights_optimizer_scheduler_RNG=True,exact_batch_order=True,
                    checkpoint_strict=True,fresh_eval_prediction_hash=reports['eval']['prediction_hash'],
                    mean_loss=True,normalizer_no_refit=True,limits='synthetic prefix + CPU transfers; no real loader'))

    def test_all_new_launcher_actions_profiles_and_old_scripts_static(self):
        for task in SHORT:
            for action in ('train','eval'):
                script=ROOT/f'tran_evaluate/linearno/{task}_{action}.sh'
                subprocess.run(['bash','-n',str(script)],check=True)
                for profile in ('paper_table8_on_release_model','official_release'):
                    proc=subprocess.run(['bash',str(script),'--dry-run','--linearno-profile',profile,
                                         '--experiment-dir','space in path','--gpu','1'],cwd=ROOT,text=True,capture_output=True)
                    self.assertEqual(proc.returncode,0,proc.stderr)
                    self.assertIn('--model LinearNO_',proc.stdout)
                    self.assertIn('--eval '+('0' if action=='train' else '1'),proc.stdout)
                    self.assertIn('DRY RUN',proc.stdout)
        baseline=json.loads((ROOT/'docs/linearno_audit/l4/baseline.json').read_text())
        old_scripts=[r for r in baseline['files'] if r['path'].endswith('.sh') and r['kind']=='tracked']
        for row in old_scripts:
            path=ROOT/row['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['sha256'])
            subprocess.run(['bash','-n',str(path)],check=True,capture_output=True)
        self.rows.append(dict(kind='launchers',new_previews=16,old_shells_byte_equal_and_syntax=len(old_scripts)))
