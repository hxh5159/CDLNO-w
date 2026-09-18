"""L5 temporal contracts: actual loops, parameter counts and process continuation."""
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

from linearno.temporal_worker import ROOT, PROJECT, SHORT, args_for, parser_for, synthetic_values
from model_dict import get_model
from cdlno_entry import parse_args
from cdlno.linearno.standard_entry import model_kwargs, verify_data
from cdlno.linearno.checkpoint import inspect_checkpoint, read_pair, strict_load
from cdlno.linearno.schema import unpack_state, validate_metadata
from cdlno.linearno.profiles import validate_resolved
from cdlno.training_state import _same
from linearno_entry_projection import strip_linearno


class TemporalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1); cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if target := os.environ.get('LINEARNO_L5_REPORT'):
            Path(target).write_text(json.dumps(cls.rows, indent=2)+'\n')

    def test_profiles_real_counts_and_explicit_override(self):
        for task, count in [('ns',3377921), ('plasticity',1799428)]:
            for profile in ('paper_table8_on_release_model', 'official_release'):
                args = args_for(task, '--linearno-profile', profile)
                model = get_model(args).Model(**model_kwargs(args))
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                self.assertFalse(any('temperature' in k for k in model.state_dict()))
                self.assertEqual((args.n_layers,args.n_heads),(8,8))
                if task == 'ns':
                    self.assertEqual((args.n_hidden,args.linearno_rank,args.mlp_ratio,args.ref,args.unified_pos),(256,32,2,10,True))
                    self.assertEqual(args.linearno_variant,'plain')
                else:
                    self.assertEqual((args.n_hidden,args.linearno_rank,args.mlp_ratio),(128,64,1))
                    self.assertEqual(args.linearno_variant,'conv')
                    self.assertEqual((model_kwargs(args)['H'],model_kwargs(args)['W'],model_kwargs(args)['out_dim']),(101,31,4))
                    self.assertTrue(model_kwargs(args)['Time_Input'])
                validate_resolved(args._linearno_config)
                self.assertTrue(args._linearno_config['values']['objective']['reduction'].startswith('sum'))
                self.rows.append(dict(kind='formal_construction',task=task,profile=profile,parameters=count))
            explicit=args_for(task,'--ref',6,'--mlp_ratio',3)
            self.assertEqual((explicit.ref,explicit.mlp_ratio),(6,3))
            old=parse_args(parser_for(task),task,[])
            self.assertEqual((old.n_hidden,old.ref,old.mlp_ratio),(64,8,1))
            self.assertFalse(hasattr(old,'linearno_family'))
        with self.assertRaises(SystemExit): args_for('ns','--downsample',2)
        with self.assertRaises(SystemExit): args_for('plasticity','--slice_num',32)

    def test_time_condition_gradient_and_historical_point_order(self):
        args=args_for('plasticity','--n-hidden',8,'--n-heads',2,'--n-layers',2,'--linearno-rank',4)
        model=get_model(args).Model(**model_kwargs(args)).eval()
        xx,yy=np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,31))
        x=torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float)[None].repeat(2,1,1)
        fx=torch.randn(2,3131,1)
        t=torch.tensor([[.1],[.8]],requires_grad=True)
        before=(x.clone(),fx.clone(),t.clone())
        output=model(x,fx,T=t)
        self.assertEqual(tuple(output.shape),(2,3131,4))
        self.assertFalse(torch.allclose(output,model(x,fx,T=t.flip(0)),atol=1e-9,rtol=1e-7))
        output.square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.time_fc.parameters()),0)
        self.assertTrue(torch.isfinite(t.grad).all());self.assertGreater(t.grad.abs().sum().item(),0)
        for a,b in zip((x,fx,t),before):torch.testing.assert_close(a,b,atol=0,rtol=0)
        # Raw field row-major101x31, meshgrid row-major31x101: preserve mismatch.
        field=torch.arange(3131).float().reshape(1,3131,1).expand(1,3131,8).contiguous()
        seen=[]
        h=model.blocks[0].Attn.in_project_x.register_forward_pre_hook(lambda m,i:seen.append(i[0].clone()))
        model.blocks[0].Attn(field);h.remove()
        torch.testing.assert_close(seen[0],field.transpose(1,2).reshape(1,8,101,31),atol=0,rtol=0)
        self.assertEqual(x[0,1,0].item(),np.float32(.01))
        self.rows.append(dict(kind='time_point_order',B=2,N=3131,T_shape=[2,1],time_grad=True,field_grid=[101,31],meshgrid=[31,101]))

    def test_unchanged_complete_temporal_AST_and_real_collate(self):
        baseline=json.loads((ROOT/'docs/linearno_audit/l5/baseline.json').read_text())
        for task,short in SHORT.items():
            path=Path(f'PDE-Solving-StandardBenchmark/exp_{short}.py')
            old=ast.parse((Path(baseline['snapshot'])/'source'/path).read_text())
            now=ast.parse((ROOT/path).read_text())
            self.assertEqual(ast.dump(strip_linearno(now)),ast.dump(old))
        tree=ast.parse((PROJECT/'exp_plas.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='random_collate_fn')
        scope=dict(torch=torch)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual collate>','exec'),scope)
        tim=torch.arange(20).float()
        batch=[(torch.zeros(3131,2),tim,torch.zeros(3131,1),tim.repeat(3131,4,1)) for _ in range(2)]
        torch.manual_seed(97); expected=[torch.randperm(20) for _ in range(2)]
        torch.manual_seed(97)
        with patch.object(np.random,'permutation',side_effect=AssertionError('not the actual RNG')):
            actual=scope['random_collate_fn'](batch)
        for i in range(2):
            torch.testing.assert_close(actual[1][i],tim[expected[i]],atol=0,rtol=0)
            torch.testing.assert_close(actual[3][i,0,0],tim[expected[i]],atol=0,rtol=0)
        self.assertFalse(torch.equal(actual[1][0],actual[1][1]))
        self.rows.append(dict(kind='collate_and_legacy_AST',per_sample_torch_rng=True,old_complete_AST_equal=True))

    def test_native_temporal_cycles_and_negative_checkpoints(self):
        root=Path(os.environ.get('LINEARNO_L5_ARTIFACT_ROOT',tempfile.mkdtemp(prefix='linearno-l5-')))
        root.mkdir(parents=True,exist_ok=True)
        for task in SHORT:
            reports={}
            for action,label,subdir in [('train','continuous','continuous'),('interrupt','interrupt','split'),('resume','resume','split'),('eval','eval','split')]:
                report=root/f'{task}-{label}.json';log=root/f'{task}-{label}.log'
                command=[sys.executable,'-B',str(ROOT/'tests/linearno/temporal_worker.py'),task,action,str(root/f'{task}-{subdir}'),str(report)]
                with log.open('w') as stream:
                    proc=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,timeout=240)
                self.assertEqual(proc.returncode,0,log.read_text()[-6000:])
                reports[label]=json.loads(report.read_text())
            cls=get_model(args_for(task)).Model
            mc,pc=inspect_checkpoint(root/f'{task}-continuous','final',cls)
            ms,ps=inspect_checkpoint(root/f'{task}-split','final',cls)
            _,wc=read_pair(pc,cls);_,ws=read_pair(ps,cls)
            self.assertTrue(_same(wc,ws))
            for k in ('optimizer','scheduler','rng','dataloader_generators','sampler_state'):
                self.assertEqual(mc['resume_state'][k],ms['resume_state'][k],k)
            for k in ('batches','time_queries'):
                self.assertEqual(reports['continuous'][k],reports['interrupt'][k]+reports['resume'][k])
            self.assertEqual(reports['continuous']['numpy_next'],reports['resume']['numpy_next'])
            for label in ('resume','eval'):
                self.assertEqual(reports['continuous']['prediction_hash'],reports[label]['prediction_hash'])
                self.assertTrue(reports[label]['immutable_metadata'])
            self.assertEqual(ms['resume_state']['global_step'],120 if task=='plasticity' else 6)
            self.assertEqual(unpack_state(ms['resume_state']['scheduler'])['last_epoch'],6)
            @functools.wraps(cls.__init__)
            def forbidden(*a,**kw): raise AssertionError('construction before metadata validation')
            for flag,value in [('--linearno-profile','official_release'),('--linearno-rank',7),('--linearno-variant','temp')]:
                with patch.object(cls,'__init__',new=forbidden),self.assertRaises(SystemExit):
                    args_for(task,'--eval',1,'--experiment-dir',root/f'{task}-split',flag,value)
            model=cls(**ms['model_spec']['constructor_kwargs'])
            for kind in ('missing','extra','shape'):
                bad=copy.deepcopy(ws);key=next(k for k,v in bad.items() if v.ndim and v.shape[0]>1)
                if kind=='missing':bad.pop(key)
                elif kind=='extra':bad['unexpected']=bad[key].clone()
                else:bad[key]=bad[key][:-1]
                with self.assertRaises(ValueError):strict_load(model,bad)
            bad=copy.deepcopy(ms);bad['family']='CDLNO'
            with self.assertRaisesRegex(ValueError,'family'):validate_metadata(bad,constructor=cls)
            self.rows.append(dict(kind='native_cycle',task=task,artifacts=str(root/f'{task}-split'),
                reports=reports,exact_all_state=True,strict_negatives=7,epochs=3,synthetic=True))

    def test_launchers_and_file_path_contract(self):
        for task in SHORT:
            for action in ('train','eval'):
                path=ROOT/f'tran_evaluate/linearno/{task}_{action}.sh'
                subprocess.run(['bash','-n',str(path)],check=True)
                for profile in ('paper_table8_on_release_model','official_release'):
                    p=subprocess.run(['bash',str(path),'--dry-run','--linearno-profile',profile,'--experiment-dir','path with spaces','--gpu','1'],capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stderr)
                    self.assertIn('--model LinearNO_Structured_Mesh_2D',p.stdout)
                    self.assertIn('--eval '+('0' if action=='train' else '1'),p.stdout)
                    self.assertIn('exp_'+SHORT[task]+'.py',p.stdout)
        # Check path assembly without creating a fake .mat dataset.
        args=args_for('plasticity');args.data_path='/virtual/renamed-data.mat'
        seen=[]
        with patch('cdlno.linearno.checkpoint.sha256',side_effect=lambda p:seen.append(p) or 'a'*64):verify_data(args)
        self.assertEqual(seen,[Path('/virtual/renamed-data.mat')])
        self.rows.append(dict(kind='launcher_previews',count=8,plasticity_path='whole MAT path'))
