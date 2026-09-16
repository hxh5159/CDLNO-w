"""M4 selection/explicit objective/core checkpoint foundation; no task entry imports."""
import argparse
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.core import MSARLNO
from cdlno.msar_lno.entry import core_model_kwargs
from cdlno.msar_lno.registry import core_class
from cdlno.msar_lno.options import explicit_arguments, parser_for_family, resolve_training
from cdlno.msar_lno.metadata import MSARMetadata, MSARMetadataMismatch
from cdlno.msar_lno.checkpoint import save_core_checkpoint, load_core_checkpoint
from cdlno.msar_lno.objective import training_forward, training_objective
from test_msar_core import small
from test_msar_config import parser_only, ENTRIES

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    torch.set_num_threads(1)


def meta(task='darcy', training=None, fmt=None):
    fmt = fmt or ('whole_model' if task == 'car' else 'model_list' if task == 'airfrans' else 'state_dict')
    return MSARMetadata(task, small(), fmt, training=training or MSARTrainingConfig())


def model_for(metadata, output_dim=2):
    return MSARLNO(metadata.architecture, output_dim=output_dim, training_config=metadata.training)


def hashes(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in Path(directory).rglob('*') if p.is_file()}


class MSARIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(916404)

    def test_real_factory_routes_and_old_kwargs_are_empty(self):
        code = '''
from argparse import Namespace
from model_dict import get_model
from cdlno.msar_lno.core import MSARLNO
from cdlno.msar_lno.entry import core_model_kwargs
assert get_model(Namespace(model='msar_lno')).Model is MSARLNO
for key in ('Transolver_Irregular_Mesh','Transolver_Structured_Mesh_2D','Transolver_Structured_Mesh_3D','kcdno','lrsa_matched'):
    module=get_model(Namespace(model=key))
    assert module.Model is not MSARLNO
for task in ('darcy','elasticity','airfoil','pipe','ns','plasticity'):
    assert get_model(Namespace(model='CDLNO',cdlno_task=task)).Model is not MSARLNO
print('actual factory mappings passed')
'''
        result = subprocess.run([sys.executable,'-B','-c',code], cwd=ROOT/'PDE-Solving-StandardBenchmark',
                                capture_output=True, text=True, env=dict(os.environ,PYTHONPATH=str(ROOT)))
        self.assertEqual(result.returncode,0,result.stderr)
        for key in ('CDLNO','kcdno','lrsa_matched','Transolver_Structured_Mesh_2D','GraphSAGE','unknown'):
            self.assertIsNone(core_class(key))
            # Invalid new-family input is not even inspected on a legacy branch.
            self.assertEqual(core_model_kwargs(key,task='unused',explicit=None,output_dim=-1),{})
        self.assertIs(core_class('msar_lno'),MSARLNO)

    def test_real_parser_config_priority_forwarding_and_light_defaults(self):
        parser=parser_only(ENTRIES[0]); before=vars(parser.parse_args([])).copy()
        new=parser_for_family(parser,'msar_lno')
        kw=core_model_kwargs('msar_lno',task='darcy',explicit=explicit_arguments(new,['--model','msar_lno']))
        self.assertEqual(kw['config'].d,96); self.assertEqual(kw['config'].num_latents,(512,256,128,64))
        explicit=explicit_arguments(new,['--model','msar_lno','--profile','full','--d','8',
                '--heads','2','2','4','4','--num-latents','7','5','3','2','--coverage-mode','off'])
        kw=core_model_kwargs('msar_lno',task='darcy',explicit=explicit,output_dim=2)
        model=core_class('msar_lno')(**kw)
        self.assertEqual(model.config,small());self.assertFalse(model.training_config.coverage_enabled)
        self.assertEqual(model(torch.randn(2,35,8)).shape,(2,35,2))
        self.assertEqual(vars(parser.parse_args([])),before)
        for bad in ({'family':'kcdno'},{'front_blocks':2},{'d':7},{'num_latents':[3,2]}):
            with self.assertRaises(ValueError):core_model_kwargs('msar_lno',task='darcy',explicit=bad)

    def test_original_loss_then_explicit_objective_and_optimizer(self):
        # Safe original relative-L2 callable, not a copied or approximated loss.
        spec=importlib.util.spec_from_file_location('m4_original_testloss',ROOT/'PDE-Solving-StandardBenchmark/utils/testloss.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cfg=MSARTrainingConfig(coverage_kappa=1.)
        model=MSARLNO(small(),output_dim=2,training_config=cfg).train()
        x=torch.randn(2,35,8);target=torch.randn(2,35,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        forward=training_forward(model,x)
        pde=module.TestLoss(size_average=False)(forward.prediction,target)
        loss=training_objective(pde,forward)
        self.assertIs(loss.pde,pde);self.assertIs(loss.coverage_raw,forward.auxiliary.coverage_mean)
        torch.testing.assert_close(loss.total,pde+cfg.coverage_weight*forward.auxiliary.coverage_per_level.mean(),atol=0,rtol=0)
        self.assertGreater(loss.coverage_raw.item(),0)
        self.assertEqual(set(loss.log_values()),{'pde','coverage_raw','coverage_weighted','total'})
        self.assertTrue(all(not v.requires_grad for v in loss.log_values().values()))
        before=model.downs[0].latent_queries.detach().clone()
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step();self.assertFalse(torch.equal(before,model.downs[0].latent_queries))
        self.assertFalse(hasattr(model,'last_loss'))

    def test_off_and_weight_zero_keep_identical_pde_tensor_no_attention_aux(self):
        model=MSARLNO(small()).train()
        for cfg in (MSARTrainingConfig(coverage_mode='off'),MSARTrainingConfig(coverage_weight=0)):
            requests=[]
            handles=[d.register_forward_pre_hook(lambda m,a,k:requests.append(k.copy()),with_kwargs=True) for d in model.downs]
            out=training_forward(model,torch.randn(1,11,8),training_config=cfg)
            for handle in handles:handle.remove()
            self.assertIsNone(out.auxiliary)
            self.assertTrue(all(not k.get('return_aux',False) for k in requests))
            pde=out.prediction.square().mean();loss=training_objective(pde,out)
            self.assertIs(loss.total,pde);self.assertFalse(loss.coverage_active)
            self.assertFalse(loss.coverage_raw.requires_grad);self.assertEqual(loss.coverage_raw.item(),0)
            loss.total.backward();model.zero_grad(set_to_none=True)

    def test_interleaved_calls_hold_their_own_config_and_graph(self):
        model=MSARLNO(small()).train();saved=model.training_config
        x=torch.randn(2,11,8,requires_grad=True)
        on=training_forward(model,x,training_config=MSARTrainingConfig(coverage_kappa=1,coverage_weight=.3))
        off=training_forward(model,torch.randn_like(x),training_config=MSARTrainingConfig(coverage_mode='off'))
        loss_off=training_objective(off.prediction.square().mean(),off)
        loss_on=training_objective(on.prediction.square().mean(),on)
        self.assertTrue(loss_on.coverage_active);self.assertFalse(loss_off.coverage_active)
        torch.testing.assert_close(loss_on.coverage_weighted,.3*on.auxiliary.coverage_mean,atol=0,rtol=0)
        self.assertIs(model.training_config,saved)
        gradient=torch.autograd.grad(loss_on.coverage_raw,x)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(gradient.abs().sum().item(),0)

    def test_prediction_only_eval_and_negative_adapter_contracts(self):
        model=MSARLNO(small()).eval();x=torch.randn(2,35,8)
        with torch.no_grad():
            on=model(x,training_config=MSARTrainingConfig())
            off=model(x,training_config=MSARTrainingConfig(coverage_mode='off'))
        self.assertIsInstance(on,torch.Tensor);torch.testing.assert_close(on,off,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'model.train'):training_forward(model,x)
        with self.assertRaisesRegex(ValueError,'msar_lno'):training_forward(torch.nn.Linear(8,1),x)
        model.train()
        with self.assertRaises(ValueError):training_forward(model,x,return_aux=False)
        with self.assertRaises(ValueError):training_forward(model,x,training_config={})
        out=training_forward(model,x)
        with self.assertRaisesRegex(ValueError,'scalar'):training_objective(out.prediction,out)
        with self.assertRaisesRegex(ValueError,'explicit'):training_objective(out.prediction.mean(),None)
        bad=replace(out,auxiliary=None)
        with self.assertRaisesRegex(ValueError,'active coverage'):training_objective(out.prediction.mean(),bad)

    def test_same_weights_off_floor_state_dict_and_sidecars_readonly(self):
        with tempfile.TemporaryDirectory() as tmp,sdpa_kernel(SDPBackend.MATH):
            for mode in ('off','floor'):
                metadata=meta(training=MSARTrainingConfig(coverage_mode=mode));model=model_for(metadata).eval()
                # Preserve trained values, not just initializer=0 values.
                with torch.no_grad():model.fusions[0].w.add_(.1)
                directory=Path(tmp)/mode;checkpoint=save_core_checkpoint(directory,model,metadata)
                self.assertTrue(all(isinstance(v,torch.Tensor) for v in torch.load(checkpoint,weights_only=True).values()))
                frozen=hashes(directory);x=torch.randn(2,35,8);expected=model(x)
                loaded=load_core_checkpoint(directory,task='darcy',explicit={'coverage_mode':'floor' if mode=='off' else 'off'},expected_output_dim=2)
                self.assertEqual(loaded.model.training_config,metadata.training)
                self.assertFalse(loaded.model.training)
                torch.testing.assert_close(loaded.model(x),expected,atol=0,rtol=0)
                self.assertEqual(hashes(directory),frozen)
                self.assertEqual(loaded.resolution.training_differences,('coverage_mode',))

    def test_saved_structure_first_and_conflicts_precede_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';save_core_checkpoint(directory,model_for(meta()),meta());frozen=hashes(directory)
            for explicit in ({'d':16},{'num_latents':[8,6,4,2]},{'heads':[4,4,4,4]},
                             {'family':'kcdno'},{'activation':'relu'}):
                with patch('torch.load',side_effect=AssertionError('must reject BEFORE deserialization')):
                    with self.assertRaises(ValueError):load_core_checkpoint(directory,task='darcy',explicit=explicit)
            with patch('torch.load',side_effect=AssertionError('must reject BEFORE deserialization')):
                with self.assertRaisesRegex(ValueError,'output_dim'):load_core_checkpoint(directory,task='darcy',expected_output_dim=4)
            loaded=load_core_checkpoint(directory,task='darcy',explicit={'profile':'full'})
            self.assertEqual(loaded.model.config,small()) # profile provenance does not overwrite saved values
            self.assertEqual(hashes(directory),frozen)

    def test_missing_family_and_incomplete_structure_are_never_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';save_core_checkpoint(directory,model_for(meta()),meta())
            path=directory/'architecture.json';raw=json.loads(path.read_text())
            for kind in ('family','shape','wrong_family'):
                bad=copy.deepcopy(raw)
                if kind=='family':bad.pop('family')
                elif kind=='shape':bad['architecture'].pop('heads')
                else:bad['family']='kcdno'
                path.write_text(json.dumps(bad));before=path.read_bytes()
                with self.assertRaises(ValueError):load_core_checkpoint(directory,task='darcy')
                self.assertEqual(path.read_bytes(),before)

    def test_strict_missing_extra_or_wrong_shape_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';path=save_core_checkpoint(directory,model_for(meta()),meta())
            weights=torch.load(path,weights_only=True);first=next(iter(weights))
            for kind in ('missing','extra','shape'):
                bad=weights.copy()
                if kind=='missing':bad.pop(first)
                elif kind=='extra':bad['made_up']=torch.zeros(1)
                else:bad[first]=bad[first][:-1]
                torch.save(bad,path)
                with self.assertRaises(RuntimeError):load_core_checkpoint(directory,task='darcy')

    def test_core_contract_schema_shape_and_dtype_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';save_core_checkpoint(directory,model_for(meta()),meta())
            path=directory/'core.json';original=json.loads(path.read_text())
            invalid=[dict(family='kcdno'),dict(input_contract='raw-task-data'),dict(output_dim=True),
                     dict(members=2),dict(schema_version=True),dict(weight_dtype=[]),dict(filename='../model.pt')]
            invalid.append({'model_class':'models.Transolver.Model'})
            for change in invalid:
                path.write_text(json.dumps(original | change));frozen=path.read_bytes()
                with patch('torch.load',side_effect=AssertionError('invalid contract must not load')):
                    with self.assertRaises(ValueError):load_core_checkpoint(directory,task='darcy')
                self.assertEqual(path.read_bytes(),frozen)

    def test_exclusive_save_and_validation_before_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'run';save_core_checkpoint(directory,model_for(meta()),meta());before=hashes(directory)
            with self.assertRaises(FileExistsError):save_core_checkpoint(directory,model_for(meta()),meta())
            self.assertEqual(hashes(directory),before)
            other=Path(tmp)/'bad'
            with self.assertRaises(ValueError):save_core_checkpoint(other,model_for(meta()),meta(training=MSARTrainingConfig(coverage_mode='off')))
            self.assertFalse(other.exists())
            with self.assertRaises(ValueError):save_core_checkpoint(other,model_for(meta()),meta(),filename='../model.pt')
            self.assertFalse(other.exists())

    def test_whole_and_list_protocols_and_trusted_local_boundary(self):
        with tempfile.TemporaryDirectory() as tmp,sdpa_kernel(SDPBackend.MATH):
            for task,fmt in (('car','whole_model'),('airfrans','model_list'),('airfrans','whole_model')):
                metadata=meta(task,fmt=fmt);models=[model_for(metadata).eval() for _ in range(2 if fmt=='model_list' else 1)]
                payload=models if fmt=='model_list' else models[0]
                directory=Path(tmp)/(task+fmt)
                filename='model_200.pth' if task=='car' else None
                save_core_checkpoint(directory,payload,metadata,filename=filename);frozen=hashes(directory)
                with patch('torch.load',side_effect=AssertionError('untrusted must reject before pickle')):
                    with self.assertRaisesRegex(ValueError,'trusted local'):load_core_checkpoint(directory,task=task)
                loaded=load_core_checkpoint(directory,task=task,trusted_local=True,explicit={'coverage_mode':'off'}).model
                actual=loaded if fmt=='model_list' else [loaded]
                self.assertEqual(type(loaded),list if fmt=='model_list' else MSARLNO)
                x=torch.randn(1,11,8)
                for a,b in zip(actual,models):torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)
                self.assertEqual(hashes(directory),frozen)

    def test_whole_object_hidden_behavior_and_list_count_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata=meta('car');model=model_for(metadata);directory=Path(tmp)/'car'
            path=save_core_checkpoint(directory,model,metadata,filename='model_200.pth')
            model.downs[0].heads=4 # same weights/config but a different forward
            torch.save(model,path)
            with self.assertRaisesRegex(ValueError,'behavior'):load_core_checkpoint(directory,task='car',trusted_local=True)
            metadata=meta('airfrans');directory=Path(tmp)/'air'
            path=save_core_checkpoint(directory,[model_for(metadata),model_for(metadata)],metadata)
            torch.save([model_for(metadata)],path)
            with self.assertRaisesRegex(ValueError,'member count'):load_core_checkpoint(directory,task='airfrans',trusted_local=True)

    def test_parameter_aliasing_and_dtype_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata=meta();model=model_for(metadata)
            model.downs[1].to_k.weight=model.downs[0].to_k.weight
            with self.assertRaisesRegex(ValueError,'shared'):save_core_checkpoint(Path(tmp)/'alias',model,metadata)
            model=model_for(metadata).double().eval();directory=Path(tmp)/'double'
            save_core_checkpoint(directory,model,metadata)
            loaded=load_core_checkpoint(directory,task='darcy').model;x=torch.randn(1,11,8,dtype=torch.float64)
            self.assertEqual(next(loaded.parameters()).dtype,torch.float64)
            torch.testing.assert_close(loaded(x),model(x),atol=0,rtol=0)

    def test_load_preserves_rng_and_does_not_initialize_over_trained_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata=meta();model=model_for(metadata).eval()
            with torch.no_grad():model.fusions[2].w.fill_(.42)
            directory=Path(tmp)/'run';rng=torch.get_rng_state().clone();save_core_checkpoint(directory,model,metadata)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            actual=load_core_checkpoint(directory,task='darcy').model
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            torch.testing.assert_close(actual.fusions[2].w,model.fusions[2].w,atol=0,rtol=0)

    def test_three_project_cwd_fresh_process_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs=[]
            for project,task in (('PDE-Solving-StandardBenchmark','darcy'),('Car-Design-ShapeNetCar','car'),('Airfoil-Design-AirfRANS','airfrans')):
                metadata=meta(task);model=model_for(metadata).eval();directory=Path(tmp)/task
                payload=[model] if task=='airfrans' else model
                save_core_checkpoint(directory,payload,metadata,filename='model_200.pth' if task=='car' else None)
                x=torch.randn(2,11,8)
                torch.save({'input':x,'output':model(x).detach()},Path(tmp)/(task+'.pt'))
                dirs.append((project,task,directory))
            code='''
import sys,torch
from cdlno.msar_lno.checkpoint import load_core_checkpoint
torch.set_num_threads(1)
directory,task,fixture=sys.argv[1:]
result=load_core_checkpoint(directory,task=task,trusted_local=task!='darcy')
model=result.model[0] if isinstance(result.model,list) else result.model
sample=torch.load(fixture,weights_only=True)
torch.testing.assert_close(model(sample['input']),sample['output'],atol=0,rtol=0)
print(task+' strict same-weight fresh-cwd load passed')
'''
            for project,task,directory in dirs:
                result=subprocess.run([sys.executable,'-B','-c',code,str(directory),task,str(Path(tmp)/(task+'.pt'))],
                    cwd=ROOT/project,env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable: M4 adapter GPU check not run')
    def test_gpu_explicit_objective_step_and_checkpoint(self):
        cfg=MSARTrainingConfig(coverage_kappa=1.)
        metadata=meta(training=cfg);model=model_for(metadata).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        with sdpa_kernel(SDPBackend.MATH):
            x=torch.randn(2,11,8,device='cuda')
            result=training_forward(model,x)
            # Explicit synthetic MSE, not a real task loop or metric.
            losses=training_objective(result.prediction.square().mean(),result)
            losses.total.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            with tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/'gpu';save_core_checkpoint(directory,model,metadata)
                loaded=load_core_checkpoint(directory,task='darcy',device='cuda').model
                torch.testing.assert_close(model.eval()(x),loaded(x),atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main(verbosity=2)
