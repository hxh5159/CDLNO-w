"""MSAR M1: real configuration/metadata and safe parser ASTs, no fake model."""
import argparse
import ast
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cdlno.msar_lno import (FAMILY, MSARArchitectureConfig, MSARTrainingConfig, MSARRuntimeConfig,
                             resolve_profile, input_layout, family_for_model_key, checkpoint_family)
from cdlno.msar_lno.profiles import TASKS, profile_values
from cdlno.msar_lno.options import (explicit_arguments, parser_for_family, parse_training_options,
                                   architecture_overrides, resolve_training)
from cdlno.msar_lno.metadata import (MSARMetadata, MSARMetadataMismatch, load_metadata, save_metadata,
                                    compare_architecture, resolve_evaluation, new_run_path, reserve_run_directory)

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [f'PDE-Solving-StandardBenchmark/exp_{name}.py' for name in ('darcy','elas','airfoil','pipe','ns','plas')]
ENTRIES += [f'{project}/{entry}.py' for project in ('Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS')
            for entry in ('main','main_evaluation')]


def parser_only(name):
    nodes = [n for n in ast.parse((ROOT/name).read_text()).body if
             (isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'parser') or
             (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) == 'parser.add_argument')]
    scope = {'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name+':parser-AST-only','exec'),scope)
    # Existing helpers add family/profile options, including Air eval's model
    # selector. Execute only declarations before their first parse_args call;
    # no entry/helper model selection, filesystem access, or data imports.
    project = name.split('/')[0]
    helper = ROOT/project/('models/cdlno_run.py' if project.startswith('Car-') else 'cdlno_entry.py')
    tree = ast.parse(helper.read_text())
    prefix = [n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == 'MODEL_OPTIONS']
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'parse_args')
    for node in function.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in ('tokens','args'):
            break
        prefix.append(node)
    scope.update(Path=Path, evaluation=name.endswith('main_evaluation.py'))
    exec(compile(ast.Module(body=prefix,type_ignores=[]),str(helper)+':declarations-only','exec'),scope)
    return scope['parser']


def sidecar(task='darcy', profile='light', training=None):
    fmt = 'whole_model' if task == 'car' else ('model_list' if task == 'airfrans' else 'state_dict')
    return MSARMetadata(task, resolve_profile(profile), fmt, profile,
                        training=training or MSARTrainingConfig())


class MSARConfigTests(unittest.TestCase):
    def test_resolved_profiles_all_tasks_and_independent_schema(self):
        for task in TASKS:
            for profile,d,tokens in (('light',96,[512,256,128,64]),('full',192,[1024,512,256,128])):
                with self.subTest(task=task,profile=profile):
                    resolved = resolve_training(task,{'profile':profile})
                    cfg = resolved.architecture
                    self.assertEqual(cfg.family,FAMILY)
                    self.assertEqual(cfg.d,d);self.assertEqual(list(cfg.num_latents),tokens)
                    self.assertEqual(cfg.heads,(4,4,8,8))
                    self.assertEqual(cfg.encoder_depths,cfg.decoder_depths)
                    self.assertEqual(cfg.encoder_depths,(3,1,1,1))
                    self.assertEqual(sum(cfg.encoder_depths)+sum(cfg.decoder_depths),12)
                    self.assertEqual(cfg.latent_ffn_hidden,2*d)
                    self.assertEqual((cfg.activation,cfg.point_module),('gelu','pointwise_mlp'))
                    self.assertEqual((resolved.training.coverage_mode,resolved.training.coverage_weight,
                                      resolved.training.coverage_kappa),('floor',.01,.2))
                    self.assertEqual(resolved.to_dict()['profile_overrides'],[])
                    for key in ('L','F','P','M','h','front_latent_mode','cdpa_mode','history_mode','kernel_rank','coverage_mode'):
                        self.assertNotIn(key,cfg.to_dict())
        self.assertEqual(resolve_training('darcy',{}).architecture,MSARArchitectureConfig())

    def test_list_shapes_integer_types_and_fixed_depths(self):
        for name in ('num_latents','heads','encoder_depths','decoder_depths'):
            for value in ([],[1,2,3],[1]*5,'1,2,3,4',None):
                with self.subTest(name=name,value=value),self.assertRaisesRegex(ValueError,'exactly four'):
                    MSARArchitectureConfig(**{name:value})
            for invalid in (True,1.0,'1',float('inf'),float('nan'),-1):
                values=list(getattr(MSARArchitectureConfig(),name));values[0]=invalid
                with self.subTest(name=name,invalid=invalid),self.assertRaises(ValueError):
                    MSARArchitectureConfig(**{name:values})
        for depths in ((0,1,1,1),(1,1,1,1),(3,1,1,0)):
            with self.assertRaisesRegex(ValueError,'v1 requires'):MSARArchitectureConfig(encoder_depths=depths)
        with self.assertRaises(ValueError):MSARArchitectureConfig(heads=(0,4,8,8))
        with self.assertRaises(ValueError):MSARArchitectureConfig(num_latents=(4,4,2,1))
        with self.assertRaises(ValueError):MSARArchitectureConfig(num_latents=(1,2,3,4))
        for d in (0,True,96.0,95):
            with self.assertRaises(ValueError):MSARArchitectureConfig(d=d)

    def test_fixed_behaviors_and_json_strict_roundtrip(self):
        for key,value in dict(family='kcdno',architecture_version='future',activation='geglu',
                              latent_ffn_ratio=4,norm='layernorm',qk_norm='none',output_norm='rmsnorm',
                              norm_eps=1e-5,point_module='conv_ffn',fusion='history',fusion_scale=1.).items():
            with self.subTest(key=key),self.assertRaises(ValueError):MSARArchitectureConfig(**{key:value})
        for cls in (MSARArchitectureConfig,MSARTrainingConfig,MSARRuntimeConfig):
            cfg=cls();raw=json.loads(json.dumps(cfg.to_dict()))
            self.assertEqual(cls.from_dict(raw),cfg)
            for key in raw:
                bad=raw.copy();bad.pop(key)
                with self.assertRaisesRegex(ValueError,'missing'):cls.from_dict(bad)
            with self.assertRaisesRegex(ValueError,'unknown'):cls.from_dict(raw|{'extra':1})
        values=profile_values();cfg=MSARArchitectureConfig(**values);values['num_latents'][0]=2
        exported=cfg.to_dict();exported['num_latents'][0]=1
        self.assertEqual(cfg.num_latents,(512,256,128,64))
        self.assertEqual(profile_values()['num_latents'],[512,256,128,64])

    def test_training_objective_validation_and_effective_off(self):
        for name in ('coverage_weight','coverage_kappa','coverage_eps'):
            for value in (True,'0.1',float('nan'),float('inf'),-1):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):MSARTrainingConfig(**{name:value})
        for values in (dict(coverage_mode='balance'),dict(coverage_kappa=1.1),dict(coverage_eps=0),dict(diagnostics=1)):
            with self.assertRaises(ValueError):MSARTrainingConfig(**values)
        for values in (dict(coverage_mode='off'),dict(coverage_mode='floor',coverage_weight=0),dict(coverage_mode='off',coverage_weight=0)):
            cfg=MSARTrainingConfig(**values)
            self.assertFalse(cfg.coverage_enabled);self.assertEqual(cfg.effective_coverage_mode,'off')
        self.assertTrue(MSARTrainingConfig(coverage_kappa=0).coverage_enabled)  # zero penalty, not a third mode
        self.assertEqual(MSARTrainingConfig(coverage_kappa=1).coverage_kappa,1)
        with self.assertRaises(ValueError):MSARRuntimeConfig(batch_size=True)
        with self.assertRaises(ValueError):MSARRuntimeConfig(amp=1)

    def test_expansion_record_does_not_clip_full_or_change_config(self):
        cfg=resolve_profile('full');before=cfg.to_dict()
        layout=input_layout(cfg,972)
        self.assertTrue(layout['first_down_expands']);self.assertEqual(layout['token_path'],[972,1024,512,256,128])
        self.assertEqual(cfg.to_dict(),before)
        self.assertFalse(input_layout(cfg,2048)['first_down_expands'])
        for n in (0,-1,True,2.5):
            with self.assertRaises(ValueError):input_layout(cfg,n)

    def test_real_parser_styles_cli_priority_and_no_original_mutation(self):
        for name in ENTRIES:
            with self.subTest(entry=name):
                parser=parser_only(name)
                # An original required --my_path remains required; supply only safe parser values.
                required=[]
                for action in parser._actions:
                    if action.required:required.extend([action.option_strings[-1],'synthetic-path-not-opened'])
                defaults=vars(parser.parse_args(required)).copy()
                before=[(a.dest,copy.deepcopy(a.default),copy.deepcopy(a.choices)) for a in parser._actions]
                flag='--cfd_model' if name.startswith('Car-') else '--model'
                resolved=parse_training_options(parser,[*required,flag,FAMILY],task='darcy',model_key=FAMILY)
                self.assertEqual(resolved.architecture.d,96)
                self.assertEqual(resolved.architecture.num_latents,(512,256,128,64))
                full=parse_training_options(parser,[*required,flag,FAMILY,'--profile','full'],task='darcy',model_key=FAMILY)
                self.assertEqual(full.architecture.d,192)
                two=parse_training_options(parser,[*required,flag,FAMILY,'--latent-ffn-ratio','2'],task='darcy',model_key=FAMILY)
                self.assertEqual(two.architecture.latent_ffn_hidden,192)
                override=parse_training_options(parser,[*required,flag,FAMILY,'--profile','full','--d','64','--d=128',
                    '--num-latents','64','32','16','8','--heads','4','4','8','8',
                    '--coverage-mode','off','--coverage-weight','0','--diagnostics','--no-diagnostics'],
                    task='darcy',model_key=FAMILY)
                self.assertEqual((override.architecture.d,override.architecture.num_latents),(128,(64,32,16,8)))
                self.assertFalse(override.training.coverage_enabled);self.assertFalse(override.training.diagnostics)
                self.assertEqual(vars(parser.parse_args(required)),defaults)
                self.assertEqual([(a.dest,a.default,a.choices) for a in parser._actions],before)

    def test_existing_profile_defaults_are_not_promoted_to_explicit(self):
        parser=parser_only(ENTRIES[0])
        parser.set_defaults(n_hidden=64,n_layers=3,slice_num=32)
        copy_parser=parser_for_family(parser,FAMILY)
        self.assertEqual(explicit_arguments(copy_parser,[]),{})
        self.assertEqual(resolve_training('pipe',explicit_arguments(copy_parser,[])).architecture.d,96)
        cfg=resolve_training('pipe',explicit_arguments(copy_parser,['--profile','full','--n-hidden','64']))
        self.assertEqual(cfg.architecture.d,64);self.assertEqual(cfg.architecture.num_latents,(1024,512,256,128))
        self.assertEqual(parser.parse_args([]).profile,'kcdno_v1')
        with self.assertRaisesRegex(ValueError,'not applicable'):
            resolve_training('pipe',explicit_arguments(copy_parser,['--front-blocks','2']))
        with self.assertRaisesRegex(ValueError,'conflicting'):
            architecture_overrides(dict(n_hidden=64,d=96))

    def test_old_family_delegation_and_inapplicable_flags(self):
        parser=parser_only(ENTRIES[0])
        for key in ('Transolver_Structured_Mesh_2D','CDLNO','kcdno','lrsa_matched','unknown'):
            self.assertIsNone(family_for_model_key(key))
            self.assertIsNone(parse_training_options(parser,['--coverage-mode','floor'],task='darcy',model_key=key))
        self.assertEqual(family_for_model_key(FAMILY),FAMILY)
        for key in ('L','F','P','M','n_layers','slice_num','n_heads','front_latent_mode','cdpa_mode','history_mode','kernel_rank'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'not applicable'):
                resolve_training('darcy',{key:1})
        for key in ('model','cfd_model','family'):
            with self.assertRaises(ValueError):resolve_training('darcy',{key:'kcdno'})
        with self.assertRaisesRegex(ValueError,'unknown'):resolve_training('darcy',{'coverage_balance':True})
        with self.assertRaises(ValueError):resolve_training('unknown',{})
        with self.assertRaises(ValueError):resolve_training('darcy',{'profile':'unknown'})
        with self.assertRaises(ValueError):resolve_training('darcy',argparse.Namespace())

    def test_metadata_all_protocols_roundtrip_without_weight_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in TASKS:
                for profile in ('light','full'):
                    meta=sidecar(task,profile);path=Path(directory)/task/(profile+'.json')
                    save_metadata(path,meta);self.assertEqual(load_metadata(path),meta)
                    before=path.read_bytes()
                    self.assertEqual(resolve_evaluation(path,{},task=task).saved.architecture,meta.architecture)
                    self.assertEqual(path.read_bytes(),before)
                    with self.assertRaises(FileExistsError):save_metadata(path,meta)
            self.assertEqual(replace(sidecar('airfrans'),checkpoint_format='whole_model').checkpoint_format,'whole_model')
            for task in TASKS:
                with self.assertRaises(MSARMetadataMismatch):replace(sidecar(task),checkpoint_format='made_up_format')

    def test_evaluation_reads_first_coverage_changes_allowed_structure_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'architecture.json'
            with self.assertRaises(FileNotFoundError):resolve_evaluation(path,{'model':'wrong'},task='darcy')
            saved=sidecar(profile='full');save_metadata(path,saved);before=path.read_bytes()
            old_runtime=MSARRuntimeConfig(device='cpu',batch_size=2,sdpa_backend='math')
            resolved=resolve_evaluation(path,dict(model=FAMILY,coverage_mode='off',coverage_weight=0),task='darcy',runtime=old_runtime)
            self.assertEqual(resolved.saved.architecture.d,192)
            self.assertEqual(resolved.saved.training.coverage_mode,'floor')
            self.assertFalse(resolved.requested_training.coverage_enabled)
            self.assertEqual(set(resolved.training_differences),{'coverage_mode','coverage_weight'})
            self.assertEqual(set(resolved.runtime_differences),{'device','batch_size','sdpa_backend'})
            self.assertEqual(path.read_bytes(),before)
            for kw in (dict(d=96),dict(num_latents=[512,256,128,64]),dict(heads=[8,8,8,8]),dict(model='CDLNO'),dict(activation='relu')):
                with self.subTest(kw=kw),self.assertRaises(ValueError):resolve_evaluation(path,kw,task='darcy')
                self.assertEqual(path.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'task mismatch'):resolve_evaluation(path,{},task='pipe')
            # Profile provenance never expands new defaults over resolved saved structure.
            resolved=resolve_evaluation(path,{'profile':'light'},task='darcy')
            self.assertEqual(resolved.saved.architecture.d,192);self.assertEqual(resolved.requested_profile,'light')
            self.assertEqual(compare_architecture(saved.architecture,resolve_profile('light')),['d','num_latents'])

    def test_corrupt_metadata_family_schema_and_legacy_dispatch(self):
        raw=sidecar().to_dict()
        for key in raw:
            bad=copy.deepcopy(raw);del bad[key]
            with self.assertRaises(ValueError):MSARMetadata.from_dict(bad)
        for value in (True,2,'1'):
            with self.assertRaises(ValueError):MSARMetadata.from_dict(raw|{'schema_version':value})
        for value in ('kcdno','CDLNO',None,''):
            with self.assertRaises(ValueError):MSARMetadata.from_dict(raw|{'family':value})
        bad=copy.deepcopy(raw);bad['architecture']['family']='kcdno'
        with self.assertRaises(ValueError):MSARMetadata.from_dict(bad)
        self.assertIsNone(checkpoint_family({'architecture':{'family':FAMILY}}))
        self.assertEqual(checkpoint_family({'family':'kcdno'}),'kcdno')
        from cdlno.checkpoint import save_sidecar,load_sidecar
        from cdlno.config import CDLNOArchitectureConfig
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.json';save_sidecar(path,CDLNOArchitectureConfig())
            before=path.read_bytes();payload=json.loads(before)
            self.assertIsNone(checkpoint_family(payload));self.assertEqual(load_sidecar(path)['architecture']['model_name'],'CDLNO')
            with self.assertRaises(ValueError):load_metadata(path)
            self.assertEqual(path.read_bytes(),before)

    def test_isolated_paths_named_runs_exclusive_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for profile in ('light','full'):
                for mode in ('floor','off'):
                    resolved=resolve_training('darcy',dict(profile=profile,coverage_mode=mode))
                    path=new_run_path(directory,resolved,save_name='user_seed0')
                    self.assertEqual(path,Path(directory)/'darcy'/FAMILY/profile/('coverage_'+mode)/'user_seed0')
                    self.assertFalse(path.exists());reserve_run_directory(path)
                    (path/'keep.txt').write_text('preserve')
                    with self.assertRaises(FileExistsError):reserve_run_directory(path)
                    self.assertEqual((path/'keep.txt').read_text(),'preserve');paths.append(path)
            self.assertEqual(len(set(paths)),4)
            resolved=resolve_training('darcy',{'coverage_weight':0})
            self.assertIn('coverage_off',new_run_path(directory,resolved).parts)
            self.assertNotEqual(new_run_path(directory,resolved),new_run_path(directory,resolved))
            for name in ('../outside','x/y','', '.', '..'):
                with self.assertRaises(ValueError):new_run_path(directory,resolved,save_name=name)

    def test_fresh_cwds_torch_free_import_package_discovery_and_preview(self):
        code="""
import json,sys
from cdlno.msar_lno import resolve_profile
from cdlno.msar_lno.options import resolve_training
from cdlno.msar_lno.metadata import MSARMetadata
assert 'torch' not in sys.modules
assert resolve_profile('full').d == 192
assert resolve_training('pipe', {}).architecture.num_latents[0] == 512
print('configuration import passed')
"""
        for cwd in ('PDE-Solving-StandardBenchmark','Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS'):
            p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT/cwd,env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
        with tempfile.TemporaryDirectory() as directory:
            p=subprocess.run([sys.executable,'-B','-m','cdlno.msar_lno','--task','elasticity','--profile','full',
                              '--input-tokens','972','--coverage-mode','off','--coverage-weight','0'],cwd=directory,
                             env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);data=json.loads(p.stdout)
            self.assertEqual(data['architecture']['num_latents'][0],1024)
            self.assertTrue(data['input_layout']['first_down_expands'])
            self.assertEqual(list(Path(directory).iterdir()),[])
        from setuptools import find_packages
        self.assertIn('cdlno.msar_lno',find_packages(ROOT,include=['cdlno*']))
        for p in (ROOT/'cdlno/msar_lno').glob('*.py'):ast.parse(p.read_text(),feature_version=(3,10))


if __name__=='__main__':unittest.main(verbosity=2)
