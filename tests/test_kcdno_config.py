"""K1 config/provenance checks only. No exp/main imports or new-model math."""

import argparse
import ast
import copy
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cdlno.config import CDLNOArchitectureConfig
from cdlno.checkpoint import load_sidecar, save_sidecar
from cdlno.kcdno import KCDNOArchitectureConfig, KCDNOInitializationConfig, KCDNORuntimeConfig
from cdlno.kcdno.metadata import (KCDNOMetadata, KCDNOMetadataMismatch, checkpoint_family,
                                 compare_architecture, family_for_model_key, load_metadata,
                                 new_run_path, resolve_evaluation, save_metadata)
from cdlno.kcdno.options import architecture_overrides, explicit_arguments, resolve_training
from cdlno.kcdno.profiles import PROFILE_NAMES, TASKS, resolve_profile


ROOT = Path(__file__).resolve().parents[1]


def original_parser(relative):
    # Use only original argparse declarations; exp/main never get imported.
    nodes = []
    for node in ast.parse((ROOT / relative).read_text()).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'parser':
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'parser.add_argument':
            nodes.append(node)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-parser-only>', 'exec'), scope)
    parser = scope['parser']
    # Future integration contract, local test parser ONLY. No production entry edits.
    parser.add_argument('--profile', choices=PROFILE_NAMES, default='kcdno_v1')
    parser.add_argument('--kernel-rank', type=int, default=16)
    parser.add_argument('--history-mode', choices=('all', 'off'), default='all')
    parser.add_argument('--front-blocks', type=int, default=2)
    return parser


def metadata(task='darcy', profile='kcdno_v1', **changes):
    format = 'whole_model' if task == 'car' else ('model_list' if task == 'airfrans' else 'state_dict')
    return KCDNOMetadata(task, resolve_profile(task, profile, **changes), format, profile)


class KCDNOConfigTests(unittest.TestCase):
    def test_default_and_no_cdlno_fields(self):
        cfg = KCDNOArchitectureConfig()
        self.assertEqual((cfg.family, cfg.L, cfg.kernel_rank, cfg.history_mode), ('kcdno', 8, 16, 'all'))
        self.assertEqual((cfg.ffn1_hidden, cfg.ffn2_hidden, cfg.point_hidden), (256, 256, 256))
        self.assertEqual(cfg.gate_parameterization, 'unconstrained_scalar')
        self.assertEqual(cfg.norm_eps, cfg.kernel_clamp)
        self.assertEqual(cfg.kernel_denominator_eps, 1e-6)
        init = KCDNOInitializationConfig()
        self.assertEqual((init.gamma_init, init.scorer_init, init.norm_scale_init), (.1, 0, 1))
        self.assertEqual(init.kernel_init, 'xavier_uniform_gain1')
        self.assertFalse(cfg.kernel_projection_bias)
        for name in ('F', 'P', 'front_latent_mode', 'cdpa_mode', 'rear_depth'):
            self.assertNotIn(name, cfg.to_dict())
            with self.assertRaises(TypeError):
                KCDNOArchitectureConfig(**{name: 2})
        self.assertEqual(CDLNOArchitectureConfig().P, 6)

    def test_integers_and_unsupported_behaviors(self):
        for name in ('L', 'd', 'h', 'M', 'kernel_rank', 'ffn1_hidden', 'ffn2_hidden', 'point_hidden'):
            for value in (0, -1, True, 2.0, '2', float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    KCDNOArchitectureConfig(**{name: value})
        bad = dict(family='CDLNO', architecture_version='future', history_mode='entry',
                   latent_activation='geglu', point_activation='relu', point_module='depthwise',
                   norm='layernorm', qk_norm='none', output_norm='rmsnorm', norm_eps=1e-5,
                   kernel_phi='softmax', kernel_clamp=0., kernel_denominator_eps=0.,
                   kernel_projection_bias=True, gate_parameterization='sigmoid', attention_dropout=.1)
        for name, value in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                KCDNOArchitectureConfig(**{name: value})
        with self.assertRaisesRegex(ValueError, 'divisible'):
            KCDNOArchitectureConfig(d=127)
        # r is NOT a head dimension; no artificial rank/head or F<=6 restriction.
        cfg = KCDNOArchitectureConfig(L=1, d=12, h=3, M=1, kernel_rank=5, history_mode='off')
        self.assertEqual(cfg.ffn1_hidden, 24)
        self.assertEqual(KCDNOArchitectureConfig(L=16).L, 16)

    def test_two_profiles_all_eight_tasks(self):
        main = [(128,8,64,'conv_ffn'), (128,8,64,'point_ffn'), (128,4,64,'conv_ffn'),
                (128,4,32,'conv_ffn'), (256,8,64,'conv_ffn'), (128,8,64,'conv_ffn'),
                (256,8,64,'point_ffn'), (256,8,64,'point_ffn')]
        for task, (d,h,m,point) in zip(TASKS, main):
            for profile in PROFILE_NAMES:
                cfg = resolve_profile(task, profile)
                want_h = 8 if profile == 'transolver_shape_match' and task in ('airfoil','pipe') else h
                want_m = (64 if task == 'pipe' else 32) if profile == 'transolver_shape_match' and task in ('pipe','ns','car','airfrans') else m
                self.assertEqual((cfg.d,cfg.h,cfg.M,cfg.point_module), (d,want_h,want_m,point))
                self.assertEqual((cfg.L,cfg.kernel_rank,cfg.ffn1_hidden,cfg.ffn2_hidden,cfg.point_hidden), (8,16,2*d,2*d,2*d))
        with self.assertRaises(ValueError): resolve_profile('unknown')
        with self.assertRaises(ValueError): resolve_profile('darcy', 'unknown')

    def test_explicit_cli_uses_argparse_and_keeps_original_parser(self):
        parser = original_parser('PDE-Solving-StandardBenchmark/exp_darcy.py')
        defaults = vars(parser.parse_args([])).copy()
        parser.set_defaults(gpu='3')
        self.assertEqual(explicit_arguments(parser, []), {})
        default_cfg = resolve_training('darcy', explicit_arguments(parser, []))
        self.assertEqual((default_cfg.L,default_cfg.d,default_cfg.h,default_cfg.M), (8,128,8,64))
        explicit = explicit_arguments(parser, ['--model','kcdno','--profile','transolver_shape_match',
                                              '--n-layers','12','--n-hidden','64','--n-heads','8',
                                              '--slice_num=80','--slice_num','96','--kernel-rank','7','--history-mode','off'])
        cfg = resolve_training('pipe', explicit)
        self.assertEqual((cfg.L,cfg.d,cfg.h,cfg.M,cfg.kernel_rank,cfg.history_mode), (12,64,8,96,7,'off'))
        self.assertEqual((cfg.ffn1_hidden,cfg.ffn2_hidden,cfg.point_hidden), (128,128,128))
        now = vars(parser.parse_args([]))
        self.assertEqual({k:v for k,v in now.items() if k!='gpu'}, {k:v for k,v in defaults.items() if k!='gpu'})
        self.assertEqual(now['gpu'], '3')
        # Profile switch is effective when dimensions were not explicitly passed.
        for task in ('pipe','ns','car','airfrans'):
            cfg = resolve_training(task, explicit_arguments(parser, ['--profile','transolver_shape_match']))
            self.assertEqual(cfg.M, 64 if task=='pipe' else 32)

    def test_three_original_parser_styles_and_forbidden_explicit_options(self):
        for entry in ('PDE-Solving-StandardBenchmark/exp_darcy.py', 'Car-Design-ShapeNetCar/main.py',
                      'Airfoil-Design-AirfRANS/main.py'):
            with self.subTest(entry=entry):
                parser = original_parser(entry)
                self.assertEqual(explicit_arguments(parser, []), {})
                flag = '--cfd_model' if entry.startswith('Car-Design') else '--model'
                cfg = resolve_training('car', explicit_arguments(parser, [flag,'kcdno']))
                self.assertEqual((cfg.L,cfg.d,cfg.M), (8,256,64))
                with self.assertRaisesRegex(ValueError, 'not applicable'):
                    resolve_training('car', explicit_arguments(parser, ['--front-blocks','2']))
        for key in ('F','P','front_blocks','front_latent_mode','rear_depth','latent_blocks',
                    'cdpa_mode','cdpa_source_chunk_size','latent_ffn_ratio','mlp_ratio'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not applicable'):
                architecture_overrides({key: 2})
        with self.assertRaisesRegex(ValueError, 'conflicting explicit aliases'):
            architecture_overrides({'L':8,'n_layers':3})
        self.assertEqual(architecture_overrides({'L':8,'n_layers':8}), {'L':8})
        with self.assertRaises(ValueError): resolve_training('car', {'model':'CDLNO'})
        with self.assertRaises(ValueError): resolve_training('car', {'cfd_model':'CDLNO'})

    def test_complete_configuration_roundtrips_and_rejects_missing_fields(self):
        for cls in (KCDNOArchitectureConfig, KCDNOInitializationConfig, KCDNORuntimeConfig):
            value = cls()
            self.assertEqual(cls.from_dict(json.loads(json.dumps(value.to_dict()))), value)
            for key in value.to_dict():
                incomplete = value.to_dict(); del incomplete[key]
                with self.subTest(cls=cls, key=key), self.assertRaises(ValueError):
                    cls.from_dict(incomplete)
            with self.assertRaises(ValueError): cls.from_dict(dict(value.to_dict(), accidental='ignored'))
        for key in ('ffn1_hidden','ffn2_hidden','point_hidden'):
            with self.assertRaises(ValueError):
                KCDNOArchitectureConfig.from_dict(dict(KCDNOArchitectureConfig().to_dict(), **{key:None}))

    def test_metadata_formats_roundtrip_for_all_profiles_and_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in TASKS:
                for profile in PROFILE_NAMES:
                    saved = metadata(task, profile)
                    path = Path(directory)/task/profile/'architecture.json'
                    save_metadata(path, saved)
                    self.assertEqual(load_metadata(path), saved)
                    raw = json.loads(path.read_text())
                    self.assertEqual(checkpoint_family(raw), 'kcdno')
                    self.assertEqual(raw['architecture']['family'], 'kcdno')
                    self.assertEqual(raw['architecture']['point_hidden'], 2*saved.architecture.d)
            member = replace(metadata('airfrans'), checkpoint_format='whole_model')
            self.assertEqual(KCDNOMetadata.from_dict(member.to_dict()), member)
            with self.assertRaises(ValueError): replace(metadata('car'), checkpoint_format='state_dict')

    def test_eval_reconstructs_saved_config_before_explicit_compare(self):
        parser = original_parser('PDE-Solving-StandardBenchmark/exp_darcy.py')
        saved = metadata('pipe', 'transolver_shape_match', L=12, kernel_rank=32, history_mode='off')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'architecture.json'; save_metadata(path,saved)
            before = path.read_bytes()
            for tokens in ([], ['--profile','kcdno_v1'], ['--kernel-rank','32','--history-mode','off']):
                resolved = resolve_evaluation(path,explicit_arguments(parser,tokens),task='pipe')
                self.assertEqual(resolved.saved, saved)
                self.assertEqual(resolved.runtime_differences, ())
                self.assertEqual(path.read_bytes(), before)
            for name, value in (('L',8), ('M',32), ('kernel_rank',16), ('history_mode','all'), ('family','CDLNO'), ('latent_activation','relu')):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    resolve_evaluation(path, {name:value}, task='pipe')
                self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(KCDNOMetadataMismatch): resolve_evaluation(path,{},task='darcy')
            with self.assertRaises(FileNotFoundError):
                resolve_evaluation(Path(directory)/'missing.json', {'F':-1}, task='pipe')
            with self.assertRaises(FileExistsError): save_metadata(path,metadata())
            self.assertEqual(path.read_bytes(), before)

    def test_reproduction_and_runtime_are_not_structure(self):
        saved = metadata()
        different = replace(saved, profile='transolver_shape_match', initialization=KCDNOInitializationConfig(
            initialization_version='recorded-other-init', gamma_init=-.2, seed=42))
        self.assertEqual(compare_architecture(saved.architecture,different.architecture), [])
        runtime = KCDNORuntimeConfig(device='cpu',dtype='bfloat16',batch_size=2,sdpa_backend='math',amp=True,tf32=True,compile=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'architecture.json'; save_metadata(path,different)
            before=path.read_bytes()
            result=resolve_evaluation(path,{},task='darcy',runtime=runtime)
            self.assertEqual(result.saved.initialization.gamma_init,-.2)
            self.assertEqual(result.saved.initialization.initialization_version,'recorded-other-init')
            self.assertEqual(set(result.runtime_differences), {f.name for f in fields(runtime)})
            self.assertEqual(result.runtime,runtime)
            self.assertEqual(path.read_bytes(),before)
        for change in (dict(gamma_init=float('nan')),dict(gamma_init=True),dict(seed=-1)):
            with self.assertRaises(ValueError): KCDNOInitializationConfig(**change)
        for change in (dict(batch_size=True),dict(batch_size=0),dict(amp=1),dict(device='')):
            with self.assertRaises(ValueError): KCDNORuntimeConfig(**change)

    def test_missing_family_stays_legacy_and_never_guessed(self):
        self.assertIsNone(checkpoint_family({'architecture':{'family':'kcdno'}}))
        self.assertEqual(family_for_model_key('kcdno'),'kcdno')
        for name in ('CDLNO','Transolver','Transolver_Structured_Mesh_2D','unknown'):
            self.assertIsNone(family_for_model_key(name))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'architecture.json'
            save_sidecar(path,CDLNOArchitectureConfig())
            before=path.read_bytes()
            self.assertIsNone(checkpoint_family(json.loads(before)))
            self.assertEqual(load_sidecar(path)['architecture']['model_name'],'CDLNO')
            with self.assertRaises(KCDNOMetadataMismatch): load_metadata(path)
            self.assertEqual(path.read_bytes(),before)
            # The exact existing pre-A1 JSON exception remains in the old loader.
            legacy=json.loads(before);del legacy['architecture']['front_latent_mode']
            legacy['architecture']['history_rule']='front-t-after-ffn2-before-up-v1'
            path.write_text(json.dumps(legacy));before=path.read_bytes()
            self.assertIsNone(checkpoint_family(legacy))
            self.assertEqual(load_sidecar(path)['architecture']['front_latent_mode'],'full')
            self.assertEqual(path.read_bytes(),before)

    def test_tampered_new_sidecar_fields_rejected(self):
        original=metadata().to_dict()
        for key in original:
            bad=copy.deepcopy(original);del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError): KCDNOMetadata.from_dict(bad)
        for family in ('CDLNO','transolver',None,'',False):
            with self.subTest(family=family), self.assertRaises(ValueError):
                KCDNOMetadata.from_dict(dict(original,family=family))
        for section,key,value in (('architecture','family','CDLNO'),('architecture','kernel_rank',0),
                                  ('architecture','history_mode','window'),('architecture','latent_activation','geglu')):
            bad=copy.deepcopy(original);bad[section][key]=value
            with self.assertRaises(ValueError): KCDNOMetadata.from_dict(bad)
        bad=copy.deepcopy(original);bad['schema_version']=True
        with self.assertRaises(ValueError): KCDNOMetadata.from_dict(bad)

    def test_paths_are_new_family_only_and_no_constructor_placeholder(self):
        import cdlno.kcdno as package
        self.assertFalse(hasattr(package,'KCDNO'))
        with tempfile.TemporaryDirectory() as directory:
            for task in TASKS:
                path=new_run_path(directory,task,resolve_profile(task))
                self.assertEqual(path.parent,Path(directory)/task/'kcdno')
                self.assertIn('_L8_',path.name)
                self.assertIn('_r16_all',path.name)
                self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_three_fresh_cwd_config_imports_and_schema_loading_without_torch(self):
        script='''
import sys
from cdlno.kcdno import KCDNOArchitectureConfig
from cdlno.kcdno.metadata import load_metadata, resolve_evaluation
x=load_metadata(sys.argv[1])
assert resolve_evaluation(sys.argv[1],{},task='darcy').saved == x
assert 'torch' not in sys.modules
assert not any(k in sys.modules for k in ('main','main_evaluation','exp_darcy','model_dict'))
print('new config import and metadata load passed; no model or torch imported')
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'architecture.json'; save_metadata(path,metadata())
            for project in ('PDE-Solving-StandardBenchmark','Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS'):
                env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
                result=subprocess.run([sys.executable,'-B','-c',script,str(path)],cwd=ROOT/project,
                                      env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
