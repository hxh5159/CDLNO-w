import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from cdlno.linearno.profiles import (PROFILES, TASKS, DEFAULT_PROFILE, resolve_config,
    validate_resolved, validate_model, parse_overrides, require_resolved_objective, digest)

ROOT = Path(__file__).resolve().parents[2]


class ProfileChecks(unittest.TestCase):
    def test_all_24_match_independent_L0_inventory_and_roundtrip(self):
        audit = json.loads((ROOT/'docs/linearno_audit/l0/profile-inventory.json').read_text())['rows']
        self.assertEqual(len(audit), 24)
        for row in audit:
            with self.subTest(task=row['task'], profile=row['profile']):
                # L0 records source facts, before the user's AirfRANS objective
                # decision. Keep testing that historical interpretation too.
                c = resolve_config(row['task'], row['profile'], contract=None)
                self.assertEqual(validate_resolved(json.loads(json.dumps(c))), c)
                m = c['values']['model']; old = row['official_constructor_kwargs']
                for new, key in [('hidden','n_hidden'),('heads','n_head'),('layers','n_layers'),('ffn_ratio','mlp_ratio'),
                                 ('fun_dim','fun_dim'),('space_dim','space_dim'),('out_dim','out_dim'),('ref','ref')]:
                    self.assertEqual(m[new], old[key])
                self.assertEqual(m['linearno_rank'], row['model_spec_summary']['actual_M'])
                self.assertEqual(m['linearno_variant'], row['model_spec_summary']['variant'])
                for new, key in [('training','training_config'),('objective','objective_spec'),('evaluation','evaluation_spec'),('data','data_spec')]:
                    self.assertEqual(c['values'][new], row[key])
                self.assertNotIn('legacy_default', c['field_sources'].values())
                from cdlno.linearno.schema import validate_release_descriptor
                validate_release_descriptor(dict(class_path=row['official_class_path'],
                    constructor_kwargs=row['official_constructor_kwargs']), c)

    def test_explicit_priority_no_legacy_defaults_and_alias(self):
        old = dict(n_hidden=17, slice_num=2, mlp_ratio=4, batch_size=900)
        profile, explicit = parse_overrides(['--linearno-profile','official_release','--linearno-rank','48',
                                           '--linearno-batch-size','3','--linearno-save-name','my exact name'])
        result = resolve_config('darcy', profile, explicit=explicit, legacy_defaults=old)
        self.assertEqual(result['values']['model']['hidden'], 128)
        self.assertEqual(result['values']['model']['linearno_rank'], 48)
        self.assertEqual(result['values']['training']['batch_size'], 3)
        self.assertEqual(result['values']['runtime']['save_name'], 'my exact name')
        self.assertEqual(result['field_sources']['model.linearno_rank'], 'cli_explicit')
        self.assertEqual(result['field_sources']['model.hidden'], 'profile')
        self.assertEqual(result['field_sources']['runtime.device'], 'family_default')
        self.assertEqual(result['ignored_legacy_defaults'], old)
        self.assertEqual(parse_overrides([]), (DEFAULT_PROFILE, {}))
        self.assertEqual(resolve_config('ns', explicit={'model.qk_dim':19})['values']['model']['linearno_rank'], 19)
        with self.assertRaises(ValueError):
            resolve_config('ns', explicit={'model.qk_dim':19,'model.linearno_rank':19})

    def test_invalid_values_and_non_square_grid(self):
        for key, values in {'hidden':[True,0,-2,127,8.5], 'heads':[0,3,False],
                            'linearno_rank':[0,-1,2.5,True], 'layers':[0], 'dropout':[1,float('nan'),float('inf')],
                            'H':[0,None], 'unified_pos':[1], 'activation':['relu'], 'linearno_variant':['LinearNO']}.items():
            for value in values:
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    resolve_config('darcy', explicit={'model.'+key:value})
        for kwargs in ({'task':'burgers'}, {'task':'ns','profile':'released_eval_exact'},
                       {'task':'ns','explicit':{'slice_num':8}}, {'task':'car','explicit':{'model.linearno_rank':1}},
                       {'task':'airfrans','explicit':{'model.linearno_variant':'plain'}},
                       {'task':'ns','explicit':{'training.lr':float('nan')}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): resolve_config(**kwargs)
        c=resolve_config('darcy', explicit={'model.H':5,'model.W':7})
        validate_model('darcy',c['values']['model'],N=35)
        with self.assertRaisesRegex(ValueError,'N = H'): validate_model('darcy',c['values']['model'],N=36)
        # Unstructured N is arbitrary; M>N does not imply automatic clipping.
        c=resolve_config('elasticity');validate_model('elasticity',c['values']['model'],N=5)
        self.assertEqual(c['values']['model']['linearno_rank'],64)

    def test_industrial_pending_decisions_and_objective_axes(self):
        for task in ('car','airfrans'):
            c=resolve_config(task, contract=None)
            with self.assertRaisesRegex(ValueError,'reviewed decisions'): require_resolved_objective(c)
            official=resolve_config(task,'official_release');require_resolved_objective(official)
            self.assertEqual(official['values']['objective']['kind'],'MSE')
            self.assertEqual(c['values']['objective']['kind'],'relative_L2')
            for axis in ('prediction_sampling','field_metric','force_metric','force_input','split','aggregation'):
                self.assertIn(axis,c['values']['evaluation'])
        self.assertEqual(resolve_config('car')['derived_model']['rank_mapping']['original_value'],1)
        self.assertTrue(resolve_config('airfrans')['derived_model']['temperature']['dead_parameter'])
        # Official Car's dormant unified-position branch is a 2D replacement;
        # it must not inherit old Transolver Car's appended 3D position encoding.
        car=resolve_config('car')['derived_model']
        self.assertEqual(car['position_mode'],'replace_coordinates')
        self.assertEqual(car['grid_range'],[[0,1],[0,1]])
        with self.assertRaisesRegex(ValueError,'time condition'):
            resolve_config('airfrans',explicit={'model.time_input':True})

    def test_hash_and_resolution_provenance_cannot_be_forged_by_rehash(self):
        c=resolve_config('pipe');c['values']['model']['hidden']=64
        with self.assertRaises(ValueError): validate_resolved(c)
        c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
        with self.assertRaises(ValueError): validate_resolved(c)

    def test_schema_namespace_does_not_import_torch_or_task_factories(self):
        code="import sys; from cdlno.linearno.profiles import resolve_config; from cdlno.linearno import schema; resolve_config('ns'); assert 'torch' not in sys.modules; assert 'model_dict' not in sys.modules"
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
