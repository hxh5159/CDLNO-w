import copy
import json
from pathlib import Path
import unittest
from linearno_loop.v3.config import resolve_config, validate_config, run_directory_id
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR, V3SchemaError

ROOT=Path(__file__).resolve().parents[2]
def cfg(task='darcy', **options):
    return resolve_config(task, options={'architecture':ARCHITECTURE_SELECTOR,**options})

class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_effective_width_grid(self):
        c=cfg();s=c['loop_spec'];m=c['profile_spec']['values']['model'];k=c['model_spec']['constructor_kwargs']
        self.assertEqual([s[x] for x in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks')],[2,4,2,2])
        self.assertEqual((s['hidden_width'],s['latent_width'],s['actual_M']),(104,704,64))
        self.assertEqual(m['hidden_width'],104);self.assertEqual(k['hidden_width'],104)
        self.assertEqual((m['grid_height'],k['grid_height']),(85,85))
        self.assertNotIn('hidden',m);self.assertNotIn('H',k)
        self.assertEqual(validate_config(json.loads(json.dumps(c))),c)
    def test_architecture_is_required(self):
        for o in ({},{'cost_profile':'matched_v1'},{'architecture':'loop_linearno_v1'}):
            with self.assertRaises(V3SchemaError):resolve_config('darcy',options=o)
    def test_profile_conflicts_and_strict_types(self):
        for o in ({'hidden_width':128},{'latent_width':512},{'actual_M':128},{'heads':4},
                  {'executed_depth':16},{'executed_depth':True},{'latent_enabled':1},
                  {'adapter_rank':True},{'adapter_alpha':float('nan')},{'adapter_alpha':'4'},
                  {'hidden_width':104.0},{'unknown':0},{'linearno_latent_attnres':False}):
            with self.subTest(o=o),self.assertRaises(V3SchemaError):cfg(**o)
    def test_custom_car_and_old_validation(self):
        c=cfg('car',cost_profile='custom',hidden_width=200,latent_width=30,actual_M=31,
              topology_preset='custom',prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,
              suffix_blocks=1,adapter_mode='none')
        self.assertEqual(c['loop_spec']['head_dim'],25)
        self.assertEqual(c['loop_spec']['actual_M'],31)
        with self.assertRaises(V3SchemaError):cfg('car',cost_profile='custom',hidden_width=201,latent_width=30,actual_M=31,executed_depth=12)
    def test_resealed_derived_forgery_fails(self):
        from linearno_loop.v3.contracts import seal
        c=cfg();c['loop_spec']['unique_depth']=999
        with self.assertRaises(V3SchemaError):validate_config(seal(c,'config_hash'))

class StrictBoundaryTests(unittest.TestCase):
    def test_checkpoint_version_explicit(self):
        from linearno_loop.v3.schema import restore_config
        from .support import metadata
        c=cfg();self.assertEqual(c['checkpoint_version'],3)
        m=metadata(c);self.assertEqual(m['checkpoint_version'],3)
        with self.assertRaises(ValueError):restore_config(m,explicit={'checkpoint_version':2})
    def test_profile_conflicts_require_custom_and_do_not_touch_grid(self):
        for task in ('darcy','airfoil','elasticity','pipe','ns','plasticity','car','airfrans'):
            c=cfg(task);before=c['base_profile_spec']['values']['model']
            for key in ('model.hidden','model.H','model.W','model.layers','model.heads','model.linearno_rank','data.sampling'):
                with self.assertRaises(ValueError):resolve_config(task,options={'architecture':ARCHITECTURE_SELECTOR},profile_overrides={key:17})
            self.assertEqual(c['loop_spec']['grid_height'],before['H'])
            self.assertEqual(c['loop_spec']['grid_width'],before['W'])
    def test_profile_explicit_assertion_provenance(self):
        c=cfg(hidden_width=104,latent_width=704)
        self.assertEqual(c['field_sources']['explicit_profile_assertions'],{'hidden_width':104,'latent_width':704})
    def test_custom_preset_and_even_depth_rules(self):
        common=dict(cost_profile='custom',hidden_width=96,latent_width=40)
        for preset,unique,executed in [('p1_c3_r2_s1',5,8),('p2_c2_r2_s2',6,8)]:
            c=cfg(**common,topology_preset=preset)
            self.assertEqual((c['loop_spec']['unique_depth'],c['loop_spec']['executed_depth']),(unique,executed))
        for depth in (6,10,16,40):
            c=cfg(**common,executed_depth=depth)
            self.assertEqual(c['loop_spec']['recurrent_core_blocks'],(depth-4)//2)
        for o in [common,{**common,'executed_depth':5},{**common,'executed_depth':7},
                  {**common,'topology_preset':'custom','prefix_blocks':2},
                  {**common,'topology_preset':'p2_c2_r2_s2','prefix_blocks':2},
                  {**common,'topology_preset':'d12','executed_depth':12}]:
            with self.assertRaises(ValueError):cfg(**o)
    def test_derived_fields_cannot_be_supplied(self):
        for key in ('unique_depth','head_dim','grid_height','grid_width','rank_multiplier','schema_version','config_hash'):
            with self.assertRaises(ValueError):cfg(**{key:1})
    def test_explicit_null_topology_is_not_an_omitted_default(self):
        with self.assertRaisesRegex(ValueError,'topology'):
            cfg(topology_preset=None)
    def test_all_structural_integer_fields_reject_nonintegers(self):
        for key in ('hidden_width','latent_width','actual_M','heads','adapter_rank','executed_depth'):
            for value in (True,False,1.0,'1',None,0,-1,float('inf')):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):cfg(**{key:value})
        for key in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'):
            for value in (True,1.0,'2',None,-1):
                options=dict(cost_profile='custom',hidden_width=96,latent_width=24,topology_preset='custom',
                             prefix_blocks=2,recurrent_core_blocks=2,loop_repeats=2,suffix_blocks=2)
                options[key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):cfg(**options)
