import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .support import config,metadata,packed
from linearno_loop.v3.contracts import canonical_json,seal,V3SchemaError,ARCHITECTURE_SELECTOR
from linearno_loop.v3.schema import validate_metadata,restore_config,write_metadata,read_metadata,validate_constructor

class SchemaTests(unittest.TestCase):
    def test_all_wrappers_features_modes_roundtrip(self):
        for task in ('darcy','airfrans','car'):
            for mode in ('sr_1_over_r','rb_attnres','lb_attnres_1_over_r'):
                for latent in (True,False):
                    m=metadata(config(task,residual_mode=mode,latent_enabled=latent))
                    self.assertEqual(validate_metadata(json.loads(canonical_json(m))),m)
                    self.assertEqual(restore_config(m)['config'],m['resolved_config'])
                    self.assertEqual(restore_config(m,explicit={'architecture':ARCHITECTURE_SELECTOR,'adapter_alpha':4})['config'],m['resolved_config'])
    def test_saved_restore_never_resolves_current_profiles(self):
        m=metadata()
        with patch('linearno_loop.v3.config.resolve_config',side_effect=AssertionError('resolver called')),\
             patch('linearno_loop.v3.config._profile',side_effect=AssertionError('profile called')),\
             patch('linearno_loop.v3.config.width_for',side_effect=AssertionError('table called')):
            self.assertEqual(restore_config(m)['config'],m['resolved_config'])
    def test_complete_fields_hash_versions_and_resealed_tampering(self):
        m=metadata()
        for k in m:
            x=copy.deepcopy(m);del x[k]
            with self.subTest(missing=k),self.assertRaises(ValueError):validate_metadata(x)
        for section,key,val in [(None,'family','linearno'),(None,'architecture_extension','loop_linearno_ffn_v2'),
                (None,'schema_version',True),(None,'checkpoint_format','linearno-loop-epoch-pair-v1'),
                ('loop_spec','hidden_width',128),('loop_spec','residual_mode','rb_attnres'),
                ('model_spec','class_path','bad.Model'),('load_policy','strict',False),
                ('data_spec','scope','fake'),('provenance_spec','dirty',1)]:
            x=copy.deepcopy(m);(x if section is None else x[section])[key]=val
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
    def test_explicit_conflicts_and_runtime_only_changes(self):
        m=metadata();before=canonical_json(m)
        invalid=[{'family':'linearno'},{'architecture':'operator_latent_adapter_v2'},{'cost_profile':'efficient_v1'},
                 {'hidden_width':128},{'latent_width':1},{'actual_M':128},{'heads':4},{'task':'pipe'},
                 {'residual_mode':'rb_attnres'},{'recurrent_core_blocks':2},{'adapter_mode':'none'},
                 {'latent_enabled':False},{'adapter_rank':8},{'adapter_alpha':8},{'schema_version':2},
                 {'core_ffn_mode':'round_specific'},{'linearno_latent_attnres':False},{'unknown':0}]
        for fields in invalid:
            with self.subTest(fields=fields),self.assertRaises(ValueError):restore_config(m,explicit=fields)
        for strict in (False,1,'true',None):
            with self.assertRaises(ValueError):restore_config(m,strict=strict)
        with self.assertRaises(ValueError):restore_config(m,runtime={'epochs':1})
        a=restore_config(m,runtime={'device':'cuda:1','experiment_dir':'run with spaces'})
        self.assertEqual(set(a['runtime_changes']),{'device','experiment_dir'});self.assertEqual(before,canonical_json(m))
    def test_measurement_pending_scaler_resume_and_normalizer_contracts(self):
        m=metadata()
        x=copy.deepcopy(m);x['resume_state']['epoch']=1
        with self.assertRaisesRegex(ValueError,'measurement'):validate_metadata(seal(x,'metadata_hash'))
        for field in m['resume_state']:
            x=copy.deepcopy(m);del x['resume_state'][field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
        for key,val in [('optimizer',packed({})),('scheduler',packed({})),('rng',packed({})),('scaler',packed(2)),
                        ('dataloader_generators',packed({'train':{'device':'cpu','state':None}})),('sampler_state',packed(None))]:
            x=copy.deepcopy(m);x['resume_state'][key]=val
            with self.subTest(field=key),self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
        x=copy.deepcopy(m);x['parameter_measurement']['total']=123
        with self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
        x=copy.deepcopy(m);x['resume_state']['scaler']=packed({'scale':65536.0})
        validate_metadata(seal(x,'metadata_hash'))
        x=copy.deepcopy(m);x['normalizer_spec']={'policy':'saved_train_fit','records':{}}
        with self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
    def test_create_only_files_and_duplicate_nan_json(self):
        m=metadata()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'architecture.json';write_metadata(p,m)
            self.assertEqual(read_metadata(p),m)
            with self.assertRaises(FileExistsError):write_metadata(p,m)
            for raw in ('{"family":1,"family":2}','{"alpha":NaN}'):
                p.write_text(raw)
                with self.assertRaises(ValueError):read_metadata(p)
    def test_constructor_is_inspected_never_invoked(self):
        c=config()['model_spec'];scope={}
        args=','.join(c['constructor_kwargs'])
        exec('def Sentinel(*,'+args+'):\n raise AssertionError("invoked")',scope)
        fn=scope['Sentinel'];fn.__module__=c['class_path'].rsplit('.',1)[0];fn.__qualname__=c['class_path'].rsplit('.',1)[1]
        validate_constructor(c,fn)
        x=copy.deepcopy(c);x['constructor_kwargs']['extra']=1
        with self.assertRaises(ValueError):validate_constructor(x,fn)
    def test_ensemble_safe_order_and_state_dict_only(self):
        m=metadata(config('airfrans'));row={'member_id':'member_000','order':0,'path':'member_000/weights/epoch_0001.pt','sha256':'4'*64,'format':'state_dict'}
        m['ensemble_manifest']=[row];validate_metadata(seal(m,'metadata_hash'))
        for key,value in [('path','../a.pt'),('path','/tmp/a.pt'),('order',True),('format','whole_object'),('sha256','bad')]:
            x=copy.deepcopy(m);x['ensemble_manifest'][0][key]=value
            with self.assertRaises(ValueError):validate_metadata(seal(x,'metadata_hash'))
