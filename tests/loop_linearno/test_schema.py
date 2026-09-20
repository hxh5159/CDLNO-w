import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

from linearno_loop.contracts import LoopSchemaError, RESIDUAL_MODES, canonical_json, seal
from linearno_loop.schema import (
    validate_metadata, restore_config, read_metadata, write_metadata, validate_constructor,
)
from linearno_loop.state import numeric, inspect_state
from loop_linearno.support import config, metadata, rehash


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = metadata()

    def rejects(self, obj, message=None):
        with self.assertRaisesRegex(LoopSchemaError, message or '.'):
            validate_metadata(rehash(obj))

    def test_three_wrapper_contracts_three_modes_roundtrip_without_model(self):
        for task in ('darcy','airfrans','car'):
            for mode in RESIDUAL_MODES:
                m=metadata(config(task,residual_mode=mode))
                restored=validate_metadata(json.loads(canonical_json(m)))
                self.assertEqual(restored,m)
                self.assertEqual(restore_config(m)['config'],m['resolved_config'])
                self.assertEqual(m['model_spec']['constructor_kwargs']['linearno_rank'],
                                 m['loop_spec']['resolved_rank'])
                self.assertNotIn('n_layers',m['model_spec']['constructor_kwargs'])
                self.assertTrue(m['load_policy']['strict'])

    def test_closed_metadata_required_fields_and_legacy_rejection(self):
        for key in self.saved:
            m=copy.deepcopy(self.saved);del m[key]
            # A missing seal is rejected as well, without synthesizing one.
            with self.subTest(key=key), self.assertRaises(LoopSchemaError): validate_metadata(m)
        m=copy.deepcopy(self.saved);m['unexpected']=1;self.rejects(m,'unknown')
        for family in ('linearno','linearno_history','transolver',None):
            m=copy.deepcopy(self.saved);m['family']=family;self.rejects(m,'family')
        m=copy.deepcopy(self.saved);m['schema_version']=True;self.rejects(m,'schema_version')
        m=copy.deepcopy(self.saved);m['load_policy']['strict']=False;self.rejects(m,'strict')

    def test_rehashed_structure_protocol_and_provenance_corruption(self):
        for name,value in (('variant','plain'),('resolved_rank',64),('head_dim',32),
                ('loop_repeats',3),('residual_mode','sr_1_over_r'),('schema_version',2)):
            m=copy.deepcopy(self.saved);m['loop_spec'][name]=value
            with self.subTest(name=name):self.rejects(m,name)
        for section,field,value in (
            ('model_spec','class_path','other.Model'),
            ('objective_spec','unexpected','loss'),('evaluation_spec','unexpected','test'),
            ('data_spec','scope','unknown'),('provenance_spec','dirty',1),
            ('provenance_spec','linearno_sha','0'*40),('provenance_spec','source_sha256','bad'),
            ('provenance_spec','command',[]),('provenance_spec','environment',{})):
            m=copy.deepcopy(self.saved);m[section][field]=value
            with self.subTest(section=section,field=field):self.rejects(m)
        m=copy.deepcopy(self.saved);m['model_spec']['constructor_kwargs']['executed_depth']=8
        self.rejects(m,'executed_depth')
        m=copy.deepcopy(self.saved);m['data_spec']['protocol']['invented']=True
        self.rejects(m,'protocol')

    def test_restore_assertions_accumulate_and_runtime_does_not_override(self):
        before=canonical_json(self.saved)
        result=restore_config(self.saved, explicit={'linearno_loop':True,'linearno_rank':128,
            'prefix_blocks':1,'head_dim':16,'variant':'conv_temp','family':'linearno_loop'},
            runtime={'device':'cuda:1','experiment_dir':'remote/output/run'})
        self.assertEqual(result['config'],self.saved['resolved_config'])
        self.assertEqual(set(result['runtime_changes']),{'device','experiment_dir'})
        with self.assertRaises(LoopSchemaError) as raised:
            restore_config(self.saved,explicit={'residual_mode':'sr_1_over_r','linearno_rank':64,'loop_repeats':3})
        for key in ('residual_mode','linearno_rank','loop_repeats'):
            self.assertIn('explicit.'+key,str(raised.exception))
        for invalid in (False,0,1,None,'true'):
            with self.assertRaises(LoopSchemaError):restore_config(self.saved,strict=invalid)
        for explicit in ({'rank_multiplier':2,'linearno_rank':128},{'unknown':1},
            {'linearno_latent_attnres':False},{'linearno_history_k_conditioning':False},
            {'linearno_attnres_history_dropout_p':None},{'linearno_loop':1}):
            with self.assertRaises(LoopSchemaError):restore_config(self.saved,explicit=explicit)
        with self.assertRaises(LoopSchemaError):restore_config(self.saved,runtime={'epochs':1})
        self.assertEqual(before,canonical_json(self.saved))

    def test_normalizer_numeric_bytes_shapes_dtype_hash_and_fit_evidence(self):
        from cdlno.linearno.schema import numerical_state
        import torch
        import numpy as np
        for tensor in (torch.tensor([1.5],dtype=torch.bfloat16),np.array([1.,2.],dtype='>f8'),
                       torch.tensor(4,dtype=torch.int64),np.array([],dtype=np.float32)):
            spec=numerical_state(tensor)
            self.assertEqual(numeric(spec,'fixture').spec,spec)
        for field,value in (('dtype','object'),('shape',[True]),('shape',[999]),
                            ('data','not-base64'),('backend','pickle')):
            m=copy.deepcopy(self.saved)
            s=m['normalizer_spec']['records']['input']['states']['mean'];s[field]=value
            m['normalizer_spec']['records']['input']['states']['mean']=seal(s,'sha256')
            with self.subTest(field=field):self.rejects(m)
        m=copy.deepcopy(self.saved)
        m['normalizer_spec']['records']['input']['states']['mean']['data']=base64.b64encode(b'changed').decode()
        self.rejects(m,'checksum')
        for key,value in (('fit_split',''),('data_checksum','0'*64),('states',{})):
            m=copy.deepcopy(self.saved);m['normalizer_spec']['records']['input'][key]=value;self.rejects(m)
        m=copy.deepcopy(self.saved);m['normalizer_spec']={'policy':'none','records':{}}
        self.assertEqual(validate_metadata(rehash(m))['normalizer_spec']['policy'],'none')

    def test_resume_state_required_rng_generator_scheduler_and_selection(self):
        from cdlno.linearno.schema import pack_state, unpack_state
        for key in self.saved['resume_state']:
            m=copy.deepcopy(self.saved);del m['resume_state'][key];self.rejects(m,key)
        for key,value in (('epoch',True),('global_step',-1),('optimizer',pack_state({})),
                          ('scheduler',pack_state({})),('sampler_state',pack_state(None)),
                          ('dataloader_generators',pack_state({'train':{'device':'cpu','state':None}}))):
            m=copy.deepcopy(self.saved);m['resume_state'][key]=value;self.rejects(m)
        for field in ('python','numpy','torch_cpu','torch_cuda'):
            m=copy.deepcopy(self.saved);rng=unpack_state(m['resume_state']['rng']);rng[field]=None
            m['resume_state']['rng']=pack_state(rng);self.rejects(m,'rng')
        m=copy.deepcopy(self.saved);r=m['resume_state']
        r.update(checkpoint_role='best_validation',selection_split='test',selection_metric='loss')
        self.rejects(m,'selection_split')
        r['selection_split']='validation';validate_metadata(rehash(m))
        r.update(checkpoint_role='final');self.rejects(m,'selection')

    def test_state_encoding_rejects_duplicate_keys_and_unknown_types(self):
        from cdlno.linearno.schema import pack_state
        node=pack_state({'a':1});node['value'].append(node['value'][0])
        with self.assertRaisesRegex(LoopSchemaError,'duplicate'):inspect_state(node)
        for node in ({'type':'pickle','value':'x'},{'type':'scalar','value':[]},
                     {'type':'numeric','value':{}},{'type':'tuple','value':None}):
            with self.assertRaises(LoopSchemaError):inspect_state(node)

    def test_ensemble_member_order_hash_safe_paths_and_state_dict_only(self):
        m=copy.deepcopy(self.saved)
        member=dict(member_id='member_000',order=0,path='member_000/weights.pt',sha256='1'*64,format='state_dict')
        m['ensemble_manifest']=[member];validate_metadata(rehash(m))
        for key,value in (('order',True),('order',1),('sha256','no'),('format','whole_object'),
                          ('path','../weights.pt'),('path','/tmp/weights.pt'),('path','a\\b'),('path','.')):
            bad=copy.deepcopy(m);bad['ensemble_manifest'][0][key]=value;self.rejects(bad)
        m['ensemble_manifest'].append({**member,'order':1});self.rejects(m,'id/order')

    def test_disk_roundtrip_is_create_only_and_bad_json_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.json'
            write_metadata(path,self.saved);before=path.read_bytes()
            self.assertEqual(read_metadata(path),self.saved)
            with self.assertRaises(FileExistsError):write_metadata(path,self.saved)
            self.assertEqual(before,path.read_bytes())
            with self.assertRaises(FileNotFoundError):write_metadata(Path(directory)/'missing/meta.json',self.saved)
            for raw in ('{"family":1,"family":2}','{"x":NaN}'):
                path.write_text(raw)
                with self.assertRaises(LoopSchemaError):read_metadata(path)

    def test_callable_contract_inspection_does_not_execute_callable(self):
        # This is a sentinel function for inspect.signature, never a factory/model.
        def signature_only(*, prefix_blocks, suffix_blocks):
            raise AssertionError('schema must never invoke a constructor')
        spec={'class_path':signature_only.__module__+'.'+signature_only.__qualname__,
              'constructor_kwargs':{'prefix_blocks':1,'suffix_blocks':1}}
        validate_constructor(spec,signature_only)
        spec['constructor_kwargs']['unknown']=1
        with self.assertRaisesRegex(LoopSchemaError,'fields'):validate_constructor(spec,signature_only)


if __name__=='__main__':unittest.main()
