"""Metadata-only roundtrips, not an unimplemented LinearNO model success path."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from cdlno.linearno.profiles import resolve_config, digest
from cdlno.linearno.schema import (make_metadata, validate_metadata, validate_model_spec, checkpoint_family,
    numerical_state, restore_numerical_state, normalizer_record, pack_state, unpack_state,
    read_metadata, write_metadata, compare_metadata)
from cdlno.training_state import _same

ROOT=Path(__file__).resolve().parents[2]


def sample_metadata(explicit=None):
    profile=resolve_config('darcy','official_release',explicit=explicit)
    # Actual official constructor descriptor from L0, never instantiate or load it.
    row=next(r for r in json.loads((ROOT/'docs/linearno_audit/l0/profile-inventory.json').read_text())['rows']
             if r['task']=='darcy' and r['profile']=='official_release')
    row['official_constructor_kwargs']['key_ratio']=profile['values']['model']['linearno_rank']
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=[])
    data_hash=hashlib.sha256(b'synthetic L1 fixture, no dataset').hexdigest()
    profile_values=profile['values']
    return make_metadata(profile_spec=profile,
        model_spec=dict(class_path=row['official_class_path'],constructor_kwargs=row['official_constructor_kwargs']),
        data_spec=dict(split='synthetic training fixture only',checksums={'synthetic':data_hash},sampling='fixed synthetic array'),
        objective_spec=profile_values['objective'],evaluation_spec=profile_values['evaluation'],
        provenance_spec=dict(target_sha='bb73b3099d3b8ce45bd939156b737453b9ca5454',base_commit='bb73b3099d3b8ce45bd939156b737453b9ca5454',
            transolver_sha='75e0f67643806a81cd1d3f6adc88dd8c02416fe7',linearno_sha='3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
            paper_version='2511.06294v3',paper_sha256='637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd',
            dirty=True,source_sha256=hashlib.sha256((ROOT/'cdlno/linearno/schema.py').read_bytes()).hexdigest(),
            normalized_patch_sha256=hashlib.sha256(b'no operator patch; L1 metadata-only test').hexdigest(),
            command=['python','-B','-m','unittest','linearno.test_schema'],environment={'torch':str(torch.__version__),'device':'cpu','scope':'metadata only'}),
        normalizer_spec=dict(policy='saved_train_fit',records={'input':normalizer_record(
            {'mean':torch.tensor([1.,2.]),'std':np.array([.1,.2],dtype=np.float64)},fit_split='synthetic train',
            data_checksum=data_hash,algorithm='existing UnitTransformer std+1e-8')}),
        resume_state=dict(checkpoint_role='final',selection_split=None,selection_metric=None,epoch=0,global_step=0,
            optimizer=pack_state({'state':{},'param_groups':[]}),scheduler=pack_state({'last_epoch':0}),rng=pack_state(rng),
            dataloader_generators=pack_state({'loader':{'device':'cpu','state':torch.Generator().manual_seed(3).get_state()}}),sampler_state=pack_state({})),
        ensemble_manifest=[dict(member_id='member0',order=0,path='members/000/state.pt',sha256=data_hash,format='state_dict')])


def rehash(metadata):
    metadata['metadata_hash']=digest({k:v for k,v in metadata.items() if k!='metadata_hash'})
    return metadata


class SchemaChecks(unittest.TestCase):
    def test_roundtrip_is_read_only_and_no_overwrite(self):
        m=sample_metadata()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'metadata.json';write_metadata(p,m)
            before=p.read_bytes()
            self.assertEqual(read_metadata(p),m);self.assertEqual(p.read_bytes(),before)
            with self.assertRaises(FileExistsError):write_metadata(p,m)
            self.assertEqual(p.read_bytes(),before)
            p.write_text('{"family":"linearno","family":"other"}')
            with self.assertRaisesRegex(ValueError,'duplicate'):read_metadata(p)

    def test_numerical_normalizers_and_rng_optimizer_tree_are_lossless(self):
        values=[torch.tensor(1.5,dtype=torch.float64),torch.arange(6,dtype=torch.bfloat16).reshape(2,3),
                torch.zeros(0,3),np.array([1.,2.],dtype='>f8'),np.array(2,dtype='int32'),torch.tensor([True,False])]
        for x in values:
            y=restore_numerical_state(numerical_state(x))
            if torch.is_tensor(x):self.assertTrue(torch.equal(x,y));self.assertEqual(x.dtype,y.dtype)
            else:np.testing.assert_array_equal(x,y);self.assertEqual(x.dtype,y.dtype)
        state={'optimizer':{0:{'exp_avg':torch.arange(3.),'step':torch.tensor(2.)}},
               'python':random.getstate(),'numpy':np.random.get_state(),'generators':{'loader':torch.Generator().get_state()}}
        restored=unpack_state(json.loads(json.dumps(pack_state(state))))
        np.testing.assert_array_equal(state['numpy'][1],restored['numpy'][1])
        self.assertEqual(state['numpy'][0],restored['numpy'][0])
        self.assertEqual(state['numpy'][2:],restored['numpy'][2:])
        state.pop('numpy');restored.pop('numpy')
        self.assertTrue(_same(state,restored))
        bad=numerical_state(torch.ones(2));bad['data']='AAAA'
        with self.assertRaisesRegex(ValueError,'checksum'):restore_numerical_state(bad)
        bad=numerical_state(torch.ones(2));bad['shape']=[3];bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'byte size'):restore_numerical_state(bad)
        with self.assertRaises(ValueError):numerical_state(np.array([object()]))

    def test_real_constructor_signature_rejects_runtime_fields(self):
        # Generic signature validator exercised on a REAL PyTorch class, no fake LinearNO model.
        cls=torch.nn.Linear
        spec={'class_path':cls.__module__+'.'+cls.__qualname__,'constructor_kwargs':{'in_features':3,'out_features':2}}
        validate_model_spec(spec,cls)
        for change in ({'lr':.01},{'family':'linearno'}):
            bad=copy.deepcopy(spec);bad['constructor_kwargs'].update(change)
            with self.assertRaisesRegex(ValueError,'unknown constructor'):validate_model_spec(bad,cls)
        bad=copy.deepcopy(spec);bad['constructor_kwargs'].pop('in_features')
        with self.assertRaises(ValueError):validate_model_spec(bad,cls)
        bad=copy.deepcopy(spec);bad['class_path']='models.Nonexistent'
        with self.assertRaises(ValueError):validate_model_spec(bad,cls)

    def test_conflicts_required_state_and_legacy_recognition(self):
        m=sample_metadata()
        self.assertIsNone(checkpoint_family({'weight':torch.ones(1)}))
        self.assertIsNone(checkpoint_family({'profile_spec':{'family':'linearno'}}))
        self.assertEqual(checkpoint_family({'family':'cdlno'}),'cdlno')
        mutations=[lambda p:p.update(family='cdlno'),lambda p:p.update(schema_version=9),lambda p:p.update(schema_version=True),
                   lambda p:p['resume_state'].pop('optimizer'),lambda p:p['provenance_spec'].pop('normalized_patch_sha256'),
                   lambda p:p['data_spec'].update(checksums=None),lambda p:p['normalizer_spec']['records']['input'].pop('fit_split'),
                   lambda p:p['resume_state'].update(checkpoint_role='best_validation',selection_split='test',selection_metric='rL2'),
                   lambda p:p['ensemble_manifest'][0].update(order=1),lambda p:p['ensemble_manifest'][0].update(path='../bad.pt'),
                   lambda p:p['provenance_spec'].update(linearno_sha='main'),
                   lambda p:p['model_spec']['constructor_kwargs'].update(lr=.001),
                   lambda p:p['model_spec']['constructor_kwargs'].pop('n_hidden'),
                   lambda p:p['model_spec'].update(class_path='models.NotImplemented'),
                   lambda p:p['resume_state'].update(rng=pack_state(dict(python=None,numpy=None,torch_cpu=None,torch_cuda=[]))),
                   lambda p:p['resume_state'].update(optimizer=pack_state({})),
                   lambda p:p['resume_state'].update(dataloader_generators=pack_state({'loader':{'device':'cpu','state':None}})),
                   lambda p:p['evaluation_spec'].pop('force_input')]
        for mutate in mutations:
            bad=copy.deepcopy(m);mutate(bad);rehash(bad)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):validate_metadata(bad)
        bad=copy.deepcopy(m);bad['resume_state']['epoch']=9
        with self.assertRaisesRegex(ValueError,'checksum'):validate_metadata(bad)

    def test_structural_rejection_runtime_reporting_and_resume_rules(self):
        saved=sample_metadata();runtime=sample_metadata({'runtime.device':'cuda:1'})
        self.assertEqual(set(compare_metadata(saved,runtime)),{'runtime'})
        compare_metadata(saved,runtime,resume=True)
        different=sample_metadata({'training.batch_size':2})
        self.assertIn('training',compare_metadata(saved,different))
        with self.assertRaisesRegex(ValueError,'resume protocol'):compare_metadata(saved,different,resume=True)
        changed=copy.deepcopy(saved);changed['model_spec']['constructor_kwargs']['n_hidden']=64;rehash(changed)
        with self.assertRaisesRegex(ValueError,'structural'):compare_metadata(saved,changed)
        changed=sample_metadata({'model.linearno_rank':32})
        with self.assertRaisesRegex(ValueError,'structural'):compare_metadata(saved,changed)
