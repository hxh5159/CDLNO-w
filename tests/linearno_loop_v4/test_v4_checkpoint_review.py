"""Metadata-first negative matrix and allocation-free CLI conflicts."""
import copy
import json
from unittest.mock import patch

import pytest
import torch
from linearno_loop.v4.config import resolve_config
from linearno_loop.v4.contracts import digest
from linearno_loop.v4.schema import validate_metadata
from cdlno.linearno_loop.v4.checkpoint import save_pair,load_model,read_metadata
from cdlno.linearno_loop.v4.construction import build_from_config
from .test_v4_entry_contracts import parse,BASE


@pytest.fixture
def saved(tmp_path):
    cfg=resolve_config('elasticity',options=dict(architecture='resmlp_dual_temp_v4',temperature_mode='point_k_point_q',seed=7))
    save_pair(tmp_path,build_from_config(cfg),cfg)
    return tmp_path,read_metadata(tmp_path/'architecture.json')


@pytest.mark.parametrize('field,value',[
    ('architecture','operator_latent_adapter_v3'),('architecture_version',3),
    ('checkpoint_schema','linearno-loop-epoch-pair-v2'),('checkpoint_version',2),
    ('config_hash','0'*64),('loop_spec',{}),('rmlp_spec',{}),('temperature_spec',{}),
    ('model_spec',{}),('profile_spec',{}),('resume_state',{'epoch':1,'checkpoint_role':'final'}),
    ('parameter_measurement',{'status':'measured','total':1,'trainable':1,'groups':{'stem':1}}),
])
def test_modified_sidecar_rejected_before_construction_or_tensors(saved,field,value):
    directory,meta=saved
    meta[field]=value;meta['metadata_hash']=digest({k:v for k,v in meta.items() if k!='metadata_hash'})
    (directory/'architecture.json').write_text(json.dumps(meta))
    with patch('torch.load',side_effect=AssertionError('premature tensor read')) as tensor, \
         patch('cdlno.linearno_loop.v4.construction.build_from_config',side_effect=AssertionError('premature construction')) as constructor:
        with pytest.raises((ValueError,KeyError)):load_model(directory)
    tensor.assert_not_called();constructor.assert_not_called()


def test_explicit_mode_seed_profile_architecture_conflicts(saved):
    directory,meta=saved
    for extra in (['--seed','8'],['--linearno-loop-temperature-mode','latent_k_point_q'],
                  ['--linearno-loop-architecture','operator_latent_adapter_v3']):
        with patch('torch.load',side_effect=AssertionError('premature tensor read')):
            with pytest.raises(SystemExit):parse('elasticity','eval',['--experiment-dir',directory,*extra])


def test_config_metadata_import_rng_and_no_torch():
    import os,subprocess,sys
    code="""
import sys,random,json
before=random.getstate()
from linearno_loop.v4.config import resolve_config,validate_config
from linearno_loop.v4.schema import validate_metadata
for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'):
 for mode in ('base','latent_k_point_q','point_k_point_q'):
  c=resolve_config(task,options={'architecture':'resmlp_dual_temp_v4','temperature_mode':mode})
  assert validate_config(json.loads(json.dumps(c)))==c
assert before==random.getstate()
assert 'torch' not in sys.modules
print('24 JSON roundtrips, RNG unchanged, torch absent')
"""
    process=subprocess.run([sys.executable,'-B','-c',code],env={**os.environ,'PYTHONPATH':'.:cdlno'},capture_output=True,text=True)
    assert process.returncode==0,process.stdout+process.stderr
