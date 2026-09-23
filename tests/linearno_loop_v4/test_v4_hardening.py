"""Contract failures discovered during the independent delivery review."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from linearno_loop.v4.config import resolve_config, validate_config
from linearno_loop.v4.contracts import digest
from cdlno.linearno_loop.v4.construction import build_from_config
from cdlno.linearno_loop.v4.checkpoint import save_pair, inspect_checkpoint


def config(task='elasticity', mode='latent_k_point_q'):
    return resolve_config(task, options=dict(architecture='resmlp_dual_temp_v4',
                                            temperature_mode=mode, seed=19))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_factory_and_predictors_preserve_initialized_cuda_rng():
    torch.cuda.init()
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    cpu = torch.get_rng_state().clone()
    build_from_config(config())
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all()))


def test_full_tree_initialization_after_installation_is_rejected():
    from cdlno.linearno.attention import initialize_release_weights
    model = build_from_config(config())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match='initializ'):
        model.apply(initialize_release_weights)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


def test_resealed_profile_cannot_change_frozen_task_dimensions():
    from cdlno.linearno.profiles import resolve_config as base_config
    original = config()
    original['profile_spec'] = base_config('elasticity', original['profile'],
        explicit={'model.hidden': 64}, contract='standard_static_l4')
    # Re-sealing any individual profile is not permission to alter the v4 contract.
    with pytest.raises(ValueError):
        validate_config({**original, 'config_hash': digest({k:v for k,v in original.items() if k!='config_hash'})})


def test_version_dispatch_resolves_explicit_v4():
    from linearno_loop.versioning import resolve_config as dispatch
    direct = config()
    assert dispatch('elasticity', direct['profile'], options=dict(
        architecture='resmlp_dual_temp_v4', temperature_mode='latent_k_point_q', seed=19),
        profile_overrides={}) == direct


def test_manifest_epoch_must_match_filename_before_tensor_load(tmp_path):
    c = config(); model = build_from_config(c); path = save_pair(tmp_path, model, c)
    wrong = path.with_name('epoch_9999.json')
    payload = json.loads(path.read_text())
    # Valid file hashes cannot make a manifest for another epoch valid.
    wrong.write_text(json.dumps(payload))
    with patch('torch.load', side_effect=AssertionError('premature tensor read')):
        with pytest.raises(ValueError):
            inspect_checkpoint(tmp_path, 'epoch_9999')


def test_provenance_is_portable_and_covers_configuration_and_dispatch():
    from cdlno.linearno_loop.versioning import provenance
    record = provenance(config(), 'elasticity')
    assert 'sources' in record
    assert 'linearno_loop/v4/config.py' in record['sources']
    assert 'cdlno/linearno_loop/standard_entry.py' in record['sources']
    assert all(not Path(key).is_absolute() for key in record['sources'])


def test_resealed_unknown_config_field_is_rejected():
    c=config();c['undeclared_feature']=True
    c['config_hash']=digest({k:v for k,v in c.items() if k!='config_hash'})
    with pytest.raises(ValueError,match='fields'):validate_config(c)
