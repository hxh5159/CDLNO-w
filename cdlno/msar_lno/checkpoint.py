"""Strict M4 core checkpoint foundation, retaining each task's payload format.

These snapshots contain an already-lifted-input core, NOT a task wrapper or an
optimizer/RNG resume archive. Task stages add their own field/protocol metadata.
Only the local industrial whole-object/list branch opts into trusted pickle;
state_dict always uses weights_only=True. Evaluation never writes any file.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .config import FAMILY, positive_int
from .core import MSARLNO
from .metadata import (MSARMetadata, MSARMetadataMismatch, EvaluationResolution,
                       reserve_run_directory, resolve_evaluation, save_metadata)

_CLASS = 'cdlno.msar_lno.core.MSARLNO'
_INPUT = 'already-lifted-B-N-d-v1'
_DTYPES = {name: getattr(torch, name) for name in ('float32', 'float64', 'float16', 'bfloat16')}


def _filename(value):
    if (type(value) is not str or not value or Path(value).name != value
            or value in ('.', '..', 'architecture.json', 'core.json') or '\\' in value):
        raise MSARMetadataMismatch('checkpoint filename must be a distinct local filename')
    return value


def _members(payload, fmt):
    if fmt == 'model_list':
        if type(payload) is not list or not payload:
            raise MSARMetadataMismatch('expected nonempty model_list')
        return payload
    if type(payload) is not MSARLNO:
        raise MSARMetadataMismatch('expected the stable MSARLNO core class')
    return [payload]


def _reference(meta, output_dim, dtype):
    # Validation/reconstruction must not advance the caller's initialization RNG.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        return MSARLNO(meta.architecture, output_dim=output_dim,
                       training_config=meta.training).to(dtype=dtype)


def _validate_model(model, meta, output_dim, dtype):
    if (type(model) is not MSARLNO or getattr(model, 'config', None) != meta.architecture
            or getattr(model, 'output_dim', None) != output_dim
            or getattr(model, 'training_config', None) != meta.training):
        raise MSARMetadataMismatch('core class/architecture/output/training metadata mismatch')
    values = list(model.parameters())
    if not values or any(p.dtype != dtype for p in values):
        raise MSARMetadataMismatch('core parameter dtype mismatch')
    named = list(model.named_parameters(remove_duplicate=False))
    if len(named) != len({(p.device, p.untyped_storage().data_ptr()) for _, p in named}):
        raise MSARMetadataMismatch('unexpected shared parameter storage in MSAR core')
    expected = _reference(meta, output_dim, dtype)
    # Reuse the existing read-only type/module/behavior/buffer + strict-state
    # validator; never copy its initialized parameters into the saved object.
    from ..kcdno.loading import validate_whole_model
    try:
        validate_whole_model(model, expected)
    except (ValueError, RuntimeError) as error:
        raise MSARMetadataMismatch(str(error)) from error


def save_core_checkpoint(directory, payload, metadata: MSARMetadata, *, filename=None) -> Path:
    """Exclusive NEW snapshot; no periodic save policy, task Run or resume added.

    PDE payload stays a bare state_dict. Car stays one whole object (caller
    supplies model_<epochs>.pth); Air stays a list or one member object.
    """
    if type(metadata) is not MSARMetadata:
        raise MSARMetadataMismatch('MSARMetadata required')
    metadata.validate()
    fmt = metadata.checkpoint_format
    models = _members(payload, fmt)
    first = models[0]
    if type(first) is not MSARLNO:
        raise MSARMetadataMismatch('only the real lifted-feature MSARLNO core is supported in M4')
    output_dim, dtype = first.output_dim, next(first.parameters()).dtype
    dtype_name = str(dtype).removeprefix('torch.')
    if dtype_name not in _DTYPES:
        raise MSARMetadataMismatch('unsupported saved parameter dtype')
    for model in models:
        _validate_model(model, metadata, output_dim, dtype)
    if filename is None:
        if metadata.task == 'car':
            raise MSARMetadataMismatch('Car requires its explicit model_<epochs>.pth filename')
        filename = 'model.pt' if fmt == 'state_dict' else (FAMILY if fmt == 'model_list' else 'model')
    filename = _filename(filename)
    record = dict(schema_version=1, family=FAMILY, model_class=_CLASS, input_contract=_INPUT,
                  output_dim=output_dim, members=len(models), weight_dtype=dtype_name, filename=filename)
    directory = reserve_run_directory(directory)
    save_metadata(directory / 'architecture.json', metadata)
    with (directory / filename).open('xb') as stream:
        torch.save(first.state_dict() if fmt == 'state_dict' else payload, stream)
    with (directory / 'core.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return directory / filename


@dataclass(frozen=True)
class LoadedCoreCheckpoint:
    model: MSARLNO | list[MSARLNO]
    resolution: EvaluationResolution


def load_core_checkpoint(directory, *, task, explicit=None, expected_output_dim=None,
                         device='cpu', trusted_local=False) -> LoadedCoreCheckpoint:
    """Read saved architecture BEFORE construction/loading; return eval models.

    No inference from legacy files lacking family. Explicit coverage changes
    are reported in resolution, never substituted into saved training metadata.
    trusted_local=True is required only for an already trusted industrial run;
    it is not a general unpickling fallback and never affects old loaders.
    """
    directory = Path(directory)
    resolution = resolve_evaluation(directory / 'architecture.json',
                                    {} if explicit is None else explicit, task=task)
    meta = resolution.saved
    record = json.loads((directory / 'core.json').read_text(encoding='utf-8'))
    required = {'schema_version', 'family', 'model_class', 'input_contract', 'output_dim',
                'members', 'weight_dtype', 'filename'}
    if (type(record) is not dict or record.keys() != required
            or type(record['schema_version']) is not int or record['schema_version'] != 1
            or record['family'] != FAMILY or record['model_class'] != _CLASS
            or record['input_contract'] != _INPUT):
        raise MSARMetadataMismatch('unsupported or incomplete core checkpoint contract')
    positive_int('output_dim', record['output_dim'])
    positive_int('members', record['members'])
    if expected_output_dim is not None:
        positive_int('expected_output_dim', expected_output_dim)
        if expected_output_dim != record['output_dim']:
            raise MSARMetadataMismatch('checkpoint output_dim mismatch')
    if type(record['weight_dtype']) is not str or record['weight_dtype'] not in _DTYPES:
        raise MSARMetadataMismatch('unsupported weight_dtype')
    fmt = meta.checkpoint_format
    if fmt != 'model_list' and record['members'] != 1:
        raise MSARMetadataMismatch('single-model format requires members=1')
    filename = _filename(record['filename'])
    if type(trusted_local) is not bool:
        raise ValueError('trusted_local must be bool')
    if fmt != 'state_dict' and not trusted_local:
        raise ValueError('whole objects/lists require an explicitly trusted local industrial checkpoint')
    dtype = _DTYPES[record['weight_dtype']]
    if fmt == 'state_dict':
        model = _reference(meta, record['output_dim'], dtype)
        weights = torch.load(directory / filename, weights_only=True, map_location='cpu')
        if (not isinstance(weights, dict) or not all(isinstance(v, torch.Tensor) and v.dtype == dtype
                                                    for v in weights.values())):
            raise MSARMetadataMismatch('expected bare state_dict with the recorded weight dtype')
        model.load_state_dict(weights, strict=True)
        payload = model.to(device).eval()
    else:
        payload = torch.load(directory / filename, weights_only=False, map_location='cpu')
        models = _members(payload, fmt)
        if len(models) != record['members']:
            raise MSARMetadataMismatch('whole-model list member count mismatch')
        for model in models:
            _validate_model(model, meta, record['output_dim'], dtype)
            model.to(device).eval()
    return LoadedCoreCheckpoint(payload, resolution)
