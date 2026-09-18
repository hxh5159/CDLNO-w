"""Allowlisted internal factory. No production registry or task entry imports."""
import importlib
import inspect
from pathlib import Path
import sys

from .config import PATHS, task_kind, validate_config, baseline_model_spec
from linearno_history.schema import HistorySchemaError


def class_for(path):
    if path not in {p for row in PATHS.values() for p in row.values()}:
        raise HistorySchemaError(f'unrecognized LinearNO class_path: {path}')
    module, name = path.rsplit('.', 1)
    if module.startswith('model.'):
        # Preserve the actual legacy class path; reject an unrelated `model`
        # package already occupying that name instead of silently shadowing it.
        standard = Path(__file__).resolve().parents[2]/'PDE-Solving-StandardBenchmark'
        existing = sys.modules.get('model')
        if existing is not None:
            locations = {Path(p).resolve() for p in getattr(existing, '__path__', ())}
            if locations != {standard/'model'}:
                raise HistorySchemaError('Standard model package is shadowed by another project')
        added = str(standard) not in sys.path
        if added:
            sys.path.insert(0, str(standard))
        try:
            return getattr(importlib.import_module(module), name)
        finally:
            if added:
                sys.path.remove(str(standard))
    return getattr(importlib.import_module(module), name)


def validate_constructor(config):
    c = validate_config(config)
    spec = c['model_spec']
    pure = class_for(baseline_model_spec(c['profile_spec'])['class_path'])
    kwargs = dict(spec['constructor_kwargs'])
    if c['family'] == 'linearno_history':
        kwargs.pop('feature_seed')
    signature = inspect.signature(pure)
    if set(kwargs) != set(signature.parameters):
        raise HistorySchemaError('constructor kwargs must contain exactly the complete base signature')
    signature.bind(**kwargs)
    cls = class_for(spec['class_path'])
    inspect.signature(cls).bind(**spec['constructor_kwargs'])
    return cls


def build_model(config):
    c = validate_config(config)
    cls = validate_constructor(c)
    # Pure class is returned directly for A0K0. No wrapper, feature module or raw
    # history context is created, and no metadata is attached to the model.
    model = cls(**c['model_spec']['constructor_kwargs'])
    if c['family'] == 'linearno_history':
        model._history_model_spec = c['model_spec']
    return model


def build_fair_model(config):
    """Copy a canonical fresh pure backbone without advancing its RNG twice.

    The caller seeds the normal task initialization stream. This is not a
    trained-checkpoint warm start or resume; feature RNG remains independent.
    A0K0 still constructs the original class with its original RNG evolution.
    """
    import torch
    c = validate_config(config)
    if c['family'] == 'linearno':
        return build_model(c)
    spec = baseline_model_spec(c['profile_spec'])
    with torch.random.fork_rng(devices=[]):
        public = class_for(spec['class_path'])(**spec['constructor_kwargs'])
    model = build_model(c)
    common = public.state_dict()
    actual = model.state_dict()
    extra = set(actual) - set(common)
    if any(not k.startswith(('latent_attnres.', 'history_k.')) for k in extra):
        raise HistorySchemaError('unexpected non-feature key in research backbone')
    with torch.no_grad():
        for key, value in common.items():
            if key not in actual or actual[key].shape != value.shape or actual[key].dtype != value.dtype:
                raise HistorySchemaError('public backbone key/shape/dtype mismatch: '+key)
            actual[key].copy_(value)
    return model
