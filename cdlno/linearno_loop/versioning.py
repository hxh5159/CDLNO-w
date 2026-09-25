"""Runtime version dispatch after metadata/config validation."""

import hashlib
import importlib
from pathlib import Path

import torch

from linearno_loop.contracts import digest
from linearno_loop.versioning import is_v2, is_v3, is_v4, is_v5


def construction_api(config):
    if is_v5(config):return importlib.import_module("cdlno.linearno_loop.v5.construction")
    if is_v4(config):return importlib.import_module("cdlno.linearno_loop.v4.construction")
    if is_v3(config):
        return importlib.import_module("cdlno.linearno_loop.v3.construction")
    return importlib.import_module("cdlno.linearno_loop.v2.construction" if is_v2(config)
                                   else "cdlno.linearno_loop.construction")


def checkpoint_api(config):
    if is_v5(config):return importlib.import_module("cdlno.linearno_loop.v5.checkpoint")
    if is_v4(config):return importlib.import_module("cdlno.linearno_loop.v4.checkpoint")
    if is_v3(config):
        return importlib.import_module("cdlno.linearno_loop.v3.checkpoint")
    return importlib.import_module("cdlno.linearno_loop.v2.checkpoint" if is_v2(config)
                                   else "cdlno.linearno_loop.checkpoint")


def construct(config, *, member_seed=None):
    if is_v5(config):return construction_api(config).build_from_config(config,initialization_seed=member_seed)
    if is_v4(config):return construction_api(config).build_from_config(config,initialization_seed=member_seed)
    if member_seed is None:
        return construction_api(config).build_from_config(config)
    if is_v3(config):
        # V3 wrappers validate their complete constructor contract through the
        # construction context; bypassing it would make industrial member
        # construction differ from the normal metadata-first path.
        return construction_api(config).build_from_config(config,
                                                          initialization_seed=member_seed)
    spec = config["model_spec"]
    module_name, class_name = spec["class_path"].rsplit(".", 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    from linearno_loop.versioning import validate_constructor
    validate_constructor(spec, constructor, version=config["config_version"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(member_seed)
        model = constructor(**spec["constructor_kwargs"])
    model.config_hash = config["config_hash"]
    model.task = config["loop_spec"]["task"]
    return model


def provenance(config, task=None):
    if is_v5(config):
        from .v5.provenance import provenance as v5_provenance
        return v5_provenance(config, task=task)
    if is_v4(config):
        from .v4.provenance import provenance as v4_provenance
        return v4_provenance(config, task=task)
    if is_v3(config):
        from .v3.provenance import provenance as v3_provenance
        return v3_provenance(config, task=task)
    if not is_v2(config):
        if task is None:
            from .provenance import provenance as standard
            return standard()
        from .industrial_state import provenance as industrial
        return industrial(task)
    from .v2.provenance import provenance as v2_provenance
    return v2_provenance(config, task=task)
