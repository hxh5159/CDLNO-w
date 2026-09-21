"""Runtime version dispatch after metadata/config validation."""

import hashlib
import importlib
from pathlib import Path

import torch

from linearno_loop.contracts import digest
from linearno_loop.versioning import is_v2


def construction_api(config):
    return importlib.import_module("cdlno.linearno_loop.v2.construction" if is_v2(config)
                                   else "cdlno.linearno_loop.construction")


def checkpoint_api(config):
    return importlib.import_module("cdlno.linearno_loop.v2.checkpoint" if is_v2(config)
                                   else "cdlno.linearno_loop.checkpoint")


def construct(config, *, member_seed=None):
    if member_seed is None:
        return construction_api(config).build_from_config(config)
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
    if not is_v2(config):
        if task is None:
            from .provenance import provenance as standard
            return standard()
        from .industrial_state import provenance as industrial
        return industrial(task)
    from .v2.provenance import provenance as v2_provenance
    return v2_provenance(config, task=task)

