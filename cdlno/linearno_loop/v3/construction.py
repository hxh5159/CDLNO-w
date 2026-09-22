"""Config-first V3 wrapper construction without production task routing."""
from contextvars import ContextVar
import importlib

import torch

from linearno_loop.v3.config import validate_config
from linearno_loop.v3.contracts import require_equal
from linearno_loop.v3.schema import validate_constructor


_ACTIVE_CONFIG = ContextVar("linearno_loop_v3_active_config", default=None)


def constructor_config(kwargs):
    """Return the validated config installed by build_from_config.

    The frozen model signature contains only JSON constructor fields. The full
    config remains the authority for core ownership and feature seeds, so direct
    constructor calls are rejected instead of reconstructing it from globals or
    current profile tables.
    """
    config = _ACTIVE_CONFIG.get()
    if config is None:
        raise RuntimeError("V3 wrappers must be constructed from validated config metadata")
    require_equal(config["model_spec"]["constructor_kwargs"], kwargs,
                  "model.constructor_kwargs")
    return config


class LoopForwardViewV3:
    family = "linearno_loop"
    architecture_extension = "loop_linearno_latent_adapter_v3"

    @property
    def blocks(self):
        return (self.loop,)


def build_from_config(config, *, initialization_seed=None):
    checked = validate_config(config)
    seed = (checked["fair_comparison"]["public_backbone_seed"]
            if initialization_seed is None else initialization_seed)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("initialization_seed must be a Torch uint64 integer")
    spec = checked["model_spec"]
    module_name, class_name = spec["class_path"].rsplit(".", 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    validate_constructor(spec, constructor)
    token = _ACTIVE_CONFIG.set(checked)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = constructor(**spec["constructor_kwargs"])
    finally:
        _ACTIVE_CONFIG.reset(token)
    model.config_hash = checked["config_hash"]
    model.task = checked["loop_spec"]["task"]
    model.profile = checked["loop_spec"]["profile"]
    model.cost_profile = checked["loop_spec"]["cost_profile"]
    model.topology_preset = checked["loop_spec"]["topology_preset"]
    model.initialization_seed = seed
    return model


def common_state(config, *, initialization_seed=None):
    model = build_from_config(config, initialization_seed=initialization_seed)
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if ".latent_processor." not in key and ".adapter." not in key and
            not key.startswith("loop.rb_") and not key.startswith("loop.lb_")}

