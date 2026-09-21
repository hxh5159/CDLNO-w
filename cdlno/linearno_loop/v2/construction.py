"""Synthetic v2 construction from the closed version-2 configuration."""

import importlib

import torch

from linearno_loop.v2.config import validate_config
from linearno_loop.v2.schema import validate_constructor


class LoopForwardViewV2:
    family = "linearno_loop"
    architecture_extension = "loop_linearno_ffn_v2"

    @property
    def blocks(self):
        return (self.loop,)


def build_from_config(config):
    checked = validate_config(config)
    seed = checked["profile_spec"]["values"]["runtime"]["seed"]
    if not 0 <= seed < 2**63:
        raise ValueError("initialization seed must fit signed Torch seed range")
    spec = checked["model_spec"]
    module_name, class_name = spec["class_path"].rsplit(".", 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    validate_constructor(spec, constructor)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = constructor(**spec["constructor_kwargs"])
    model.config_hash = checked["config_hash"]
    model.task = checked["loop_spec"]["task"]
    model.profile = checked["loop_spec"]["profile"]
    model.topology_preset = checked["loop_spec"]["topology_preset"]
    model.initialization_seed = seed
    return model


def common_state(config):
    model = build_from_config(config)
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if not key.startswith("loop.latent_ffns.") and
            not key.startswith("loop.rb_") and not key.startswith("loop.lb_")}

