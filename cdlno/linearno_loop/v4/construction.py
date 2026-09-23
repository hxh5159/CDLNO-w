"""Explicit v4 config-first constructors."""
import importlib
import torch
from linearno_loop.v4.config import validate_config

CLASS_PATHS={"airfrans":"cdlno.linearno_loop.v4.airfrans.LoopedAirfRANSModelV4",
             "car":"cdlno.linearno_loop.v4.shapenet.LoopedShapeNetModelV4",
             "standard":"cdlno.linearno_loop.v4.standard.LoopedStandardModelV4"}

def constructor_kwargs(config):
    return dict(validate_config(config)['model_spec']['constructor_kwargs'])

def build_from_config(config, *, initialization_seed=None):
    c=validate_config(config); task=c["task"]; key="standard" if task in ("airfoil","darcy","elasticity","pipe","ns","plasticity") else task
    path=CLASS_PATHS[key]; mod,name=path.rsplit(".",1); cls=getattr(importlib.import_module(mod),name)
    kwargs=constructor_kwargs(c)
    init_seed=c['seed'] if initialization_seed is None else initialization_seed
    kwargs['public_seed']=init_seed
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(init_seed); model=cls(**kwargs)
    model.config_hash=c["config_hash"]; model.v4_config=c
    return model
