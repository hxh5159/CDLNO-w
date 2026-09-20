"""Internal synthetic construction from LL1 contracts; no production registry."""
import importlib

import torch

from cdlno.linearno.profiles import validate_model
from linearno_loop.config import validate_config
from linearno_loop.contracts import integer
from linearno_loop.schema import validate_constructor
from .core import require_implemented, topology


def validate_arguments(kind,kw):
    require_implemented(kw['residual_mode'])
    U,_=topology(kw['prefix_blocks'],kw['recurrent_core_blocks'],kw['loop_repeats'],kw['suffix_blocks'])
    variant=kw.get('linearno_variant',{'airfrans':'airfrans','car':'shapenet'}.get(kind))
    m=dict(hidden=kw['n_hidden'],heads=kw['n_head'],layers=U,linearno_rank=kw['linearno_rank'],
        ffn_ratio=kw['mlp_ratio'],space_dim=kw['space_dim'],out_dim=kw['out_dim'],ref=kw['ref'],
        fun_dim=kw['fun_dim'],linearno_variant=variant,dropout=kw['dropout'],activation=kw['act'],
        unified_pos=kw['unified_pos'],time_input=kw.get('Time_Input',False),H=kw.get('H'),W=kw.get('W'))
    validate_model('darcy' if kind=='standard' else kind,m)
    if kind!='airfrans':
        integer(kw['H'],'H',1);integer(kw['W'],'W',1)
    if m['time_input'] and m['hidden']%2:raise ValueError('time embedding requires even hidden')
    if kind=='airfrans':
        if (m['space_dim'],m['fun_dim'],m['out_dim'])!=(7,0,4) or type(kw['linear']) is not bool:
            raise ValueError('AirfRANS requires channels 7/0/4 and boolean linear')
    if kind=='car':
        if m['space_dim']+m['fun_dim']!=7 or m['out_dim']!=4 or type(kw['isregular']) is not bool:
            raise ValueError('Car requires input7/output4 and boolean isregular')
        if m['unified_pos'] and m['fun_dim']:raise ValueError('Car unified position requires fun_dim=0')


class LoopForwardView:
    family='linearno_loop'
    architecture_extension='loop_linearno_v1'

    @property
    def blocks(self):
        # Native forward reuses its exact stem/position/time/validation code,
        # then invokes this one loop. No second module registration.
        return (self.loop,)


def build_from_config(config):
    """Validate before import/allocation, initialize under the recorded CPU seed.

    Intended for synthetic verification only. No parser, filesystem output,
    task data, weight loading or legacy dispatch. Unsupported modes fail first.
    """
    checked=validate_config(config)
    require_implemented(checked['loop_spec']['residual_mode'])
    seed=checked['profile_spec']['values']['runtime']['seed']
    if seed>=2**64:raise ValueError('initialization seed must fit the Torch uint64 range')
    spec=checked['model_spec'];path,name=spec['class_path'].rsplit('.',1)
    cls=getattr(importlib.import_module(path),name)
    validate_constructor(spec,cls)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=cls(**spec['constructor_kwargs'])
    model.config_hash=checked['config_hash']
    model.task=checked['loop_spec']['task'];model.profile=checked['loop_spec']['profile']
    model.topology_preset=checked['loop_spec']['topology_preset']
    model.initialization_seed=seed
    return model


def common_initial_state(config):
    """Produce only common backbone weights for all three declared modes.

    This returns a detached backbone state_dict, never a residual-mode model.
    RB's zero/one receiver initialization consumes no RNG; LB remains absent.
    """
    from linearno_loop.config import resolve_config
    c=validate_config(config);r=c['request']
    sr=resolve_config(r['task'],r['profile'],options={**r['options'],'residual_mode':'sr_1_over_r'},
                      profile_overrides=r['profile_overrides'])
    return {k:v.detach().clone() for k,v in build_from_config(sr).state_dict().items()}
