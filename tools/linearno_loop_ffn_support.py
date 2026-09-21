"""Data-free v2 configurations and inputs for LF7 accounting.

This module does not import benchmark entry points or load datasets.
"""

import torch

from linearno_loop.v2.config import resolve_config


TASKS = ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')
PRESETS = ('p1_c3_r2_s1','p2_c2_r2_s2')
RESIDUALS = ('sr_1_over_r','rb_attnres','lb_attnres_1_over_r')
CORE_FFN_MODES = ('round_specific','round_specific_latent')


def configuration(task,preset,residual,core_ffn_mode,*,multiplier=1,small=False,
                  canonical=False,seed=17):
    overrides={'runtime.seed':seed}
    options=dict(topology_preset=preset,residual_mode=residual,
                 core_ffn_mode=core_ffn_mode,rank_multiplier=multiplier)
    if small:
        overrides.update({'model.hidden':8,'model.heads':2,'model.ref':3})
        options.pop('rank_multiplier');options['linearno_rank']=4*multiplier
        if not canonical and task not in ('airfrans','car'):
            overrides.update({'model.H':5,'model.W':7})
    return resolve_config(task,options=options,profile_overrides=overrides)


def point_count(config,canonical=True):
    model=config['profile_spec']['values']['model'];task=config['loop_spec']['task']
    if task in ('airfrans','car'):
        return {'airfrans':32000,'car':32186}[task] if canonical else 37
    if task=='elasticity' and canonical:return 972
    return model['H']*model['W']


def inputs(config,*,canonical=False,batch=2,device='cpu',dtype=torch.float32):
    model=config['profile_spec']['values']['model'];task=config['loop_spec']['task']
    n=point_count(config,canonical);generator=torch.Generator().manual_seed(902)
    def rand(*shape):
        return torch.randn(*shape,generator=generator,dtype=dtype).to(device)
    if task in ('airfrans','car'):
        from torch_geometric.data import Data
        data=Data(x=rand(n,7),pos=rand(n,2),
                  batch=torch.zeros(n,dtype=torch.long,device=device),
                  ptr=torch.tensor([0,n],device=device))
        return (data,) if task=='airfrans' else ((data,None),)
    x=rand(batch,n,model['space_dim'])
    fx=rand(batch,n,model['fun_dim']) if model['fun_dim'] else None
    time=(torch.linspace(.1,.8,batch,dtype=dtype,device=device).reshape(batch,1)
          if model['time_input'] else None)
    return x,fx,time
