"""Data-free LL9 configurations and inputs, independent of benchmark entrypoints."""
import torch
from linearno_loop.config import resolve_config

TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')
PRESETS=('p1_c3_r2_s1','p2_c2_r2_s2')
MODES=('sr_1_over_r','rb_attnres','lb_attnres_1_over_r')


def configuration(task,preset,mode,*,multiplier=2,small=False,canonical=False,seed=17):
    overrides={'runtime.seed':seed}
    options=dict(topology_preset=preset,residual_mode=mode,rank_multiplier=multiplier)
    if small:
        overrides.update({'model.hidden':8,'model.heads':2,'model.ref':3})
        options.pop('rank_multiplier');options['linearno_rank']=4*multiplier
        if not canonical and task not in ('airfrans','car'):
            overrides.update({'model.H':5,'model.W':7})
    return resolve_config(task,options=options,profile_overrides=overrides)


def point_count(config,canonical=True):
    m=config['profile_spec']['values']['model'];task=config['loop_spec']['task']
    if task in ('airfrans','car'):return {'airfrans':32000,'car':32186}[task] if canonical else 37
    if task=='elasticity' and canonical:return 972
    return m['H']*m['W']


def inputs(config,*,canonical=False,batch=2,device='cpu',dtype=torch.float32):
    m=config['profile_spec']['values']['model'];task=config['loop_spec']['task']
    n=point_count(config,canonical);g=torch.Generator().manual_seed(902)
    def rand(*shape):return torch.randn(*shape,generator=g,dtype=dtype).to(device)
    if task in ('airfrans','car'):
        from torch_geometric.data import Data
        data=Data(x=rand(n,7),pos=rand(n,2),batch=torch.zeros(n,dtype=torch.long,device=device),
                  ptr=torch.tensor([0,n],device=device))
        return (data,) if task=='airfrans' else ((data,None),)
    x=rand(batch,n,m['space_dim'])
    fx=rand(batch,n,m['fun_dim']) if m['fun_dim'] else None
    t=torch.linspace(.1,.8,batch,dtype=dtype,device=device).reshape(batch,1) if m['time_input'] else None
    return x,fx,t
