"""LL3 synthetic fixtures and accounting; never a task launcher."""
import importlib
import torch

from linearno_loop.config import resolve_config

TOPOLOGIES=('p1_c3_r2_s1','p2_c2_r2_s2','custom')


def config(task='darcy',preset='p1_c3_r2_s1',*,variant=None,R=None,seed=7):
    options=dict(topology_preset=preset,residual_mode='sr_1_over_r',linearno_rank=4)
    if preset=='custom':options.update(prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3 if R is None else R,suffix_blocks=1)
    overrides={'model.hidden':8,'model.heads':2,'model.ref':3,'runtime.seed':seed}
    if task not in ('airfrans','car'):overrides.update({'model.H':3,'model.W':5})
    if variant is not None:overrides['model.linearno_variant']=variant
    return resolve_config(task,options=options,profile_overrides=overrides)


def pure_constructor(c):
    task=c['loop_spec']['task']
    if task=='airfrans':return importlib.import_module('cdlno.linearno.airfrans').AirfRANSLinearNO
    if task=='car':return importlib.import_module('cdlno.linearno.shapenet').ShapeNetLinearNO
    return importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO').Model


def pure_kwargs(c):
    kw=dict(c['model_spec']['constructor_kwargs'])
    for key in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks','residual_mode'):kw.pop(key)
    kw['n_layers']=c['loop_spec']['unique_depth']
    return kw


def to_pure_state(state,P,C,S):
    result={}
    offsets={'prefix':0,'core':P,'suffix':P+C}
    for key,value in state.items():
        if key.startswith('loop.'):
            _,group,index,block,suffix=key.split('.',4)
            assert block=='block'
            result[f'blocks.{offsets[group]+int(index)}.{suffix}']=value
        else:result[key]=value
    return result


def inputs(c,dtype=torch.float32):
    from torch_geometric.data import Data
    m=c['profile_spec']['values']['model'];task=c['loop_spec']['task']
    g=torch.Generator().manual_seed(902)
    if task in ('airfrans','car'):
        data=Data(x=torch.randn(13,7,generator=g,dtype=dtype),pos=torch.randn(13,2,generator=g,dtype=dtype),
                  batch=torch.zeros(13,dtype=torch.long),ptr=torch.tensor([0,13]))
        return (data,) if task=='airfrans' else ((data,object()),)
    N=m['H']*m['W'];x=torch.randn(2,N,m['space_dim'],generator=g,dtype=dtype)
    fx=torch.randn(2,N,m['fun_dim'],generator=g,dtype=dtype) if m['fun_dim'] else None
    T=torch.tensor([[.1],[.8]],dtype=dtype) if m['time_input'] else None
    return x,fx,T


def counts(c,*,B=1,N=None):
    """Analytic dense matrix MACs only; biases/norm/nonlinear/softmax excluded."""
    s=c['loop_spec'];m=c['profile_spec']['values']['model'];task=s['task']
    d,h,M,r=m['hidden'],m['heads'],s['resolved_rank'],m['ffn_ratio'];dh=d//h
    variant=m['linearno_variant'];conv=variant in ('conv','conv_temp');outlayers=2 if conv or task=='car' else 1
    temp=2*h if variant in ('temp','conv_temp','shapenet') else h if task=='airfrans' else 0
    body_params=(9 if conv else 1)*d*d+d+2*dh*M+dh*dh+outlayers*(d*d+d)+temp+4*d+2*r*d*d+(r+1)*d
    head_params=2*d+d*m['out_dim']+m['out_dim']
    lift=(m['space_dim']+m['ref']**2 if task=='airfrans' and m['unified_pos'] else
          m['ref']**2 if task=='car' and m['unified_pos'] else
          m['fun_dim']+(m['ref']**2 if m['unified_pos'] else m['space_dim']))
    stem_params=lift*2*d+2*d+2*d*d+d+d+(2*d*d+2*d if m['time_input'] else 0)
    if N is None:N={'airfrans':32000,'car':32186,'elasticity':972}.get(task,m['H']*m['W'] if m['H'] else 0)
    body_mac=B*N*((9 if conv else 1)*d*d+outlayers*d*d+2*r*d*d+d*dh+4*d*M)
    stem_mac=B*N*(lift*2*d+2*d*d+(2*d*d if m['time_input'] else 0))
    head_mac=B*N*d*m['out_dim']
    return dict(B=B,N=N,unique_depth=s['unique_depth'],executed_depth=s['executed_depth'],
        unique_parameters=stem_params+s['unique_depth']*body_params+head_params,
        executed_parameter_uses=stem_params+s['executed_depth']*body_params+head_params,
        one_visit_per_physical_block_macs=stem_mac+s['unique_depth']*body_mac+head_mac,
        executed_macs=stem_mac+s['executed_depth']*body_mac+head_mac,
        dead_temperature_parameters=s['unique_depth']*h if task=='airfrans' else 0,
        mac_scope='Linear/Conv projections, K^T V, QC; no norm/GELU/softmax/bias/residual/sinusoid/storage cost')
