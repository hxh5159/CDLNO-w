"""Independent synthetic-tool inputs; never import dataset/experiment modules."""
import torch
from cdlno.linearno.profiles import resolve_config as profile
from cdlno.linearno_history.config import resolve_config
from cdlno.linearno_history.factory import build_fair_model

MODES=('A0K0','A1K0','A0K1','A1K1')

def configuration(mode='A0K0',layers=4,task='elasticity',*,hidden=32,heads=4,rank=16,seed=17,variant=None):
    overrides={'model.layers':layers,'model.hidden':hidden,'model.heads':heads,
               'model.linearno_rank':rank,'runtime.seed':seed,'runtime.device':'cpu'}
    if task not in ('airfrans','car'):
        overrides.update({'model.H':3,'model.W':5,'model.ref':3,'model.unified_pos':False})
        if variant is not None:overrides['model.linearno_variant']=variant
    p=profile(task,'paper_table8_on_release_model',explicit=overrides)
    return resolve_config(p,family='linearno_history',features={
        'linearno_latent_attnres':mode[1]=='1','linearno_history_k_conditioning':mode[3]=='1'})

def construct(c):
    torch.manual_seed(c['profile_spec']['values']['runtime']['seed'])
    return build_fair_model(c)

def inputs(c,*,n=972,batch=2):
    kw=c['model_spec']['constructor_kwargs'];v=c['base_linearno']['variant']
    gen=torch.Generator().manual_seed(8417)
    def rand(*shape):return torch.randn(*shape,generator=gen)
    if v in ('airfrans','shapenet'):
        from torch_geometric.data import Data
        data=Data(x=rand(n,7),pos=rand(n,2))
        return (data,) if v=='airfrans' else ((data,None),)
    if v in ('conv','conv_temp'):n=kw['H']*kw['W']
    x=rand(batch,n,kw['space_dim'])
    fx=rand(batch,n,kw['fun_dim']) if kw['fun_dim'] else None
    t=rand(batch,1) if kw['Time_Input'] else None
    return x,fx,t
