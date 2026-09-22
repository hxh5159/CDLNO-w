"""LAA4 synthetic saved-profile fixtures, factories and transient hook traces."""
import importlib
from unittest.mock import patch

import torch
from cdlno.linearno.attention import initialize_release_weights
from cdlno.linearno.profiles import resolve_config as base_resolve
from cdlno.linearno_loop.core import LinearNOLoopCore
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR

MODES=('sr_1_over_r','rb_attnres','lb_attnres_1_over_r')
TASKS=('ns','elasticity','plasticity','darcy','airfrans','car')
ROWS=[];COST_ROWS=[];GPU_ROWS=[];PROFILE_ROWS=[]


def configuration(task='elasticity',preset='p2_c2_r2_s2',mode=MODES[0],latent=True,adapter=True,
                  dropout=0.,**changes):
    # Synthetic catalog snapshot ONLY in tests: H/W=2x3, unchanged protocol.
    # Resolve it before V3, which must consistently replay the saved grid facts.
    contract={} if task in ('airfrans','car') else {'contract':'standard_temporal_l5' if task in ('ns','plasticity') else 'standard_static_l4'}
    base=base_resolve(task,'paper_table8_on_release_model',explicit={'model.H':2,'model.W':3,'model.dropout':dropout},**contract)
    opts=dict(architecture=ARCHITECTURE_SELECTOR,cost_profile='custom',hidden_width=6,heads=2,actual_M=5,
              latent_width=7,topology_preset=preset,residual_mode=mode,latent_enabled=latent,
              adapter_mode='bilateral_qk_lowrank_second_visit' if adapter else 'none',adapter_rank=2,adapter_alpha=3.)
    opts.update(changes)
    with patch('linearno_loop.v3.config._profile',return_value=base):return resolve_config(task,options=opts)


def factory(c,records=None):
    s=c['loop_spec'];m=c['profile_spec']['values']['model'];task=s['task']
    def create(*,last_layer):
        if task in ('airfrans','car'):
            module=importlib.import_module('cdlno.linearno.'+('airfrans' if task=='airfrans' else 'shapenet'))
            cls=getattr(module,'AirfRANSBlock' if task=='airfrans' else 'ShapeNetBlock')
            b=cls(s['hidden_width'],s['heads'],s['actual_M'],s['ffn_ratio'],m['dropout'],m['out_dim'],last_layer)
        else:
            cls=importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO').LinearNOBlock
            b=cls(hidden=s['hidden_width'],heads=s['heads'],rank=s['actual_M'],variant=s['variant'],
                dropout=m['dropout'],mlp_ratio=s['ffn_ratio'],H=s['grid_height'],W=s['grid_width'],last_layer=last_layer,out_dim=m['out_dim'])
        if records is not None:records.append((id(b),id(b.Attn),{k:id(v) for k,v in b.named_parameters()}))
        return b
    return create


def make(c,*,v1=False,dtype=torch.float64,device='cpu',records=None):
    from cdlno.linearno_loop.v3.core import LinearNOLoopCoreV3
    s=c['loop_spec']
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(c['fair_comparison']['public_backbone_seed'])
        m=(LinearNOLoopCore(**{k:s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks','residual_mode')},block_factory=factory(c,records))
           if v1 else LinearNOLoopCoreV3(config=c,block_factory=factory(c,records)))
        m.apply(initialize_release_weights)
    m.to(device=device,dtype=dtype)
    if not v1:m.install_features()
    return m


def excite(m):
    with torch.no_grad():
        for i,(name,p) in enumerate(m.named_parameters()):
            if name.endswith('Attn.temperature'):continue
            v=torch.arange(p.numel(),dtype=p.dtype,device=p.device).reshape_as(p)
            if 'temperature_' in name or 'tempreature_' in name:p.fill_(.55)
            else:
                p.copy_(torch.sin(v*.71+i)*.17)
                if name.endswith(('norm_scale','norm.weight','ln_1.weight','ln_2.weight','ln_3.weight')):p.add_(1.)
    return m


def inputs(c,*,dtype=torch.float64,device='cpu',batch=2):
    H=c['loop_spec']['hidden_width'];v=torch.arange(batch*6*H,dtype=dtype,device=device)
    return (v*.37).sin().reshape(batch,6,H).requires_grad_()


def close(a,b,name,dtype,errors):
    atol,rtol=(2e-11,2e-9) if dtype==torch.float64 else (5e-6,1e-4)
    errors[name]=float((a.detach().double()-b.detach().double()).abs().max())
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


class Trace:
    def __init__(self,core):
        self.core=core;self.steps=[];self.routes=[];self.visits=[];self.events=[];self.handles=[];self.raw_sources=[]
    def __enter__(self):
        for group in ('prefix','core','suffix'):
            for i,physical in enumerate(getattr(self.core,group)):
                for branch,ln,mod in ((0,physical.block.ln_1,physical.block.Attn),(1,physical.block.ln_2,physical.block.mlp)):
                    def pre(m,a,group=group,i=i,branch=branch):
                        self.events.append((group,i,branch))
                        if group=='core':self.steps.append(dict(position=i,branch=branch,h=a[0]))
                    def post(m,a,y,group=group):
                        if group=='core':self.steps[-1]['u']=y
                    self.handles.extend([ln.register_forward_pre_hook(pre),mod.register_forward_hook(post)])
                self.handles.append(physical.register_forward_hook(lambda m,a,y,group=group,i=i:self.visits.append((group,i,a[0],y))))
        for name,mod in self.core.named_modules():
            if name.startswith(('rb_receivers.','rb_output','lb_boundaries.','lb_output')) and hasattr(mod,'source_weights'):
                def receive(m,a,y,name=name):
                    self.routes.append(dict(key=name,sources=a[0],weights=m.source_weights(a[0]).permute(1,2,0),h=y))
                self.handles.append(mod.register_forward_hook(receive))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
