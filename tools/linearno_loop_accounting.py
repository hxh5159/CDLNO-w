"""Independent closed-form counts plus actual module/ATen accounting for LL9.

Dense MAC: one multiplication with accumulation; matrix FLOPs = 2*MAC.
Scalar, bias, residual, norm, activation, positional distance/sinusoid and
softmax work is explicitly NOT included in matrix FLOPs. Router contraction
MAC equivalents are separate, since the implementation uses mul+sum, not mm.
"""
from collections import Counter
from contextlib import contextmanager
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from .linearno_loop_support import point_count


def router_sources(spec):
    c,r=spec['recurrent_core_blocks'],spec['loop_repeats']
    mode=spec['residual_mode']
    if mode=='sr_1_over_r':return []
    if mode=='lb_attnres_1_over_r':return list(range(2,r+2))
    return [s for j in range(r) for s in [j+1]+[j+2]*(2*c-1)]+[r+1]


def analytic(config,*,B=1,N=None):
    """Only integer algebra from schema; no model construction/module introspection."""
    s=config['loop_spec'];m=config['profile_spec']['values']['model'];task=s['task']
    d,h,M,f=m['hidden'],m['heads'],s['resolved_rank'],m['ffn_ratio'];dh=d//h
    P,C,R,S=(s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
    N=point_count(config) if N is None else N
    if type(B) is not int or type(N) is not int or min(B,N)<1:raise ValueError('B,N must be positive integers')
    if task in ('airfrans','car') and B!=1:raise ValueError('industrial wrapper is single graph: B=1')
    conv=m['linearno_variant'] in ('conv','conv_temp');kernel=9 if conv else 1
    outputs=2 if conv or task=='car' else 1
    tau=2*h if m['linearno_variant'] in ('temp','conv_temp','shapenet') else h if task=='airfrans' else 0
    # Shared-across-head Q/K/V weights, in contrast to executed per-head MACs.
    attention=kernel*d*d+d+2*dh*M+dh*dh+outputs*(d*d+d)+tau
    ffn=2*f*d*d+(f+1)*d
    body=attention+4*d+ffn
    head=2*d+d*m['out_dim']+m['out_dim']
    pos=m['ref']**2 if m['unified_pos'] else m['space_dim']
    if task=='airfrans' and m['unified_pos']:pos+=m['space_dim']
    channels=pos+m['fun_dim']
    stem=2*d*channels+2*d+2*d*d+d+d
    if m['time_input']:stem+=2*(d*d+d)
    receiver_count=len(router_sources(s));router=2*d*receiver_count
    parts=dict(stem=stem,prefix=P*body,shared_core=C*body,suffix_body=S*body,head=head,router=router)
    matrices=dict(input_projection=B*N*kernel*d*d,Q_projection=B*N*d*M,K_projection=B*N*d*M,
                  V_projection=B*N*d*dh,K_transpose_V=B*N*d*M,QC=B*N*d*M,
                  output_projection=B*N*outputs*d*d,FFN=B*N*2*f*d*d)
    body_mac=sum(matrices.values());stem_mac=B*N*(2*d*channels+2*d*d+(2*d*d if m['time_input'] else 0))
    head_mac=B*N*d*m['out_dim']
    sources=router_sources(s);active=sum(k for k in sources if k>1)
    return dict(B=B,N=N,hidden=d,heads=h,head_dim=dh,M=M,variant=m['linearno_variant'],
        unique_depth=P+C+S,executed_depth=P+C*R+S,parameter_parts=parts,parameters=sum(parts.values()),
        dead_temperature_parameters=(P+C+S)*h if task=='airfrans' else 0,
        body_matrix_macs=matrices,stem_matrix_macs=stem_mac,head_matrix_macs=head_mac,
        unique_block_once_matrix_macs=stem_mac+(P+C+S)*body_mac+head_mac,
        executed_matrix_macs=stem_mac+(P+C*R+S)*body_mac+head_mac,
        executed_matrix_flops=2*(stem_mac+(P+C*R+S)*body_mac+head_mac),
        router_receivers=receiver_count,router_sources=sources,
        router_contraction_mac_equivalents=2*B*N*d*active,
        router_rms_square_elements=B*N*d*active,
        matrix_flops_are_total_flops=False)


def measured_parameters(model):
    parts=Counter({k:0 for k in ('stem','prefix','shared_core','suffix_body','head','router')})
    for name,p in model.named_parameters():
        if name.startswith(('loop.rb_','loop.lb_')):part='router'
        elif '.ln_3.' in name or '.mlp2.' in name:part='head'
        elif name.startswith('loop.prefix.'):part='prefix'
        elif name.startswith('loop.core.'):part='shared_core'
        elif name.startswith('loop.suffix.'):part='suffix_body'
        else:part='stem'
        parts[part]+=p.numel()
    return dict(parts)


class Ledger(TorchDispatchMode):
    def __init__(self,scope,attentions,B,N):
        super().__init__();self.scope=scope;self.attentions=attentions;self.B=B;self.N=N
        self.macs=Counter();self.inventory={};self.contractions=[];self.softmax=[];self.violations=[]
        self.attention_bmm=Counter();self.router_contractions=0;self.peak_tensor_elements=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));name=str(func)
        ts=[t for t in tree_flatten(out)[0] if isinstance(t,torch.Tensor)]
        loc=self.scope[-1] if self.scope else '<wrapper>'
        entry=self.inventory.setdefault(loc+'::'+name,dict(calls=0,output_elements=0))
        entry['calls']+=1;entry['output_elements']+=sum(t.numel() for t in ts)
        self.peak_tensor_elements=max(self.peak_tensor_elements,max((t.numel() for t in ts),default=0))
        if name in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            lhs,rhs=args[1:3] if name=='aten.addmm.default' else args[:2]
            self.macs[loc]+=out.numel()*lhs.shape[-1]
            if name=='aten.bmm.default':
                a=self.attentions.get(loc);kind='UNEXPECTED'
                if a is not None:
                    bh=self.B*a.heads;n=self.N;m=a.rank;d=a.dim_head
                    ktv=((bh,m,n),(bh,n,d));qc=((bh,n,m),(bh,m,d))
                    expected=ktv if self.attention_bmm[loc]%2==0 else qc
                    if (tuple(lhs.shape),tuple(rhs.shape))==expected:kind='K^T V' if self.attention_bmm[loc]%2==0 else 'QC'
                    self.attention_bmm[loc]+=1
                row=dict(module=loc,kind=kind,lhs=list(lhs.shape),rhs=list(rhs.shape),output=list(out.shape))
                self.contractions.append(row)
                if kind=='UNEXPECTED':self.violations.append(row)
        elif name=='aten.convolution.default':self.macs[loc]+=out.numel()*math.prod(args[1].shape[1:])
        if 'softmax' in name:
            row=dict(module=loc,shape=list(out.shape),axis=args[1]);self.softmax.append(row)
            if loc in self.attentions:
                a=self.attentions[loc]
                if tuple(out.shape)!=(self.B,a.heads,self.N,a.rank) or args[1] not in (-1,-2,2,3):self.violations.append(row)
            elif not loc.startswith(('loop.rb_','loop.lb_')):self.violations.append(row)
        # Two contractions: normalized-key/query dot over H, raw-source mix over S.
        if name=='aten.sum.dim_IntList' and loc.startswith(('loop.rb_','loop.lb_')):
            self.router_contractions+=args[0].numel()
        return out


@contextmanager
def scoped_modules(model):
    scope=[];handles=[]
    for name,module in model.named_modules():
        def enter(m,a,tag=name or '<model>'):scope.append(tag)
        def leave(m,a,y):scope.pop()
        handles.extend([module.register_forward_pre_hook(enter),module.register_forward_hook(leave,always_call=True)])
    try:yield scope
    finally:
        for handle in handles:handle.remove()


def audit(model,args,*,B,N):
    """Explicit opt-in forward profiler; no backward/timing hooks survive exit."""
    from cdlno.linearno.attention import LinearNOAttention
    attentions={n:m for n,m in model.named_modules() if isinstance(m,LinearNOAttention)}
    training={module:module.training for module in model.modules()};model.eval()
    try:
        with scoped_modules(model) as scope,torch.no_grad():
            ledger=Ledger(scope,attentions,B,N)
            with ledger:model(*args)
    finally:
        for module,state in training.items():module.training=state
    return dict(parameter_parts=measured_parameters(model),matrix_macs=sum(ledger.macs.values()),
        matrix_flops=2*sum(ledger.macs.values()),matrix_macs_by_module=dict(ledger.macs),
        router_contraction_mac_equivalents=ledger.router_contractions,
        scalar_and_tensor_operator_inventory=ledger.inventory,contractions=ledger.contractions,
        softmax=ledger.softmax,forbidden_attention=ledger.violations,
        largest_observed_tensor_elements=ledger.peak_tensor_elements,
        peak_memory_note='largest single forward tensor, NOT allocator peak/live activation memory')
