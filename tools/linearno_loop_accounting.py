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


def _linearno_terms(config, B, N):
    s=config['loop_spec'];m=config['profile_spec']['values']['model'];task=s['task']
    d,h,M,f=m['hidden'],m['heads'],s['resolved_rank'],m['ffn_ratio'];dh=d//h
    conv=m['linearno_variant'] in ('conv','conv_temp');kernel=9 if conv else 1
    outputs=2 if conv or task=='car' else 1
    tau=2*h if m['linearno_variant'] in ('temp','conv_temp','shapenet') else h if task=='airfrans' else 0
    attention=kernel*d*d+d+2*dh*M+dh*dh+outputs*(d*d+d)+tau
    ffn=2*f*d*d+(f+1)*d
    matrices=dict(input_projection=B*N*kernel*d*d,Q_projection=B*N*d*M,K_projection=B*N*d*M,
                  V_projection=B*N*d*dh,K_transpose_V=B*N*d*M,QC=B*N*d*M,
                  output_projection=B*N*outputs*d*d,FFN=B*N*2*f*d*d)
    return dict(d=d,h=h,M=M,f=f,dh=dh,attention=attention,ffn=ffn,
                body=attention+4*d+ffn,matrices=matrices,body_mac=sum(matrices.values()))


def analytic_v2(config,*,B=1,N=None):
    """Independent v2 parameter/call/MAC algebra; no module introspection."""
    s=config['loop_spec'];m=config['profile_spec']['values']['model'];task=s['task']
    N=point_count(config) if N is None else N
    if type(B) is not int or type(N) is not int or min(B,N)<1:raise ValueError('B,N must be positive integers')
    if task in ('airfrans','car') and B!=1:raise ValueError('industrial wrapper is single graph: B=1')
    terms=_linearno_terms(config,B,N);d,h,M=terms['d'],terms['h'],terms['M']
    P,C,R,S=(s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
    operator=terms['attention']+2*d;point_ffn=terms['ffn']+2*d
    head=2*d+d*m['out_dim']+m['out_dim']
    pos=m['ref']**2 if m['unified_pos'] else m['space_dim']
    if task=='airfrans' and m['unified_pos']:pos+=m['space_dim']
    channels=pos+m['fun_dim'];stem=2*d*channels+2*d+2*d*d+d+d
    if m['time_input']:stem+=2*(d*d+d)
    receiver_count=len(router_sources(s));router=2*d*receiver_count
    latent_enabled=s['core_ffn_mode']=='round_specific_latent'
    width=s['latent_ffn']['width'];latent_per=2*d*width+width+3*d if latent_enabled else 0
    parts=dict(stem=stem,prefix=P*terms['body'],shared_operators=C*operator,
               round_specific_point_ffns=C*R*point_ffn,suffix_body=S*terms['body'],
               head=head,routers=router,latent_ffns=C*latent_per)
    stem_mac=B*N*(2*d*channels+2*d*d+(2*d*d if m['time_input'] else 0))
    head_mac=B*N*d*m['out_dim'];sources=router_sources(s);active=sum(k for k in sources if k>1)
    latent_visit_mac=2*B*M*d*width if latent_enabled else 0
    latent_executed_mac=C*R*latent_visit_mac
    # Every round-specific FFN is a unique module, whereas operators and latent
    # FFNs are reused. This number calls each registered module exactly once.
    unique_once=(stem_mac+(P+S)*terms['body_mac']+C*(terms['body_mac']-terms['matrices']['FFN'])
                 +C*R*terms['matrices']['FFN']+C*latent_visit_mac+head_mac)
    executed=stem_mac+(P+C*R+S)*terms['body_mac']+latent_executed_mac+head_mac
    return dict(B=B,N=N,hidden=d,heads=h,head_dim=terms['dh'],M=M,
        variant=m['linearno_variant'],core_ffn_mode=s['core_ffn_mode'],
        unique_depth=P+C+S,executed_depth=P+C*R+S,
        unique_operator_calls=P+C+S,executed_operator_calls=P+C*R+S,
        unique_point_ffn_modules=P+C*R+S,executed_point_ffn_calls=P+C*R+S,
        parameter_parts=parts,parameters=sum(parts.values()),state_key_count=None,
        body_matrix_macs=terms['matrices'],stem_matrix_macs=stem_mac,head_matrix_macs=head_mac,
        latent_width=width,latent_context_shape=[B,h,M,terms['dh']],
        latent_matrix_macs_per_visit=latent_visit_mac,
        latent_matrix_macs=latent_executed_mac,
        unique_module_once_matrix_macs=unique_once,executed_matrix_macs=executed,
        executed_matrix_flops=2*executed,router_receivers=receiver_count,
        router_sources=sources,router_contraction_mac_equivalents=2*B*N*d*active,
        router_rms_square_elements=B*N*d*active,matrix_flops_are_total_flops=False)


def analytic_v3(config, *, B=None, N=None):
    """Dispatch to the independent, tensor-free V3 cost contract.

    V3 keeps its own parameter partitions and matrix/non-matrix accounting;
    this adapter deliberately does not reuse the V1/V2 algebra below.
    ``B`` and ``N`` are passed through only when explicitly requested so the
    frozen representative shapes remain the default.
    """
    from linearno_loop.v3.costs import analytic_cost
    kwargs = {}
    if B is not None:
        kwargs['batch'] = B
    if N is not None:
        kwargs['points'] = N
    return analytic_cost(config, **kwargs)

def analytic_v4(config, *, B=1, N=None):
    """Independent v4 parameter/MAC ledger; never imports the v4 model."""
    from linearno_loop.v4.config import validate_config
    c=validate_config(config); m=c['model']; task=c['task']; H,heads,M=m['hidden_width'],m['heads'],m['actual_M']; dh=H//heads; ratio=m['ffn_ratio']
    if N is None:
        N = 1 if task in ('airfrans','car') else (m['grid_height']*m['grid_width'] if m['variant'] in ('conv','conv_temp') else 32)
    if B<1 or N<1 or (task in ('airfrans','car') and B!=1): raise ValueError('invalid v4 representative shape')
    structured=m['variant'] in ('conv','conv_temp'); kernel=9 if structured else 1
    inproj=kernel*H*H+H
    outproj=2*H*H+2*H if structured or task=='car' else H*H+H
    attn=inproj + 2*dh*M+dh*dh + outproj + 4*H
    if m['variant'] in ('temp','conv_temp','shapenet'): attn += 2*heads
    if task=='airfrans': attn += heads
    body=[attn]*8;head_parameters=2*H+H*m['out_dim']+m['out_dim']
    rmlp={}
    for owner,L in (('first',2),('A',3),('B',3),('C',3),('last',2)):
        W=H*ratio; rmlp[owner]=H*W+W+L*(W*W+W)+W*H+H
    channels=m['fun_dim']+(m['ref']**2 if m['unified_pos'] else m['space_dim'])
    if task=='airfrans' and m['unified_pos']: channels += m['space_dim']
    stem=channels*2*H+2*H+2*H*H+H+H # preprocess + placeholder
    time_parameters=2*(H*H+H) if m['time_input'] else 0
    predictor=0
    if c['temperature_mode']!='base':
        q=dh*M+M+M+1
        k=(dh*M+M+M*M+M) if c['temperature_mode']=='latent_k_point_q' else q
        predictor=8*(q+k)
    output_layers=2 if structured or task=='car' else 1
    attention_macs=B*N*8*(kernel*H*H+4*H*M+H*dh+output_layers*H*H)
    W=H*ratio
    rmlp_macs=B*N*(16*H*W+22*W*W)
    stem_macs=B*N*(2*H*channels+2*H*H+(2*H*H if m['time_input'] else 0))
    head_macs=B*N*H*m['out_dim']
    q_predictor=0;k_predictor=0
    if c['temperature_mode']!='base':
        q_predictor=8*B*heads*N*(dh*M+M)
        k_predictor=(8*B*heads*(dh*M+M*M) if c['temperature_mode']=='latent_k_point_q'
                     else 8*B*heads*N*(dh*M+M))
    matrix_per=attention_macs+rmlp_macs+stem_macs+head_macs+q_predictor+k_predictor
    return dict(version=4,task=task,temperature_mode=c['temperature_mode'],B=B,N=N,hidden=H,heads=heads,head_dim=dh,M=M,
      unique_depth=8,executed_depth=8,parameter_parts=dict(stem=stem,time=time_parameters,
        operators=sum(body),rmlp=sum(rmlp.values()),head=head_parameters,temperature_predictors=predictor),rmlp_owners=rmlp,
      parameters=stem+time_parameters+sum(body)+sum(rmlp.values())+head_parameters+predictor,
      matrix_mac_parts=dict(stem=stem_macs,operators=attention_macs,rmlp=rmlp_macs,head=head_macs,
                            q_temperature=q_predictor,k_temperature=k_predictor),
      matrix_macs=matrix_per, matrix_flops=2*matrix_per,
      non_matrix_operations='LayerNorm/GELU/softmax/residual/reshape counted separately',router_parameters=0,
      no_nxn_or_mxm_attention=True)


def analytic_v5(config, *, B=1, N=None):
    """Independent V5 parameter and matrix-MAC ledger; imports no V5 model."""
    from linearno_loop.v5.config import validate_config
    c = validate_config(config); spec = c['loop_spec']; m = c['profile_spec']['values']['model']
    task = c['task']; H = spec['hidden_width']; heads = spec['heads']; M = spec['actual_M']
    dh = H // heads; E = spec['expert_count']; F = spec['expert_width']
    if N is None:
        N = point_count(c)
    if type(B) is not int or type(N) is not int or min(B, N) < 1:
        raise ValueError('B,N must be positive integers')
    if task in ('airfrans', 'car') and B != 1:
        raise ValueError('industrial wrapper is single graph: B=1')
    P, C, R, S = (spec[key] for key in
                   ('prefix_blocks', 'recurrent_core_blocks', 'loop_repeats', 'suffix_blocks'))
    unique, executed = P + C + S, P + C * R + S
    structured = spec['variant'] in ('conv', 'conv_temp')
    kernel = 9 if structured else 1
    output_layers = 2 if structured or task == 'car' else 1
    active_temperature = 2 * heads if spec['variant'] in ('temp', 'conv_temp', 'shapenet') else 0
    inert_temperature = heads if task == 'airfrans' else 0
    shared_operator = (kernel * H * H + H + dh * dh +
                       output_layers * (H * H + H) + inert_temperature)
    shared_norms = 4 * H
    shared_body = shared_operator + shared_norms
    visit_norms = (4 * H * C * (R - 1)
                   if spec['core_norm_mode'] == 'visit_independent' else 0)
    expert_per_position = E * (2 * H * F + F + H)
    visit_qk_temperature = 2 * dh * M + active_temperature
    router_per_visit = H * E + E
    channels = m['fun_dim'] + (m['ref'] ** 2 if m['unified_pos'] else m['space_dim'])
    if task == 'airfrans' and m['unified_pos']:
        channels += m['space_dim']
    stem = 2 * H * channels + 2 * H + 2 * H * H + H + H
    time = 2 * (H * H + H) if m['time_input'] else 0
    head = 2 * H + H * m['out_dim'] + m['out_dim']
    parts = dict(
        stem=stem, time=time, prefix=P * shared_body, shared_core=C * shared_body,
        suffix_body=S * shared_body, head=head, experts=unique * expert_per_position,
        visit_norms=visit_norms,
        visit_qk_temperature=executed * visit_qk_temperature,
        routers=executed * router_per_visit,
    )
    stem_mac = B * N * (2 * H * channels + 2 * H * H +
                        (2 * H * H if m['time_input'] else 0))
    attention_per_visit = B * N * (kernel * H * H + 4 * H * M + H * dh +
                                   output_layers * H * H)
    router_per_visit_mac = B * N * H * E
    experts_per_visit_mac = B * N * 2 * E * H * F
    head_mac = B * N * H * m['out_dim']
    matrix_parts = dict(
        stem=stem_mac,
        operators=executed * attention_per_visit,
        routers=executed * router_per_visit_mac,
        dense_experts=executed * experts_per_visit_mac,
        head=head_mac,
    )
    matrix_macs = sum(matrix_parts.values())
    return dict(
        version=5, task=task, profile=c['profile'], topology=c['topology_preset'],
        B=B, N=N, hidden=H, heads=heads, head_dim=dh, M=M,
        expert_count=E, expert_width=F, unique_depth=unique,
        executed_depth=executed, parameter_parts=parts, parameters=sum(parts.values()),
        matrix_mac_parts=matrix_parts, matrix_macs=matrix_macs,
        matrix_flops=2 * matrix_macs,
        expert_mixture_mac_equivalents=executed * B * N * E * H,
        matrix_flops_are_total_flops=False,
        non_matrix_operations=(
            'LayerNorm, GELU, Q/K and expert softmax, temperature clamp/divide, '
            'expert weighting/sum, bias, residual, reshape and position/time features excluded'),
        no_nxn_or_mxm_attention=True,
    )

def analytic_pure_linearno(task, *, B=1, N=None, profile='paper_table8_on_release_model'):
    """Pure LinearNO reference under the same matrix-MAC convention as v4."""
    from cdlno.linearno.profiles import resolve_config
    base=resolve_config(task,profile);m=base['values']['model'];H,heads,M,ratio=m['hidden'],m['heads'],m['linearno_rank'],m['ffn_ratio'];dh=H//heads
    if N is None:N=1 if task in ('airfrans','car') else (m['H']*m['W'] if m['linearno_variant'] in ('conv','conv_temp') or m['unified_pos'] else 32)
    structured=m['linearno_variant'] in ('conv','conv_temp');kernel=9 if structured else 1;outputs=2 if structured or task=='car' else 1
    temperature=2*heads if m['linearno_variant'] in ('temp','conv_temp','shapenet') else heads if task=='airfrans' else 0
    attention=kernel*H*H+H+2*dh*M+dh*dh+outputs*(H*H+H)+temperature
    W=H*ratio;point=2*H*W+W+H;body=attention+4*H+point;head=2*H+H*m['out_dim']+m['out_dim']
    channels=m['fun_dim']+(m['ref']**2 if m['unified_pos'] else m['space_dim'])
    if task=='airfrans' and m['unified_pos']:channels+=m['space_dim']
    stem=2*H*channels+2*H+2*H*H+H+H+(2*(H*H+H) if m['time_input'] else 0)
    params=stem+8*body+head
    stem_mac=B*N*(2*H*channels+2*H*H+(2*H*H if m['time_input'] else 0))
    attention_mac=B*N*8*(kernel*H*H+4*H*M+H*dh+outputs*H*H)
    point_mac=B*N*8*2*H*W;head_mac=B*N*H*m['out_dim'];mac=stem_mac+attention_mac+point_mac+head_mac
    return dict(task=task,B=B,N=N,parameters=params,matrix_macs=mac,matrix_flops=2*mac,
      matrix_mac_parts=dict(stem=stem_mac,operators=attention_mac,point_ffn=point_mac,head=head_mac),matrix_flops_are_total_flops=False)


def analytic(config,*,B=1,N=None):
    """Only integer algebra from schema; no model construction/module introspection."""
    if config.get('architecture')=='partial_share_feature_gate_v5':
        return analytic_v5(config,B=B,N=N)
    if config.get('architecture')=='resmlp_dual_temp_v4':
        return analytic_v4(config,B=B,N=N)
    if config.get('architecture_extension')=='loop_linearno_latent_adapter_v3':
        # V3's public default is its frozen representative batch/point shape;
        # explicit B/N still select a synthetic shape.  V1/V2 keep their
        # historical B=1 default below.
        return analytic_v3(config, B=None if B == 1 and N is None else B, N=N)
    if config.get('architecture_extension')=='loop_linearno_ffn_v2':
        return analytic_v2(config,B=B,N=N)
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
    if getattr(model, 'architecture', None) == 'partial_share_feature_gate_v5':
        from cdlno.linearno_loop.v5.checkpoint import _parameter_groups
        return _parameter_groups(model)
    if getattr(model, 'architecture', None) == 'resmlp_dual_temp_v4':
        parts=Counter({k:0 for k in ('stem','time','operators','rmlp','head','temperature_predictors')})
        for name,p in model.named_parameters():
            if name.startswith('time_fc.'):
                part='time'
            elif name.startswith('loop.blocks.'):
                part='temperature_predictors' if ('.q_temperature.' in name or '.k_temperature.' in name) else 'operators'
            elif name.startswith('loop.rmlp.'):
                part='rmlp'
            elif name.startswith(('final_norm.','head.')):
                part='head'
            else:
                part='stem'
            parts[part]+=p.numel()
        return dict(parts)
    if getattr(model, 'architecture_extension', None) == 'loop_linearno_latent_adapter_v3':
        parts=Counter({k:0 for k in ('stem','time','prefix','shared_core','suffix','head','latent','adapter','router')})
        for name,p in model.named_parameters():
            if name.startswith(('loop.rb_', 'loop.lb_')):
                part='router'
            elif '.latent_processor.' in name:
                part='latent'
            elif '.adapter.' in name:
                part='adapter'
            elif name.startswith('loop.prefix.'):
                part='prefix'
            elif name.startswith('loop.core.'):
                part='shared_core'
            elif name.startswith('loop.suffix.'):
                part='head' if ('.ln_3.' in name or '.mlp2.' in name) else 'suffix'
            elif name.startswith('time_fc.'):
                part='time'
            else:
                part='stem'
            parts[part]+=p.numel()
        return dict(parts)
    if hasattr(model.loop,'core_operators'):
        parts=Counter({k:0 for k in ('stem','prefix','shared_operators',
            'round_specific_point_ffns','suffix_body','head','routers','latent_ffns')})
        for name,p in model.named_parameters():
            if name.startswith(('loop.rb_','loop.lb_')):part='routers'
            elif name.startswith('loop.latent_ffns.'):part='latent_ffns'
            elif '.ln_3.' in name or '.mlp2.' in name:part='head'
            elif name.startswith('loop.prefix.'):part='prefix'
            elif name.startswith('loop.core_operators.'):part='shared_operators'
            elif name.startswith('loop.core_ffns.'):part='round_specific_point_ffns'
            elif name.startswith('loop.suffix.'):part='suffix_body'
            else:part='stem'
            parts[part]+=p.numel()
        return dict(parts)
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
    def __init__(self,scope,attentions,B,N,expert_blocks=None):
        super().__init__();self.scope=scope;self.attentions=attentions;self.B=B;self.N=N
        self.expert_blocks=expert_blocks or {}
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
            elif loc in self.expert_blocks:
                if tuple(out.shape)!=(self.B,self.N,self.expert_blocks[loc]) or args[1] not in (-1,2):
                    self.violations.append(row)
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
    expert_blocks={}
    try:
        from cdlno.linearno_loop.v5.operator import PartialSharedLinearNOOperator
        from cdlno.linearno_loop.v5.core import V5PhysicalBlock
        attentions.update({n:m for n,m in model.named_modules()
                           if isinstance(m,PartialSharedLinearNOOperator)})
        expert_blocks.update({n:m.Attn.visits[0].router.out_features
                              for n,m in model.named_modules()
                              if isinstance(m,V5PhysicalBlock)})
    except ImportError:
        pass
    # The v2 owner deliberately calls ``Attn.forward_with_context`` directly,
    # so ATen operations are scoped to SharedCoreOperator rather than Attn.
    # Register that exact outer scope with the same LinearNO signature.
    try:
        from cdlno.linearno_loop.v2.core import SharedCoreOperator
        attentions.update({n:m.Attn for n,m in model.named_modules()
                           if isinstance(m,SharedCoreOperator)})
    except ImportError:
        pass
    training={module:module.training for module in model.modules()};model.eval()
    try:
        with scoped_modules(model) as scope,torch.no_grad():
            ledger=Ledger(scope,attentions,B,N,expert_blocks)
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
