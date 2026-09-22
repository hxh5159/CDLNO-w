"""Tensor-free analytic parameter/MAC oracle; no model or accounting imports.

Matrix operations use 1 MAC=2 FLOPs. Scalar/non-matrix work is enumerated,
never silently counted as zero runtime. Forward only, no backward/optimizer.
"""
from .config import validate_config
from .contracts import integer

COST_VERSION='linearno-v3-matrix-cost-v1'
POINTS={'airfoil':11271,'darcy':7225,'elasticity':972,'pipe':16641,
        'plasticity':3131,'ns':4096,'airfrans':32000,'car':32186}
BATCHES={'airfoil':4,'darcy':4,'elasticity':1,'pipe':4,'plasticity':8,'ns':2,'airfrans':1,'car':1}


def source_counts(C,R,mode):
    if mode=='rb_attnres':
        return [s for r in range(R) for s in [r+1]+[r+2]*(2*C-1)]+[R+1]
    if mode=='lb_attnres_1_over_r':return list(range(2,R+2))
    return []


def analytic_cost(config, *, batch=None, points=None):
    c=validate_config(config);s=c['loop_spec'];m=c['profile_spec']['values']['model'];task=s['task']
    B=integer(BATCHES[task] if batch is None else batch,'batch',1)
    N=integer(POINTS[task] if points is None else points,'points',1)
    H,h,M,Dz=s['hidden_width'],s['heads'],s['actual_M'],s['latent_width'];d=H//h
    P,C,R,S=(s[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
    U,D=P+C+S,P+C*R+S;f=m['ffn_ratio'];variant=m['linearno_variant'];out=m['out_dim']
    kernel=9 if variant in ('conv','conv_temp') else 1
    output_layers=2 if kernel==9 or task=='car' else 1
    temperature=2*h if variant in ('temp','conv_temp','shapenet') else h if task=='airfrans' else 0
    channels=(m['ref']**2 if m['unified_pos'] else m['space_dim'])+m['fun_dim']
    if task=='airfrans' and m['unified_pos']:channels+=m['space_dim']
    operator=kernel*H*H+H+2*d*M+d*d+output_layers*(H*H+H)+temperature
    point_ffn=2*f*H*H+(f+1)*H
    body=dict(operator=operator,operator_norm=2*H,point_ffn=point_ffn,point_ffn_norm=2*H)
    mac_body=dict(input_projection=B*N*kernel*H*H,q_projection=B*N*H*M,
        k_projection=B*N*H*M,v_projection=B*N*H*d,KtV=B*N*H*M,
        Q_readout=B*N*H*M,output_projection=B*N*output_layers*H*H,point_ffn=2*B*N*f*H*H)
    latent=s['latent_enabled'];adapter=s['adapter_mode']!='none';r=s['adapter_rank']
    # Disjoint groups: latent/adapter/router outside the shared backbone count.
    parameters=dict(stem=dict(preprocess=2*channels*H+2*H*H+3*H,placeholder=H),
        time=dict(projections=2*(H*H+H) if m['time_input'] else 0),
        prefix={k:P*v for k,v in body.items()},shared_core={k:C*v for k,v in body.items()},
        suffix={k:S*v for k,v in body.items()},head=dict(norm=2*H,projection=H*out+out),
        latent=dict(total=C*(2*H*Dz+Dz+3*H) if latent else 0),
        adapter=dict(total=C*2*r*(d+M) if adapter else 0),
        router=dict(total=s['attnres']['receiver_count']*2*H))
    macs=dict(stem=B*N*(2*channels*H+2*H*H),time=2*B*N*H*H if m['time_input'] else 0,
        prefix={k:P*v for k,v in mac_body.items()},shared_core={k:C*R*v for k,v in mac_body.items()},
        suffix={k:S*v for k,v in mac_body.items()},head=B*N*H*out,
        latent=2*B*C*R*M*H*Dz if latent else 0,
        adapter=2*B*C*h*N*r*(d+M) if adapter else 0)
    counts=source_counts(C,R,s['residual_mode']);router=2*B*N*H*sum(counts)
    group_params={k:sum(v.values()) for k,v in parameters.items()}
    group_macs={k:sum(v.values()) if isinstance(v,dict) else v for k,v in macs.items()}
    non_matrix=dict(included_in_matrix_macs=False,
        operations=['Q softmax over M','K softmax over N','temperature/clamp if active',
                    'LayerNorm/RMSNorm','GELU','bias/residual additions','dropout if active',
                    'source softmax','elementwise residual scaling','reshape/transpose/indexing',
                    'position/reference distances','time sin/cos if active'],
        point_norm_calls=2*D+1,latent_norm_calls=C*R if latent else 0,
        query_softmax_rows=B*h*N*D,key_softmax_rows=B*h*M*D,
        source_softmax_rows=B*N*len(counts),router_normalized_scalars=B*N*H*sum(counts),
        note='Operation inventory/selected element counts only; no latency or complete scalar FLOPs estimate')
    canonical=(s['cost_profile']!='custom' and s['residual_mode']=='sr_1_over_r'
               and latent and adapter and r==4 and s['adapter_alpha']==4.0
               and s['profile']=='paper_table8_on_release_model'
               and B==BATCHES[task] and N==POINTS[task])
    return dict(cost_version=COST_VERSION,scope='analytic_forward_only_not_measured',task=task,batch=B,points=N,
        shape_source='frozen_representative_B_N' if batch is None and points is None else 'explicit_synthetic_B_N',
        unique_depth=U,executed_depth=D,operator_instances=U,point_ffn_instances=U,
        operator_visits=D,point_ffn_visits=D,head_visits=1,
        parameters=parameters,parameter_groups=group_params,total_parameters=sum(group_params.values()),
        matrix_mac_parts=macs,matrix_mac_groups=group_macs,matrix_macs=sum(group_macs.values()),
        matrix_flops=2*sum(group_macs.values()),router_contraction_macs=router,
        forward_macs_including_router=sum(group_macs.values())+router,
        router_source_counts=counts,non_matrix=non_matrix,
        profile_match_claim_applies=canonical,
        profile_match_scope='paper_table8_on_release_model,representative_B_N,on/on,r4,alpha4,SR,matrix_only; approximate, not exact',
        actual_parameters=None,latency=None,peak_memory=None,epoch_time=None)
