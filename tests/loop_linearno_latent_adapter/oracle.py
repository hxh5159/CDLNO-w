"""Independent tensor-shape inventory oracle, written before V3 cost code.

No production cost helper or model import. This enumerates conceptual tensors
and dense contractions, including repeated visits only on the MAC side.
"""
from math import prod

def count(task, H, Dz, M, heads, P, C, R, S, f, B, N, input_channels,
          outputs, variant, time_input, latent=True, adapter=True, rank=4,
          residual='sr_1_over_r'):
    d=H//heads
    tensors=[(2*H,input_channels),(2*H,),(H,2*H),(H,),(H,)]
    macs=B*N*(2*H*input_channels+2*H*H)
    if time_input:
        tensors += [(H,H),(H,),(H,H),(H,)]
        macs += 2*B*N*H*H
    block=[(H,H,3,3) if variant.startswith('conv') else (H,H),(H,),
           (M,d),(M,d),(d,d),(H,H),(H,),
           (H,),(H,),(H,),(H,), (f*H,H),(f*H,),(H,f*H),(H,)]
    if variant in ('conv','conv_temp','shapenet'): block += [(H,H),(H,)]
    if variant in ('temp','conv_temp','shapenet'):block += [(heads,1,1),(heads,1,1)]
    if variant=='airfrans':block += [(heads,1,1)]
    for _ in range(P+C+S):tensors.extend(block)
    visits=P+C*R+S
    projection=H*H*(9 if variant.startswith('conv') else 1)
    out=H*H*(2 if variant in ('conv','conv_temp','shapenet') else 1)
    macs+=visits*B*N*(projection+2*H*M+H*d+2*H*M+out+2*f*H*H)
    if latent:
        for _ in range(C): tensors += [(H,),(H,),(Dz,H),(Dz,),(H,Dz),(H,)]
        macs+=C*R*B*M*(H*Dz+Dz*H)
    if adapter:
        for _ in range(C): tensors += [(rank,d),(M,rank),(rank,d),(M,rank)]
        macs+=C*B*heads*N*(d*rank+rank*M)*2
    tensors += [(H,),(H,),(outputs,H),(outputs,)]
    macs+=B*N*H*outputs
    receivers=2*C*R+1 if residual=='rb_attnres' else R if residual=='lb_attnres_1_over_r' else 0
    tensors.extend([(H,),(H,)]*receivers)
    sources=[]
    if residual=='rb_attnres':
        for r in range(R):sources += [r+1]+[r+2]*(2*C-1)
        sources += [R+1]
    if residual=='lb_attnres_1_over_r':sources=list(range(2,R+2))
    return dict(parameters=sum(prod(s) for s in tensors),matrix_macs=macs,
                router_contraction_macs=2*B*N*H*sum(sources),source_counts=sources)
