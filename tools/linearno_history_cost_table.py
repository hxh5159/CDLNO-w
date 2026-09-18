#!/usr/bin/env python3
"""Analytic matrix work from registered modules, checked against live operators."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import torch
from torch import nn
from cdlno.linearno.profiles import resolve_config as profile
from cdlno.linearno_history.config import resolve_config
from linearno_history_support import MODES,configuration,construct,inputs
from linearno_history_performance import audit

POINTS=dict(airfoil=221*51,darcy=85*85,elasticity=972,pipe=129*129,ns=64*64,plasticity=101*31,airfrans=32000,car=32186)

def theoretical(model,b,n):
    f=model.blocks[0].Attn;h,d,m=f.heads,f.dim_head,f.rank;L=len(model.blocks)
    counts=Counter();parts=Counter()
    for name,p in model.named_parameters():parts[name.split('.')[0]]+=p.numel()
    for name,module in model.named_modules():
        if name.startswith(('latent_attnres','history_k')):continue
        tokens=b*n*(h if any(name.endswith('.Attn.'+x) for x in ('to_q','to_k','to_v')) else 1)
        if isinstance(module,nn.Linear):counts[name]=tokens*module.in_features*module.out_features
        if isinstance(module,nn.Conv2d):counts[name]=b*n*module.out_channels*(module.in_channels//module.groups)*math.prod(module.kernel_size)
    counts['base_KtV_QC']=2*b*h*n*m*d*L
    if hasattr(model,'latent_attnres'):
        for l in range(1,L):
            counts[f'A.receiver{l}.query']=b*h*m*d*d
            counts[f'A.receiver{l}.per_source_KVO']=3*l*b*h*m*d*d
            counts[f'A.receiver{l}.cross_QK_AV']=2*l*b*h*m*m*d
    if hasattr(model,'history_k'):
        for l in range(1,L):
            counts[f'K.receiver{l}.slot_query']=m*d*d
            counts[f'K.receiver{l}.bank_KV']=2*b*h*l*m*d*d
            counts[f'K.receiver{l}.cross_EK_AV']=2*b*h*m*l*m*d
            counts[f'K.receiver{l}.point_slot']=b*h*n*m*d
    return dict(parameters=sum(parts.values()),parameters_by_group=dict(parts),matrix_macs=sum(counts.values()),matrix_flops=2*sum(counts.values()),macs_by_operation=dict(counts))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    parity=[]
    for v in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
        task={'airfrans':'airfrans','shapenet':'car'}.get(v,'elasticity')
        for mode in MODES:
            c=configuration(mode,task=task,variant=v if task=='elasticity' else None)
            model=construct(c);arg=inputs(c,n=15);b=1 if task in ('airfrans','car') else 2
            expected=theoretical(model,b,15);actual=audit(model,arg,15)
            assert expected['matrix_macs']==actual['matrix_macs'],(v,mode,expected['matrix_macs'],actual['matrix_macs'])
            parity.append(dict(variant=v,mode=mode,matrix_macs=actual['matrix_macs'],status='EXACT'))
    rows=[]
    for task,n in POINTS.items():
        for L in range(4,9):
            for mode in MODES:
                c=resolve_config(profile(task,explicit={'model.layers':L}),family='linearno_history',features={
                    'linearno_latent_attnres':mode[1]=='1','linearno_history_k_conditioning':mode[3]=='1'})
                model=construct(c);b=c['profile_spec']['values']['training']['batch_size'];f=model.blocks[0].Attn
                result=theoretical(model,b,n)
                rows.append(dict(task=task,depth=L,mode=mode,B=b,N=n,d=f.dim,heads=f.heads,head_dim=f.dim_head,M=f.rank,
                    variant=f.variant,**result,
                    temporal='one model call; NS training10 calls/one update; Plasticity20calls/20updates' if task in ('ns','plasticity') else 'one model call'))
    (out/'results.json').write_text(json.dumps(dict(note='Actual registered parameter counts; forward matrix MAC/FLOPs only. Norm, softmax, scalar fusion, point centering, GELU etc are separately inventoried by live profiler. No claim of total scalar FLOPs.',parity=parity,rows=rows),indent=2))
    print('24 analytic/live matrix MAC equalities PASS; 160 official task/mode/depth costs recorded')
if __name__=='__main__':main()
