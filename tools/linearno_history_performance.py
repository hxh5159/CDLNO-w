#!/usr/bin/env python3
"""Bounded CPU synthetic performance and operation ledger; never load data.

Matrix FLOPs use 2 per MAC. Scalar/reduction/transcendental work is inventoried
separately, including all softmax/norm/GELU/bias/gates; 2*MAC is not total FLOPs.
Controls are predeclared global-rank nearest-count matches, never fit to test.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from linearno_history_support import configuration,construct,inputs,MODES

class Ledger(TorchDispatchMode):
    def __init__(self,scope,n):
        super().__init__();self.scope=scope;self.n=n;self.macs=Counter();self.ops={};self.matrices=[];self.softmax=[];self.nn=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));name=str(func)
        tensors=[t for t in tree_flatten(out)[0] if isinstance(t,torch.Tensor)]
        loc=self.scope[-1] if self.scope else '<wrapper>'
        entry=self.ops.setdefault(name,dict(calls=0,output_elements=0,input_elements=0))
        entry['calls']+=1;entry['output_elements']+=sum(t.numel() for t in tensors)
        entry['input_elements']+=sum(t.numel() for t in tree_flatten(args)[0] if isinstance(t,torch.Tensor))
        if name in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            lhs,rhs=(args[1:3] if name=='aten.addmm.default' else args[:2]);mac=out.numel()*lhs.shape[-1]
            self.macs[loc]+=mac
            if name=='aten.bmm.default':self.matrices.append(dict(scope=loc,lhs=list(lhs.shape),rhs=list(rhs.shape),output=list(out.shape),macs=mac))
        elif name=='aten.convolution.default':
            self.macs[loc]+=out.numel()*math.prod(args[1].shape[1:])
        if 'softmax' in name:self.softmax.append(dict(scope=loc,shape=list(out.shape),dim=args[1]))
        for t in tensors:
            if t.ndim>=2 and tuple(t.shape[-2:])==(self.n,self.n):self.nn.append(dict(op=name,scope=loc,shape=list(t.shape)))
        return out


def audit(model,arg,n,*,training=False):
    names={id(m):name or '<model>' for name,m in model.named_modules()};scope=[];original=nn.Module._call_impl
    def called(m,*a,**kw):
        location=names.get(id(m),type(m).__name__)
        if type(m).__name__ in ('LatentSummaryAttnRes','HistoryConditionedK'):
            location += '.receiver_'+str(a[0])
        scope.append(location)
        try:return original(m,*a,**kw)
        finally:scope.pop()
    ledger=Ledger(scope,n)
    with patch.object(nn.Module,'_call_impl',called),ledger,torch.no_grad():model.train(training)(*arg)
    parts=Counter()
    for name,p in model.named_parameters():
        prefix=name.split('.')[0]
        if prefix=='blocks':prefix='.'.join(name.split('.')[:2])
        parts[prefix]+=p.numel()
    return dict(parameters=sum(p.numel() for p in model.parameters()),parameter_components=dict(parts),
        matrix_macs=sum(ledger.macs.values()),matrix_flops=2*sum(ledger.macs.values()),macs_by_module=dict(ledger.macs),
        scalar_reduction_transcendental_inventory=ledger.ops,bmm=ledger.matrices,softmax=ledger.softmax,N_by_N=ledger.nn)


def measure(c,n,batch,warmup,steps):
    model=construct(c);arg=inputs(c,n=n,batch=batch)
    costs=audit(model,arg,n)
    assert not costs['N_by_N']
    rng=torch.get_rng_state().clone()
    train_costs=audit(model,arg,n,training=True)
    torch.set_rng_state(rng)
    assert costs['matrix_macs']==train_costs['matrix_macs']
    costs['train_matrix_macs']=train_costs['matrix_macs']
    costs['dropout_compute_saving']=False
    costs['train_scalar_inventory']=train_costs['scalar_reduction_transcendental_inventory']
    model.eval();times=[]
    with torch.no_grad():
        for _ in range(warmup):model(*arg)
        for _ in range(steps):
            start=time.perf_counter_ns();model(*arg);times.append((time.perf_counter_ns()-start)/1e6)
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    def step():
        opt.zero_grad(set_to_none=True);y=model(*arg);loss=(y-.31).square().mean();loss.backward();opt.step()
    for _ in range(warmup):step()
    train_times=[]
    for _ in range(steps):
        start=time.perf_counter_ns();step();train_times.append((time.perf_counter_ns()-start)/1e6)
    return dict(config=c,**costs,inference_median_ms=statistics.median(times),inference_p90_ms=sorted(times)[math.ceil(.9*len(times))-1],
        inference_samples_ms=times,train_step_median_ms=statistics.median(train_times),train_samples_per_second=batch/(statistics.mean(train_times)/1000),
        train_samples_ms=train_times,train_peak_GPU_bytes=None,peak_memory_status='NOT RUN: CPU-only authorization',
        accuracy=None,accuracy_status='NOT RUN: synthetic MSE timing is not a task metric')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--points',type=int,default=972);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=12)
    a=p.parse_args()
    if min(a.points,a.batch,a.warmup,a.steps)<1:p.error('all bounds positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    declaration=dict(task='elasticity',scope='synthetic CPU float32; d32/h4/M16; one thread; no data/GPU',
        seeds=[17,29,43],measurement_seed=17,N=a.points,B=a.batch,warmup=a.warmup,steps=a.steps,
        control='for each enhanced mode/depth: pure same-depth global rank nearest parameter or matrix-MAC count; ties smaller rank; width unchanged',
        timing='diagnostics/profiler OFF; CPU synchronous perf_counter_ns; initialized AdamW; synthetic MSE',
        flops='2 per dense matrix/convolution MAC; explicit separate full operator inventory, not claimed total scalar FLOPs',
        paired_accuracy='NOT RUN for all seeds; final checkpoint only; no test selection',torch=str(torch.__version__),python=platform.python_version(),cpu=platform.processor())
    (out/'predeclared.json').write_text(json.dumps(declaration,indent=2))
    rows=[];cache={}
    for depth in range(4,9):
        for mode in MODES:
            row=measure(configuration(mode,depth),a.points,a.batch,a.warmup,a.steps)
            row.update(kind='core20',mode=mode,depth=depth)
            if mode=='A0K0':base=row;cache[(depth,16)]=row
            row['extra_parameters']=row['parameters']-base['parameters']
            row['extra_matrix_macs']=row['matrix_macs']-base['matrix_macs']
            rows.append(row);(out/'results.json').write_text(json.dumps(dict(protocol=declaration,rows=rows),indent=2))
    # Predeclared matching rule, depending only on size/cost (never timing/test).
    for target in list(rows):
        if target['mode']=='A0K0':continue
        depth=target['depth'];base=next(r for r in rows if r['kind']=='core20' and r['mode']=='A0K0' and r['depth']==depth)
        for metric,slope in [('parameters',2*8*depth),('matrix_macs',4*a.batch*a.points*32*depth)]:
            ideal=16+(target[metric]-base[metric])/slope
            rank=min({max(16,math.floor(ideal)),max(16,math.ceil(ideal))},key=lambda r:(abs((base[metric]+(r-16)*slope)-target[metric]),r))
            if (depth,rank) not in cache:
                cache[(depth,rank)]=measure(configuration('A0K0',depth,rank=rank),a.points,a.batch,a.warmup,a.steps)
            row=dict(cache[(depth,rank)]);row.update(kind='matched_control',mode='A0K0',depth=depth,global_rank=rank,
                target_mode=target['mode'],matched_metric=metric,target_value=target[metric],relative_mismatch=(row[metric]-target[metric])/target[metric])
            rows.append(row)
            (out/'results.json').write_text(json.dumps(dict(protocol=declaration,rows=rows),indent=2))
    pure8=next(r for r in rows if r['kind']=='core20' and r['mode']=='A0K0' and r['depth']==8)
    comparisons=[dict(mode=r['mode'],depth=r['depth'],parameters_vs_pure8=r['parameters']/pure8['parameters'],
        macs_vs_pure8=r['matrix_macs']/pure8['matrix_macs'],latency_vs_pure8=r['inference_median_ms']/pure8['inference_median_ms'],
        accuracy_parameter_pareto='NOT RUN',accuracy_flops_pareto='NOT RUN',accuracy_latency_pareto='NOT RUN') for r in rows if r['kind']=='core20']
    (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
    print('PASS: 20 mode/depth rows and 30 predeclared matched-control rows; no N by N')
if __name__=='__main__':main()
