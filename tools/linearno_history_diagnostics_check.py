#!/usr/bin/env python3
"""Standalone diagnostic invariance audit, not a benchmark or training entry."""
import argparse
import copy
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from linearno_history_support import configuration,construct,inputs,MODES
from monitor.history_diagnostics import HistoryDiagnostics
from monitor.kernels import kernel_cosine
from cdlno.training_state import capture_rng,restore_rng,_same

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);rows=[]
    for v in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
        task={'airfrans':'airfrans','shapenet':'car'}.get(v,'elasticity')
        for mode in MODES:
            for train in (False,True):
                c=configuration(mode,task=task,variant=v if task=='elasticity' else None)
                m=construct(c).train(train)
                with torch.no_grad():
                    if hasattr(m,'latent_attnres'):
                        for r in m.latent_attnres.receivers.values():r.gamma.fill_(.4);r.w.fill_(.2)
                    if hasattr(m,'history_k'):
                        for gate in m.history_k.raw_gates.values():gate.fill_(.3)
                arg=inputs(c,n=15)
                random.seed(51);np.random.seed(51);torch.manual_seed(51)
                rng=capture_rng()
                y=m(*arg);y.square().sum().backward()
                grads={k:p.grad.clone() if p.grad is not None else None for k,p in m.named_parameters()}
                end=capture_rng();m.zero_grad(set_to_none=True);restore_rng(rng)
                original=nn.Module._call_impl
                directory=out/(v+'-'+mode+'-'+str(train))
                # One true rendering case proves plotting preserves RNG too.
                with HistoryDiagnostics(directory,every=1,max_points=9,plots=(v=='plain' and mode=='A1K1' and train)) as observer:
                    z=m(*arg);z.square().sum().backward()
                    buffer=io.BytesIO();torch.save(m,buffer) # no closure attached to model
                assert nn.Module._call_impl is original
                torch.testing.assert_close(y,z,atol=0,rtol=0)
                assert _same(grads,{k:p.grad for k,p in m.named_parameters()})
                assert _same(end,capture_rng())
                assert len(observer.rows)==1 and not observer.frames and observer.pending is None
                row=observer.rows[0];assert len(row['layers'])==4
                assert len(row['A'])==(3 if mode[1]=='1' else 0)
                assert len(row['K'])==(3 if mode[3]=='1' else 0)
                rows.append(dict(variant=v,mode=mode,train=train,forward_gradient_rng='EXACT',pickle='PASS'))
    # Independent dense tiny oracle solely in this audit, never monitor runtime.
    q=torch.rand(2,3,7,4,dtype=torch.float64);k=torch.rand_like(q)
    r=torch.rand_like(q);s=torch.rand_like(q)
    x=q@k.transpose(-1,-2);y=r@s.transpose(-1,-2)
    oracle=(x*y).sum((-1,-2))/(x.norm(dim=(-1,-2))*y.norm(dim=(-1,-2))+1e-12)
    observed=kernel_cosine(q,k,r,s)
    torch.testing.assert_close(oracle,observed,atol=1e-12,rtol=1e-12)
    original=nn.Module._call_impl
    try:
        with HistoryDiagnostics(out/'exception',plots=False):raise RuntimeError('intentional')
    except RuntimeError:pass
    assert nn.Module._call_impl is original
    (out/'result.json').write_text(json.dumps(dict(rows=rows,oracle_max_abs=(oracle-observed).abs().max().item(),exception_restore='PASS'),indent=2))
    print('48 mode/variant/train cases EXACT; dense oracle and exception restore PASS')
if __name__=='__main__':main()
