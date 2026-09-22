"""Opt-in test-side traces; no diagnostic state is added to production modules."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import torch
from loop_linearno_latent_adapter.core_oracle import full_reference
from loop_linearno_latent_adapter.core_support import MODES,Trace,configuration,excite,inputs,make


def encode(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.detach().cpu().tolist())
    if isinstance(value,dict):return {key:encode(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(v) for v in value]
    return value


def main():
    torch.set_num_threads(1);start=time.perf_counter();records=[]
    for mode in MODES:
        c=configuration(mode=mode);m=excite(make(c));visits=[];handles=[]
        for i,p in enumerate(m.core):
            handles.append(p.register_forward_pre_hook(lambda mod,args,kw,i=i:visits.append(dict(position=i,**kw)),with_kwargs=True))
        x=inputs(c);state={k:p.detach().clone() for k,p in m.named_parameters()}
        expected,ref=full_reference(x.detach(),state,c['loop_spec'])
        try:
            with Trace(m) as t:actual=m(x)
        finally:
            for h in handles:h.remove()
        torch.testing.assert_close(actual,expected,atol=2e-11,rtol=2e-9)
        steps=[];rounds=[];C=m.recurrent_core_blocks;R=m.loop_repeats
        core_visits=[v for v in t.visits if v[0]=='core']
        for j,s in enumerate(t.steps):
            r=j//(2*C);row=dict(round_index=r,position=s['position'],branch='operator' if s['branch']==0 else 'point_ffn',h=s['h'],raw=s['u'])
            if mode=='rb_attnres':
                row.update(scale=None,sources=t.routes[j]['sources'],receiver=t.routes[j]['key'],
                    weights=t.routes[j]['weights'],partial=t.routes[j+1]['sources'][-1])
                torch.testing.assert_close(row['partial'],ref['steps'][j]['partial'],atol=2e-11,rtol=2e-9)
            else:
                after=t.steps[j+1]['h'] if j%2==0 else core_visits[j//2][3]
                scale=visits[j//2]['scale'];row.update(scale=scale,after=after)
                torch.testing.assert_close(after,s['h']+scale*s['u'],atol=0,rtol=0)
                assert scale==1/R
            steps.append(row)
        for r in range(R):
            if mode=='rb_attnres':rounds.append(dict(round_index=r,b_r=t.routes[(r+1)*2*C]['sources'][-1]))
            else:
                entry=core_visits[r*C][2];end=core_visits[(r+1)*C-1][3];row=dict(round_index=r,H=entry,Y=end)
                if mode=='lb_attnres_1_over_r':
                    delta=t.routes[r]['sources'][-1];torch.testing.assert_close(delta,end-entry,atol=0,rtol=0)
                    row.update(Delta=delta,boundary=t.routes[r])
                rounds.append(row)
        records.append(encode(dict(mode=mode,topology=[2,2,2,2],features='nonzero latent/adapter/router parameters',
            input=x,steps=steps,rounds=rounds,routes=t.routes,output=actual,
            max_output_error=float((actual.detach()-expected).abs().max()))))
    result=dict(status='PASS',stage='LAA4',scope='test-side FP64 B2/N6/H6 traces',
        command=[sys.executable,'-B',*sys.argv],wall_seconds=time.perf_counter()-start,records=records)
    with (OUT/'round-traces.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
