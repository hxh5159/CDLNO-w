"""Bounded synchronized synthetic timings; never a real epoch estimate."""
import argparse
import gc
import io
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import torch
from linearno_loop_v4.cost_review import pure_model
from linearno_loop.v4.config import resolve_config
from cdlno.linearno_loop.v4.construction import build_from_config
from tools.linearno_loop_support import configuration,inputs
from cdlno.linearno_loop.construction import build_from_config as loop_model


def measure(model,args,device):
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def forward():
        with torch.no_grad():return model(*args)
    def train():
        optimizer.zero_grad(set_to_none=True);loss=model(*args).square().mean();loss.backward();optimizer.step()
        assert torch.isfinite(loss)
    for _ in range(3):forward();train()
    sync()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    rows={}
    for name,fn in [('forward',forward),('forward_backward_adamw',train)]:
        samples=[]
        for _ in range(10):
            sync();start=time.perf_counter();fn();sync();samples.append(1000*(time.perf_counter()-start))
        rows[name]=dict(median_ms=statistics.median(samples),p90_ms=sorted(samples)[8],samples_ms=samples)
    if device=='cuda':rows.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    else:rows.update(peak_allocated=None,peak_reserved=None)
    file=io.BytesIO();torch.save(model.state_dict(),file);rows['weights_bytes']=file.tell()
    return rows


def main(path):
    torch.set_num_threads(1);rows=[]
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        for task in ('elasticity',):
            c=resolve_config(task,options=dict(architecture='resmlp_dual_temp_v4',temperature_mode='base',seed=17))
            arg=inputs(c,canonical=True,batch=1,device=device)
            factories=[('pure_linearno',lambda:pure_model(c)),
                       ('v1_p1c3r2s1_Mbase',lambda:loop_model(configuration(task,'p1_c3_r2_s1','sr_1_over_r',multiplier=1)))]
            for mode in ('base','latent_k_point_q','point_k_point_q'):
                factories.append(('v4_'+mode,lambda mode=mode:build_from_config(resolve_config(task,options=dict(
                    architecture='resmlp_dual_temp_v4',temperature_mode=mode,seed=17)))))
            for label,factory in factories:
                model=factory().to(device);row=dict(task=task,model=label,device=device,dtype='float32',batch=1,N=972,H=128,heads=8,M=64,
                    parameters=sum(p.numel() for p in model.parameters()),**measure(model,arg,device))
                rows.append(row);print(label,device,row['forward']['median_ms'],flush=True)
                del model;gc.collect()
                if device=='cuda':torch.cuda.empty_cache()
    path.write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,threads=1,warmup=3,measured=10,
        synthetic_objective='mean squared output, NOT task loss or epoch timing',tf32=torch.backends.cuda.matmul.allow_tf32,
        compile=False,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
