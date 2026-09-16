#!/usr/bin/env python3
"""Finite MSAR synthetic comparison; original models and timing tools reused.

No data loader, task loss, actual training entry or checkpoint import. MSAR
uses unmodified Light/Full; the other models use their task d/h/M/L presets.
This is NOT an equal-width/parameter comparison or an epoch-speed estimate.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import torch
from tools.cdlno_perf.models import Case,TASKS,build,synthetic_inputs
from tools.cdlno_perf.costs import audit
from tools.cdlno_perf.measure import (benchmark,backend_probe,contexts,environment,fingerprint,precision_settings)
from tools.cdlno_perf.msar import (MODELS,build_msar,live_audit,coverage_path_audit,synthetic_loss,task_inventory)

CONTROLS=('transolver','lrsa_matched_trainable','kcdno_all','kcdno_off')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=TASKS,default='elasticity')
    p.add_argument('--models',nargs='+',choices=MODELS+CONTROLS,default=list(MODELS)+list(CONTROLS[:3]))
    p.add_argument('--B',type=int,default=1)
    p.add_argument('--N',type=int)
    p.add_argument('--grid',type=int,nargs=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--backend',choices=('math','auto','flash','efficient','cudnn'),default='math')
    p.add_argument('--precision',choices=('fp32','amp-bf16','amp-fp16'),default='fp32')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--audit-only',action='store_true',help='live synthetic forward/backward cost; no timing')
    p.add_argument('--inventory-only',action='store_true',help='all8 actual Light/Full construction + formulas; no forward/timing')
    p.add_argument('--output',type=Path,required=True,help='new JSON only; never overwrite')
    return p


def run(a):
    if a.output.exists():raise FileExistsError(f'refusing to overwrite {a.output}')
    if a.threads<1 or a.warmup<1 or a.iterations<2:raise ValueError('threads/warmup>=1, iterations>=2')
    device=torch.device(a.device)
    if device.type not in ('cpu','cuda'):raise ValueError('CPU/CUDA only')
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; GPU not run')
        torch.cuda.set_device(device)
    if device.type=='cpu' and (a.precision=='amp-fp16' or a.backend not in ('math','auto')):
        raise ValueError('CPU supports fp32/amp-bf16 with math/auto')
    torch.set_num_threads(a.threads)
    report=dict(schema='msar-performance-v1',scope='M9 finite synthetic model costs/timing',
        environment=environment(device),settings=dict(vars(a),output=str(a.output),tf32=False,compile=False),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),*(ROOT/'tools/cdlno_perf').glob('*.py'),*(ROOT/'cdlno').rglob('*.py')]},
        interpretation='Task-preset controls differ from MSAR in widths/latent counts/blocks; NOT fair speed/accuracy or LRSA-paper reproduction.',
        limits='No real data, graph construction, task losses, time rollout, scheduler, epoch/convergence/accuracy/SOTA claim.',
        results=[],errors=[])
    with precision_settings(False):
        if a.inventory_only:
            report['inventory']=task_inventory()
        else:
            overrides=dict(B=a.B)
            if a.N is not None:overrides['N']=a.N
            if a.grid is not None:
                if a.N is not None and a.N!=a.grid[0]*a.grid[1]:raise ValueError('N disagrees with grid')
                overrides['grid']=a.grid
            case=Case.preset(a.task,'task',**overrides)
            report['case']=case.description()
            args,target=synthetic_inputs(case,device)
            context=contexts(device,a.precision,a.backend)
            initial={};eval_outputs={}
            for name in dict.fromkeys(a.models):
                row=dict(model=name,status='running');model=None
                try:
                    if name in MODELS:
                        profile=name.split('_')[1]
                        model=build_msar(case,name,device,a.seed)
                        if profile not in initial:
                            initial[profile]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        # Explicit common weights, never reliance on matching RNG consumption.
                        model.load_state_dict(initial[profile],strict=True)
                        row.update(architecture=model.config.to_dict(),adapter=model.adapter_architecture(),
                            objective=model.training_config.to_dict(),initial_sha256=fingerprint(initial[profile]))
                        model.eval()
                        with torch.no_grad(),context():prediction=model(*args).detach().cpu()
                        if profile in eval_outputs:
                            torch.testing.assert_close(prediction,eval_outputs[profile],atol=0,rtol=0)
                            row['same_weight_eval_off_floor']=dict(passed=True,atol=0,rtol=0)
                        else:eval_outputs[profile]=prediction
                        row['cost']=live_audit(model,args,target,context,case.B,case.N)
                        row['training_path']=coverage_path_audit(model,args,target,context)
                    else:
                        model,actual,source=build(case,name,device,seed=a.seed)
                        row.update(architecture=actual.description(),source=source,
                                   initial_sha256=fingerprint(model.state_dict()))
                        row['cost']=audit(model,args,target,context)
                        if not row['cost']['finite_output'] or row['cost']['parameters']['nonfinite_grad_names']:
                            raise RuntimeError('nonfinite baseline cost audit')
                        if name!='transolver' and row['cost']['parameters']['missing_grad_names']:
                            raise RuntimeError('unused control parameters')
                    if not a.audit_only:
                        row['untimed_eager_profiler']=backend_probe(model,args,context,device)
                        row['measurement']=benchmark(model,args,target,context,device,
                            warmup=a.warmup,iterations=a.iterations,precision=a.precision,
                            loss_closure=synthetic_loss if name in MODELS else None)
                    row['status']='passed'
                except Exception as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    report['errors'].append(dict(model=name,error=row['error']))
                finally:
                    report['results'].append(row)
                    print(f'{name}: {row["status"]}'+(' '+row['error'] if row['status']=='failed' else ''),flush=True)
                    del model;gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    return report


if __name__=='__main__':
    result=run(parser().parse_args());sys.exit(1 if result['errors'] else 0)
