#!/usr/bin/env python3
"""Bounded LF7 v2 accounting and synthetic measurements.

No stage imports benchmark entry scripts or loads benchmark data. Matrix and
timing use reduced synthetic tensors; counts use full task profiles.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import torch

from cdlno.linearno_loop.v2.construction import build_from_config
from tools.linearno_loop_accounting import analytic,audit,measured_parameters
from tools.linearno_loop_ffn_diagnostics import V2LoopDiagnostics
from tools.linearno_loop_ffn_support import (
    CORE_FFN_MODES,PRESETS,RESIDUALS,TASKS,configuration,inputs,point_count,
)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def environment():
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),platform.processor())
    return dict(python=platform.python_version(),torch=str(torch.__version__),
        cuda_build=torch.version.cuda,platform=platform.platform(),cpu=cpu,
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),seed=17,input_seed=902,
        compile=False,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','<unset>'))


def state_hash(model,*,public=False):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if public and name.startswith(('loop.latent_ffns.','loop.rb_','loop.lb_')):continue
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def descriptor(config):
    loop=config['loop_spec']
    return dict(task=loop['task'],preset=loop['topology_preset'],residual=loop['residual_mode'],
        core_ffn_mode=loop['core_ffn_mode'],rank_multiplier=loop['rank_multiplier'],
        resolved_rank=loop['resolved_rank'],hidden=loop['hidden'],heads=loop['heads'],
        P=loop['prefix_blocks'],C=loop['recurrent_core_blocks'],R=loop['loop_repeats'],
        S=loop['suffix_blocks'],config_hash=config['config_hash'])


def counts(output):
    rows=[];strict_reload=0
    for task in TASKS:
        for preset in PRESETS:
            for residual in RESIDUALS:
                public=[]
                for mode in CORE_FFN_MODES:
                    for multiplier in (1,2):
                        config=configuration(task,preset,residual,mode,multiplier=multiplier)
                        model=build_from_config(config);expected=analytic(config)
                        actual=measured_parameters(model)
                        assert expected['parameter_parts']==actual,(descriptor(config),expected['parameter_parts'],actual)
                        state_keys=list(model.state_dict())
                        state_key_sha256=hashlib.sha256('\n'.join(state_keys).encode()).hexdigest()
                        expected['state_key_count']=len(state_keys)
                        if multiplier==1:
                            clone=build_from_config(config);clone.load_state_dict(model.state_dict(),strict=True)
                            strict_reload+=1;del clone
                        if multiplier==1:public.append((mode,state_hash(model,public=True)))
                        rows.append(dict(**descriptor(config),parameter_parts=actual,
                            parameters=sum(actual.values()),state_key_count=len(model.state_dict()),
                            state_keys=state_keys,state_key_sha256=state_key_sha256,
                            calls={key:expected[key] for key in ('unique_depth','executed_depth',
                                'unique_operator_calls','executed_operator_calls',
                                'unique_point_ffn_modules','executed_point_ffn_calls')},
                            macs={key:expected[key] for key in ('unique_module_once_matrix_macs',
                                'executed_matrix_macs','executed_matrix_flops','latent_matrix_macs')},
                            public_backbone_sha256=state_hash(model,public=True)))
                        del model
                # Common tensors are equal between the two v2 modes for each M.
                assert public[0][1]==public[1][1],(task,preset,residual,public)
    value=dict(environment=environment(),status='PASS',rows=rows,
        full_profile_constructors=len(rows),strict_state_reloads=strict_reload,
        scope='full task profile parameter/state/MAC algebra; no full-profile forward; Mx1 and Mx2',
        flop_scope='dense matrix FLOPs=2*MAC; excludes normalization, softmax, GELU, bias and residual scalar work')
    write(output,value);return value


def expected_schedule(loop):
    operators=[];point=[];latent=[]
    for round_index in range(loop.loop_repeats):
        for position in range(loop.recurrent_core_blocks):
            operators.append((position,round_index));point.append((position,round_index))
            if hasattr(loop,'latent_ffns'):latent.append((position,round_index))
    return operators,point,latent


def synthetic_matrix(output,*,diagnostics=False):
    rows=[]
    for task in TASKS:
        for preset in PRESETS:
            for residual in RESIDUALS:
                for mode in CORE_FFN_MODES:
                    config=configuration(task,preset,residual,mode,small=True)
                    model=build_from_config(config);args=inputs(config)
                    B=1 if task in ('airfrans','car') else 2;N=point_count(config,False)
                    expected=analytic(config,B=B,N=N);measured=audit(model,args,B=B,N=N)
                    assert measured['parameter_parts']==expected['parameter_parts']
                    assert measured['matrix_macs']==expected['executed_matrix_macs']
                    assert measured['router_contraction_mac_equivalents']==expected['router_contraction_mac_equivalents']
                    assert not measured['forbidden_attention']
                    operator_calls=[];point_calls=[];latent_calls=[];head_calls=[];handles=[]
                    for position,module in enumerate(model.loop.core_operators):
                        handles.append(module.register_forward_hook(
                            lambda m,a,y,p=position:operator_calls.append((p,len(operator_calls)//model.loop.recurrent_core_blocks))))
                    for position,row in enumerate(model.loop.core_ffns):
                        for round_index,module in enumerate(row):
                            handles.append(module.register_forward_hook(
                                lambda m,a,y,p=position,r=round_index:point_calls.append((p,r))))
                    for position,module in enumerate(getattr(model.loop,'latent_ffns',())):
                        handles.append(module.register_forward_hook(
                            lambda m,a,y,p=position:latent_calls.append((p,len(latent_calls)//model.loop.recurrent_core_blocks))))
                    handles.append(model.loop.suffix[-1].block.mlp2.register_forward_hook(lambda *a:head_calls.append(1)))
                    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                    before_rng=torch.get_rng_state().clone()
                    with V2LoopDiagnostics(model,enabled=diagnostics) as observation:
                        output_tensor=model(*args);loss=(output_tensor-.31).square().mean();loss.backward()
                        gradients=observation.gradients()
                    after_rng=torch.get_rng_state().clone()
                    for handle in handles:handle.remove()
                    schedule=expected_schedule(model.loop)
                    assert operator_calls==schedule[0] and point_calls==schedule[1] and latent_calls==schedule[2]
                    assert head_calls==[1]
                    assert torch.isfinite(output_tensor).all() and torch.isfinite(loss)
                    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                               if parameter.grad is not None)
                    optimizer.step();model.eval()
                    with torch.no_grad():prediction=model(*args)
                    clone=build_from_config(config).eval();clone.load_state_dict(model.state_dict(),strict=True)
                    with torch.no_grad():restored=clone(*args)
                    torch.testing.assert_close(restored,prediction,atol=0,rtol=0)
                    rows.append(dict(**descriptor(config),B=B,N=N,output_shape=list(output_tensor.shape),
                        loss=float(loss.detach()),parameter_parts=measured.pop('parameter_parts'),
                        measured_costs=measured,operator_calls=[list(x) for x in operator_calls],
                        point_ffn_calls=[list(x) for x in point_calls],latent_calls=[list(x) for x in latent_calls],
                        head_calls=len(head_calls),strict_reload_max_abs=float((restored-prediction).abs().max()),
                        diagnostic_enabled=diagnostics,diagnostic_records=observation.records if diagnostics else None,
                        diagnostic_gradients=gradients,rng_changed_by_train_forward=not torch.equal(before_rng,after_rng)))
                    del model,clone,optimizer,output_tensor,loss,prediction,restored,args
    value=dict(environment=environment(),status='PASS',rows=rows,
        scope='96 reduced CPU FP32 synthetic forward/backward/AdamW/strict reload cases; no task data',
        diagnostics='explicitly enabled' if diagnostics else 'OFF; no diagnostic hooks installed')
    write(output,value);return value


def summary(samples):
    return dict(median_ms=statistics.median(samples),
                p90_ms=sorted(samples)[math.ceil(.9*len(samples))-1],samples_ms=samples)


def cpu(output,*,warmup=1,steps=3):
    rows=[];env=environment();env['system_load_start']=os.getloadavg()
    for task in TASKS:
        for preset in PRESETS:
            for residual in RESIDUALS:
                for mode in CORE_FFN_MODES:
                    config=configuration(task,preset,residual,mode,small=True)
                    model=build_from_config(config);args=inputs(config);forward=[];train=[]
                    model.eval()
                    with torch.no_grad():
                        for _ in range(warmup):model(*args)
                        for _ in range(steps):
                            start=time.perf_counter_ns();model(*args)
                            forward.append((time.perf_counter_ns()-start)/1e6)
                    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                    def step():
                        optimizer.zero_grad(set_to_none=True);out=model(*args)
                        (out-.31).square().mean().backward();optimizer.step()
                    for _ in range(warmup):step()
                    for _ in range(steps):
                        start=time.perf_counter_ns();step();train.append((time.perf_counter_ns()-start)/1e6)
                    rows.append(dict(**descriptor(config),N=point_count(config,False),
                                     forward=summary(forward),forward_backward_adamw=summary(train)))
                    del model,args,optimizer
    value=dict(environment=env,status='PASS',warmup=warmup,measured=steps,rows=rows,
        system_load_at_write=os.getloadavg(),
        scope='reduced CPU FP32 synthetic tensors; hidden8/heads2/M4; not real epoch throughput',
        timing='perf_counter_ns; synchronous CPU; one intra/inter-op thread; diagnostics OFF')
    write(output,value);return value


def cuda(output):
    env=environment()
    if not torch.cuda.is_available():
        value=dict(environment=env,status='NOT_RUN_CUDA_UNAVAILABLE',rows=[]);write(output,value);return value
    env.update(device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))
    precisions=[('FP32',None),('AMP_FP16',torch.float16)]
    if torch.cuda.is_bf16_supported():precisions.append(('AMP_BF16',torch.bfloat16))
    rows=[]
    for task in TASKS:
        for preset in PRESETS:
            for residual in RESIDUALS:
                for mode in CORE_FFN_MODES:
                    for label,dtype in precisions:
                        config=configuration(task,preset,residual,mode,small=True)
                        model=build_from_config(config).cuda();args=inputs(config,device='cuda')
                        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                        scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16,init_scale=256)
                        row=dict(**descriptor(config),precision=label,N=point_count(config,False))
                        try:
                            torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):
                                result=model(*args);loss=(result.float()-.31).square().mean()
                            scaler.scale(loss).backward();scaler.unscale_(optimizer)
                            assert torch.isfinite(result).all() and torch.isfinite(loss)
                            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
                            row.update(status='PASS',loss=float(loss.detach()),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        except (RuntimeError,TypeError,ValueError,AssertionError) as error:
                            row.update(status='FAIL',error_type=type(error).__name__,error=str(error),
                                       traceback=traceback.format_exc())
                        rows.append(row);del model,args,optimizer,scaler;torch.cuda.empty_cache()
    value=dict(environment=env,status='FAIL' if any(x['status']=='FAIL' for x in rows) else 'PASS',rows=rows,
        scope='reduced synthetic GPU smoke and allocator peak; no timing, data, full width or real training')
    write(output,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('counts','matrix','cpu','cuda'))
    parser.add_argument('--output',required=True);parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args()
    if Path(args.output).exists():parser.error('output already exists; choose a fresh path')
    if args.diagnostics and args.stage!='matrix':parser.error('--diagnostics is valid only for matrix')
    if args.warmup<1 or args.steps<2:parser.error('warmup>=1 and steps>=2 required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.manual_seed(17)
    value=(counts(args.output) if args.stage=='counts' else
           synthetic_matrix(args.output,diagnostics=args.diagnostics) if args.stage=='matrix' else
           cpu(args.output,warmup=args.warmup,steps=args.steps) if args.stage=='cpu' else cuda(args.output))
    if value['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()
