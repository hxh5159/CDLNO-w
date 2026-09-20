#!/usr/bin/env python3
"""Bounded synthetic LL9 audit. Never load task data, checkpoints or benchmark CLI.

Stages counts/matrix/cpu/cuda are independent. --diagnostics explicitly enables
external observations only for matrix; timing always has all hooks OFF.
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
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.config import validate_config
from tools.linearno_loop_support import TASKS,PRESETS,MODES,configuration,inputs,point_count
from tools.linearno_loop_accounting import analytic,measured_parameters,audit
from tools.linearno_loop_diagnostics import LoopDiagnostics


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def environment():
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    return dict(python=platform.python_version(),torch=str(torch.__version__),cuda_build=torch.version.cuda,
        platform=platform.platform(),cpu=cpu,cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        seed=17,input_seed=902,compile=False,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','<unset>'))


def common_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('loop.rb_','loop.lb_')):continue
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def count_matrix(output):
    rows=[]
    for task in TASKS:
        for preset in PRESETS:
            for multiplier in (1,2):
                hashes=[]
                for mode in MODES:
                    c=configuration(task,preset,mode,multiplier=multiplier);m=build_from_config(c)
                    expected=analytic(c);actual=measured_parameters(m)
                    assert actual==expected['parameter_parts'],(task,preset,mode,actual,expected)
                    hashes.append(common_hash(m))
                    rows.append(dict(task=task,preset=preset,mode=mode,rank_multiplier=multiplier,
                        config=c,analytic=expected,measured_parameter_parts=actual,backbone_sha256=hashes[-1]))
                    del m
                assert len(set(hashes))==1
    write(output,dict(environment=environment(),scope='96 full-profile constructors, canonical N, B1 analytical MACs; no full-width forward',rows=rows))
    print('counts: 96 full-profile parameter decompositions exact',flush=True)


def call_hooks(model):
    calls=[];head=[];handles=[]
    for group in ('prefix','core','suffix'):
        for index,physical in enumerate(getattr(model.loop,group)):
            for branch,module in (('operator',physical.block.Attn),('MLP',physical.block.mlp)):
                handles.append(module.register_forward_hook(lambda m,a,y,g=group,i=index,b=branch:calls.append([g,i,b])))
    handles.append(model.loop.suffix[-1].block.mlp2.register_forward_hook(lambda *args:head.append(1)))
    return calls,head,handles


def expected_calls(model):
    loop=model.loop
    schedule=[('prefix',i) for i in range(loop.prefix_blocks)]
    schedule += [('core',i) for r in range(loop.loop_repeats) for i in range(loop.recurrent_core_blocks)]
    schedule += [('suffix',i) for i in range(loop.suffix_blocks)]
    return [[g,i,b] for g,i in schedule for b in ('operator','MLP')]


def synthetic_matrix(output,diagnostics=False):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='loop-ll9-state-') as tmp:
        for task in TASKS:
            for preset in PRESETS:
                for mode in MODES:
                    c=configuration(task,preset,mode,small=True,canonical=True);m=build_from_config(c)
                    args=inputs(c,canonical=True);B=1 if task in ('airfrans','car') else 2;N=point_count(c)
                    expected=analytic(c,B=B,N=N);actual=audit(m,args,B=B,N=N)
                    assert actual['matrix_macs']==expected['executed_matrix_macs']
                    assert actual['router_contraction_mac_equivalents']==expected['router_contraction_mac_equivalents']
                    assert not actual['forbidden_attention']
                    params=list(m.named_parameters(remove_duplicate=False));assert len(params)==len({id(p) for _,p in params})
                    assert not any('round' in k for k in m.state_dict())
                    calls,head,handles=call_hooks(m);opt=torch.optim.AdamW(m.parameters(),lr=.001)
                    with LoopDiagnostics(m,enabled=diagnostics) as observation:
                        out=m(*args);loss=(out-.31).square().mean();loss.backward()
                        assert torch.isfinite(out).all()
                        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                        gradients=observation.gradients()
                    for handle in handles:handle.remove()
                    assert calls==expected_calls(m);assert head==[1]
                    opt.step();m.eval()
                    with torch.no_grad():y=m(*args);y2=m(*args)
                    torch.testing.assert_close(y,y2,atol=0,rtol=0)
                    # Metadata read/validation precedes constructing/restoring a NEW object.
                    # Standalone synthetic state fixture, not a production resume archive.
                    path=Path(tmp);write(path/'config.json',c)
                    torch.save(m.state_dict(),path/'weights.pt');torch.save(opt.state_dict(),path/'optimizer.pt')
                    restored_config=validate_config(json.loads((path/'config.json').read_text()))
                    restored=build_from_config(restored_config).eval()
                    restored.load_state_dict(torch.load(path/'weights.pt',weights_only=True),strict=True)
                    restored_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
                    from cdlno.linearno_loop.checkpoint import validate_optimizer_state
                    state=torch.load(path/'optimizer.pt',weights_only=True);validate_optimizer_state(state,restored_opt);restored_opt.load_state_dict(state)
                    with torch.no_grad():prediction=restored(*args)
                    torch.testing.assert_close(prediction,y,atol=0,rtol=0)
                    row=dict(task=task,preset=preset,mode=mode,config=c,B=B,N=N,
                        parameter_parts=actual.pop('parameter_parts'),costs=actual,call_schedule=calls,head_calls=len(head),
                        loss=float(loss.detach()),output_shape=list(y.shape),strict_reload_max_abs=float((prediction-y).abs().max()),
                        repeat_forward_max_abs=float((y-y2).abs().max()),optimizer_restored=True,
                        unique_parameter_objects=True,no_round_specific_weights=True,
                        diagnostics=observation.records if diagnostics else None,shared_gradients=gradients)
                    rows.append(row);write(output,dict(environment=environment(),scope='canonical spatial N with synthetic values; hidden8/heads2/M8, ref3; B2 Standard, single industrial graph; synthetic MSE',rows=rows))
                    print('matrix:',task,preset,mode,'PASS',flush=True)
                    del m,restored,opt,restored_opt,out,loss,y,y2,args
    return rows


def summary(samples):
    return dict(median_ms=statistics.median(samples),p90_ms=sorted(samples)[math.ceil(.9*len(samples))-1],samples_ms=samples)


def cpu_measure(output,warmup=3,steps=12):
    rows=[];env=environment();env['system_load_start']=os.getloadavg()
    for task in TASKS:
        for preset in PRESETS:
            for mode in MODES:
                c=configuration(task,preset,mode,small=True,canonical=True);m=build_from_config(c);args=inputs(c,canonical=True)
                assert all(not x._forward_hooks and not x._forward_pre_hooks for x in m.modules())
                m.eval();forward=[]
                with torch.no_grad():
                    for _ in range(warmup):m(*args)
                    for _ in range(steps):
                        start=time.perf_counter_ns();m(*args);forward.append((time.perf_counter_ns()-start)/1e6)
                m.train();optimizer=torch.optim.AdamW(m.parameters(),lr=.001);train=[]
                def step():
                    optimizer.zero_grad(set_to_none=True);out=m(*args);loss=(out-.31).square().mean();loss.backward();optimizer.step()
                for _ in range(warmup):step()
                for _ in range(steps):
                    start=time.perf_counter_ns();step();train.append((time.perf_counter_ns()-start)/1e6)
                rows.append(dict(task=task,preset=preset,mode=mode,config=c,N=point_count(c),B=1 if task in ('airfrans','car') else 2,
                    forward=summary(forward),forward_backward_adamw=summary(train)))
                write(output,dict(environment=env,system_load_at_write=os.getloadavg(),warmup=warmup,measured=steps,
                    scope='CPU FP32 canonical spatial N; d8/h2/M8/ref3 synthetic fixture, not formal-width timing or epoch performance',
                    timing='perf_counter_ns; synchronous CPU; one intra/inter-op thread; no hooks/diagnostics/profiler; warmed AdamW',rows=rows))
                print('cpu:',task,preset,mode,'PASS',flush=True)
    return rows


def cuda_smoke(output):
    env=environment()
    if not torch.cuda.is_available():
        write(output,dict(environment=env,status='NOT RUN: CUDA unavailable',rows=[]));return True
    env.update(device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))
    rows=[]
    # Small N and small width only. Each task, mode and preset is exercised.
    precisions=[('FP32',None),('AMP_FP16',torch.float16)]
    if torch.cuda.is_bf16_supported():precisions.append(('AMP_BF16',torch.bfloat16))
    for task in TASKS:
        for preset in PRESETS:
            for mode in MODES:
                for label,dtype in precisions:
                    c=configuration(task,preset,mode,small=True);m=build_from_config(c).cuda();args=inputs(c,device='cuda')
                    opt=torch.optim.AdamW(m.parameters(),lr=.001)
                    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16,init_scale=256)
                    def step(*,check=True):
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=dtype,enabled=dtype is not None):
                            out=m(*args);loss=(out.float()-.31).square().mean()
                        scaler.scale(loss).backward();scaler.unscale_(opt)
                        if check:
                            assert torch.isfinite(out).all() and torch.isfinite(loss)
                            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                            before={name:p.detach().clone() for name,p in m.named_parameters() if p.grad is not None}
                        scaler.step(opt);scaler.update()
                        if check:
                            assert any(not torch.equal(before[name],p.detach()) for name,p in m.named_parameters() if name in before),'optimizer step was skipped'
                        return float(loss.detach())
                    row=dict(task=task,preset=preset,mode=mode,precision=label,N=point_count(c,False),hidden=8,M=8)
                    try:
                        step();torch.cuda.synchronize();opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                        baseline=torch.cuda.memory_allocated();loss=step(check=False);torch.cuda.synchronize()
                        peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                        # Validate AFTER capturing peaks; clone/isfinite buffers are not measurement work.
                        assert math.isfinite(loss)
                        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
                        assert all(float(state['step'])==2 for state in opt.state.values()),'measured optimizer step skipped'
                        row.update(status='PASS',finite_forward_gradients=True,optimizer_updated=True,loss=loss,
                            baseline_allocated_bytes=baseline,peak_allocated_bytes=peak_allocated,
                            peak_reserved_bytes=peak_reserved,
                            incremental_peak_allocated_bytes=peak_allocated-baseline,scaler=float(scaler.get_scale()))
                    except (RuntimeError,TypeError,ValueError,AssertionError) as error:
                        row.update(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                        # An incompatible precision is reported, never silently retried as FP32.
                    rows.append(row)
                    del m,args,opt,scaler;torch.cuda.empty_cache()
                    write(output,dict(environment=env,status='PARTIAL' if any(r['status']=='FAIL' for r in rows) else 'PASS',
                        scope='small synthetic FP32/AMP smoke only; no GPU timing or full-width/real training',
                        peak_protocol='warm up one AdamW step; synchronize, zero_grad, empty_cache, reset_peak_memory_stats; one step with diagnostic/check buffers OFF; synchronize; capture peaks BEFORE finite/state checks; process allocator bytes include model/optimizer/input',rows=rows))
                print('cuda:',task,preset,mode,[(r['precision'],r['status']) for r in rows[-len(precisions):]],flush=True)

    return not any(row['status']=='FAIL' for row in rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('counts','matrix','cpu','cuda'));p.add_argument('--output',required=True)
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--warmup',type=int,default=3);p.add_argument('--steps',type=int,default=12)
    a=p.parse_args()
    if a.warmup<1 or a.steps<2:p.error('warmup>=1 and steps>=2 required')
    if a.diagnostics and a.stage!='matrix':p.error('--diagnostics only with matrix; timing/smoke observation OFF')
    if Path(a.output).exists():p.error('output already exists; choose a fresh path to preserve evidence')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.manual_seed(17)
    if a.stage=='counts':count_matrix(a.output)
    elif a.stage=='matrix':synthetic_matrix(a.output,a.diagnostics)
    elif a.stage=='cpu':cpu_measure(a.output,a.warmup,a.steps)
    elif not cuda_smoke(a.output):raise SystemExit(1)

if __name__=='__main__':main()
