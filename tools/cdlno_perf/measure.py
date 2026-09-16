"""Synchronized synthetic timing; instrumentation never enters timed iterations."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import platform
import statistics
import subprocess
import time

import torch
from torch.profiler import ProfilerActivity, profile
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {'math': SDPBackend.MATH, 'flash': SDPBackend.FLASH_ATTENTION,
            'efficient': SDPBackend.EFFICIENT_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION}


def environment(device):
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                         text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        driver = 'unavailable'
    try:
        pyg = importlib.metadata.version('torch-geometric')
    except importlib.metadata.PackageNotFoundError:
        pyg = None
    return dict(python=platform.python_version(), torch=torch.__version__, cuda_build=torch.version.cuda,
                pyg=pyg, device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                driver=driver, cpu_threads=torch.get_num_threads(), cudnn=torch.backends.cudnn.version())


@contextmanager
def precision_settings(tf32=False):
    old = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.set_float32_matmul_precision(old[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[1:3]
        torch.backends.cudnn.benchmark = old[3]


def contexts(device, precision='fp32', backend='auto'):
    @contextmanager
    def combined():
        dtype = {'fp32': torch.float32, 'amp-fp16': torch.float16, 'amp-bf16': torch.bfloat16}[precision]
        with (nullcontext() if backend == 'auto' else sdpa_kernel(BACKENDS[backend])):
            with torch.autocast(device.type, dtype=dtype, enabled=precision != 'fp32'):
                yield
    return combined


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def quantiles(samples):
    values = sorted(samples)
    position = .9 * (len(values) - 1)
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return dict(median_ms=statistics.median(values), p90_ms=values[lo] + (values[hi] - values[lo]) * (position - lo),
                samples_ms=samples)


def fingerprint(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def output_gradients(model, args, target, context):
    model.train()
    model.zero_grad(set_to_none=True)
    with context():
        out = model(*args)
        loss = (out.float() - target).square().mean()
    loss.backward()
    result = (out.detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None})
    model.zero_grad(set_to_none=True)
    return result


def compare(reference, actual, precision):
    # FP32 reductions differ across folded-batch backend shapes. AMP receives
    # a separate tolerance and is never claimed to be FP32/reference parity.
    atol, rtol = (2e-5, 5e-4) if precision == 'fp32' else (4e-3, 3e-2)
    torch.testing.assert_close(actual[0], reference[0], atol=atol, rtol=rtol)
    if actual[1].keys() != reference[1].keys():
        raise AssertionError('chunk gradient participation differs')
    for name in actual[1]:
        torch.testing.assert_close(actual[1][name], reference[1][name], atol=atol, rtol=rtol, msg=name)
    return dict(passed=True, atol=atol, rtol=rtol,
                output_max_abs=float((actual[0] - reference[0]).abs().max()),
                gradient_max_abs=max((float((actual[1][n] - reference[1][n]).abs().max()) for n in actual[1]), default=0.),
                parameter_gradients_compared=len(actual[1]))


def backend_probe(model, args, context, device, *, regions=True):
    """One untimed eager forward to observe actual SDPA operators and costs.

    Profiler FLOPs are explicitly partial; analytical shape-based MACs are the
    authoritative matrix count. Per-region times here include instrumentation.
    """
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    handles, ranges = [], []
    selected = []
    for name, module in model.named_modules():
        if name in ('preprocess', 'time_fc', 'core.bridge', 'core.readout') or name.endswith(('.reader','.writer')) or (
            name.startswith(('core.front_blocks.', 'core.latent_blocks.', 'core.cdpa_at.', 'core.blocks.')) and name.count('.') == 2
        ) or (name.startswith('blocks.') and name.count('.') == 1):
            selected.append((name, module))
    def enter(name):
        ctx = torch.profiler.record_function('region::' + name)
        ctx.__enter__()
        ranges.append(ctx)
    def leave():
        ranges.pop().__exit__(None, None, None)
    for name, module in selected if regions else []:
        handles.append(module.register_forward_pre_hook(lambda m, a, n=name: enter(n)))
        handles.append(module.register_forward_hook(lambda m, a, o: leave()))
    try:
        model.eval()
        synchronize(device)
        with profile(activities=activities, record_shapes=True, with_flops=True, profile_memory=True) as prof:
            with torch.no_grad(), context():
                model(*args)
            synchronize(device)
        events = prof.key_averages()
        backends = sorted({e.key for e in events if 'scaled_dot_product' in e.key})
        def record(e):
            return dict(op=e.key, calls=e.count, device_type=str(e.device_type), cpu_total_us=e.cpu_time_total,
                        device_total_us=getattr(e, 'device_time_total', 0),
                        self_device_us=getattr(e, 'self_device_time_total', 0))
        return dict(observed_sdpa_ops=backends,
                    profiler_partial_flops=sum(e.flops for e in events),
                    partial_flops_warning='not total FLOPs; SDPA and other unsupported ops can be omitted',
                    regions=[record(e) for e in events if e.key.startswith('region::') and e.device_type == torch.autograd.DeviceType.CPU],
                    top_self_device_ops=[record(e) for e in sorted(
                        [e for e in events if e.key.startswith('aten::') and e.device_type == torch.autograd.DeviceType.CPU],
                        key=lambda e: getattr(e, 'self_device_time_total', 0), reverse=True)[:12]])
    finally:
        for h in handles:
            h.remove()


def benchmark(model, args, target, context, device, *, warmup=5, iterations=20,
              precision='fp32', compile_model=False, loss_closure=None):
    if warmup < 1 or iterations < 2:
        raise ValueError('warmup>=1 and iterations>=2 required for initialized-state quantiles')
    if loss_closure is not None and compile_model:
        raise ValueError('custom training loss timing supports eager models only')
    # Model-only compile is common to every variant. Optimizer/loss stay eager.
    # Compile first-forward and first-backward costs are excluded and reported.
    run_model = torch.compile(model) if compile_model else model
    compile_cold = {}
    eager_check = output_gradients(model, args, target, context) if compile_model else None
    if compile_model:
        run_model.eval()
        synchronize(device); start = time.perf_counter()
        with torch.no_grad(), context():
            run_model(*args)
        synchronize(device)
        compile_cold['eval_first_call_seconds_including_execution'] = time.perf_counter() - start
    def forward():
        with torch.no_grad(), context():
            return run_model(*args)

    def measure(fn):
        for _ in range(warmup):
            fn()
        synchronize(device)
        base = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter()
            fn()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.)
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        return dict(**quantiles(samples), allocated_before_bytes=base, peak_allocated_bytes=peak,
                    peak_increment_bytes=peak - base if peak is not None else None)

    run_model.eval()
    model.zero_grad(set_to_none=True)
    inference = measure(forward)
    # Always a fresh optimizer at the same initial weights for each chunk.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, foreach=False)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == 'amp-fp16')
    def step():
        optimizer.zero_grad(set_to_none=True)
        with context():
            if loss_closure is None:
                out = run_model(*args)
                loss = (out.float() - target.float()).square().mean()
            else:
                # Explicit per-call objective; no model-owned auxiliary cache.
                loss = loss_closure(run_model, args, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    run_model.train()
    if compile_model:
        synchronize(device); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with context():
            out = run_model(*args)
            loss = (out.float() - target.float()).square().mean()
        loss.backward()  # compile backward without changing initial weights/state
        synchronize(device)
        compile_cold['train_first_forward_backward_seconds_including_execution'] = time.perf_counter() - start
        compiled_check = (out.detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None})
        compile_cold['eager_equivalence'] = compare(eager_check, compiled_check, precision)
        del eager_check, compiled_check
        del out, loss
        optimizer.zero_grad(set_to_none=True)
    training = measure(step)
    # Verify FP16 did not turn the benchmark into skipped optimizer steps.
    observed_steps = sorted({int(s['step'].item()) for s in optimizer.state.values() if 'step' in s})
    if observed_steps != [warmup + iterations]:
        raise RuntimeError(f'optimizer updates missing/skipped: {observed_steps}, expected {warmup + iterations}')
    nonfinite = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()
                 or (p.grad is not None and not torch.isfinite(p.grad).all())]
    if nonfinite:
        raise RuntimeError('nonfinite benchmark parameters/gradients: ' + ','.join(nonfinite))
    state_bytes = sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
    cuda_state_bytes = sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor) and v.device.type == 'cuda')
    # Untimed steady training profiler identifies optimizer/backward overhead.
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else [])
    synchronize(device)
    with profile(activities=activities) as train_prof:
        with torch.profiler.record_function('synthetic_train_step'):
            step()
        synchronize(device)
    train_events = train_prof.key_averages()
    train_profile = dict(
        diagnostic_extra_optimizer_steps=1,
        aten_calls=sum(e.count for e in train_events if e.key.startswith('aten::')),
        top_cpu_ops=[dict(op=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                          self_device_us=getattr(e, 'self_device_time_total', 0))
                     for e in sorted([e for e in train_events if e.device_type == torch.autograd.DeviceType.CPU],
                                     key=lambda e:e.self_cpu_time_total, reverse=True)[:12]])
    compiled_probe = backend_probe(run_model, args, context, device, regions=False) if compile_model else None
    return dict(forward=inference, train_step=training, untimed_train_profiler=train_profile, compiled_backend_probe=compiled_probe,
                optimizer=dict(name='AdamW', lr=.001, weight_decay=1e-5, foreach=False,
                               state_initialized_before_measured_train_steps=True, state_tensor_bytes=state_bytes,
                               cuda_state_tensor_bytes=cuda_state_bytes, active_parameter_tensors=len(optimizer.state), successful_steps_per_active_parameter=observed_steps,
                               grad_scaler_enabled=scaler.is_enabled()),
                timing='synchronized perf_counter; includes host launch/wrapper overhead; no data I/O',
                loss=('synthetic FP32 mean squared error, one model call/update; not task loss/temporal rollout'
                      if loss_closure is None else
                      'explicit caller loss_closure; one model call/update, see caller objective metadata'),
                memory_scope='forward: model+buffers+inputs+output/temporaries, no grads/optimizer; train: also grads+initialized AdamW states',
                compile_cold=compile_cold)
