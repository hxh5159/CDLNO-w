#!/usr/bin/env python3
"""Finite synthetic performance comparison. No loaders or experiment imports.

Run --help for explicit task/matched scopes. Each invocation covers ONE case,
not a dataset/depth/width matrix. JSON reports retain individual timing samples.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tools.cdlno_perf.models import Case, MODELS, TASKS, build, synthetic_inputs
from tools.cdlno_perf.costs import audit
from tools.cdlno_perf.measure import (backend_probe, benchmark, compare, contexts, environment,
                                     fingerprint, output_gradients, precision_settings)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, default='elasticity')
    p.add_argument('--comparison', choices=('task', 'matched'), default='matched',
                   help='task: original/new task presets; matched: common dimensions, architecture overrides allowed')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--chunks', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--front-latent-mode', '--front_latent_mode', choices=('full', 'no_sa', 'identity'), default='full',
                   help='CDLNO front processor only; matched LRSA always remains full')
    for name in ('B', 'N', 'd', 'h', 'M', 'L', 'F', 'ratio'):
        p.add_argument('--' + name, type=int)
    p.add_argument('--grid', nargs=2, type=int, metavar=('H', 'W'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=('fp32', 'amp-fp16', 'amp-bf16'), default='fp32',
                   help='parameters and inputs FP32; optional common autocast')
    p.add_argument('--backend', choices=('auto', 'math', 'flash', 'efficient', 'cudnn'), default='auto')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--compile', action='store_true', dest='compile_model')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--audit-only', action='store_true', help='cost/gradient/chunk audit, no timing/profiler')
    p.add_argument('--output', type=Path, required=True, help='new JSON file; existing files are rejected')
    return p


def run(a):
    if a.output.exists():
        raise FileExistsError(f'refusing to overwrite {a.output}')
    if a.threads < 1 or a.warmup < 1 or a.iterations < 2 or not a.chunks or min(a.chunks) < 0:
        raise ValueError('threads/warmup>=1, iterations>=2 and nonnegative chunks required')
    overrides = {k: getattr(a, k) for k in ('B','N','d','h','M','L','F','ratio','grid') if getattr(a, k) is not None}
    overrides['front_latent_mode'] = a.front_latent_mode
    if a.grid is not None and a.N is not None and a.N != a.grid[0] * a.grid[1]:
        raise ValueError('--N disagrees with --grid')
    case = Case.preset(a.task, a.comparison, **overrides)
    device = torch.device(a.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('only CPU/CUDA supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('requested CUDA unavailable; GPU measurements not run')
        torch.cuda.set_device(device)
    if device.type == 'cpu' and (a.precision == 'amp-fp16' or a.backend not in ('auto', 'math')):
        raise ValueError('CPU supports fp32/amp-bf16 and auto/math here')
    torch.set_num_threads(a.threads)
    report = dict(schema='cdlno-performance-v1', phase=9, supplement='A4', scope='synthetic model structure only',
                  case=case.description(), environment=environment(device),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),*(ROOT/'tools/cdlno_perf').glob('*.py'),*(ROOT/'cdlno').glob('*.py')]},
                  settings=dict(parameter_dtype='float32', input_dtype='float32', precision=a.precision,
                                requested_sdpa_backend=a.backend, tf32=a.tf32, compile=a.compile_model,
                                warmup=a.warmup, measured_iterations=a.iterations, seed=a.seed,
                                cudnn_benchmark=False, cudnn_deterministic=torch.backends.cudnn.deterministic,
                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                float32_matmul_precision='high' if a.tf32 else 'highest',
                                audit_only=a.audit_only),
                  cost_convention='Forward dense matrix MAC: one multiply-accumulate=2 FLOPs. Conv includes padded taps. '
                  'Matrix FLOPs exclude bias/norm/activation/softmax/reductions/copies; these have separate inventories. '
                  'SDPA QK+AV counted from actual shapes. No claim of exact all-operation FLOPs. '
                  'Storage payload/activation ledgers are not summed into measured allocator peaks.',
                  exclusions='No data, graphs, task loss, temporal rollout, scheduler, epoch estimate or LRSA paper reproduction.',
                  results=[], errors=[])
    args, target = synthetic_inputs(case, device)
    context = contexts(device, a.precision, a.backend)
    with precision_settings(a.tf32):
        for name in dict.fromkeys(a.models):
            chunks = list(dict.fromkeys(a.chunks)) if name.startswith('cdlno_') else [0]
            reference = initial = initial_hash = None
            for chunk in chunks:
                row = dict(model=name, chunk=chunk, status='running')
                model = None
                try:
                    model, actual_case, provenance = build(case, name, device, chunk, a.seed)
                    if initial is None:
                        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        initial_hash = fingerprint(initial)
                    model.load_state_dict(initial, strict=True)
                    row.update(config=actual_case.description(), source=provenance, initial_weights_sha256=initial_hash)
                    row['effective_structure'] = (dict(physics_blocks=actual_case.L) if name=='transolver' else
                        dict(full_lrsa_blocks=actual_case.L, persistent_blocks=0, bridge=False, extra_readout=False) if name=='lrsa_matched' else
                        dict(front_blocks=actual_case.F, front_latent_mode=actual_case.front_latent_mode,
                             full_lrsa_blocks=actual_case.F if actual_case.front_latent_mode=='full' else 0,
                             persistent_blocks=actual_case.L-actual_case.F, bridge=True, extra_readout=True))
                    row['cost'] = audit(model, args, target, context)
                    if not row['cost']['finite_output'] or row['cost']['parameters']['nonfinite_grad_names']:
                        raise RuntimeError('nonfinite synthetic audit')
                    if name != 'transolver' and row['cost']['parameters']['missing_grad_names']:
                        raise RuntimeError('unexpected unused new-model parameters')
                    actual = output_gradients(model, args, target, context)
                    if reference is None:
                        reference = actual
                        row['chunk_equivalence'] = dict(reference_chunk=chunk, baseline=True)
                    else:
                        row['chunk_equivalence'] = dict(reference_chunk=chunks[0], **compare(reference, actual, a.precision))
                    del actual
                    if not a.audit_only:
                        # Backend probe is eager and labeled so even in compile mode.
                        row['untimed_eager_profiler'] = backend_probe(model, args, context, device)
                        row['measurement'] = benchmark(model, args, target, context, device,
                                                       warmup=a.warmup, iterations=a.iterations,
                                                       precision=a.precision, compile_model=a.compile_model)
                    row['status'] = 'passed'
                except Exception as exc:
                    row['status'] = 'failed'
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    report['errors'].append(dict(model=name, chunk=chunk, error=row['error']))
                finally:
                    report['results'].append(row)
                    print(f'{name} chunk={chunk}: {row["status"]}', flush=True)
                    if row['status'] == 'failed':
                        print(row['error'], flush=True)
                    del model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if a.compile_model:
                        torch.compiler.reset()
            del initial, reference
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    result = run(parser().parse_args())
    sys.exit(1 if result['errors'] else 0)
