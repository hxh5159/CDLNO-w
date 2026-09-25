#!/usr/bin/env python3
"""Data-free V5 parameter, matrix-MAC and bounded timing report.

Matrix FLOPs are exactly two times dense matrix/convolution MACs. LayerNorm,
GELU, softmax, temperature operations, expert weighting, residual adds and
reshapes are inventoried separately and excluded from that FLOP number.
"""
from __future__ import annotations
import argparse
import gc
import importlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdlno.linearno_loop.construction import build_from_config as build_v1
from cdlno.linearno_loop.v5.construction import build_from_config as build_v5
from linearno_loop.v5.config import resolve_config as resolve_v5
from tools.linearno_loop_accounting import analytic, analytic_pure_linearno, analytic_v5, audit
from tools.linearno_loop_support import configuration as v1_config, inputs

TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
TOPOLOGIES = ("p1_c3_r2_s1", "p2_c2_r2_s2")
EXPERT_COUNTS = (2, 3, 4, 8)
POINTS = dict(airfoil=11271, darcy=7225, elasticity=972, pipe=16641, ns=4096, plasticity=3131, airfrans=32000, car=32186)
ARCHITECTURE = "partial_share_feature_gate_v5"
RESIDUAL = "operator_1_expert_1_over_r"

def pure_model(config):
    task, base = config["task"], config["profile_spec"]
    if task == "car":
        from cdlno.linearno.car_entry import constructor_kwargs
        from cdlno.linearno.shapenet import ShapeNetLinearNO
        return ShapeNetLinearNO(**constructor_kwargs(base))
    if task == "airfrans":
        from cdlno.linearno.air_entry import _constructor_kwargs
        from cdlno.linearno.airfrans import AirfRANSLinearNO
        return AirfRANSLinearNO(**_constructor_kwargs(base))
    from cdlno.linearno.standard_entry import constructor_kwargs
    cls = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO").Model
    return cls(**constructor_kwargs(base))

def _count(model):
    return sum(parameter.numel() for parameter in model.parameters())

def parameter_rows(tasks=TASKS, counts=EXPERT_COUNTS):
    rows = []
    for task in tasks:
        n = POINTS[task]
        reference = resolve_v5(task, options={"architecture": ARCHITECTURE})
        pure = analytic_pure_linearno(task, B=1, N=n)
        live = pure_model(reference)
        assert _count(live) == pure["parameters"]
        rows.append(dict(task=task, model="pure_linearno", topology=None, expert_count=None, expert_width=None,
                         actual_M=reference["actual_M"], analytic=pure, measured_parameters=_count(live)))
        del live
        default_f = reference["expert_width"]
        for topology in TOPOLOGIES:
            legacy_config = v1_config(task, topology, "sr_1_over_r", multiplier=1)
            legacy = analytic(legacy_config, B=1, N=n)
            live = build_v1(legacy_config)
            assert _count(live) == legacy["parameters"]
            rows.append(dict(task=task, model="loop_v1_m_base", topology=topology, expert_count=None,
                             expert_width=None, actual_M=reference["actual_M"], analytic=legacy,
                             measured_parameters=_count(live)))
            del live
            widths = [(count, default_f, "profile_default") for count in counts]
            widths.append((2, max(1, default_f // 2), "custom_half_default"))
            for count, width, width_source in widths:
                config = resolve_v5(task, options=dict(architecture=ARCHITECTURE, topology_preset=topology,
                    residual_mode=RESIDUAL, expert_count=count, expert_width=width))
                expected = analytic_v5(config, B=1, N=n)
                live = build_v5(config); measured = _count(live)
                assert measured == expected["parameters"]
                rows.append(dict(task=task, model=ARCHITECTURE, topology=topology, expert_count=count,
                    expert_width=width, expert_width_source=width_source, actual_M=config["actual_M"],
                    analytic=expected, measured_parameters=measured,
                    parameters_vs_pure=measured / pure["parameters"],
                    matrix_flops_vs_pure=expected["matrix_flops"] / pure["matrix_flops"]))
                del live
        gc.collect()
    return rows

def _sync(device):
    if device == "cuda": torch.cuda.synchronize()

def _measure(model, arguments, device, warmup, steps):
    model = model.to(device).eval(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def forward():
        with torch.no_grad(): return model(*arguments)
    def train_step():
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model(*arguments).square().mean(); loss.backward(); optimizer.step()
        if not torch.isfinite(loss): raise ValueError("non-finite synthetic timing loss")
        model.eval()
    for _ in range(warmup): forward(); train_step()
    _sync(device)
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    result = {}
    for label, function in (("forward", forward), ("forward_backward_adamw", train_step)):
        samples = []
        for _ in range(steps):
            _sync(device); start = time.perf_counter_ns(); function(); _sync(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        result[label] = dict(median_ms=statistics.median(samples),
            p90_ms=sorted(samples)[math.ceil(.9 * len(samples)) - 1], samples_ms=samples)
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if device == "cuda" else None
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); result["weights_bytes"] = buffer.tell()
    return result

def measurement_rows(*, devices=None, warmup=3, steps=10):
    devices = devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    task, n = "elasticity", POINTS["elasticity"]; rows = []
    for device in devices:
        reference = resolve_v5(task, options={"architecture": ARCHITECTURE})
        factories = [("pure_linearno", reference, lambda: pure_model(reference))]
        for topology in TOPOLOGIES:
            legacy = v1_config(task, topology, "sr_1_over_r", multiplier=1)
            factories.append(("loop_v1_m_base_" + topology, legacy, lambda legacy=legacy: build_v1(legacy)))
            config = resolve_v5(task, options=dict(architecture=ARCHITECTURE, topology_preset=topology,
                residual_mode=RESIDUAL, expert_count=2))
            factories.append(("v5_e2_default_f_" + topology, config, lambda config=config: build_v5(config)))
        for label, config, factory in factories:
            model = factory(); arguments = inputs(config, canonical=True, batch=1, device=device); trace = None
            if label == "v5_e2_default_f_p2_c2_r2_s2" and device == "cpu":
                trace = audit(model, arguments, B=1, N=n); expected = analytic_v5(config, B=1, N=n)
                assert trace["matrix_macs"] == expected["matrix_macs"] and not trace["forbidden_attention"]
            rows.append(dict(task=task, N=n, B=1, model=label, device=device, dtype="float32",
                parameters=_count(model), trace=trace, measurement=_measure(model, arguments, device, warmup, steps)))
            del model, arguments; gc.collect()
            if device == "cuda": torch.cuda.empty_cache()
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if min(args.warmup, args.steps) < 1: parser.error("warmup and steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    devices = None if args.device == "auto" else [args.device]
    torch.set_num_threads(1)
    protocol = dict(scope="data-free parameter matrix plus bounded synthetic timing", tasks=list(TASKS),
        topologies=list(TOPOLOGIES), expert_counts=list(EXPERT_COUNTS),
        custom_width="half the selected profile default at K=2", canonical_points=POINTS, batch=1,
        warmup=args.warmup, measured_steps=args.steps, timing_case="Elasticity canonical N=972, synchronized float32",
        objective="synthetic squared output only; not a task metric or epoch estimate",
        flops="1 MAC = 2 FLOPs for matrix/convolution operations only",
        non_matrix="LayerNorm/GELU/softmax/temperature/expert weighting/residual/reshape separate",
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    result = dict(protocol=protocol, parameter_rows=parameter_rows(),
        measurement_rows=measurement_rows(devices=devices, warmup=args.warmup, steps=args.steps),
        real_data="NOT RUN", convergence="NOT RUN", accuracy="NOT RUN")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + chr(10))
    print(f"PASS: {len(result['parameter_rows'])} parameter rows; {len(result['measurement_rows'])} bounded timing rows")
    return 0

if __name__ == "__main__": raise SystemExit(main())
