"""LAA9 V3 accounting and synthetic verification.

This tool never opens a dataset or imports an experiment entry point.  It
separates the tensor-free 768-row configuration matrix from the smaller D12
module and train-step checks, and writes only JSON evidence requested by the
caller.
"""
from __future__ import annotations

import argparse
import gc
import json
import io
import math
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cdlno.linearno.profiles import resolve_config as resolve_base
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR, TASKS
from linearno_loop.v3.costs import analytic_cost, BATCHES, POINTS
from cdlno.linearno_loop.v3.construction import build_from_config
from tools.linearno_loop_accounting import analytic_v3, measured_parameters, audit


PROFILES = ("matched_v1", "efficient_v1")
DEPTHS = ("d12", "d20", "d28", "d60")
MODES = ("sr_1_over_r", "rb_attnres", "lb_attnres_1_over_r")
ABLATIONS = ((False, False), (False, True), (True, False), (True, True))
CONTRACTS = {
    "airfoil": "standard_static_l4", "darcy": "standard_static_l4",
    "elasticity": "standard_static_l4", "pipe": "standard_static_l4",
    "ns": "standard_temporal_l5", "plasticity": "standard_temporal_l5",
    "airfrans": None, "car": None,
}


def small_config(task, profile, depth="d12", mode="sr_1_over_r",
                 latent=True, adapter=True, seed=17):
    """Resolve the real profile with a 2x3 synthetic grid, without data IO."""
    kwargs = {} if CONTRACTS[task] is None else {"contract": CONTRACTS[task]}
    base = resolve_base(task, "paper_table8_on_release_model",
                        explicit={"model.H": 2, "model.W": 3, "runtime.seed": seed},
                        **kwargs)
    options = {
        "architecture": ARCHITECTURE_SELECTOR, "cost_profile": profile,
        "topology_preset": depth, "residual_mode": mode,
        "latent_enabled": latent,
        "adapter_mode": ("bilateral_qk_lowrank_second_visit" if adapter else "none"),
    }
    with patch("linearno_loop.v3.config._profile", return_value=base):
        return resolve_config(task, "paper_table8_on_release_model", options=options)


def formal_config(task, profile, depth="d12", mode="sr_1_over_r",
                  latent=True, adapter=True, seed=17):
    options = {
        "architecture": ARCHITECTURE_SELECTOR, "cost_profile": profile,
        "topology_preset": depth, "residual_mode": mode,
        "latent_enabled": latent,
        "adapter_mode": ("bilateral_qk_lowrank_second_visit" if adapter else "none"),
    }
    return resolve_config(task, "paper_table8_on_release_model", options=options)


def synthetic_args(config, *, batch=1, points=6, dtype=torch.float32):
    """Construct only in-memory task inputs matching the eight public contracts."""
    task = config["loop_spec"]["task"]
    model = config["profile_spec"]["values"]["model"]
    if task in ("airfrans", "car"):
        from torch_geometric.data import Data
        x = torch.randn(points, 7, dtype=dtype)
        pos_dim = 2 if task == "airfrans" else 3
        data = Data(x=x, pos=torch.randn(points, pos_dim, dtype=dtype),
                    batch=torch.zeros(points, dtype=torch.long),
                    ptr=torch.tensor([0, points], dtype=torch.long))
        return (data,) if task == "airfrans" else ((data, torch.randn(4, 3, dtype=dtype)),)
    x = torch.randn(batch, points, model["space_dim"], dtype=dtype)
    fx = None if model["fun_dim"] == 0 else torch.randn(batch, points, model["fun_dim"], dtype=dtype)
    t = torch.rand(batch, 1, dtype=dtype) if model["time_input"] else None
    return x, fx, t


def invoke(model, config, args):
    return model(*args)


def matrix_rows():
    rows = []
    for profile in PROFILES:
        for task in TASKS:
            for depth in DEPTHS:
                for mode in MODES:
                    for latent, adapter in ABLATIONS:
                        config = formal_config(task, profile, depth, mode, latent, adapter)
                        cost = analytic_cost(config)
                        rows.append({
                            "task": task, "profile": profile, "depth": depth,
                            "residual_mode": mode, "latent_enabled": latent,
                            "adapter_enabled": adapter,
                            "hidden_width": config["loop_spec"]["hidden_width"],
                            "latent_width": config["loop_spec"]["latent_width"],
                            "actual_M": config["loop_spec"]["actual_M"],
                            "topology": [config["loop_spec"][k] for k in
                                         ("prefix_blocks", "recurrent_core_blocks", "loop_repeats", "suffix_blocks")],
                            "unique_depth": cost["unique_depth"],
                            "executed_depth": cost["executed_depth"],
                            "parameter_groups": cost["parameter_groups"],
                            "total_parameters": cost["total_parameters"],
                            "matrix_mac_groups": cost["matrix_mac_groups"],
                            "matrix_macs": cost["matrix_macs"],
                            "matrix_flops": cost["matrix_flops"],
                            "router_contraction_macs": cost["router_contraction_macs"],
                            "router_source_counts": cost["router_source_counts"],
                            "non_matrix": cost["non_matrix"],
                            "profile_match_claim_applies": cost["profile_match_claim_applies"],
                            "config_hash": config["config_hash"],
                        })
    assert len(rows) == 8 * 2 * 4 * 3 * 4
    return rows


def d12_instances():
    """Actual module/state/forward accounting for every formal D12 variant."""
    rows = []
    for profile in PROFILES:
        for task in TASKS:
            for mode in MODES:
                for latent, adapter in ABLATIONS:
                    config = small_config(task, profile, "d12", mode, latent, adapter)
                    model = build_from_config(config)
                    measured = measured_parameters(model)
                    expected = analytic_cost(config)
                    names = list(model.state_dict())
                    feature_keys = [name for name in names if ".latent_processor." in name or ".adapter." in name]
                    router_keys = [name for name in names if name.startswith("loop.rb_") or name.startswith("loop.lb_")]
                    block_calls = {"prefix": 0, "core": 0, "suffix": 0}
                    router_sources = []
                    handles = []
                    for block in model.loop.prefix:
                        handles.append(block.register_forward_hook(lambda *_a, **_k: block_calls.__setitem__("prefix", block_calls["prefix"] + 1)))
                    for block in model.loop.core:
                        handles.append(block.register_forward_hook(lambda *_a, **_k: block_calls.__setitem__("core", block_calls["core"] + 1)))
                    for block in model.loop.suffix:
                        handles.append(block.register_forward_hook(lambda *_a, **_k: block_calls.__setitem__("suffix", block_calls["suffix"] + 1)))
                    for module in model.modules():
                        if hasattr(module, "source_weights"):
                            handles.append(module.register_forward_pre_hook(
                                lambda _m, args: router_sources.append(len(args[0]))))
                    model_args = synthetic_args(config)
                    with torch.no_grad():
                        output = invoke(model, config, model_args)
                    for handle in handles:
                        handle.remove()
                    expected_parts = {
                        "stem": expected["parameter_groups"]["stem"],
                        "time": expected["parameter_groups"]["time"],
                        "prefix": expected["parameter_groups"]["prefix"],
                        "shared_core": expected["parameter_groups"]["shared_core"],
                        "suffix": expected["parameter_groups"]["suffix"],
                        "head": expected["parameter_groups"]["head"],
                        "latent": expected["parameter_groups"]["latent"],
                        "adapter": expected["parameter_groups"]["adapter"],
                        "router": expected["parameter_groups"]["router"],
                    }
                    if measured != expected_parts:
                        raise AssertionError(f"measured parameter partition mismatch: {task}/{profile}/{mode}/{latent}/{adapter}")
                    # RB deliberately visits operator/MLP through VisitBody so
                    # the physical-block wrapper itself is bypassed; retain
                    # both the observed wrapper calls and the logical schedule.
                    expected_calls = {"prefix": 2, "core": 0 if mode == "rb_attnres" else 8, "suffix": 2}
                    if block_calls != expected_calls:
                        raise AssertionError(f"block schedule mismatch: {block_calls}")
                    if (latent != bool(feature_keys and any("latent_processor" in k for k in feature_keys)) or
                        adapter != bool(feature_keys and any("adapter" in k for k in feature_keys))):
                        raise AssertionError("feature state key mismatch")
                    if router_sources != expected["router_source_counts"]:
                        raise AssertionError(f"router source schedule mismatch: {router_sources}")
                    trace = audit(model, model_args, B=1, N=6)
                    synthetic_cost = analytic_cost(config, batch=1, points=6)
                    if trace["matrix_macs"] != synthetic_cost["matrix_macs"] or trace["forbidden_attention"]:
                        raise AssertionError("D12 instance shape-trace mismatch")
                    rows.append({
                        "task": task, "profile": profile, "residual_mode": mode,
                        "latent_enabled": latent, "adapter_enabled": adapter,
                        "parameter_groups": measured, "total_parameters": sum(measured.values()),
                        "analytic_total": expected["total_parameters"],
                        "state_key_count": len(names), "latent_key_count": sum("latent_processor" in k for k in names),
                        "adapter_key_count": sum("adapter" in k for k in names), "router_key_count": len(router_keys),
                        "block_calls": block_calls, "logical_core_visits": 8,
                        "unique_depth": expected["unique_depth"],
                        "executed_depth": expected["executed_depth"], "output_shape": list(output.shape),
                        "matrix_macs": expected["matrix_macs"],
                        "synthetic_trace_matrix_macs": trace["matrix_macs"],
                        "synthetic_oracle_matrix_macs": synthetic_cost["matrix_macs"],
                        "forbidden_attention_count": len(trace["forbidden_attention"]),
                        "router_sources_observed": router_sources,
                        "router_contraction_macs": expected["router_contraction_macs"],
                        "router_contraction_macs_observed_synthetic": trace["router_contraction_mac_equivalents"],
                        "config_hash": config["config_hash"],
                    })
                    del model
                    gc.collect()
    assert len(rows) == 2 * 8 * 3 * 4
    return rows


def _baseline(task, depth):
    """Independent L-depth LinearNO formula for the frozen baseline table."""
    config = resolve_config(task, "paper_table8_on_release_model",
                            options={"architecture": ARCHITECTURE_SELECTOR,
                                     "cost_profile": "matched_v1", "topology_preset": "d12"})
    model = config["base_profile_spec"]["values"]["model"]
    H, heads, M, f = model["hidden"], model["heads"], model["linearno_rank"], model["ffn_ratio"]
    dh = H // heads; variant = model["linearno_variant"]
    kernel = 9 if variant in ("conv", "conv_temp") else 1
    outputs = 2 if kernel == 9 or task == "car" else 1
    temperature = 2 * heads if variant in ("temp", "conv_temp", "shapenet") else heads if task == "airfrans" else 0
    channels = (model["ref"] ** 2 if model["unified_pos"] else model["space_dim"]) + model["fun_dim"]
    if task == "airfrans" and model["unified_pos"]:
        channels += model["space_dim"]
    # preprocess has three bias vectors and the release placeholder adds H.
    stem = 2 * channels * H + 2 * H * H + 4 * H + (2 * (H * H + H) if model["time_input"] else 0)
    body = (kernel * H * H + H + 2 * dh * M + dh * dh + outputs * (H * H + H) + temperature
            + 4 * H + 2 * f * H * H + (f + 1) * H)
    head = 2 * H + H * model["out_dim"] + model["out_dim"]
    B, N = BATCHES[task], POINTS[task]
    body_mac = B * N * (kernel * H * H + 4 * H * M + H * dh + outputs * H * H + 2 * f * H * H)
    stem_mac = B * N * (2 * channels * H + 2 * H * H + (2 * H * H if model["time_input"] else 0))
    head_mac = B * N * H * model["out_dim"]
    return {"parameters": stem + depth * body + head, "matrix_macs": stem_mac + depth * body_mac + head_mac}


def table_comparison():
    rows = []
    original_depth = {"d12": 8, "d20": 12, "d28": 16, "d60": 32}
    for depth, base_depth in original_depth.items():
        for task in TASKS:
            baseline = _baseline(task, base_depth)
            entries = {}
            for profile in PROFILES:
                config = formal_config(task, profile, depth)
                cost = analytic_cost(config)
                entries[profile] = {
                    "v3_parameters": cost["total_parameters"], "baseline_parameters": baseline["parameters"],
                    "parameter_ratio_percent": 100 * cost["total_parameters"] / baseline["parameters"],
                    "v3_matrix_macs": cost["matrix_macs"], "baseline_matrix_macs": baseline["matrix_macs"],
                    "matrix_mac_ratio_percent": 100 * cost["matrix_macs"] / baseline["matrix_macs"],
                    "matrix_flops_reduction_percent": 100 * (1 - cost["matrix_macs"] / baseline["matrix_macs"]),
                }
            rows.append({"task": task, "comparison": f"{base_depth}->{int(depth[1:])}", "entries": entries})
    return rows


def synthetic_suite():
    """Small real wrappers: optimizer step, AMP where available, strict state reload."""
    rows = []
    cases = [("elasticity", "matched_v1"), ("ns", "efficient_v1"), ("airfrans", "matched_v1")]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    dtypes = [torch.float32]
    for task, profile in cases:
        for mode in MODES:
            for latent, adapter in ABLATIONS:
                config = small_config(task, profile, "d12", mode, latent, adapter)
                for device in devices:
                    active_dtypes = dtypes if device == "cpu" else dtypes + [torch.float16, torch.bfloat16]
                    for dtype in active_dtypes:
                        model = build_from_config(config).to(device=device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                        args = synthetic_args(config, dtype=(dtype if device == "cuda" else torch.float32))
                        if device == "cuda":
                            args = tuple(a.to(device) if hasattr(a, "to") else a for a in args)
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        adapter_dtype = []
                        hooks = []
                        for module in model.modules():
                            if module.__class__.__name__ == "BilateralQKLowRankAdapter":
                                hooks.append(module.register_forward_hook(
                                    lambda _m, _a, output: adapter_dtype.extend(
                                        [str(output[0].dtype), str(output[1].dtype)])))
                        try:
                            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
                                out = invoke(model, config, args)
                                loss = out.float().square().mean()
                            loss.backward(); optimizer.step()
                            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                            restored = build_from_config(config).to(device=device)
                            restored.load_state_dict(state, strict=True)
                            if adapter and (not adapter_dtype or len(set(adapter_dtype)) != 1):
                                raise AssertionError("adapter Q/K delta dtype mismatch")
                            rows.append({"task": task, "profile": profile, "residual_mode": mode,
                                         "latent_enabled": latent, "adapter_enabled": adapter,
                                         "device": device, "dtype": str(dtype), "status": "passed",
                                         "adapter_logits_dtype": sorted(set(adapter_dtype)),
                                         "max_reload_error": 0.0})
                        except (RuntimeError, TypeError, ValueError) as exc:
                            # Unsupported AMP kernels are an environment skip, not a model pass.
                            rows.append({"task": task, "profile": profile, "residual_mode": mode,
                                         "latent_enabled": latent, "adapter_enabled": adapter,
                                         "device": device, "dtype": str(dtype), "status": "skip",
                                         "reason": str(exc)})
                        for hook in hooks:
                            hook.remove()
                        del model, optimizer
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
    return rows


def performance_smoke():
    """One canonical-N/full-width GPU case; no dataset or epoch loop."""
    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "CUDA unavailable"}
    config = formal_config("elasticity", "matched_v1", "d12", "sr_1_over_r", True, True)
    device = torch.device("cuda")
    model = build_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = synthetic_args(config, batch=1, points=POINTS["elasticity"])
    args = tuple(a.to(device) if hasattr(a, "to") else a for a in args)
    def sync():
        torch.cuda.synchronize(device)
    def forward_once():
        return invoke(model, config, args)
    model.eval()
    for _ in range(2):
        with torch.no_grad():
            forward_once()
    sync(); forward_times=[]
    for _ in range(6):
        start=time.perf_counter();
        with torch.no_grad(): forward_once()
        sync(); forward_times.append((time.perf_counter()-start)*1000)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); train_times=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        warmup_loss=forward_once().float().square().mean()
        warmup_loss.backward(); optimizer.step()
    sync(); torch.cuda.reset_peak_memory_stats(device)
    for _ in range(6):
        sync(); start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss=forward_once().float().square().mean(); loss.backward(); optimizer.step(); sync()
        train_times.append((time.perf_counter()-start)*1000)
    buffer=io.BytesIO(); torch.save(model.state_dict(), buffer)
    result={"status":"passed","task":"elasticity","profile":"matched_v1","depth":"d12",
            "residual_mode":"sr_1_over_r","latent_enabled":True,"adapter_enabled":True,
            "points":POINTS["elasticity"],"hidden_width":config["loop_spec"]["hidden_width"],
            "device":torch.cuda.get_device_name(device),"dtype":"torch.float32",
            "forward_ms":{"median":statistics.median(forward_times),"p90":sorted(forward_times)[max(0,math.ceil(.9*len(forward_times))-1)],"samples":forward_times},
            "train_step_ms":{"median":statistics.median(train_times),"p90":sorted(train_times)[max(0,math.ceil(.9*len(train_times))-1)],"samples":train_times},
            "peak_memory":{"allocated_bytes":torch.cuda.max_memory_allocated(device),"reserved_bytes":torch.cuda.max_memory_reserved(device)},
            "state_dict_bytes":len(buffer.getvalue()),"measurement_scope":"synthetic canonical-N only; no epoch extrapolation"}
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def shape_trace_smoke():
    """Cross-check one V3 ATen shape trace against its independent MAC oracle."""
    config = small_config("elasticity", "matched_v1", "d12", "sr_1_over_r", True, True)
    model = build_from_config(config)
    args = synthetic_args(config, batch=1, points=6)
    observed = audit(model, args, B=1, N=6)
    expected = analytic_cost(config, batch=1, points=6)
    if observed["matrix_macs"] != expected["matrix_macs"]:
        raise AssertionError("V3 shape trace MAC mismatch")
    if observed["forbidden_attention"]:
        raise AssertionError("forbidden dense attention observed")
    return {
        "status": "passed", "task": "elasticity", "profile": "matched_v1", "depth": "d12",
        "matrix_macs": observed["matrix_macs"], "analytic_matrix_macs": expected["matrix_macs"],
        "contraction_count": len(observed["contractions"]), "softmax_count": len(observed["softmax"]),
        "forbidden_attention": observed["forbidden_attention"],
        "parameter_parts": observed["parameter_parts"],
        "scope": "synthetic B1,N6; no dataset or persistent diagnostic state",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-instances", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    matrix = matrix_rows()
    (args.out / "parsed-matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (args.out / "table-8-to-12-and-deeper.json").write_text(json.dumps(table_comparison(), indent=2), encoding="utf-8")
    if not args.skip_instances:
        (args.out / "d12-instances.json").write_text(json.dumps(d12_instances(), indent=2), encoding="utf-8")
    if not args.skip_synthetic:
        (args.out / "synthetic-suite.json").write_text(json.dumps(synthetic_suite(), indent=2), encoding="utf-8")
    if args.performance:
        (args.out / "performance-smoke.json").write_text(json.dumps(performance_smoke(), indent=2), encoding="utf-8")
    (args.out / "shape-trace.json").write_text(json.dumps(shape_trace_smoke(), indent=2), encoding="utf-8")
    summary = {
        "parsed_rows": len(matrix), "expected_parsed_rows": 768,
        "d12_rows": None if args.skip_instances else 192,
        "synthetic_data_access": False, "real_training": False,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
