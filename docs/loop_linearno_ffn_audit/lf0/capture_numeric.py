"""Data-free pre-LF1 tensor fixtures for all six LinearNO operator variants."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT)]

import torch

from cdlno.linearno_loop.attnres import PointDepthAttnRes
from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.contracts import PRESETS, RESIDUAL_MODES
from tools.linearno_loop_support import configuration, inputs


OUT = Path(__file__).resolve().parent
ARTIFACT = Path(os.environ.get("LOOP_FFN_LF0_ARTIFACT", "/home/hwz/CDLNO-artifacts/loop-ffn-lf0-baseline"))
VARIANTS = {
    "plain": "ns",
    "temp": "elasticity",
    "conv": "plasticity",
    "conv_temp": "darcy",
    "airfrans": "airfrans",
    "shapenet": "car",
}


def raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return value.view(torch.uint8).numpy().tobytes()


def tensor_record(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw_bytes(tensor)).hexdigest(),
        "finite": bool(torch.isfinite(tensor.detach()).all()),
    }


def floating_leaves(value, prefix="input"):
    result = {}
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        value.requires_grad_(True)
        result[prefix] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(floating_leaves(item, f"{prefix}.{index}"))
    elif hasattr(value, "to_dict"):
        for name, item in value.to_dict().items():
            result.update(floating_leaves(item, f"{prefix}.{name}"))
    return result


def capture(task: str, preset: str, mode: str, device: str, precision: str):
    cpu_dtype = torch.float64 if precision == "FP64" else torch.float32
    autocast_dtype = {"FP16": torch.float16, "BF16": torch.bfloat16}.get(precision)
    config = configuration(task, preset, mode, small=True, seed=1701)
    model = build_from_config(config).to(device=device, dtype=cpu_dtype).train()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PointDepthAttnRes):
                module.query.copy_(torch.linspace(-0.13, 0.17, module.hidden,
                                                  device=device, dtype=module.query.dtype))
    args = inputs(config, canonical=False, batch=2, device=device, dtype=cpu_dtype)
    leaves = floating_leaves(args)
    calls = []
    handles = []
    for group_name, group in (("prefix", model.loop.prefix), ("core", model.loop.core),
                              ("suffix", model.loop.suffix)):
        for index, physical in enumerate(group):
            handles.append(physical.block.Attn.register_forward_hook(
                lambda _m, _a, _y, name=f"{group_name}.{index}.operator": calls.append(name)))
            handles.append(physical.block.mlp.register_forward_hook(
                lambda _m, _a, _y, name=f"{group_name}.{index}.point_ffn": calls.append(name)))
    handles.append(model.loop.suffix[-1].block.mlp2.register_forward_hook(
        lambda _m, _a, _y: calls.append("suffix.head")))

    state_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).cpu().clone() if device == "cuda" else None
    start = time.monotonic()
    try:
        torch.manual_seed(4401)
        if device == "cuda":
            torch.cuda.manual_seed_all(4401)
        run_cpu_rng_before = torch.get_rng_state().clone()
        run_cuda_rng_before = torch.cuda.get_rng_state(device).cpu().clone() if device == "cuda" else None
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            output = model(*args)
            loss = (output.float() - 0.31).square().mean()
        loss.backward()
        parameter_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        input_gradients = {
            name: None if value.grad is None else value.grad.detach().cpu().clone()
            for name, value in leaves.items()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        step_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        payload = {
            "config": config,
            "inputs": {name: value.detach().cpu().clone() for name, value in leaves.items()},
            "state_before": state_before,
            "output": output.detach().cpu().clone(),
            "loss": loss.detach().cpu().clone(),
            "input_gradients": input_gradients,
            "parameter_gradients": parameter_gradients,
            "step_state": step_state,
            "optimizer_state": optimizer.state_dict(),
            "rng": {
                "construction_cpu_before_run_seed": cpu_rng_before,
                "construction_cuda_before_run_seed": cuda_rng_before,
                "run_cpu_before": run_cpu_rng_before,
                "run_cpu_after": torch.get_rng_state().clone(),
                "run_cuda_before": run_cuda_rng_before,
                "run_cuda_after": torch.cuda.get_rng_state(device).cpu().clone() if device == "cuda" else None,
            },
            "call_sequence": calls,
        }
        name = "__".join((device, VARIANT_FOR_TASK[task], preset, mode, precision))
        path = ARTIFACT / f"{name}.pt"
        torch.save(payload, path)
        summary = {
            "name": name,
            "task": task,
            "variant": VARIANT_FOR_TASK[task],
            "preset": preset,
            "mode": mode,
            "device": device,
            "precision": precision,
            "status": "PASS",
            "seconds": round(time.monotonic() - start, 6),
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                      for name, value in state_before.items()},
            "inputs": {name: tensor_record(value) for name, value in payload["inputs"].items()},
            "output": tensor_record(output),
            "loss": tensor_record(loss),
            "input_gradients": {name: tensor_record(value) for name, value in input_gradients.items()},
            "parameter_gradients": {name: tensor_record(value) for name, value in parameter_gradients.items()},
            "step_state": {name: tensor_record(value) for name, value in step_state.items()},
            "rng": {name: tensor_record(value) for name, value in payload["rng"].items()},
            "call_sequence": calls,
            "unique_depth": model.loop.unique_depth,
            "executed_depth": model.loop.executed_depth,
        }
        return summary
    finally:
        for handle in handles:
            handle.remove()


VARIANT_FOR_TASK = {task: variant for variant, task in VARIANTS.items()}


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = []
    plans = [("cpu", "FP64"), ("cpu", "FP32")]
    if torch.cuda.is_available():
        plans.extend((("cuda", "FP32"), ("cuda", "FP16")))
        if torch.cuda.is_bf16_supported():
            plans.append(("cuda", "BF16"))
    else:
        rows.append({"device": "cuda", "status": "NOT RUN", "reason": "torch.cuda.is_available() is false"})
    for device, precision in plans:
        for variant, task in VARIANTS.items():
            for preset in PRESETS:
                for mode in RESIDUAL_MODES:
                    row = capture(task, preset, mode, device, precision)
                    rows.append(row)
                    print(row["name"], row["status"], flush=True)
    report = {
        "stage": "LF0",
        "scope": "synthetic data-free pre-change numerical fixtures",
        "artifact_root": str(ARTIFACT),
        "variants": VARIANTS,
        "presets": list(PRESETS),
        "residual_modes": list(RESIDUAL_MODES),
        "plans": plans,
        "expected_cases": len(plans) * len(VARIANTS) * len(PRESETS) * len(RESIDUAL_MODES),
        "completed_cases": sum(row.get("status") == "PASS" for row in rows),
        "rows": rows,
    }
    (OUT / "numeric-fixtures.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
