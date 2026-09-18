"""Runtime hooks and artifact writing for LinearNO kernel monitoring.

The runtime is dormant unless ``monitor.run`` sets ``LINEARNO_MONITOR_CONFIG``.
Hooks read the attention input and projection modules, reconstruct detached
Q/K factors, and collect only evaluation forwards.  No model tensor,
parameter, RNG state, loss, optimizer, or checkpoint is modified.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .kernels import pairwise_kernel_similarity
from .plotting import write_snapshot_plots


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_linearno_attention(module: nn.Module) -> bool:
    cls = type(module)
    return (
        cls.__name__ in {"LinearNOAttention", "LinearNO", "LinearNO_temp", "LinearNO_Conv", "LinearNO_Conv_temp"}
        and hasattr(module, "to_q")
        and hasattr(module, "to_k")
        and hasattr(module, "in_project_x")
        and hasattr(module, "variant")
    )


def _factor_from_input(module: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError(f"LinearNO attention input must be [B,N,d], got {tuple(x.shape)}")
    batch, points, channels = x.shape
    if channels != int(module.dim):
        raise ValueError(f"attention input width {channels} != configured dim {module.dim}")
    variant = str(module.variant)
    if variant in {"conv", "conv_temp"}:
        expected = int(module.H) * int(module.W)
        if points != expected:
            raise ValueError(f"conv attention expected N=H*W={expected}, got {points}")
        grid = x.transpose(1, 2).reshape(batch, channels, int(module.H), int(module.W))
        projected = module.in_project_x(grid)
        features = projected.reshape(batch, int(module.heads), int(module.dim_head), points).transpose(-1, -2)
    else:
        projected = module.in_project_x(x)
        features = projected.reshape(batch, points, int(module.heads), int(module.dim_head)).transpose(1, 2)
    q_logits = module.to_q(features)
    k_logits = module.to_k(features)
    if variant in {"temp", "conv_temp"}:
        q_logits = q_logits / module.temperature_q.clamp(0.01, 1.0)
        k_logits = k_logits / module.temperature_k.clamp(0.01, 1.0)
    elif variant == "shapenet":
        q_logits = q_logits / module.tempreature_q.clamp(0.1, 2.0)
        k_logits = k_logits / module.tempreature_k.clamp(0.1, 2.0)
    q = q_logits.softmax(dim=-1)
    k = k_logits.softmax(dim=-2)
    return q.detach(), k.detach()


class LinearNOKernelMonitor:
    """A process-local collector installed by :func:`install_runtime_monitor`."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.output = Path(self.config["output"]).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(self.config.get("capture_every_validations", 1)))
        self.max_samples = max(1, int(self.config.get("max_samples_per_snapshot", 8)))
        self.no_plots = bool(self.config.get("no_plots", False))
        self.eps = float(self.config.get("epsilon", 1e-12))
        self.only_eval = bool(self.config.get("eval_only", True))
        self._local = threading.local()
        self._validation_count = 0
        self._snapshot_count = 0
        self._attached: set[int] = set()
        self._layer_names: dict[int, str] = {}
        self._handles: list[Any] = []
        self._write_json(self.output / "monitor_config.json", self.config)

    def _write_json(self, path: Path, value: Any) -> None:
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _state(self) -> dict[str, Any]:
        state = getattr(self._local, "state", None)
        if state is None:
            state = {"depth": 0, "factors": [], "training": True}
            self._local.state = state
        return state

    def maybe_attach(self, module: nn.Module) -> None:
        key = id(module)
        if key in self._attached or not _is_linearno_attention(module):
            return
        self._attached.add(key)
        module_name = f"layer_{len(self._layer_names):02d}"
        self._layer_names[key] = module_name
        self._handles.append(module.register_forward_pre_hook(self._pre_hook, with_kwargs=False))

    def _pre_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        state = self._state()
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        if self.only_eval and module.training:
            state["training"] = True
            return
        state["training"] = False
        if state["factors"] is None:
            return
        try:
            q, k = _factor_from_input(module, inputs[0])
            state["factors"].append({
                "name": self._layer_names.get(id(module), f"layer_{len(state['factors']):02d}"),
                "q": q,
                "k": k,
                "n_points": int(q.shape[-2]),
                "rank": int(q.shape[-1]),
                "heads": int(q.shape[1]),
            })
        except Exception as exc:  # monitoring must never stop user training
            state.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")

    def enter_call(self, module: nn.Module) -> None:
        state = self._state()
        if state["depth"] == 0:
            state["factors"] = []
            state["errors"] = []
            state["training"] = bool(module.training)
        state["depth"] += 1

    def exit_call(self, module: nn.Module, error: BaseException | None = None) -> None:
        state = self._state()
        state["depth"] -= 1
        if state["depth"] != 0:
            return
        factors = state.get("factors") or []
        if error is not None or not factors or state.get("training", True):
            return
        self._validation_count += 1
        if self._validation_count == 1 or self._validation_count % self.every == 0:
            try:
                self._write_snapshot(factors)
            except Exception as exc:
                self._write_json(self.output / "monitor_error.json", {
                    "type": type(exc).__name__, "message": str(exc),
                })

    def _write_snapshot(self, factors: list[dict[str, Any]]) -> None:
        # Keep only a bounded number of batch samples.  All N point rows remain
        # in the factors, so rho is exact for the retained samples.
        factors = factors[: max(1, len(factors))]
        # Group the list by layer. Hooks are ordered by block execution.
        # The pre-hook list is flat only when one layer was captured per call;
        # use names and preserve the first occurrence order. Every standard
        # LinearNO forward visits each layer exactly once.
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in factors:
            if item["name"] not in seen:
                unique.append(item)
                seen.add(item["name"])
        if len(unique) != len(factors):
            # A top-level model call may contain repeated forwards (rare in
            # rollout tasks). Keep the first complete layer sequence only.
            first = {item["name"] for item in unique}
            factors = [item for item in factors if item["name"] in first]
            unique = []
            seen.clear()
            for item in factors:
                if item["name"] not in seen:
                    unique.append(item)
                    seen.add(item["name"])
        snapshot_index = self._snapshot_count + 1
        self._snapshot_count = snapshot_index
        snap = self.output / "snapshots" / f"validation_{snapshot_index:06d}"
        snap.mkdir(parents=True, exist_ok=False)
        matrices = []
        for item in unique:
            q = item["q"][: self.max_samples].cpu()
            k = item["k"][: self.max_samples].cpu()
            matrices.append((q, k))
        rho = pairwise_kernel_similarity(matrices, eps=self.eps)
        mean = rho.mean(dim=(0, 1)).numpy()
        std = rho.std(dim=(0, 1), unbiased=False).numpy()
        np.savez_compressed(
            snap / "similarity_values.npz",
            rho=rho.numpy(), rho_mean=mean, rho_std=std,
            layer_indices=np.arange(len(unique), dtype=np.int64),
            n_points=np.asarray([x["n_points"] for x in unique], dtype=np.int64),
            ranks=np.asarray([x["rank"] for x in unique], dtype=np.int64),
            heads=np.asarray([x["heads"] for x in unique], dtype=np.int64),
        )
        rows = []
        for b in range(rho.shape[0]):
            for h in range(rho.shape[1]):
                for i in range(rho.shape[-2]):
                    for j in range(rho.shape[-1]):
                        rows.append([b, h, i + 1, j + 1, float(rho[b, h, i, j])])
        with (snap / "similarity.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "head", "layer_i", "layer_j", "rho"])
            writer.writerows(rows)
        self._write_json(snap / "metadata.json", {
            "formula": "P_l=Q_l K_l^T; rho=<P_i,P_j>_F/(||P_i||_F||P_j||_F+epsilon)",
            "epsilon": self.eps,
            "validation_index": self._validation_count,
            "captured_samples": int(rho.shape[0]),
            "layers": [{"name": x["name"], "n_points": x["n_points"],
                        "rank": x["rank"], "heads": x["heads"]} for x in unique],
            "complexity": "O(N R^2) per layer pair; no N x N tensor",
        })
        if not self.no_plots:
            write_snapshot_plots(snap, mean, std)

    def close(self) -> dict[str, Any]:
        for handle in self._handles:
            handle.remove()
        return {
            "status": "completed",
            "validation_forwards": self._validation_count,
            "snapshots": self._snapshot_count,
            "output": str(self.output),
            "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }


_ACTIVE: LinearNOKernelMonitor | None = None
_ORIGINAL_CALL_IMPL = None
_PATCH_LOCK = threading.Lock()


def install_runtime_monitor(config: dict[str, Any]) -> LinearNOKernelMonitor:
    """Install the process-local hook used by ``sitecustomize``."""
    global _ACTIVE, _ORIGINAL_CALL_IMPL
    with _PATCH_LOCK:
        if _ACTIVE is not None:
            return _ACTIVE
        monitor = LinearNOKernelMonitor(config)
        _ACTIVE = monitor
        _ORIGINAL_CALL_IMPL = nn.Module._call_impl

        def wrapped_call_impl(module: nn.Module, *args: Any, **kwargs: Any):
            monitor.maybe_attach(module)
            monitor.enter_call(module)
            error = None
            try:
                return _ORIGINAL_CALL_IMPL(module, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                monitor.exit_call(module, error)

        nn.Module._call_impl = wrapped_call_impl
        return monitor


def active_monitor() -> LinearNOKernelMonitor | None:
    return _ACTIVE
