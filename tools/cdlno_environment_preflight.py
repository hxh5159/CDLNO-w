#!/usr/bin/env python3
"""Data-free CDLNO environment preflight.

This script only observes the current interpreter.  It never installs packages,
changes dependencies, imports experiment entry points, downloads data, or
starts training.  The authoritative target remains the user's remote
Python 3.10/3.11 + PyTorch 2.11 + CUDA 12.8 environment.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys


def main() -> int:
    report: dict[str, object] = {
        "purpose": "data-free observation only",
        "python": sys.version,
        "python_target_match": sys.version_info[:2] in {(3, 10), (3, 11)},
        "platform": platform.platform(),
        "target": {"python": ["3.10", "3.11"], "torch": "2.11.x", "cuda_runtime": "12.8"},
        "imports": {},
    }
    for name in ("torch", "numpy", "yaml", "einops"):
        try:
            module = importlib.import_module(name)
            report["imports"][name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:  # pragma: no cover - environment dependent
            report["imports"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    torch_info = report["imports"].get("torch")
    if isinstance(torch_info, dict) and torch_info.get("ok"):
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda_runtime"] = torch.version.cuda
        report["torch_target_match"] = torch.__version__.split("+")[0].split(".")[:2] == ["2", "11"]
        report["cuda_runtime_target_match"] = torch.version.cuda == "12.8"
        report["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        report["cuda_available"] = bool(torch.cuda.is_available())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
