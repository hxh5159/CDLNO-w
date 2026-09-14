"""Architecture sidecar serialization and compatibility checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import CDLNOArchitectureConfig, CDLNORuntimeConfig


SCHEMA_VERSION = 1


class SidecarMismatch(ValueError):
    """Raised when requested model semantics do not match a checkpoint sidecar."""


def _canonical_architecture(value: CDLNOArchitectureConfig | dict[str, Any]) -> dict[str, Any]:
    config = value if isinstance(value, CDLNOArchitectureConfig) else CDLNOArchitectureConfig.from_dict(value)
    return config.to_dict()


def compare_architecture(
    expected: CDLNOArchitectureConfig | dict[str, Any],
    actual: CDLNOArchitectureConfig | dict[str, Any],
) -> list[str]:
    """Return semantic field paths that differ; runtime fields are excluded."""

    left = _canonical_architecture(expected)
    right = _canonical_architecture(actual)
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def save_sidecar(
    path: str | Path,
    architecture: CDLNOArchitectureConfig,
    runtime: CDLNORuntimeConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a sidecar.  Callers must choose a new path for a new structure."""

    architecture.validate()
    runtime = runtime or CDLNORuntimeConfig()
    runtime.validate()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture.to_dict(),
        "derived": {"P": architecture.P},
        "runtime": runtime.to_dict(),
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing CDLNO sidecar: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate an existing sidecar before any comparison."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SidecarMismatch(f"unsupported sidecar schema: {payload.get('schema_version')!r}")
    architecture = CDLNOArchitectureConfig.from_dict(payload["architecture"])
    derived = payload.get("derived", {})
    if derived.get("P", architecture.P) != architecture.P:
        raise SidecarMismatch("sidecar derived.P does not match L-F")
    CDLNORuntimeConfig.from_dict(payload.get("runtime", {}))
    payload["architecture"] = architecture.to_dict()
    payload["derived"] = {"P": architecture.P}
    return payload


def validate_sidecar(
    path: str | Path,
    requested_architecture: CDLNOArchitectureConfig,
    requested_runtime: CDLNORuntimeConfig | None = None,
) -> dict[str, Any]:
    """Load first, compare architecture second; runtime differences are allowed."""

    payload = load_sidecar(path)
    mismatches = compare_architecture(requested_architecture, payload["architecture"])
    if mismatches:
        raise SidecarMismatch("architecture mismatch: " + ", ".join(mismatches))
    if requested_runtime is not None:
        requested_runtime.validate()
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compare a CDLNO architecture JSON sidecar")
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--architecture-json", type=Path, required=True)
    args = parser.parse_args()
    expected = CDLNOArchitectureConfig.from_dict(json.loads(args.architecture_json.read_text()))
    payload = validate_sidecar(args.sidecar, expected)
    print(json.dumps({"compatible": True, "architecture": payload["architecture"], "runtime_ignored": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
