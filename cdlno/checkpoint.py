"""Architecture sidecar serialization and compatibility checks."""

from __future__ import annotations

import argparse
from dataclasses import fields
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
    raw = payload.get("architecture")
    if not isinstance(raw, dict):
        raise SidecarMismatch('sidecar architecture must be a complete object')
    missing = {field.name for field in fields(CDLNOArchitectureConfig)} - raw.keys()
    if missing == {'front_latent_mode'} and (
            raw.get('model_name') == 'CDLNO' and raw.get('model_version') == 'cdlno-core-v1'
            and raw.get('history_rule') == 'front-t-after-ffn2-before-up-v1'):
        raw = dict(raw, front_latent_mode='full')  # Known pre-A1 full only; no file rewrite.
    elif missing:
        raise SidecarMismatch('sidecar missing architecture fields: ' + ', '.join(sorted(missing)))
    architecture = CDLNOArchitectureConfig.from_dict(raw)
    derived = payload.get("derived", {})
    if derived.get("P", architecture.P) != architecture.P:
        raise SidecarMismatch("sidecar derived.P does not match L-F")
    CDLNORuntimeConfig.from_dict(payload.get("runtime", {}))
    payload["architecture"] = architecture.to_dict()
    payload["derived"] = {"P": architecture.P}
    return payload


def resolve_front_latent_mode(path: str | Path, explicit: str | None) -> str:
    """Read an eval sidecar first, then check only a user-supplied mode override."""
    saved = load_sidecar(path)['architecture']['front_latent_mode']
    if explicit is not None and explicit != saved:
        raise SidecarMismatch(
            f'architecture mismatch: front_latent_mode (checkpoint={saved!r}, requested={explicit!r})')
    return saved


def validate_model_front_mode(model, architecture: CDLNOArchitectureConfig) -> None:
    """Check saved object semantics, including pre-A1 objects whose init is not run.

    This does not rebuild or migrate weights. Industrial callers also compare
    wrapper contracts and strictly validate the complete state_dict as before.
    """
    if not all(isinstance(value, CDLNOArchitectureConfig) for value in (model.config, model.core.config)):
        raise SidecarMismatch('checkpoint wrapper/core require complete CDLNOArchitectureConfig objects')
    if (compare_architecture(architecture, model.config)
            or compare_architecture(architecture, model.core.config)):
        raise SidecarMismatch('checkpoint wrapper/core architecture disagree (including front_latent_mode)')
    fronts = model.core.front_blocks
    if len(fronts) != architecture.F:
        raise SidecarMismatch('checkpoint front block count disagrees with F')
    for index, block in enumerate(fronts):
        # Genuine old objects have no field: their only supported path is full.
        mode = getattr(block, 'front_latent_mode', 'full')
        if mode != architecture.front_latent_mode:
            raise SidecarMismatch(f'checkpoint front_blocks.{index}.front_latent_mode disagrees with architecture')


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
