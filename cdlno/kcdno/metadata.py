"""KCDNO JSON metadata only; no torch.load/save, model, or resume implementation.

The envelope records the existing task's weight format without changing it.
K1 describes core architecture; task adapters additionally validate task.json lift/grid/output
contracts before loading task model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import FAMILY, KCDNOArchitectureConfig, KCDNOInitializationConfig, KCDNORuntimeConfig
from .options import architecture_overrides
from .profiles import PROFILE_NAMES, TASKS


SCHEMA_VERSION = 1


class KCDNOMetadataMismatch(ValueError):
    """A checkpoint cannot supply the requested new-family configuration."""


def family_for_model_key(key: str) -> str | None:
    """Only 'kcdno' is the new registry key. None delegates to the OLD selector.

    This is a dispatch contract, not a factory registration or constructor.
    Old selectors retain their own accepted names and errors for unknown names.
    """
    return FAMILY if key == FAMILY else None


def checkpoint_family(payload: dict[str, Any]) -> str | None:
    """Missing family means legacy routing; never infer kcdno from shapes/names."""
    if type(payload) is not dict:
        raise KCDNOMetadataMismatch("checkpoint metadata must be an object")
    if "family" not in payload:
        return None
    family = payload["family"]
    if type(family) is not str or not family:
        raise KCDNOMetadataMismatch("explicit checkpoint family must be a nonempty string")
    return family


def compare_architecture(expected: KCDNOArchitectureConfig, actual: KCDNOArchitectureConfig) -> list[str]:
    left, right = expected.to_dict(), actual.to_dict()
    return sorted(key for key in left if left[key] != right[key])


@dataclass(frozen=True, slots=True)
class KCDNOMetadata:
    task: str
    architecture: KCDNOArchitectureConfig
    checkpoint_format: str
    profile: str = "kcdno_v1"
    initialization: KCDNOInitializationConfig = field(default_factory=KCDNOInitializationConfig)
    runtime: KCDNORuntimeConfig = field(default_factory=KCDNORuntimeConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> KCDNOMetadata:
        if self.task not in TASKS:
            raise KCDNOMetadataMismatch(f"unknown task: {self.task!r}")
        if self.profile not in PROFILE_NAMES:
            raise KCDNOMetadataMismatch(f"unknown profile: {self.profile!r}")
        formats = ("whole_model",) if self.task == "car" else (
            ("model_list", "whole_model") if self.task == "airfrans" else ("state_dict",))
        if self.checkpoint_format not in formats:
            raise KCDNOMetadataMismatch(f"{self.task} keeps checkpoint format {formats}, got {self.checkpoint_format!r}")
        for name, kind in (("architecture", KCDNOArchitectureConfig),
                           ("initialization", KCDNOInitializationConfig), ("runtime", KCDNORuntimeConfig)):
            value = getattr(self, name)
            if type(value) is not kind:
                raise KCDNOMetadataMismatch(f"{name} must be {kind.__name__}")
            value.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(schema_version=SCHEMA_VERSION, family=FAMILY, task=self.task,
                    profile=self.profile, checkpoint_format=self.checkpoint_format,
                    architecture=self.architecture.to_dict(), initialization=self.initialization.to_dict(),
                    runtime=self.runtime.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KCDNOMetadata:
        family = checkpoint_family(payload)
        if family != FAMILY:
            raise KCDNOMetadataMismatch(
                f"expected explicit family='kcdno', got {family!r}; missing family uses existing legacy rules")
        required = {"schema_version", "family", "task", "profile", "checkpoint_format",
                    "architecture", "initialization", "runtime"}
        if payload.keys() != required:
            raise KCDNOMetadataMismatch(
                f"metadata fields: missing={sorted(required - payload.keys())}, unknown={sorted(payload.keys() - required)}")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise KCDNOMetadataMismatch("unsupported KCDNO metadata schema_version")
        return cls(task=payload["task"], profile=payload["profile"], checkpoint_format=payload["checkpoint_format"],
                   architecture=KCDNOArchitectureConfig.from_dict(payload["architecture"]),
                   initialization=KCDNOInitializationConfig.from_dict(payload["initialization"]),
                   runtime=KCDNORuntimeConfig.from_dict(payload["runtime"]))


def save_metadata(path: str | Path, metadata: KCDNOMetadata) -> Path:
    """Exclusive creation, never overwrite an existing sidecar, including legacy."""
    text = json.dumps(metadata.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return target


def load_metadata(path: str | Path) -> KCDNOMetadata:
    """Read only; fully resolved fields required. No defaults for saved records."""
    return KCDNOMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class EvaluationResolution:
    saved: KCDNOMetadata
    runtime: KCDNORuntimeConfig
    runtime_differences: tuple[str, ...]
    requested_profile: str | None


def resolve_evaluation(path: str | Path, explicit: dict[str, Any], *, task: str,
                       runtime: KCDNORuntimeConfig | None = None) -> EvaluationResolution:
    """Read FIRST, reconstruct saved config, then check explicit architecture.

    Profile names are training provenance: even an explicit eval profile never
    expands its defaults over saved resolved values. Initialization is returned
    as provenance, never reapplied. Runtime changes are reported, not rejected
    as weight incompatibilities. Callers must log runtime_differences.
    """
    saved = load_metadata(path)
    if task != saved.task:
        raise KCDNOMetadataMismatch(f"task mismatch: checkpoint={saved.task!r}, requested={task!r}")
    requested_profile = explicit.get("profile")
    if requested_profile is not None and requested_profile not in PROFILE_NAMES:
        raise KCDNOMetadataMismatch(f"unknown requested profile: {requested_profile!r}")
    requested = KCDNOArchitectureConfig.from_dict(saved.architecture.to_dict() | architecture_overrides(explicit))
    differences = compare_architecture(saved.architecture, requested)
    if differences:
        details = "; ".join(f"{key}: checkpoint={getattr(saved.architecture, key)!r}, requested={getattr(requested, key)!r}"
                            for key in differences)
        raise KCDNOMetadataMismatch("KCDNO architecture mismatch: " + details)
    actual_runtime = saved.runtime if runtime is None else runtime
    old, new = saved.runtime.to_dict(), actual_runtime.to_dict()
    return EvaluationResolution(saved, actual_runtime, tuple(sorted(k for k in old if old[k] != new[k])), requested_profile)


def new_run_path(output_root: str | Path, task: str, architecture: KCDNOArchitectureConfig) -> Path:
    """Propose output/<task>/kcdno/<timestamp>_<structure>, without creating it.

    Later entry integration must reserve the new directory exclusively, using
    the existing no-overwrite rule. Does not change old output default paths.
    """
    if task not in TASKS:
        raise ValueError(f"unknown KCDNO task: {task!r}")
    architecture.validate()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tag = (f"{stamp}_L{architecture.L}_d{architecture.d}_h{architecture.h}"
           f"_M{architecture.M}_r{architecture.kernel_rank}_{architecture.history_mode}")
    return Path(output_root) / task / FAMILY / tag
