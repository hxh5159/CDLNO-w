"""Pure argument-resolution protocol, not attached to any task entry in K1."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
from typing import Any, Sequence

from .config import FAMILY, KCDNOArchitectureConfig
from .profiles import resolve_profile


# Actual existing parser destinations map to ONE canonical resolved field.
_ALIASES = {"n_layers": "L", "n_hidden": "d", "n_heads": "h", "n_head": "h",
            "slice_num": "M", "dropout": "attention_dropout"}
_INAPPLICABLE = {"F", "P", "front_blocks", "front_latent_mode", "rear_depth", "rear_blocks",
                 "latent_blocks", "latent_ffn_ratio", "ffn_ratio", "mlp_ratio", "cdpa_mode",
                 "cdpa_source_chunk_size", "source_chunk_size", "decoder_mode"}


def explicit_arguments(parser: argparse.ArgumentParser, argv: Sequence[str]) -> dict[str, Any]:
    """Use argparse's own aliases/types/last-wins behavior, suppressing defaults.

    Call with the original parser plus new options added in a future entry stage.
    The original parser is not mutated. Only supplied option destinations appear.
    Do not pass vars(original_parser.parse_args(...)) to a resolver: that would
    promote legacy defaults (L3/d64/M32) to explicit architecture choices.
    """
    probe = copy.deepcopy(parser)
    probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    return vars(probe.parse_args(list(argv)))


def architecture_overrides(explicit: dict[str, Any]) -> dict[str, Any]:
    """Normalize explicit argparse destinations; keep unrelated task fields out."""
    result = {}
    names = {field.name for field in fields(KCDNOArchitectureConfig)}
    for name, value in explicit.items():
        if name in _INAPPLICABLE or name.startswith(("front_", "rear_", "cdpa_")):
            raise ValueError(f"{name} is not applicable to KCDNO; it has L complete point blocks, no front/rear/CDPA")
        if name in ("model", "cfd_model"):
            if value != FAMILY:
                raise ValueError(f"KCDNO registry key must be {FAMILY!r}, got {value!r}")
            continue
        target = _ALIASES.get(name, name)
        if target not in names:
            continue  # data, optimizer, runtime and path arguments belong to the task.
        if target in result and result[target] != value:
            raise ValueError(f"conflicting explicit aliases for {target}: {result[target]!r} vs {value!r}")
        result[target] = value
    return result


def resolve_training(task: str, explicit: dict[str, Any]) -> KCDNOArchitectureConfig:
    """Explicit CLI > chosen profile > KCDNO defaults. No legacy Namespace."""
    return resolve_profile(task, explicit.get("profile", "kcdno_v1"), **architecture_overrides(explicit))
