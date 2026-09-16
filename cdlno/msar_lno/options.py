"""Pure CLI protocol, not called by production task entries in M1."""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
from typing import Any, Sequence

# Reuse the existing tested explicit-only argparse extraction without changing it.
from ..kcdno.options import explicit_arguments
from .config import FAMILY, MSARArchitectureConfig, MSARTrainingConfig
from .profiles import PROFILE_NAMES, ResolvedMSARConfig, resolve_profile
from .registry import family_for_model_key

_ALIASES = {'n_hidden': 'd'}
_INAPPLICABLE = {
    'L', 'F', 'P', 'M', 'h', 'n_layers', 'slice_num', 'n_heads', 'n_head',
    'kernel_rank', 'history_mode', 'mlp_ratio', 'ffn_ratio', 'dropout',
    'source_chunk_size', 'decoder_mode', 'rear_depth', 'rear_blocks',
    'front_blocks', 'front_latent_mode', 'latent_blocks',
}


def _validate_explicit(explicit: dict[str, Any]) -> None:
    if type(explicit) is not dict:raise ValueError('explicit arguments must be a dict, not a default-filled Namespace')
    for key, value in explicit.items():
        if key in ('model', 'cfd_model', 'family') and value != FAMILY:
            raise ValueError(f'MSAR-LNO requires family/model {FAMILY!r}, got {value!r}')
        if key in _INAPPLICABLE or key.startswith(('front_', 'rear_', 'cdpa_')):
            raise ValueError(f'{key} is not applicable to MSAR-LNO; use four-scale num_latents/heads/depths')


def architecture_overrides(explicit: dict[str, Any]) -> dict[str, Any]:
    _validate_explicit(explicit)
    names = {f.name for f in fields(MSARArchitectureConfig)}
    result = {}
    for key, value in explicit.items():
        target = _ALIASES.get(key, key)
        if target not in names:continue
        if target in result and result[target] != value:raise ValueError(f'conflicting explicit aliases for {target}')
        result[target] = value
    return result


def training_overrides(explicit: dict[str, Any]) -> dict[str, Any]:
    _validate_explicit(explicit)
    names = {f.name for f in fields(MSARTrainingConfig)}
    for key in explicit:
        if (key.startswith('coverage_') or key == 'diagnostics') and key not in names:
            raise ValueError(f'unknown MSAR-LNO training option: {key}')
    return {k:v for k,v in explicit.items() if k in names}


def resolve_training(task: str, explicit: dict[str, Any]) -> ResolvedMSARConfig:
    """Explicit CLI > chosen profile > family defaults; no old Namespace defaults."""
    overrides = architecture_overrides(explicit)
    profile = explicit.get('profile', 'light')
    return ResolvedMSARConfig(task, profile, resolve_profile(profile, **overrides),
                             MSARTrainingConfig(**training_overrides(explicit)))


def parser_for_family(parser: argparse.ArgumentParser, model_key: str) -> argparse.ArgumentParser | None:
    """Only a NEW-family copy gains options. Old parsers/defaults/choices never mutate.

    Later task integration must call this after selecting the model key. M1 uses
    it only for configuration previews and parser-AST tests, not task execution.
    """
    if family_for_model_key(model_key) is None:return None
    probe = copy.deepcopy(parser)
    for action in probe._actions:
        if action.dest in ('model', 'cfd_model') and action.choices is not None:
            action.choices = tuple(action.choices) + ((FAMILY,) if FAMILY not in action.choices else ())
    profiles = [a for a in probe._actions if a.dest == 'profile']
    if profiles:
        for action in profiles:action.choices = PROFILE_NAMES; action.default = 'light'
    else:probe.add_argument('--profile', choices=PROFILE_NAMES, default='light')
    probe.add_argument('--d', type=int)
    probe.add_argument('--num-latents', '--num_latents', nargs=4, type=int)
    probe.add_argument('--heads', nargs=4, type=int)
    probe.add_argument('--encoder-depths', '--encoder_depths', nargs=4, type=int)
    probe.add_argument('--decoder-depths', '--decoder_depths', nargs=4, type=int)
    probe.add_argument('--coverage-mode', '--coverage_mode', choices=('off', 'floor'))
    probe.add_argument('--coverage-weight', '--coverage_weight', type=float)
    probe.add_argument('--coverage-kappa', '--coverage_kappa', type=float)
    probe.add_argument('--coverage-eps', '--coverage_eps', type=float)
    probe.add_argument('--diagnostics', action=argparse.BooleanOptionalAction)
    return probe


def parse_training_options(parser: argparse.ArgumentParser, argv: Sequence[str], *, task: str,
                           model_key: str) -> ResolvedMSARConfig | None:
    probe = parser_for_family(parser, model_key)
    if probe is None:return None
    return resolve_training(task, explicit_arguments(probe, argv))
