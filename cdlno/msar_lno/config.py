"""MSAR-LNO v1 configuration only; independent of torch and old model configs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

FAMILY = 'msar_lno'
MODEL_NAME = 'MSAR-LNO'
DEPTHS = (3, 1, 1, 1)


def positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer, not bool/float; got {value!r}')


def finite(name: str, value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number; got {value!r}')


class _Record:
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        # JSON-native lists on disk, immutable tuples inside the config.
        return {k:list(v) if isinstance(v, tuple) else v for k,v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        if type(data) is not dict:
            raise ValueError(f'{cls.__name__} must be an object')
        names = {f.name for f in fields(cls)}
        missing, extra = names - data.keys(), data.keys() - names
        if missing or extra:
            raise ValueError(f'{cls.__name__}: missing={sorted(missing)}, unknown={sorted(extra)}')
        return cls(**data)  # Defaults apply to new construction, never incomplete saved records.


@dataclass(frozen=True, slots=True)
class MSARArchitectureConfig(_Record):
    family: str = FAMILY
    architecture_version: str = 'msar-lno-core-v1'
    num_latents: tuple[int, ...] = (512, 256, 128, 64)
    d: int = 96
    heads: tuple[int, ...] = (4, 4, 8, 8)
    encoder_depths: tuple[int, ...] = DEPTHS
    decoder_depths: tuple[int, ...] = DEPTHS
    latent_ffn_ratio: int | float = 2
    activation: str = 'gelu'
    norm: str = 'rmsnorm'
    qk_norm: str = 'per_head_rmsnorm'
    output_norm: str = 'layernorm'
    norm_eps: float = 1e-6
    point_module: str = 'pointwise_mlp'
    fusion: str = 'query_aligned_two_source_attnres'
    fusion_scale: float = 2.0

    def __post_init__(self) -> None:
        for name in ('num_latents', 'heads', 'encoder_depths', 'decoder_depths'):
            value = getattr(self, name)
            if type(value) not in (tuple, list) or len(value) != 4:
                raise ValueError(f'{name} must contain exactly four integers')
            object.__setattr__(self, name, tuple(value))
        self.validate()

    @property
    def latent_ffn_hidden(self) -> int:
        return 2 * self.d

    def validate(self) -> MSARArchitectureConfig:
        positive_int('d', self.d)
        for name in ('num_latents', 'heads', 'encoder_depths', 'decoder_depths'):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 4:
                raise ValueError(f'{name} must contain exactly four integers')
            for i, value in enumerate(values):
                if name.endswith('depths'):
                    if type(value) is not int or value < 0:
                        raise ValueError(f'{name}[{i}] must be a nonnegative integer')
                else:
                    positive_int(f'{name}[{i}]', value)
            if name.endswith('depths') and values != DEPTHS:
                raise ValueError(f'MSAR-LNO v1 requires {name}={DEPTHS}')
        if any(self.d % h for h in self.heads):
            raise ValueError('d must be divisible by every entry of heads')
        if any(a <= b for a,b in zip(self.num_latents, self.num_latents[1:])):
            raise ValueError('num_latents must decrease across the four scales; input N is not a limit')
        finite('latent_ffn_ratio', self.latent_ffn_ratio)
        fixed = dict(family=FAMILY, architecture_version='msar-lno-core-v1', latent_ffn_ratio=2,
                     activation='gelu', norm='rmsnorm', qk_norm='per_head_rmsnorm',
                     output_norm='layernorm', norm_eps=1e-6, point_module='pointwise_mlp',
                     fusion='query_aligned_two_source_attnres', fusion_scale=2.0)
        for name, expected in fixed.items():
            value = getattr(self, name)
            if isinstance(expected, float):finite(name, value)
            if value != expected:
                raise ValueError(f'MSAR-LNO v1 requires {name}={expected!r}, got {value!r}')
        return self


@dataclass(frozen=True, slots=True)
class MSARTrainingConfig(_Record):
    """Auxiliary objective / diagnostics; not weight shape or inference behavior."""
    coverage_mode: str = 'floor'
    coverage_weight: float = 1e-2
    coverage_kappa: float = 0.2
    coverage_eps: float = 1e-6
    diagnostics: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def coverage_enabled(self) -> bool:
        return self.coverage_mode == 'floor' and self.coverage_weight > 0

    @property
    def effective_coverage_mode(self) -> str:
        return 'floor' if self.coverage_enabled else 'off'

    def validate(self) -> MSARTrainingConfig:
        if self.coverage_mode not in ('off', 'floor'):
            raise ValueError("coverage_mode must be 'off' or 'floor'")
        for name in ('coverage_weight', 'coverage_kappa', 'coverage_eps'):
            finite(name, getattr(self, name))
        if self.coverage_weight < 0:raise ValueError('coverage_weight must be >= 0')
        if not 0 <= self.coverage_kappa <= 1:raise ValueError('coverage_kappa must be in [0, 1]')
        if self.coverage_eps <= 0:raise ValueError('coverage_eps must be > 0')
        if type(self.diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        return self


@dataclass(frozen=True, slots=True)
class MSARRuntimeConfig(_Record):
    """Execution record, not eight-task defaults or architecture constraints."""
    device: str = 'cuda'
    dtype: str = 'float32'
    batch_size: int = 1
    sdpa_backend: str = 'automatic'
    amp: bool = False
    tf32: bool = False
    compile: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MSARRuntimeConfig:
        positive_int('batch_size', self.batch_size)
        for name in ('device', 'dtype', 'sdpa_backend'):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('amp', 'tf32', 'compile'):
            if type(getattr(self, name)) is not bool:raise ValueError(f'{name} must be bool')
        return self


def input_layout(architecture: MSARArchitectureConfig, input_tokens: int) -> dict:
    """Data-free log record; never clip M to N. Entry logging is a later stage."""
    architecture.validate()
    positive_int('input_tokens', input_tokens)
    return dict(input_tokens=input_tokens, num_latents=list(architecture.num_latents),
                first_down_expands=architecture.num_latents[0] > input_tokens,
                token_path=[input_tokens, *architecture.num_latents],
                note='M1>N is a legal latent expansion; no token count was clipped'
                if architecture.num_latents[0] > input_tokens else 'first Down compresses or preserves token count')
