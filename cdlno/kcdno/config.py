"""Independent KCDNO v1 contracts. No torch, task, or CDLNO config imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any


FAMILY = "kcdno"
MODEL_NAME = "KCDNO"


def _positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer (not bool/float), got {value!r}")


def _finite(name: str, value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


class _Record:
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        # Saved records must be complete. Constructor defaults are for NEW runs.
        if type(data) is not dict:
            raise ValueError(f"{cls.__name__} must be an object")
        names = {field.name for field in fields(cls)}
        missing, extra = names - data.keys(), data.keys() - names
        if missing or extra:
            raise ValueError(f"{cls.__name__}: missing={sorted(missing)}, unknown={sorted(extra)}")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class KCDNOArchitectureConfig(_Record):
    """Resolved core structure/behavior; all these fields affect compatibility.

    There is ONE width d and ONE token count M, common to all L blocks.
    There is no CDLNO F/P/front/CDPA field. Hidden widths resolve after d.
    Version v1 also fixes sum summaries, raw values, source-owned keys,
    receiver-owned queries, FP32 history arithmetic, and no latent SA.
    Task lift/grid/output semantics are checked separately by task adapters and task.json.
    """

    family: str = FAMILY
    architecture_version: str = "kcdno-core-v1"
    L: int = 8
    d: int = 128
    h: int = 8
    M: int = 64
    kernel_rank: int = 16
    history_mode: str = "all"
    ffn1_hidden: int | None = None
    ffn2_hidden: int | None = None
    point_hidden: int | None = None
    latent_activation: str = "gelu"
    point_activation: str = "gelu"
    point_module: str = "point_ffn"
    norm: str = "rmsnorm"
    qk_norm: str = "per_head_rmsnorm"
    output_norm: str = "layernorm"
    norm_eps: float = 1e-6
    kernel_phi: str = "elu_plus_one"
    kernel_clamp: float = 1e-6
    kernel_denominator_eps: float = 1e-6
    kernel_projection_bias: bool = False
    gate_parameterization: str = "unconstrained_scalar"
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        _positive_int("d", self.d)
        for name in ("ffn1_hidden", "ffn2_hidden", "point_hidden"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, 2 * self.d)
        self.validate()

    def validate(self) -> KCDNOArchitectureConfig:
        for name in ("L", "d", "h", "M", "kernel_rank", "ffn1_hidden", "ffn2_hidden", "point_hidden"):
            _positive_int(name, getattr(self, name))
        if self.d % self.h:
            raise ValueError("d must be divisible by h; kernel_rank is independent of h")
        if self.history_mode not in ("all", "off"):
            raise ValueError("history_mode must be 'all' or 'off'")
        if self.point_module not in ("point_ffn", "conv_ffn"):
            raise ValueError("point_module must be 'point_ffn' or 'conv_ffn'")
        fixed = dict(family=FAMILY, architecture_version="kcdno-core-v1",
                     latent_activation="gelu", point_activation="gelu", norm="rmsnorm",
                     qk_norm="per_head_rmsnorm", output_norm="layernorm", norm_eps=1e-6,
                     kernel_phi="elu_plus_one", kernel_clamp=1e-6, kernel_denominator_eps=1e-6,
                     kernel_projection_bias=False, gate_parameterization="unconstrained_scalar",
                     attention_dropout=0.0)
        for name, expected in fixed.items():
            value = getattr(self, name)
            if type(expected) is float:
                _finite(name, value)
            if value != expected or (type(expected) is bool and type(value) is not bool):
                raise ValueError(f"KCDNO v1 requires {name}={expected!r}, got {value!r}")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KCDNOArchitectureConfig:
        # Null is a constructor convenience, never a resolved saved width.
        if isinstance(data, dict):
            for name in ("ffn1_hidden", "ffn2_hidden", "point_hidden"):
                if name in data:
                    _positive_int(name, data[name])
        return super(KCDNOArchitectureConfig, cls).from_dict(data)


@dataclass(frozen=True, slots=True)
class KCDNOInitializationConfig(_Record):
    """Training origin only. Reading this record NEVER initializes model weights.

    public_init='cdlno-front-v1': Linear trunc_normal(.02)/bias0,
    Down query orthogonal on [M,d], native Conv, norm scale1/bias0.
    Kernel Q/K use Xavier uniform gain1 and the architecture's bias=False.
    """

    initialization_version: str = "kcdno-init-v1"
    public_init: str = "cdlno-front-v1"
    kernel_init: str = "xavier_uniform_gain1"
    gamma_init: float = 0.1
    scorer_init: float = 0.0
    norm_scale_init: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> KCDNOInitializationConfig:
        for name in ("initialization_version", "public_init", "kernel_init"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty provenance string")
        for name in ("gamma_init", "scorer_init", "norm_scale_init"):
            _finite(name, getattr(self, name))
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        return self


@dataclass(frozen=True, slots=True)
class KCDNORuntimeConfig(_Record):
    """Recorded execution choices; differences may change numerical results."""

    device: str = "cuda"
    dtype: str = "float32"
    batch_size: int = 1
    sdpa_backend: str = "automatic"
    amp: bool = False
    tf32: bool = False
    compile: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> KCDNORuntimeConfig:
        _positive_int("batch_size", self.batch_size)
        for name in ("device", "dtype", "sdpa_backend"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("amp", "tf32", "compile"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        return self
