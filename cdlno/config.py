"""Configuration contracts for the shared CDLNO package.

The architecture and runtime configurations are deliberately separate.  The
derived rear depth ``P`` is never accepted as an independent constructor
argument or serialized as an authoritative input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal


CDPAMode = Literal["off", "entry", "every_block"]


@dataclass(frozen=True, slots=True)
class CDLNOArchitectureConfig:
    """Model-semantic fields that determine checkpoint tensor structure."""

    model_name: str = "CDLNO"
    model_version: str = "cdlno-core-v1"
    history_rule: str = "front-t-after-ffn2-before-up-v1"
    task_name: str = "unspecified"
    L: int = 8
    F: int = 2
    M: int = 64
    d_model: int = 64
    num_heads: int = 4
    norm_type: str = "rmsnorm"
    # Existing ffn_ratio is the front latent + point/readout FFN ratio.
    ffn_ratio: float = 2.0
    latent_ffn_ratio: float = 2.0
    decoder_mode: str = "feature"
    cdpa_mode: CDPAMode = "entry"
    attention_dropout: float = 0.0
    structured: bool = False
    grid_shape: tuple[int, int] | None = None
    output_dim: int = 1

    @property
    def P(self) -> int:
        """Persistent rear depth, derived solely from ``L`` and ``F``."""

        return self.L - self.F

    def validate(self) -> "CDLNOArchitectureConfig":
        if self.model_name != "CDLNO":
            raise ValueError(f"model_name must be CDLNO, got {self.model_name!r}")
        if self.history_rule != "front-t-after-ffn2-before-up-v1":
            raise ValueError("unsupported CDLNO history_rule")
        for name in ("L", "F", "M", "d_model", "num_heads", "output_dim"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an int (not bool)")
        if not (0 <= self.F < self.L):
            raise ValueError(f"require 0 <= F < L, got F={self.F}, L={self.L}")
        if self.P < 1:
            raise ValueError(f"derived P=L-F must be >= 1, got P={self.P}")
        if self.M < 1 or self.d_model < 1 or self.num_heads < 1 or self.output_dim < 1:
            raise ValueError("M, d_model, num_heads and output_dim must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError(f"unsupported norm_type: {self.norm_type!r}")
        if self.decoder_mode != "feature":
            raise ValueError("CDLNO only defines decoder_mode='feature'")
        if self.cdpa_mode not in {"off", "entry", "every_block"}:
            raise ValueError(f"unsupported cdpa_mode: {self.cdpa_mode!r}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError("attention_dropout must be in [0, 1)")
        for name in ("ffn_ratio", "latent_ffn_ratio"):
            ratio = getattr(self, name)
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"{name} must be finite and positive")
            width = self.d_model * ratio
            if not math.isfinite(width) or not float(width).is_integer():
                raise ValueError(f"d_model * {name} must be an integer")
        if type(self.structured) is not bool:
            raise TypeError("structured must be bool")
        if self.grid_shape is not None:
            if (not isinstance(self.grid_shape, tuple) or len(self.grid_shape) != 2
                    or any(type(n) is not int or n < 1 for n in self.grid_shape)):
                raise ValueError("grid_shape must be a tuple of two positive integers")
        if self.structured != (self.grid_shape is not None):
            raise ValueError("structured models require grid_shape; point models omit it")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        if self.grid_shape is not None:
            data["grid_shape"] = list(self.grid_shape)
        # P is informational and is recomputed during loading/comparison.
        data.pop("P", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CDLNOArchitectureConfig":
        values = dict(data)
        if isinstance(values.get("grid_shape"), list):
            values["grid_shape"] = tuple(values["grid_shape"])
        supplied_p = values.pop("P", None)
        config = cls(**values)
        config.validate()
        if supplied_p is not None and supplied_p != config.P:
            raise ValueError(f"serialized P={supplied_p} disagrees with derived P={config.P}")
        return config


@dataclass(frozen=True, slots=True)
class CDLNORuntimeConfig:
    """Execution fields that do not change model tensor structure."""

    source_chunk_size: int = 0
    device: str = "cuda"
    dtype: str = "float32"
    sdpa_backend: str = "automatic"
    amp: bool = False
    compile: bool = False

    def validate(self) -> "CDLNORuntimeConfig":
        if type(self.source_chunk_size) is not int or self.source_chunk_size < 0:
            raise ValueError("source_chunk_size must be a non-negative integer")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CDLNORuntimeConfig":
        config = cls(**dict(data))
        return config.validate()
