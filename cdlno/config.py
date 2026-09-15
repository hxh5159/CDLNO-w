"""Configuration contracts for the shared CDLNO package.

The architecture and runtime configurations are deliberately separate.  The
derived rear depth ``P`` is never accepted as an independent constructor
argument or serialized as an authoritative input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Literal


CDPAMode = Literal["off", "entry", "every_block"]
FrontLatentMode = Literal["full", "no_sa", "identity"]
HISTORY_RULE = "front-t-after-selected-processor-before-up-v1"
_LEGACY_HISTORY_RULE = "front-t-after-ffn2-before-up-v1"
# Frozen/slotted dataclasses originally pickled this positional 18-field list.
# Keep this snapshot explicit: changing declaration order must not remap it.
_LEGACY_FIELDS = (
    "model_name", "model_version", "history_rule", "task_name", "L", "F", "M",
    "d_model", "num_heads", "norm_type", "ffn_ratio", "latent_ffn_ratio",
    "decoder_mode", "cdpa_mode", "attention_dropout", "structured", "grid_shape", "output_dim",
)


def _check_front_latent_mode(value: str) -> None:
    if type(value) is not str or value not in ("full", "no_sa", "identity"):
        raise ValueError(f"front_latent_mode must be 'full', 'no_sa' or 'identity', got {value!r}")


@dataclass(frozen=True, slots=True)
class CDLNOArchitectureConfig:
    """Model-semantic fields that determine checkpoint tensor structure."""

    model_name: str = "CDLNO"
    model_version: str = "cdlno-core-v1"
    history_rule: str = HISTORY_RULE
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
    # Architecture, not runtime: removed sublayers have no registered weights.
    front_latent_mode: FrontLatentMode = "full"

    def __post_init__(self) -> None:
        # Only the known historical full rule is an alias. Never reinterpret a
        # conflicting ablation or an unknown model version as historical full.
        if (self.history_rule == _LEGACY_HISTORY_RULE
                and self.model_version == "cdlno-core-v1"
                and self.front_latent_mode == "full"):
            object.__setattr__(self, "history_rule", HISTORY_RULE)

    def __getstate__(self) -> dict[str, Any]:
        # New object checkpoints use named fields; old 18-value lists remain
        # readable below without inserting a value into the wrong slot.
        return {"config_pickle_version": 1, "architecture": self.to_dict()}

    def __setstate__(self, state: Any) -> None:
        if type(state) in (list, tuple) and len(state) == len(_LEGACY_FIELDS):
            values = dict(zip(_LEGACY_FIELDS, state))
            if (values["model_version"] != "cdlno-core-v1"
                    or values["history_rule"] != _LEGACY_HISTORY_RULE):
                raise ValueError("unsupported legacy CDLNO configuration pickle")
        elif (type(state) is dict and set(state) == {"config_pickle_version", "architecture"}
              and type(state["config_pickle_version"]) is int and state["config_pickle_version"] == 1
              and type(state["architecture"]) is dict
              and "front_latent_mode" in state["architecture"]):
            values = state["architecture"]
        else:
            raise ValueError("unsupported CDLNO configuration pickle layout")
        restored = type(self).from_dict(values)
        for field in fields(self):
            object.__setattr__(self, field.name, getattr(restored, field.name))

    @property
    def P(self) -> int:
        """Persistent rear depth, derived solely from ``L`` and ``F``."""

        return self.L - self.F

    def validate(self) -> "CDLNOArchitectureConfig":
        if self.model_name != "CDLNO":
            raise ValueError(f"model_name must be CDLNO, got {self.model_name!r}")
        _check_front_latent_mode(self.front_latent_mode)
        if self.history_rule != HISTORY_RULE:
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
        if "front_latent_mode" not in values:
            if values.get("model_version", "cdlno-core-v1") != "cdlno-core-v1":
                raise ValueError("cannot infer historical full for an unknown CDLNO model version")
            values["front_latent_mode"] = "full"
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
