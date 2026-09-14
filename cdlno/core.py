"""Cross-Depth Latent Neural Operator over already-lifted point features.

Only depth-local raw history is managed here. Task input lifting, position/time
conditioning, data protocols and training entry points belong to wrappers.
"""

from __future__ import annotations

from torch import Tensor, nn

from .cdpa import CDPA, _check_chunk_size
from .config import CDLNOArchitectureConfig
from .modules import (
    IPOTBridge, LRSAFeatureReadout, LRSAFrontBlock, PersistentLatentBlock,
    _check_grid, _check_tokens,
)


class CDLNO(nn.Module):
    """Map H_0[B,N,d] to predictions[B,N,output_dim], without an input stem.

    The immutable architecture config determines modules and grid layout.
    ``source_chunk_size`` is an execution setting, outside the state_dict;
    callers control device/dtype/autocast using normal PyTorch conventions.
    Forward returns only a Tensor, with no retained history or debug state.
    """

    def __init__(
        self,
        config: CDLNOArchitectureConfig | None = None,
        *,
        source_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        config = CDLNOArchitectureConfig() if config is None else config
        if not isinstance(config, CDLNOArchitectureConfig):
            raise TypeError("config must be CDLNOArchitectureConfig")
        config.validate()
        # The assembled version binds the exact already-validated v1.2 modules.
        # Reject unsupported policy changes rather than silently ignoring them.
        if config.model_version != "cdlno-core-v1":
            raise ValueError("CDLNO core requires model_version='cdlno-core-v1'")
        if config.history_rule != "front-t-after-ffn2-before-up-v1":
            raise ValueError("CDLNO core requires the v1 history rule")
        if config.norm_type != "rmsnorm":
            raise ValueError("CDLNO core requires norm_type='rmsnorm' for LRSA branches")
        if config.attention_dropout != 0.0:
            raise ValueError("CDLNO core requires attention_dropout=0")
        _check_chunk_size(source_chunk_size)
        self.config = config
        self.source_chunk_size = source_chunk_size
        d, h, m = config.d_model, config.num_heads, config.M
        point_settings = dict(
            structured=config.structured, grid_shape=config.grid_shape,
            ffn_ratio=config.ffn_ratio,
        )
        self.front_blocks = nn.ModuleList([
            LRSAFrontBlock(d, h, m, **point_settings) for _ in range(config.F)
        ])
        self.bridge = IPOTBridge(d, h, m)
        self.latent_blocks = nn.ModuleList([
            PersistentLatentBlock(d, h, geglu_ratio=config.latent_ffn_ratio)
            for _ in range(config.P)
        ])
        self.readout = LRSAFeatureReadout(d, h, config.output_dim, **point_settings)
        # Keys are zero-based rear block indices. Inactive locations register
        # nothing, including F=0/entry and the first F=0/every_block location.
        self.cdpa_at = nn.ModuleDict({
            str(j): CDPA(d, h)
            for j in range(config.P)
            if ((config.cdpa_mode == "entry" and j == 0 and config.F > 0)
                or (config.cdpa_mode == "every_block" and config.F + j > 0))
        })
        # Each child initializes itself once; no recursive parent reset.

    def forward(self, h_0: Tensor, *, source_chunk_size: int | None = None) -> Tensor:
        if not isinstance(h_0, Tensor):
            raise TypeError("H_0 must be a Tensor of lifted point features")
        _check_tokens(h_0, self.config.d_model)
        if not h_0.is_floating_point():
            raise ValueError("H_0 must be floating-point")
        if self.config.structured:
            _check_grid(self.config.grid_shape, h_0.shape[1])
        chunk = self.source_chunk_size if source_chunk_size is None else source_chunk_size
        _check_chunk_size(chunk)

        # No history collection at all in off or other entirely inactive cases.
        history = [] if self.cdpa_at else None
        h_f = h_0
        for block in self.front_blocks:
            h_f, t = block(h_f)
            if history is not None:
                history.append(t)
            del t

        current = self.bridge(h_f)  # raw Z_0, never overwritten in history
        for j, block in enumerate(self.latent_blocks):
            previous = current  # raw Z_(j-1), the identity at this location
            key = str(j)
            if key in self.cdpa_at:
                block_input = self.cdpa_at[key](
                    previous, tuple(history), source_chunk_size=chunk,
                )
            else:
                block_input = previous
            current = block(block_input)  # Z_j, after the complete SA + FFN
            if history is not None:
                if self.config.cdpa_mode == "every_block" and j + 1 < self.config.P:
                    history.append(previous)  # append only after its identity use
                else:
                    history = None  # no later consumer (autograd remains live)

        return self.readout(h_f, current)


__all__ = ["CDLNO"]
