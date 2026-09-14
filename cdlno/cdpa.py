"""Single-position Cross-Depth Physics Attention (v1.2 section 3).

History selection and lifetime belong to the future model core. This module
only consumes the raw histories of one call; it never caches or detaches them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modules import RMSNorm, _check_tokens, _init_linear, _positive_int


def _check_chunk_size(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("source_chunk_size must be a non-negative integer")


class CDPA(nn.Module):
    """Align each history independently, then fuse raw candidates per token.

    Args:
        dim: Hidden width d, shared by current and historical latent tensors.
        heads: Number of Cross-attention heads; d must be divisible by heads.
        source_chunk_size: 0 batches all sources, 1 processes one at a time,
            and k>1 processes groups of at most k. This is a runtime setting.

    The default output is [B,M,d]. ``return_weights=True`` additionally returns
    FP32 source weights [B,M,S+1] for small diagnostics, with identity at index 0.
    No token-attention matrices or debug tensors are stored on the module.
    Histories may have a different floating dtype (e.g. BF16 front output and
    FP32 bridge residual under AMP); Cross input uses Z's dtype, preserving
    gradients through the cast. Sources must be on Z's device.
    """

    def __init__(self, dim: int, heads: int, *, source_chunk_size: int = 0) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("heads", heads)
        _check_chunk_size(source_chunk_size)
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.source_chunk_size = source_chunk_size
        self.ln_q = nn.LayerNorm(dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(dim, eps=1e-6)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.depth_norm = RMSNorm(dim, eps=1e-6)
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out):
            _init_linear(layer)
        # Initialize this special parameter last; never apply a parent reset.
        self.w = nn.Parameter(torch.zeros(dim))

    def _depth_fusion(self, z: Tensor, aligned: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        # Disabling autocast is essential: .float() alone does not keep matrix
        # scoring in FP32 when an outer autocast region is active.
        with torch.autocast(device_type=z.device.type, enabled=False):
            raw = torch.stack([z, *aligned], dim=2).float()  # [B,M,S+1,d]
            keys = self.depth_norm(raw)  # FP32 RMS, including scale multiplication
            scores = torch.matmul(keys, self.w.float())
            weights = scores.softmax(dim=2)  # [B,M,S+1], one distribution per token
            fused = (weights.unsqueeze(-1) * raw).sum(dim=2)
        return fused.to(z.dtype), weights

    def forward(
        self,
        z: Tensor,
        history: Sequence[Tensor],
        *,
        source_chunk_size: int | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _check_tokens(z, self.dim)
        if not z.is_floating_point():
            raise ValueError("CDPA requires floating-point latent tensors")
        chunk_size = self.source_chunk_size if source_chunk_size is None else source_chunk_size
        _check_chunk_size(chunk_size)
        if isinstance(history, Tensor) or not isinstance(history, Sequence):
            raise TypeError("history must be a sequence of raw [B,M,d] tensors")
        sources = tuple(history)  # immutable snapshot, no clone/detach
        for index, source in enumerate(sources):
            if not isinstance(source, Tensor) or source.shape != z.shape:
                raise ValueError(f"history[{index}] must have the same [B,M,d] shape as Z")
            if source.device != z.device or not source.is_floating_point():
                raise ValueError(f"history[{index}] must be floating-point on the same device as Z")
        b, m, _ = z.shape
        if not sources:
            # Also preserves exact identity for FP64/half tensors; no projection
            # or FP32 roundtrip is needed. The core must omit inactive modules.
            if return_weights:
                return z, torch.ones(b, m, 1, device=z.device, dtype=torch.float32)
            return z

        query = self.to_q(self.ln_q(z))  # computed once for this CDPA position
        query = query.reshape(b, m, self.heads, self.head_dim).transpose(1, 2)
        step = len(sources) if chunk_size == 0 else chunk_size
        aligned = []
        for start in range(0, len(sources), step):
            group = sources[start:start + step]
            k_sources = len(group)
            # Batch-major, then source-major; never concatenate on the M axis.
            contexts = torch.stack(group, dim=1).reshape(b * k_sources, m, self.dim).to(z.dtype)
            contexts = self.ln_kv(contexts)
            key = self.to_k(contexts).reshape(b * k_sources, m, self.heads, self.head_dim).transpose(1, 2)
            value = self.to_v(contexts).reshape(b * k_sources, m, self.heads, self.head_dim).transpose(1, 2)
            q = query.unsqueeze(1).expand(-1, k_sources, -1, -1, -1)
            q = q.reshape(b * k_sources, self.heads, m, self.head_dim)
            out = F.scaled_dot_product_attention(
                q, key, value, dropout_p=0.0, is_causal=False,
                scale=self.head_dim ** -0.5,
            )
            out = out.transpose(1, 2).reshape(b * k_sources, m, self.dim)
            out = self.to_out(out).reshape(b, k_sources, m, self.dim)
            aligned.extend(out.unbind(dim=1))  # each R_s already includes W_O/b_O

        fused, weights = self._depth_fusion(z, aligned)
        return (fused, weights) if return_weights else fused


__all__ = ["CDPA"]
