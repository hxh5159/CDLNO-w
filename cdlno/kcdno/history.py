"""KCDNO v1 single-source writer and single-location history reader.

Source ownership, active layer allocation, and tuple history lifetime belong
to core.py. These modules keep no inputs, caches, or debug tensors.
All M latent tokens are valid; there is no padding/mask or graph-batch policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..modules import RMSNorm
from .config import _positive_int


_EPS = 1e-6
_CLAMP = 1e-6


class KernelHistoryCache(NamedTuple):
    """Live FP32 tensors, with no raw T or source key projection stored.

    The tuple prevents replacing fields; tensor contents must also be treated
    as immutable by callers. Neither writer nor reader changes them in place.
    """

    memory: Tensor  # [B,r,d], sum over source tokens of key outer raw value
    mass: Tensor  # [B,r], sum over source tokens of positive keys


def _check_latent(x: Tensor, dim: int, name: str) -> None:
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise ValueError(f"{name} must be a floating-point [B,M,{dim}] tensor")
    if x.ndim != 3 or x.shape[-1] != dim or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"{name} must be nonempty [B,M,{dim}], got {tuple(x.shape)}")


def _phi(x: Tensor) -> Tensor:
    # Caller casts projections and disables autocast before invoking phi.
    return (F.elu(x) + 1.0).clamp_min(_CLAMP)


class KernelHistoryWriter(nn.Module):
    """Owned by source s: K=phi(RMS_k(T) Wk); return (K^T T, sum_token K).

    Input T[B,M,d] may be FP32/FP16/BF16/FP64. Production arithmetic and
    returned summaries are always FP32, preserving gradients through casts.
    No Wv/Wo, token softmax, scaling, or detach. Rank is a single total rank,
    independent of Down/Up head count. Wk's torch Linear weight is [r,d].
    """

    def __init__(self, dim: int, kernel_rank: int = 16) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("kernel_rank", kernel_rank)
        self.dim = dim
        self.kernel_rank = kernel_rank
        self.key_norm = RMSNorm(dim, eps=_EPS)
        self.to_k = nn.Linear(dim, kernel_rank, bias=False)
        nn.init.xavier_uniform_(self.to_k.weight, gain=1.0)

    def forward(self, t: Tensor) -> KernelHistoryCache:
        _check_latent(t, self.dim, "T")
        with torch.autocast(device_type=t.device.type, enabled=False):
            raw = t.float()
            # Functional Linear explicitly casts parameter storage too, allowing
            # a half-converted module without lowering the production compute.
            key = _phi(F.linear(self.key_norm(raw), self.to_k.weight.float()))
            memory = key.transpose(-1, -2) @ raw
            mass = key.sum(dim=1)
        return KernelHistoryCache(memory, mass)


class KernelHistoryReader(nn.Module):
    """Owned by receiver l: query summaries once, then gate raw source fusion.

    Accepts only KernelHistoryCache records, not raw histories for re-keying.
    Empty history returns the exact U object without norms/projections/gating.
    Default output is a Tensor[B,M,d]; optional FP32 weights[B,M,S+1] are for
    explicit diagnostics only. No token attention matrices or caches persist.
    Calling .double() does not make this production path a double oracle.
    """

    def __init__(self, dim: int, kernel_rank: int = 16) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("kernel_rank", kernel_rank)
        self.dim = dim
        self.kernel_rank = kernel_rank
        self.query_norm = RMSNorm(dim, eps=_EPS)
        self.to_q = nn.Linear(dim, kernel_rank, bias=False)
        nn.init.xavier_uniform_(self.to_q.weight, gain=1.0)
        self.depth_norm = RMSNorm(dim, eps=_EPS)
        self.w = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.tensor(0.1))  # unconstrained scalar

    def forward(self, u: Tensor, history: Sequence[KernelHistoryCache], *,
                return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        _check_latent(u, self.dim, "U")
        if isinstance(history, Tensor) or not isinstance(history, Sequence):
            raise TypeError("history must be a sequence of KernelHistoryCache summaries")
        sources = tuple(history)
        b, m, d = u.shape
        for index, source in enumerate(sources):
            if not isinstance(source, KernelHistoryCache):
                raise TypeError(f"history[{index}] must be KernelHistoryCache, not raw T")
            for name, shape in (("memory", (b, self.kernel_rank, d)), ("mass", (b, self.kernel_rank))):
                value = getattr(source, name)
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"history[{index}].{name} must have shape {shape}")
                if value.dtype != torch.float32 or value.device != u.device:
                    raise ValueError(f"history[{index}].{name} must be FP32 on U.device (use the writer output)")
        if not sources:
            if return_weights:
                return u, torch.ones(b, m, 1, dtype=torch.float32, device=u.device)
            return u

        with torch.autocast(device_type=u.device.type, enabled=False):
            current = u.float()
            query = _phi(F.linear(self.query_norm(current), self.to_q.weight.float()))  # once
            memory = torch.stack([source.memory for source in sources], dim=1)  # [B,S,r,d]
            mass = torch.stack([source.mass for source in sources], dim=1)  # [B,S,r]
            numerator = torch.einsum('bmr,bsrd->bsmd', query, memory)
            denominator = torch.einsum('bmr,bsr->bsm', query, mass).unsqueeze(-1)
            reads = numerator / (denominator + _EPS)  # [B,S,M,d]
            candidates = torch.cat([current.unsqueeze(1), reads], dim=1).transpose(1, 2)  # [B,M,S+1,d]
            scores = (self.depth_norm(candidates) * self.w.float()).sum(dim=-1)
            alpha = scores.softmax(dim=2)  # one distribution per batch/current token
            fused = (alpha.unsqueeze(-1) * candidates).sum(dim=2)  # RAW values
            output = current + self.gamma.float() * (fused - current)
        output = output.to(u.dtype)
        return (output, alpha) if return_weights else output


__all__ = ["KernelHistoryCache", "KernelHistoryWriter", "KernelHistoryReader"]
