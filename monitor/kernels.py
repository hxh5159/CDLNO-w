"""Independent LinearNO propagation-kernel math.

For one layer and head, ``Q`` and ``K`` have shape ``[N, R]`` and define the
point propagation kernel ``P = Q K.T``.  The functions below use only Gram
matrices, so their working complexity is ``O(N R^2)`` and no ``N x N`` tensor
is allocated.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _factor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape [B,N,R] or [B,H,N,R]")
    if value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have positive N and R, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value


def _as_bhnr(value: torch.Tensor, name: str) -> torch.Tensor:
    value = _factor(value, name)
    return value.unsqueeze(1) if value.ndim == 3 else value


def kernel_cosine(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    q_j: torch.Tensor,
    k_j: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return ``rho_ij`` for corresponding batches and heads.

    Inputs may be ``[B,N,R]`` or ``[B,H,N,R]``.  Ranks may differ between the
    two layers; only the point count, batch count, head count, device and dtype
    must match.  The result has shape ``[B,H]``.
    """

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    qi, ki = _as_bhnr(q_i, "q_i"), _as_bhnr(k_i, "k_i")
    qj, kj = _as_bhnr(q_j, "q_j"), _as_bhnr(k_j, "k_j")
    if qi.shape[:3] != ki.shape[:3] or qj.shape[:3] != kj.shape[:3]:
        raise ValueError("Q and K must have matching [B,H,N] dimensions")
    if qi.shape[-1] != ki.shape[-1] or qj.shape[-1] != kj.shape[-1]:
        raise ValueError("each layer's Q and K must share the same rank")
    if qi.shape[:3] != qj.shape[:3]:
        raise ValueError("layers must share batch, head and point dimensions")
    if qi.device != ki.device or qi.device != qj.device or qi.device != kj.device:
        raise ValueError("all factors must be on the same device")

    # Do arithmetic in FP32 or FP64 even when the model uses AMP.  This does
    # not affect the model graph because monitor inputs are detached first.
    work_dtype = torch.float64 if any(x.dtype == torch.float64 for x in (qi, ki, qj, kj)) else torch.float32
    qi, ki, qj, kj = (x.detach().to(dtype=work_dtype) for x in (qi, ki, qj, kj))
    qij = torch.einsum("bhni,bhnj->bhij", qi, qj)
    kji = torch.einsum("bhnj,bhni->bhji", kj, ki)
    # tr(A B) = sum_{a,b} A[a,b] B[b,a].  ``kji`` has shape
    # [b,h,rank_j,rank_i], hence the transpose before the elementwise sum.
    inner = (qij * kji.transpose(-1, -2)).sum(dim=(-1, -2))

    qii = torch.einsum("bhni,bhnj->bhij", qi, qi)
    kii = torch.einsum("bhni,bhnj->bhij", ki, ki)
    qjj = torch.einsum("bhni,bhnj->bhij", qj, qj)
    kjj = torch.einsum("bhni,bhnj->bhij", kj, kj)
    norm_i_sq = (qii * kii.transpose(-1, -2)).sum(dim=(-1, -2)).clamp_min(0)
    norm_j_sq = (qjj * kjj.transpose(-1, -2)).sum(dim=(-1, -2)).clamp_min(0)
    denominator = torch.sqrt(norm_i_sq) * torch.sqrt(norm_j_sq)
    return inner / (denominator + float(eps))


def pairwise_kernel_similarity(
    factors: Iterable[tuple[torch.Tensor, torch.Tensor]], *, eps: float = 1e-12
) -> torch.Tensor:
    """Return the layer matrix with shape ``[B,H,L,L]``.

    ``factors`` is an iterable of ``(Q, K)`` pairs.  The diagonal is computed
    with the same formula as the off-diagonal entries and is therefore close
    to one up to the requested epsilon and floating-point roundoff.
    """

    items = list(factors)
    if not items:
        raise ValueError("at least one layer factor is required")
    result = []
    for qi, ki in items:
        row = []
        for qj, kj in items:
            row.append(kernel_cosine(qi, ki, qj, kj, eps=eps))
        result.append(torch.stack(row, dim=-1))
    return torch.stack(result, dim=-1).movedim(-1, -2)
