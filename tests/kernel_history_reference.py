"""Independent K2 oracle: dtype-preserving scalar equations and source loops.

No production module import/call, no forced float32, no SDPA. Inputs and
parameter tensors supplied as float64 remain float64, including fusion.
``explicit_tokens=True`` deliberately forms QK^T ONLY in this test oracle.
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F


class ReferenceResult(NamedTuple):
    output: Tensor
    alpha: Tensor
    candidates: Tensor  # [B,M,S+1,d]
    token_weights: tuple[Tensor, ...]  # explicit oracle only, [B,M,M]


def rms_reference(x: Tensor, scale: Tensor) -> Tensor:
    return x / (x.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt() * scale


def phi_reference(x: Tensor) -> Tensor:
    return (F.elu(x) + 1).clamp_min(1e-6)


def writer_reference(t: Tensor, parameters: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    key = phi_reference(rms_reference(t, parameters['key_norm.weight']) @ parameters['to_k.weight'].T)
    memory = key.transpose(-1, -2) @ t
    mass = key.sum(dim=1)
    return memory, mass, key


def fuse_reference(u: Tensor, reads: list[Tensor], parameters: dict[str, Tensor]) -> ReferenceResult:
    if not reads:
        return ReferenceResult(u, u.new_ones((*u.shape[:2], 1)), u.unsqueeze(2), ())
    # Score each candidate independently, then normalize ALL sources once.
    raw = torch.stack([u, *reads], dim=2)
    scores = torch.stack([
        (rms_reference(candidate, parameters['depth_norm.weight']) * parameters['w']).sum(-1)
        for candidate in [u, *reads]
    ], dim=2)
    exp_scores = (scores - scores.amax(dim=2, keepdim=True)).exp()
    alpha = exp_scores / exp_scores.sum(dim=2, keepdim=True)
    fused = sum(alpha[..., s, None] * raw[:, :, s] for s in range(len(reads) + 1))
    result = u + parameters['gamma'] * (fused - u)
    return ReferenceResult(result, alpha, raw, ())


def history_reference(u: Tensor, history: list[Tensor], writer_parameters: list[dict[str, Tensor]],
                      reader_parameters: dict[str, Tensor], *, explicit_tokens: bool = False) -> ReferenceResult:
    if not history:
        return fuse_reference(u, [], reader_parameters)
    p = reader_parameters
    query = phi_reference(rms_reference(u, p['query_norm.weight']) @ p['to_q.weight'].T)
    reads, token_weights = [], []
    for t, writer in zip(history, writer_parameters, strict=True):
        memory, mass, key = writer_reference(t, writer)
        if explicit_tokens:
            kernel = query @ key.transpose(-1, -2)
            weights = kernel / (kernel.sum(-1, keepdim=True) + 1e-6)
            reads.append(weights @ t)
            token_weights.append(weights)
        else:
            # unsqueeze is essential: batched vector mass is [B,r,1].
            reads.append((query @ memory) / (query @ mass.unsqueeze(-1) + 1e-6))
    result = fuse_reference(u, reads, p)
    return result._replace(token_weights=tuple(token_weights))
