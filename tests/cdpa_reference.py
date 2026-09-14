"""Explicit single-fusion CDPA equations, written before the SDPA module.

This oracle imports only torch and consumes tensors/parameter dictionaries. It
does not import CDLNO, call a module's forward/helpers, or use SDPA. Attention
matrices are returned only here, for small mathematical tests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def _ln(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(variance + eps) * weight + bias


def cdpa_reference(
    z: Tensor,
    history: Sequence[Tensor],
    parameters: Mapping[str, Tensor],
    heads: int,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
    """Return (fused, source weights, token attention matrices, raw candidates).

    Cross equations execute in the supplied tensor dtype with autocast off.
    Depth equations always execute in FP32, including when Cross is FP64.
    The only per-source loop in this reference is explicitly over histories.
    """
    b, m, d = z.shape
    if not history:
        return z, torch.ones(b, m, 1, device=z.device, dtype=torch.float32), [], z.unsqueeze(2)
    p = parameters
    with torch.autocast(device_type=z.device.type, enabled=False):
        query = _ln(z, p['ln_q.weight'], p['ln_q.bias'], eps) @ p['to_q.weight'].T
        query = query.reshape(b, m, heads, d // heads).transpose(1, 2)
        candidates, token_weights = [z], []
        for stored_source in history:
            source = stored_source.to(z.dtype)
            normed = _ln(source, p['ln_kv.weight'], p['ln_kv.bias'], eps)
            key = (normed @ p['to_k.weight'].T).reshape(b, m, heads, d // heads).transpose(1, 2)
            value = (normed @ p['to_v.weight'].T).reshape(b, m, heads, d // heads).transpose(1, 2)
            scores = query @ key.transpose(-1, -2) / math.sqrt(d // heads)
            # Stable softmax over this source's M historical tokens only.
            token_exp = (scores - scores.amax(-1, keepdim=True)).exp()
            attention = token_exp / token_exp.sum(-1, keepdim=True)
            token_weights.append(attention)
            aligned = (attention @ value).transpose(1, 2).reshape(b, m, d)
            candidates.append(aligned @ p['to_out.weight'].T + p['to_out.bias'])

        raw = torch.stack(candidates, dim=2).float()  # [B,M,S+1,d]
        depth_keys = raw / torch.sqrt(raw.square().mean(-1, keepdim=True) + eps)
        depth_keys = depth_keys * p['depth_norm.weight'].float()
        logits = (depth_keys * p['w'].float()).sum(-1)
        source_exp = (logits - logits.amax(-1, keepdim=True)).exp()
        alpha = source_exp / source_exp.sum(-1, keepdim=True)
        fused = (alpha.unsqueeze(-1) * raw).sum(dim=2)
    return fused.to(z.dtype), alpha, token_weights, raw
