"""Explicit no-grad observations for MSAR; never part of a prediction/loss graph.

When training coverage already provides A, reuse it. Otherwise recompute only
diagnostic QK weights on demand, leaving the actual Down SDPA path unchanged.
No diagnostic tensor is cached on a module, and no CPU statistics are collected.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .modules import LearnedQueryDown, PairwiseAttnResFusion, NORM_EPS, coverage_diagnostics


@torch.no_grad()
def observe_down(module: LearnedQueryDown, source: Tensor, *,
                 valid_mask: Tensor | None = None, attention: Tensor | None = None) -> dict[str, Tensor]:
    """Called after the real Down validated the source/mask. Results are [B]."""
    if attention is None:
        # Match the active projection precision, but never build a backward graph.
        work = source if valid_mask is None else source.masked_fill(~valid_mask[..., None], 0)
        k = module.to_k(work).reshape(source.shape[0], source.shape[1], module.heads, module.head_dim)
        k = module.k_norm(k.transpose(1, 2))
        q = module.latent_queries.reshape(module.num_latents, module.heads, module.head_dim)
        q = module.q_norm(q.permute(1, 0, 2).unsqueeze(0)).to(k.dtype)
        dtype = torch.float64 if k.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=source.device.type, enabled=False):
            score = (q.to(dtype) @ k.to(dtype).transpose(-1, -2)) * module.head_dim ** -0.5
            if valid_mask is not None:
                score = score.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
            attention = score.softmax(-1)
    return coverage_diagnostics(attention, valid_mask=valid_mask)


@torch.no_grad()
def observe_fusion(module: PairwiseAttnResFusion, encoder: Tensor, up: Tensor) -> dict[str, Tensor]:
    """Return per-sample token means: alpha[B,2] and entropy[B], natural log."""
    dtype = torch.float64 if encoder.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=encoder.device.type, enabled=False):
        raw = torch.stack((encoder, up), dim=-2).to(dtype)
        keys = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + NORM_EPS)
        alpha = (keys * module.w.to(dtype)).sum(-1).softmax(-1)
        entropy = -(alpha * alpha.clamp_min(torch.finfo(dtype).tiny).log()).sum(-1)
        return dict(alpha_mean=alpha.mean(1), entropy_mean=entropy.mean(1))
