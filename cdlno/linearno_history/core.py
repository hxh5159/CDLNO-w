"""Research core: LinearNO v3 Eq.8/9 plus local raw history and independent A/K.

Operators are owned by research wrappers; no benchmark production dispatch.
ModuleList registration keeps the existing blocks.<index> state_dict layout.
The explicit observer is per-call, off by default, and intended for tests; the
core never stores traces, tensors, observers or contexts on itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from cdlno.linearno.attention import LinearNOAttention
from .context import RawHistoryContext


@dataclass(frozen=True)
class AttentionFactors:
    Z: torch.Tensor
    base_q_logits: torch.Tensor
    base_k_logits: torch.Tensor
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    C_raw: torch.Tensor
    QC_raw: torch.Tensor | None
    combined_k_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class BlockTrace:
    index: int
    history: tuple[torch.Tensor, ...]
    factors: AttentionFactors
    output: torch.Tensor
    C_tilde: torch.Tensor | None = None


def attention_factors(attention: LinearNOAttention, x: torch.Tensor, *, reconstruct=True,
                      history_k=None, index=0, history=()) -> AttentionFactors:
    """Execute the six existing variants; preserve pre-temperature logits.

    Exact same layout, temperature placement, einsum order and dtype as the
    pure module. No recomputation for diagnostics, no host copies or syncs.
    """
    if not isinstance(attention, LinearNOAttention):
        raise TypeError('research core requires a pure LinearNO attention module')
    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        raise ValueError('LinearNO attention requires a tensor [B,N,dim]')
    B, N, channels = x.shape
    if B < 1 or N < 1 or channels != attention.dim:
        raise ValueError(f'LinearNO attention expected B,N > 0 and dim={attention.dim}')
    if not x.is_floating_point():
        raise TypeError('LinearNO attention input must be floating point')
    if attention.variant in ('conv', 'conv_temp'):
        if N != attention.H * attention.W:
            raise ValueError('LinearNO conv requires N = H * W')
        grid = x.transpose(1, 2).reshape(B, channels, attention.H, attention.W)
        projected = attention.in_project_x(grid)
        Z = projected.reshape(B, attention.heads, attention.dim_head, N).transpose(-1, -2)
    else:
        projected = attention.in_project_x(x)
        Z = projected.reshape(B, N, attention.heads, attention.dim_head).transpose(1, 2)
        if attention.variant == 'airfrans':
            Z = Z.contiguous()
    base_q, base_k, V = attention.to_q(Z), attention.to_k(Z), attention.to_v(Z)
    combined_k = (base_k if history_k is None else
                  history_k(index, Z, base_k, attention.to_k.weight, history))
    q_logits, k_logits = base_q, combined_k
    if attention.variant in ('temp', 'conv_temp'):
        q_logits = q_logits / attention.temperature_q.clamp(0.01, 1.)
        k_logits = k_logits / attention.temperature_k.clamp(0.01, 1.)
    elif attention.variant == 'shapenet':
        q_logits = q_logits / attention.tempreature_q.clamp(0.1, 2.)
        k_logits = k_logits / attention.tempreature_k.clamp(0.1, 2.)
    Q, K = q_logits.softmax(dim=-1), k_logits.softmax(dim=-2)
    C_raw = torch.einsum('bhnm,bhnd->bhmd', K, V)
    QC_raw = torch.einsum('bhnm,bhmd->bhnd', Q, C_raw) if reconstruct else None
    return AttentionFactors(Z, base_q, base_k, Q, K, V, C_raw, QC_raw, combined_k)


class LinearNOHistoryCore(nn.ModuleList):
    """Own already-initialized pure blocks; do not reinitialize or add keys.

    Research wrappers register this as `blocks` after the pure constructor.
    No-op by default; research wrappers pass enabled operators per call.
    This module does not resolve feature flags or register a production model.
    """
    def __init__(self, blocks):
        blocks = tuple(blocks)
        if len(blocks) not in (4, 5, 6, 7, 8):
            raise ValueError('R2 research core requires 4..8 complete blocks')
        expected = None
        seen = set()
        for index, block in enumerate(blocks):
            if not isinstance(getattr(block, 'Attn', None), LinearNOAttention):
                raise TypeError('research core requires complete pure LinearNO blocks')
            attn = block.Attn
            shape = (attn.dim, attn.heads, attn.dim_head, attn.rank, attn.variant)
            if expected is not None and shape != expected:
                raise ValueError('all research layers must have the same width/heads/rank/variant')
            expected = shape
            if bool(block.last_layer) != (index == len(blocks) - 1):
                raise ValueError('only the last complete block must own the final head')
            for parameter in block.parameters():
                if id(parameter) in seen:
                    raise ValueError('blocks must not share parameter objects')
                seen.add(id(parameter))
        super().__init__(blocks)

    def forward(self, features: torch.Tensor, *, observe: Callable[[BlockTrace], None] | None = None,
                latent_attnres=None, history_k=None):
        if history_k is not None:
            from .history_k import HistoryConditionedK
            if not isinstance(history_k, HistoryConditionedK):
                raise TypeError('only the implemented K operator is accepted')
            if (history_k.n_layers, history_k.heads, history_k.d_h) != (
                    len(self), self[0].Attn.heads, self[0].Attn.dim_head):
                raise ValueError('K depth/heads/head dimension must match the backbone')
        if latent_attnres is not None:
            from .attnres import LatentSummaryAttnRes
            if not isinstance(latent_attnres, LatentSummaryAttnRes):
                raise TypeError('only the implemented A operator is accepted')
            if latent_attnres.n_layers != len(self) or latent_attnres.d_h != self[0].Attn.dim_head:
                raise ValueError('A depth/head dimension must match the backbone')
        context = RawHistoryContext()
        for index, block in enumerate(self):
            history = context.before_block(index)
            factors = attention_factors(block.Attn, block.ln_1(features),
                                        reconstruct=latent_attnres is None or observe is not None,
                                        history_k=history_k, index=index, history=history)
            summary = factors.C_raw
            if latent_attnres is not None:
                summary = latent_attnres(index, summary, history)
                readout = torch.einsum('bhnm,bhmd->bhnd', factors.Q, summary)
            else:
                readout = factors.QC_raw
            B, _, N, _ = readout.shape
            merged = readout.transpose(1, 2).reshape(B, N, block.Attn.heads * block.Attn.dim_head)
            update = block.Attn.to_out(merged)
            # Car release uses x + update; Standard/Air use update + x.
            if block.Attn.variant == 'shapenet':
                features = features + update
                features = features + block.mlp(block.ln_2(features))
            else:
                features = update + features
                features = block.mlp(block.ln_2(features)) + features
            if block.last_layer:
                features = block.mlp2(block.ln_3(features))
            if observe is not None:
                observe(BlockTrace(index, history, factors, features, summary))
            # Complete residuals, FFN and final head BEFORE admitting current raw.
            context = context.after_block(index, factors.C_raw)
        return features
