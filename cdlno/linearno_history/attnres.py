"""LinearNO latent-summary AttnRes v1, an original project adaptation.

One token softmax per old raw summary, followed by history-only source softmax
with fixed zero null. Not CDPA or an implementation of Kimi's architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from cdlno.linearno.attention import initialize_release_weights


class SummaryRMSNorm(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_h))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


class AttnResReceiver(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.to_q = nn.Linear(d_h, d_h, bias=False)
        self.to_o = nn.Linear(d_h, d_h, bias=False)
        self.norm = SummaryRMSNorm(d_h)
        self.w = nn.Parameter(torch.zeros(d_h))
        self.gamma = nn.Parameter(torch.zeros(()))


@dataclass(frozen=True)
class AttnResTrace:
    attention: tuple[torch.Tensor, ...]
    aligned: tuple[torch.Tensor, ...]
    scores: torch.Tensor
    drop_mask: torch.Tensor
    alpha: torch.Tensor
    H: torch.Tensor


class LatentSummaryAttnRes(nn.Module):
    """Own A parameters once. Histories and optional diagnostics are per-call.

    Production defaults to p=.1; the explicit A1K0 ablation may set p=0.
    The private evaluator's mask argument is for deterministic equation tests;
    public forward always enforces train/eval and singleton rules. No derived
    projection cache yet.
    """
    def __init__(self, n_layers: int, d_h: int, *, feature_seed: int,
                 dropout_p: float = 0.1):
        super().__init__()
        if type(n_layers) is not int or n_layers not in range(4, 9):
            raise ValueError('AttnRes requires 4..8 complete LinearNO layers')
        if type(d_h) is not int or d_h < 1:
            raise ValueError('d_h must be a positive integer')
        if type(feature_seed) is not int or not 0 <= feature_seed < 2**63:
            raise ValueError('feature_seed must be an integer in [0,2**63)')
        if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)):
            raise ValueError('dropout_p must be a finite number in [0,1)')
        if not torch.isfinite(torch.tensor(float(dropout_p))) or not 0 <= float(dropout_p) < 1:
            raise ValueError('dropout_p must be a finite number in [0,1)')
        self.n_layers, self.d_h, self.feature_seed = n_layers, d_h, feature_seed
        self.dropout_p = float(dropout_p)
        # Only CPU parameter allocation occurs here; do not seed/touch CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(feature_seed)
            self.to_k = nn.Linear(d_h, d_h, bias=False)
            self.to_v = nn.Linear(d_h, d_h, bias=False)
            self.receivers = nn.ModuleDict({str(i): AttnResReceiver(d_h) for i in range(1, n_layers)})
            self.apply(initialize_release_weights)
            with torch.no_grad():
                for receiver in self.receivers.values():
                    receiver.norm.weight.fill_(1)
                    receiver.w.zero_()
                    receiver.gamma.zero_()

    def _validate(self, index, current, history):
        if type(index) is not int or not 0 <= index < self.n_layers:
            raise ValueError('receiver index outside configured depth')
        if type(history) is not tuple or len(history) != index:
            raise ValueError('receiver must see exactly its preceding raw summaries')
        if not isinstance(current, torch.Tensor) or current.ndim != 4 or not current.is_floating_point():
            raise ValueError('current raw must be floating [B,H,M,d_h]')
        if min(current.shape) < 1 or current.shape[-1] != self.d_h:
            raise ValueError('nonempty axes and configured d_h required')
        B, heads, _, _ = current.shape
        for h in history:
            if (not isinstance(h, torch.Tensor) or h.ndim != 4 or
                    h.shape[:2] != (B, heads) or h.shape[-1] != self.d_h or h.shape[2] < 1 or
                    h.dtype != current.dtype or h.device != current.device):
                raise ValueError('history must match current B/H/d_h/device/dtype')
            if h is current:
                raise ValueError('current raw cannot be a historical source')

    def forward(self, index, current, history, *, observe: Callable[[AttnResTrace], None] | None = None):
        self._validate(index, current, history)
        if index == 0:
            return current
        return self._evaluate(index, current, history, observe=observe)

    def _evaluate(self, index, current, history, *, observe=None, _drop_mask=None):
        """Internal test seam: explicit mask bypasses sampling, not the equations."""
        receiver = self.receivers[str(index)]
        queries = receiver.to_q(current)
        aligned, scores, attention = [], [], []
        for raw in history:
            keys, values = self.to_k(raw), self.to_v(raw)
            a = ((queries @ keys.transpose(-1, -2)) / self.d_h**.5).softmax(dim=-1)
            r = receiver.to_o(a @ values)
            aligned.append(r)
            scores.append((receiver.norm(r) * receiver.w).sum(dim=-1))
            if observe is not None:
                attention.append(a)
        # [B,H,M,S_real]. All Cross and content scoring precede dropout sampling.
        raw_scores = torch.stack(scores, dim=-1)
        B, _, _, sources = raw_scores.shape
        if _drop_mask is not None:
            if (_drop_mask.shape != (B, sources) or _drop_mask.dtype != torch.bool or
                    _drop_mask.device != current.device):
                raise ValueError('internal mask must be bool [B,S_real] on current device')
            mask = _drop_mask
        elif self.training and sources > 1 and self.dropout_p > 0.0:
            mask = torch.rand(B, sources, device=current.device) < self.dropout_p
        else:
            mask = torch.zeros(B, sources, dtype=torch.bool, device=current.device)
        masked = raw_scores.masked_fill(mask[:, None, None, :], float('-inf'))
        null = torch.zeros_like(masked[..., :1])
        alpha = torch.cat((masked, null), dim=-1).softmax(dim=-1)
        H = sum(alpha[..., i, None] * r for i, r in enumerate(aligned))
        result = current + receiver.gamma * H
        if observe is not None:
            observe(AttnResTrace(tuple(attention), tuple(aligned), raw_scores, mask, alpha, H))
        return result
