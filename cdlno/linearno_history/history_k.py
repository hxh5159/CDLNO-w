"""history_conditioned_k_v1: independent K-only compression conditioning.

Queries are the current base to_k.weight rows. No A imports, learned slot
queries, source softmax, dropout, projection cache or persistent history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from cdlno.linearno.attention import initialize_release_weights


def ln0(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free last-axis normalization, population variance, eps=1e-6."""
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
        x.var(dim=-1, keepdim=True, unbiased=False) + 1e-6)


@dataclass(frozen=True)
class HistoryKTrace:
    E: torch.Tensor
    attention: torch.Tensor
    G: torch.Tensor
    uncentered: torch.Tensor
    delta: torch.Tensor
    eta: torch.Tensor


class HistoryConditionedK(nn.Module):
    """Network-shared Uq/Uk/Uv and one zero raw gate per receiver/head.

    Returns raw combined K logits; temperature and softmax_N belong to the
    original task attention. All histories and diagnostics are forward-local.
    """
    def __init__(self, n_layers: int, heads: int, d_h: int, *, feature_seed: int):
        super().__init__()
        if type(n_layers) is not int or n_layers not in range(4, 9):
            raise ValueError('history K requires 4..8 complete LinearNO layers')
        for name, value in (('heads', heads), ('d_h', d_h)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(feature_seed) is not int or not 0 <= feature_seed < 2**63:
            raise ValueError('feature_seed must be an integer in [0,2**63)')
        self.n_layers, self.heads, self.d_h = n_layers, heads, d_h
        self.feature_seed = feature_seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(feature_seed)
            self.uq = nn.Linear(d_h, d_h, bias=False)
            self.uk = nn.Linear(d_h, d_h, bias=False)
            self.uv = nn.Linear(d_h, d_h, bias=False)
            self.raw_gates = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(1, heads, 1, 1))
                                              for i in range(1, n_layers)})
            self.apply(initialize_release_weights)
            with torch.no_grad():
                for gate in self.raw_gates.values():
                    gate.zero_()

    def _validate(self, index, Z, base_logits, base_weight, history):
        if type(index) is not int or not 0 <= index < self.n_layers:
            raise ValueError('receiver index outside configured depth')
        if type(history) is not tuple or len(history) != index:
            raise ValueError('receiver must see exactly its preceding raw summaries')
        if not isinstance(Z, torch.Tensor) or Z.ndim != 4 or not Z.is_floating_point():
            raise ValueError('Z must be floating [B,H,N,d_h]')
        B, heads, N, d = Z.shape
        if min(Z.shape) < 1 or (heads, d) != (self.heads, self.d_h):
            raise ValueError('Z must have nonempty axes and configured H/d_h')
        if (not isinstance(base_weight, torch.Tensor) or base_weight.ndim != 2 or
                base_weight.shape[0] < 1 or base_weight.shape[1] != d or
                not base_weight.is_floating_point() or base_weight.device != Z.device):
            raise ValueError('base to_k.weight must be floating [M,d_h] on Z device')
        M = base_weight.shape[0]
        if (not isinstance(base_logits, torch.Tensor) or base_logits.shape != (B,heads,N,M) or
                base_logits.dtype != Z.dtype or base_logits.device != Z.device):
            raise ValueError('base K logits must be [B,H,N,M] on Z dtype/device')
        for h in history:
            if (not isinstance(h, torch.Tensor) or h.ndim != 4 or h.shape[:2] != (B, heads) or
                    h.shape[-1] != d or h.shape[2] < 1 or h.device != Z.device or h.dtype != Z.dtype):
                raise ValueError('old raw summaries must match B/H/d_h/device/dtype')

    def forward(self, index, Z, base_logits, base_weight, history, *,
                observe: Callable[[HistoryKTrace], None] | None = None):
        self._validate(index, Z, base_logits, base_weight, history)
        if index == 0:
            return base_logits
        bank = torch.cat(history, dim=-2)
        E = self.uq(ln0(base_weight))
        normalized_bank = ln0(bank)
        keys, values = self.uk(normalized_bank), self.uv(normalized_bank)
        attention = ((E @ keys.transpose(-1, -2)) / self.d_h**.5).softmax(dim=-1)
        G = attention @ values
        uncentered = Z @ G.transpose(-1, -2)
        delta = uncentered - uncentered.mean(dim=-2, keepdim=True)
        eta = self.raw_gates[str(index)].tanh()
        combined = base_logits + eta * delta
        if observe is not None:
            observe(HistoryKTrace(E, attention, G, uncentered, delta, eta))
        return combined
