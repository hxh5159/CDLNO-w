"""Small mathematical oracles which never call the production V5 forward."""
from __future__ import annotations

import torch


def dense_expert_oracle(z, probabilities, expert_outputs, repeats):
    if z.ndim != 3 or probabilities.ndim != 3 or expert_outputs.ndim != 4:
        raise ValueError("oracle tensors must be [B,N,C], [B,N,E], [B,N,E,C]")
    if probabilities.shape != expert_outputs.shape[:-1]:
        raise ValueError("probability/expert shapes differ")
    if z.shape != expert_outputs.shape[:2] + expert_outputs.shape[-1:]:
        raise ValueError("state/expert shapes differ")
    mixed = torch.zeros_like(z)
    for expert in range(probabilities.shape[-1]):
        mixed = mixed + probabilities[:, :, expert:expert + 1] * expert_outputs[:, :, expert, :]
    return z + mixed / repeats


def linearno_operator_oracle(features, q_weight, k_weight, v_weight, out_weight,
                             q_temperature=None, k_temperature=None):
    """Bias-free single-head expansion of K^T V followed by Q readout."""
    q = features @ q_weight.T
    k = features @ k_weight.T
    v = features @ v_weight.T
    if q_temperature is not None:
        q = q / q_temperature
    if k_temperature is not None:
        k = k / k_temperature
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)
    context = torch.einsum("bnm,bnd->bmd", k, v)
    readout = torch.einsum("bnm,bmd->bnd", q, context)
    return readout @ out_weight.T
