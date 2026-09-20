"""Independent LL2 equations. No imports from production loop modules.

Stack on [B,N,S,H], explicitly form key RMS, stabilize exponentials and
normalize ONLY S. Keep input precision, including float64 for gradcheck.
"""
import torch


def point_attnres(sources, query, scale):
    values = torch.stack(tuple(sources), dim=2)
    rms = torch.sqrt(torch.sum(values * values, dim=3, keepdim=True) / values.shape[3] + 1e-6)
    keys = (values / rms) * scale
    scores = torch.einsum('bnsh,h->bns', keys, query)
    exp = torch.exp(scores - scores.max(dim=2, keepdim=True).values)
    weights = exp / exp.sum(dim=2, keepdim=True)
    output = torch.einsum('bns,bnsh->bnh', weights, values)
    return output, weights
