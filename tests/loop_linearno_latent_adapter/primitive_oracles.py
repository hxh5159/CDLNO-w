"""Independent LAA2 mathematics. No production module/helper/forward imports.

Adapter implementations deliberately use three contraction orders. Latent
normalization and exact GELU are expanded arithmetically, with head merging
by concatenation instead of the production transpose/reshape path.
"""
import math
import torch


def adapter_indexed(x, a, b, alpha):
    batch, heads, points, channels = x.shape
    rank, slots = a.shape[0], b.shape[0]
    values = []
    for i in range(batch):
        for j in range(heads):
            for n in range(points):
                for m in range(slots):
                    values.append(sum(sum(x[i, j, n, d] * a[r, d] for d in range(channels))
                                      * b[m, r] for r in range(rank)) * (alpha / rank))
    return torch.stack(values).reshape(batch, heads, points, slots)


def adapter_einsum(x, a, b, alpha):
    return torch.einsum('bhnd,rd,mr->bhnm', x, a, b) * (alpha / a.shape[0])


def adapter_merged_matrix(x, a, b, alpha):
    weight = torch.matmul(b, a)
    return torch.matmul(x, weight.T) * (alpha / a.shape[0])


def latent_expanded(context, parameters):
    tokens = torch.cat([context[:, head] for head in range(context.shape[1])], dim=-1)
    centered = tokens - tokens.mean(-1, keepdim=True)
    variance = (centered * centered).mean(-1, keepdim=True)
    normalized = centered / torch.sqrt(variance + 1e-5)
    affine = normalized * parameters['norm.weight'] + parameters['norm.bias']
    first = affine @ parameters['linear1.weight'].T + parameters['linear1.bias']
    activated = 0.5 * first * (1 + torch.erf(first / math.sqrt(2)))
    result = tokens + activated @ parameters['linear2.weight'].T + parameters['linear2.bias']
    return torch.stack(result.split(context.shape[-1], dim=-1), dim=1)
