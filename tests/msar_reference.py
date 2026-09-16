"""Independent M2 oracle: raw parameter tensors, explicit per-head softmax.

No imports of production MSAR or shared modules, no SDPA, no module forward.
Operations preserve input dtype, notably float64 for numerical gradient checks.
"""
import math

import torch


def linear(x, state, prefix):
    out = x @ state[prefix + '.weight'].T
    bias = state.get(prefix + '.bias')
    return out if bias is None else out + bias


def rms(x, scale=None):
    out = x / torch.sqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + 1e-6)
    return out if scale is None else out * scale


def explicit_heads(q, k, v, state, heads, valid_mask=None):
    batch, m, dim = q.shape
    width = dim // heads
    outputs, matrices = [], []
    for b in range(batch):
        head_outputs, head_matrices = [], []
        for h in range(heads):
            sl = slice(h * width, (h + 1) * width)
            qh = rms(q[b, :, sl], state['q_norm.weight'])
            kh = rms(k[b, :, sl], state['k_norm.weight'])
            scores = (qh @ kh.T) / math.sqrt(width)
            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask[b][None], -torch.inf)
            # Explicit stable row softmax, independent of the production call.
            unnormalized = (scores - scores.max(-1, keepdim=True).values).exp()
            a = unnormalized / unnormalized.sum(-1, keepdim=True)
            head_matrices.append(a)
            head_outputs.append(a @ v[b, :, sl])
        outputs.append(torch.cat(head_outputs, -1))
        matrices.append(torch.stack(head_matrices))
    return linear(torch.stack(outputs), state, 'to_out'), torch.stack(matrices)


def down(source, state, heads, valid_mask=None):
    work = source if valid_mask is None else torch.where(valid_mask[..., None], source, 0)
    q = state['latent_queries'][None].expand(source.shape[0], -1, -1)
    return explicit_heads(q, linear(work, state, 'to_k'), linear(work, state, 'to_v'),
                          state, heads, valid_mask)


def up(encoder, decoder, state, heads):
    return explicit_heads(linear(encoder, state, 'to_q'), linear(decoder, state, 'to_k'),
                          linear(decoder, state, 'to_v'), state, heads)[0]


def ffn(x, state, prefix):
    z = linear(x, state, prefix + '.fc1')
    gelu = z * 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return linear(gelu, state, prefix + '.fc2')


def latent(x, state, heads):
    x1 = x + ffn(rms(x, state['norm1.weight']), state, 'ffn1')
    normed = rms(x1, state['norm_sa.weight'])
    sa_state = {k.removeprefix('sa.attn.'): v for k, v in state.items() if k.startswith('sa.attn.')}
    x2 = x1 + up(normed, normed, sa_state, heads)
    return x2 + ffn(rms(x2, state['norm2.weight']), state, 'ffn2')


def fusion(encoder, up_value, w):
    se = (rms(encoder) * w).sum(-1)
    su = (rms(up_value) * w).sum(-1)
    # Two-source softmax via a logistic difference, not the production stack.
    alpha_e = torch.sigmoid(se - su)
    return 2 * (alpha_e[..., None] * encoder + (1 - alpha_e[..., None]) * up_value)


def coverage(attention, kappa=0.2, eps=1e-6, valid_mask=None, source_measure=None):
    """Source SUM / batch MEAN. Multi-layer averaging is explicit in tests."""
    values = []
    for b in range(attention.shape[0]):
        valid = (torch.ones(attention.shape[-1], dtype=torch.bool, device=attention.device)
                 if valid_mask is None else valid_mask[b])
        a = attention[b, :, :, valid]
        p = a.sum((0, 1)) / (a.shape[0] * a.shape[1])
        mass = torch.ones_like(p) if source_measure is None else source_measure[b, valid]
        mu = mass / mass.sum()
        terms = torch.clamp(kappa * mu - p, min=0).square() / (mu + eps)
        values.append(terms.sum())
    return torch.stack(values).mean()
