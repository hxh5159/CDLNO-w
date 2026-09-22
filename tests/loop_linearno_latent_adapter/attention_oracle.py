"""LAA3 independent equations: no production imports or module forward calls.

Convolution is unfolded patch multiplication; batch/head matrix products and
explicit exp/sum normalization implement the attention. Only the independently
audited LAA2 arithmetic oracle is reused for tokenwise LN/erf-GELU.
"""
import math

import torch
from torch.nn import functional as F

from .primitive_oracles import latent_expanded


def normalize(x, axis):
    ex = (x - x.amax(axis, keepdim=True)).exp()
    return ex / ex.sum(axis, keepdim=True)


def attention_reference(x, weights, *, variant, heads, round_index,
                        latent=False, adapter=False, alpha=4., grid=None,
                        wrong_temperature_order=False):
    """Return output and local intermediate tensors, always with dropout off."""
    batch, points, _ = x.shape
    w = weights['in_project_x.weight']
    if variant in ('conv', 'conv_temp'):
        image = x.transpose(1, 2).reshape(batch, x.shape[-1], *grid)
        patches = F.unfold(image, kernel_size=w.shape[-1], padding=w.shape[-1] // 2)
        projected = patches.transpose(1, 2) @ w.flatten(1).T
    else:
        projected = x @ w.T
    projected = projected + weights['in_project_x.bias']
    dh = projected.shape[-1] // heads
    q_rows, k_rows, v_rows, feature_rows = [], [], [], []
    q_log_rows, k_log_rows = [], []
    for b in range(batch):
        qs, ks, vs, fs, qls, kls = [], [], [], [], [], []
        for h in range(heads):
            f = projected[b, :, h*dh:(h+1)*dh]
            q = f @ weights['to_q.weight'].T
            k = f @ weights['to_k.weight'].T
            v = f @ weights['to_v.weight'].T
            delta = {}
            for side in ('q', 'k'):
                if adapter and round_index == 1:
                    a, z = weights['adapter.A_'+side], weights['adapter.B_'+side]
                    # Deliberately premerge BA, unlike the production two-stage projection.
                    delta[side] = f @ (z @ a).T * (alpha / a.shape[0])
                else:
                    delta[side] = torch.zeros_like(q)
            tq = tk = 1.
            if variant in ('temp', 'conv_temp', 'shapenet'):
                prefix = 'tempreature' if variant == 'shapenet' else 'temperature'
                lower, upper = (.1, 2.) if variant == 'shapenet' else (.01, 1.)
                tq = weights[prefix+'_q'][0, h, 0, 0].clamp(lower, upper)
                tk = weights[prefix+'_k'][0, h, 0, 0].clamp(lower, upper)
            q = q/tq+delta['q'] if wrong_temperature_order else (q+delta['q'])/tq
            k = k/tk+delta['k'] if wrong_temperature_order else (k+delta['k'])/tk
            qs.append(normalize(q, 1)); ks.append(normalize(k, 0)); vs.append(v)
            fs.append(f); qls.append(q); kls.append(k)
        for destination, source in ((q_rows, qs), (k_rows, ks), (v_rows, vs),
                                    (feature_rows, fs), (q_log_rows, qls), (k_log_rows, kls)):
            destination.append(torch.stack(source))
    q, k, v = (torch.stack(rows) for rows in (q_rows, k_rows, v_rows))
    raw = torch.stack([torch.stack([k[b,h].T @ v[b,h] for h in range(heads)]) for b in range(batch)])
    context = latent_expanded(raw, {name.removeprefix('latent_processor.'): value
                                  for name, value in weights.items() if name.startswith('latent_processor.')}) if latent else raw
    readout = torch.stack([torch.stack([q[b,h] @ context[b,h] for h in range(heads)]) for b in range(batch)])
    merged = torch.cat([readout[:, h] for h in range(heads)], dim=-1)
    output = merged @ weights['to_out.0.weight'].T + weights['to_out.0.bias']
    if variant in ('conv', 'conv_temp', 'shapenet'):
        output = .5 * output * (1 + torch.erf(output / math.sqrt(2)))
        output = output @ weights['to_out.2.weight'].T + weights['to_out.2.bias']
    return output, dict(features=torch.stack(feature_rows), queries=q, keys=k, values=v,
                        q_logits=torch.stack(q_log_rows), k_logits=torch.stack(k_log_rows),
                        raw_context=raw, context=context, readout=readout)
