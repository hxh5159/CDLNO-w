"""Independent, differentiable LinearNO equation oracle (tests only).

No production imports/module.forward calls. Batch/head loops and explicit stable
exp/sum normalize the specified axes. Double inputs/weights remain double.
Structured input uses unfolded patches rather than the production Conv2d module.
Dense QK^T is available ONLY here for tiny mathematical tests, never production.
"""
import math
import torch
import torch.nn.functional as F


def normalize(logits, axis):
    exp = (logits - logits.amax(dim=axis, keepdim=True)).exp()
    return exp / exp.sum(dim=axis, keepdim=True)


def reference(x, state, *, heads, variant, H=None, W=None, dense=False):
    B, N, _ = x.shape
    weight = state['in_project_x.weight']
    if variant in ('conv', 'conv_temp'):
        channels = x.shape[-1]
        grid = x.transpose(1, 2).reshape(B, channels, H, W)
        kernel = weight.shape[-1]
        patches = F.unfold(grid, kernel_size=kernel, padding=kernel//2)
        projected = patches.transpose(1, 2) @ weight.flatten(1).T
    else:
        projected = x @ weight.T
    projected = projected + state['in_project_x.bias']
    dh = projected.shape[-1] // heads
    outputs, queries, keys, values, contexts = [], [], [], [], []
    for b in range(B):
        by_head, qs, ks, vs, cs = [], [], [], [], []
        for h in range(heads):
            features = projected[b, :, h*dh:(h+1)*dh]
            q_logits = features @ state['to_q.weight'].T
            k_logits = features @ state['to_k.weight'].T
            v = features @ state['to_v.weight'].T
            if variant in ('temp', 'conv_temp', 'shapenet'):
                prefix = 'tempreature' if variant == 'shapenet' else 'temperature'
                lo, hi = (0.1, 2.) if variant == 'shapenet' else (0.01, 1.)
                q_logits = q_logits / state[prefix+'_q'][0,h,0,0].clamp(lo, hi)
                k_logits = k_logits / state[prefix+'_k'][0,h,0,0].clamp(lo, hi)
            q, k = normalize(q_logits, 1), normalize(k_logits, 0)
            context = k.T @ v
            by_head.append((q @ k.T) @ v if dense else q @ context)
            qs.append(q);ks.append(k);vs.append(v);cs.append(context)
        outputs.append(torch.cat(by_head, dim=-1))
        queries.append(torch.stack(qs));keys.append(torch.stack(ks))
        values.append(torch.stack(vs));contexts.append(torch.stack(cs))
    result = torch.stack(outputs) @ state['to_out.0.weight'].T + state['to_out.0.bias']
    if variant in ('conv', 'conv_temp', 'shapenet'):
        result = 0.5 * result * (1. + torch.erf(result / math.sqrt(2.)))
        result = result @ state['to_out.2.weight'].T + state['to_out.2.bias']
    # Oracle is dropout-free; training dropout is compared against fixed source
    # with restored RNG in separate tests, not silently approximated here.
    return result, dict(Q=torch.stack(queries), K=torch.stack(keys), V=torch.stack(values), C=torch.stack(contexts))
