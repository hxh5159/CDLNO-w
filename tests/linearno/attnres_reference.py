"""Independent R3 oracle: explicit per-sample/head/source/slot equations.

No production forward, norm, projection or mask helper is imported here.
Linear weights use PyTorch's stored [out,in] convention.
"""
import torch


def probabilities(values):
    shifted = values - values.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True)


def reference(current, history, state, receiver, drop_mask=None):
    if not history:
        return current, dict(H=torch.zeros_like(current), attention=(), alpha=None)
    B, heads, slots, d = current.shape
    prefix = f'receivers.{receiver}.'
    q_weight, o_weight = state[prefix+'to_q.weight'], state[prefix+'to_o.weight']
    scale, w, gamma = state[prefix+'norm.weight'], state[prefix+'w'], state[prefix+'gamma']
    keys = [h @ state['to_k.weight'].T for h in history]
    values = [h @ state['to_v.weight'].T for h in history]
    per_sample_H, per_sample_alpha = [], []
    attention = [[] for _ in history]
    for b in range(B):
        per_head_H, per_head_alpha = [], []
        per_head_attention = [[] for _ in history]
        for head in range(heads):
            aligned = []
            for source in range(len(history)):
                scores = (current[b,head] @ q_weight.T) @ keys[source][b,head].T / d**.5
                a = probabilities(scores)
                per_head_attention[source].append(a)
                aligned.append(a @ values[source][b,head] @ o_weight.T)
            slot_H, slot_alpha = [], []
            for slot in range(slots):
                scores = []
                for source, r in enumerate(aligned):
                    v = r[slot]
                    rms = torch.sqrt(sum(v[j]**2 for j in range(d))/d + 1e-6)
                    score = sum(w[j]*scale[j]*v[j]/rms for j in range(d))
                    if drop_mask is not None and bool(drop_mask[b,source]):
                        score = score * 0 + float('-inf')
                    scores.append(score)
                scores.append(current.new_zeros(()))
                alpha = probabilities(torch.stack(scores))
                slot_alpha.append(alpha)
                slot_H.append(sum(alpha[s]*aligned[s][slot] for s in range(len(history))))
            per_head_H.append(torch.stack(slot_H))
            per_head_alpha.append(torch.stack(slot_alpha))
        per_sample_H.append(torch.stack(per_head_H))
        per_sample_alpha.append(torch.stack(per_head_alpha))
        for source in range(len(history)):
            attention[source].append(torch.stack(per_head_attention[source]))
    H = torch.stack(per_sample_H)
    return current + gamma*H, dict(H=H, alpha=torch.stack(per_sample_alpha),
                                  attention=tuple(torch.stack(a) for a in attention))
