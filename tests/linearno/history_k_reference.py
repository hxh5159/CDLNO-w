"""R4 explicit equation oracle, independent of production code and helpers."""
import torch


def ln0(x):
    d = x.shape[-1]
    mean = sum(x[..., j:j+1] for j in range(d)) / d
    variance = sum((x[..., j:j+1] - mean)**2 for j in range(d)) / d
    return (x - mean) / torch.sqrt(variance + 1e-6)


def softmax(x):
    exp = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return exp / exp.sum(dim=-1, keepdim=True)


def reference(index, Z, base_logits, base_weight, history, state):
    if index == 0:
        return base_logits, {}
    B, heads, N, d = Z.shape
    E = ln0(base_weight) @ state['uq.weight'].T
    samples_G, samples_A, samples_delta = [], [], []
    for b in range(B):
        heads_G, heads_A, heads_delta = [], [], []
        for head in range(heads):
            # A single bank, including all tokens from all preceding sources.
            bank = torch.cat([h[b, head] for h in history], dim=0)
            normalized = ln0(bank)
            keys = normalized @ state['uk.weight'].T
            values = normalized @ state['uv.weight'].T
            weights = softmax(E @ keys.T / d**.5)
            G = weights @ values
            delta = torch.stack([torch.stack([sum(Z[b,head,n,j]*G[m,j] for j in range(d))
                                  for m in range(E.shape[0])]) for n in range(N)])
            heads_G.append(G); heads_A.append(weights); heads_delta.append(delta)
        samples_G.append(torch.stack(heads_G)); samples_A.append(torch.stack(heads_A))
        samples_delta.append(torch.stack(heads_delta))
    uncentered = torch.stack(samples_delta)
    mean_N = sum(uncentered[:,:,n:n+1,:] for n in range(N)) / N
    delta = uncentered - mean_N
    eta = torch.tanh(state[f'raw_gates.{index}'])
    combined = base_logits + eta*delta
    return combined, dict(E=E, A=torch.stack(samples_A), G=torch.stack(samples_G),
                          uncentered=uncentered, delta=delta, eta=eta)
