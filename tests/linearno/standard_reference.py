"""Independent complete Standard equations, no production Module.forward calls.

Use explicit LN moments, erf GELU, and the L2 per-head mathematical oracle.
Only dropout-free eval is modeled; stochastic training uses fixed-source parity.
"""
import math
import torch

from linearno.attention_reference import reference as attention_reference


def gelu(x):
    return .5 * x * (1 + torch.erf(x / math.sqrt(2)))


def reference(x, fx, time, state, config, positions=None):
    def linear(z, prefix):
        return z @ state[prefix + '.weight'].T + state[prefix + '.bias']

    def mlp(z, prefix):
        return linear(gelu(linear(z, prefix + '.linear_pre.0')), prefix + '.linear_post')

    def norm(z, prefix):
        center = z - z.mean(dim=-1, keepdim=True)
        return center / (center.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt() * state[prefix+'.weight'] + state[prefix+'.bias']

    if config['unified_pos']:
        x = positions.reshape(1, config['H'] * config['W'], config['ref'] ** 2).expand(x.shape[0], -1, -1)
    if fx is None:
        z = mlp(x, 'preprocess') + state['placeholder']
    else:
        z = mlp(torch.cat((x, fx), dim=-1), 'preprocess')
    if time is not None:
        half = config['n_hidden'] // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=time.device) / half)
        angles = time.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1).to(z.dtype)
        expanded = embedding.expand(-1, x.shape[1], -1)
        hidden = linear(expanded, 'time_fc.0')
        z = z + linear(hidden * hidden.sigmoid(), 'time_fc.2')
    blocks = []
    for i in range(config['n_layers']):
        prefix = f'blocks.{i}'
        atom = {k[len(prefix+'.Attn.'):]: v for k, v in state.items() if k.startswith(prefix+'.Attn.')}
        branch, _ = attention_reference(norm(z, prefix+'.ln_1'), atom, heads=config['n_head'],
                                        variant=config['args']['model'], H=config['H'], W=config['W'])
        z = z + branch
        z = z + mlp(norm(z, prefix+'.ln_2'), prefix+'.mlp')
        if i == config['n_layers'] - 1:
            z = linear(norm(z, prefix+'.ln_3'), prefix+'.mlp2')
        blocks.append(z)
    return z, blocks
