"""M3 hand composition from tensor-only M2 oracle, never production forwards."""
import torch

import msar_reference as primitives


def subset(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def stack(x, state, prefix, depth, heads):
    for index in range(depth):
        x = primitives.latent(x, subset(state, f'{prefix}{index}.'), heads)
    return x


def core(e0, state, heads, encoder_depths, decoder_depths, *, valid_mask=None,
         source_measure=None, kappa=.2, eps=1e-6):
    encoded = {0: e0}
    coverage = []
    for level in (1, 2, 3, 4):
        mask = valid_mask if level == 1 else None
        s, a = primitives.down(encoded[level - 1], subset(state, f'downs.{level-1}.'),
                               heads[level-1], mask)
        coverage.append(primitives.coverage(a, kappa, eps, mask,
                                             source_measure if level == 1 else None))
        encoded[level] = stack(s, state, f'encoders.{level-1}.', encoder_depths[level-1], heads[level-1])
    decoded = {4: stack(encoded[4], state, 'decoders.3.', decoder_depths[3], heads[3])}
    for level in (3, 2, 1):
        u = primitives.up(encoded[level], decoded[level+1], subset(state, f'ups.{level-1}.'), heads[level-1])
        fused = primitives.fusion(encoded[level], u, state[f'fusions.{level-1}.w'])
        decoded[level] = stack(fused, state, f'decoders.{level-1}.', decoder_depths[level-1], heads[level-1])
    u0 = primitives.up(e0, decoded[1], subset(state, 'final_up.'), heads[0])
    # Original task head contract: LayerNorm, then pointwise linear. No E0 add.
    centered = u0 - u0.mean(-1, keepdim=True)
    normed = centered / (centered.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    normed = normed * state['output_norm.weight'] + state['output_norm.bias']
    return primitives.linear(normed, state, 'output'), torch.stack(coverage)
