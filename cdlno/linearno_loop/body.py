"""Non-owning adapter over existing pure LinearNO block submodules.

No block imports/construction/copy/init, no registration or tensor cache. The
future owner registers each physical block once; this plain Python view retains
only that exact block reference. Internal call forms are NOT residual modes.
"""
import math

import torch
from torch import nn


class LinearNOBlockBody:
    __slots__ = ('block',)

    def __init__(self, block):
        if not isinstance(block, nn.Module):
            raise TypeError('body requires an existing LinearNO block module')
        for name in ('ln_1', 'Attn', 'ln_2', 'mlp'):
            if not isinstance(getattr(block, name, None), nn.Module):
                raise ValueError(f'block is missing required submodule {name}')
        if type(getattr(block, 'last_layer', None)) is not bool:
            raise ValueError('block.last_layer must be bool')
        if block.last_layer:
            for name in ('ln_3', 'mlp2'):
                if not isinstance(getattr(block, name, None), nn.Module):
                    raise ValueError(f'final block is missing {name}')
        self.block = block

    def operator(self, x):
        """Raw operator branch, without point residual or head."""
        return self.block.Attn(self.block.ln_1(x))

    def mlp(self, x):
        """Raw MLP branch, without point residual or head."""
        return self.block.mlp(self.block.ln_2(x))

    def native(self, x):
        """Two original unscaled point residuals, always excluding the head."""
        x = self.operator(x) + x
        return self.mlp(x) + x

    def scaled(self, x, scale):
        """Scale each raw branch, never the identity path. No head call."""
        if type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('branch scale must be a finite positive number, not bool')
        x = x + scale * self.operator(x)
        return x + scale * self.mlp(x)

    def finalize(self, x):
        """Explicit terminal ln_3/head; the future topology reserves final suffix.

        LL2 has no topology and cannot assign suffix roles. Non-final blocks
        fail here; core/suffix ownership is checked by the future loop builder.
        """
        if not self.block.last_layer:
            raise ValueError('finalize requires last_layer=True on the last suffix block')
        return self.block.mlp2(self.block.ln_3(x))
