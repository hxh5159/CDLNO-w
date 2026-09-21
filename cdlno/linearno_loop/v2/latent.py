"""Tokenwise residual FFN for current-call LinearNO latent contexts."""

import torch
from torch import nn
from timm.layers import trunc_normal_


class LatentContextFFN(nn.Module):
    def __init__(self, hidden, inner_width, *, eps=1e-5):
        super().__init__()
        if type(hidden) is not int or hidden < 1 or type(inner_width) is not int or inner_width < 1:
            raise ValueError("latent FFN hidden and inner_width must be positive integers")
        if type(eps) is not float or eps != 1e-5:
            raise ValueError("latent FFN eps is frozen at 1e-5")
        self.hidden, self.inner_width = hidden, inner_width
        self.norm = nn.LayerNorm(hidden, eps=eps, elementwise_affine=True)
        self.linear1 = nn.Linear(hidden, inner_width, bias=True)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(inner_width, hidden, bias=True)

    def initialize_release_identity(self):
        nn.init.ones_(self.norm.weight); nn.init.zeros_(self.norm.bias)
        trunc_normal_(self.linear1.weight, std=0.02); nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight); nn.init.zeros_(self.linear2.bias)

    def forward(self, context):
        if not isinstance(context, torch.Tensor) or context.ndim != 4:
            raise ValueError("latent context must have shape [B,h,M,d_h]")
        B, heads, rank, head_dim = context.shape
        if min(B, heads, rank, head_dim) < 1 or heads * head_dim != self.hidden:
            raise ValueError(f"latent context last head product must equal hidden={self.hidden}")
        if not context.is_floating_point():
            raise TypeError("latent context must be floating point")
        tokens = context.transpose(1, 2).contiguous().reshape(B, rank, self.hidden)
        update = self.linear2(self.activation(self.linear1(self.norm(tokens))))
        result = tokens + update
        return result.reshape(B, rank, heads, head_dim).transpose(1, 2).contiguous()

