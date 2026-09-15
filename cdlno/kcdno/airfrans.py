"""Stable AirfRANS class, reusing the accepted x/pos/single-graph contract."""
import numpy as np
import torch
from torch import nn
from ..airfrans import AirfRANSModel as OriginalAdapter, adapter_architecture
from ..modules import _init_linear
from .matched import make_core


class AirfRANSModel(OriginalAdapter):
    def __init__(self, *, config):
        nn.Module.__init__(self)
        config.validate()
        if config.point_module != 'point_ffn':
            raise ValueError('AirfRANS requires point_ffn')
        self.config = config
        d = config.d
        self.preprocess = nn.Sequential(nn.Linear(71, 2*d), nn.GELU(), nn.Linear(2*d, d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        self.placeholder = nn.Parameter(torch.rand(d) / d)
        gx = torch.tensor(np.linspace(-2, 4, 8), dtype=torch.float32)
        gy = torch.tensor(np.linspace(-1.5, 1.5, 8), dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), -1).reshape(64, 2)
        self.register_buffer('reference', grid, persistent=False)
        self.core = make_core(config)
        self.output_norm = nn.LayerNorm(d, eps=config.norm_eps)
        self.output = nn.Linear(d, 4)
        _init_linear(self.output)

    def adapter_architecture(self):
        return adapter_architecture() | {'version': 'kcdno-airfrans-v1', 'output_head': 'layernorm-linear-v1'}

    def forward(self, data):
        return self.output(self.output_norm(super().forward(data)))
