"""AirfRANS input adapter; all neural-operator mathematics stays in core.py."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .config import CDLNOArchitectureConfig
from .core import CDLNO
from .modules import _init_linear


def architecture(n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2,
                 dropout=0., slice_num=64, front_blocks=2,
                 latent_ffn_ratio=2., cdpa_mode='entry', front_latent_mode='full'):
    if dropout != 0.:
        raise ValueError('the confirmed CDLNO core requires dropout=0')
    return CDLNOArchitectureConfig(
        task_name='airfrans', L=n_layers, F=front_blocks, M=slice_num,
        d_model=n_hidden, num_heads=n_head, ffn_ratio=mlp_ratio,
        latent_ffn_ratio=latent_ffn_ratio, cdpa_mode=cdpa_mode, front_latent_mode=front_latent_mode,
        attention_dropout=dropout, structured=False, output_dim=4,
    ).validate()


def adapter_architecture():
    return dict(version='airfrans-v1', input_channels=7, stem_width=71,
                input_order=['xy2', 'Uinf2', 'sdf1', 'normal2'],
                output_order=['vx', 'vy', 'p', 'nut'], output_channels=4,
                ref=8, reference_domain=[[-2., 4.], [-1.5, 1.5]],
                reference_rule='append-distances-from-data.pos', placeholder=True,
                Time_Input=False, batch_policy='single-graph-sampled-ptr-v1')


def _single_graph(data, n):
    error = 'CDLNO AirfRANS accepts one physical graph per forward; multi-graph input is unsupported'
    integers = (torch.int32, torch.int64)
    batch = getattr(data, 'batch', None)
    if batch is not None:
        if not isinstance(batch, Tensor) or batch.shape != (n,) or batch.dtype not in integers:
            raise ValueError('data.batch must be an integer tensor of shape [N]')
        if torch.any(batch != 0).item():
            raise ValueError(error + '; single-graph batch must be all zero')
    ptr = getattr(data, 'ptr', None)
    if ptr is not None:
        if not isinstance(ptr, Tensor) or ptr.ndim != 1 or ptr.dtype not in integers:
            raise ValueError('data.ptr must be a one-dimensional integer tensor')
        if ptr.numel() > 2:
            raise ValueError(error)
        if ptr.numel() != 2 or ptr[0].item() != 0:
            raise ValueError('single-graph ptr must contain [0,original_N]')
        original_n = ptr[1].item()
        # Original Infer_test slices batch along with x/pos, but retains ptr.
        # A single original graph with an all-zero sampled batch is still one
        # field. Do not rewrite its metadata or change the sampling pipeline.
        if original_n != n and (batch is None or original_n < n):
            raise ValueError('ptr must match N, or the original single graph before sampling')


class AirfRANSModel(nn.Module):
    """forward(data) -> [N,4], preserving original x7 and position semantics."""

    def __init__(self, n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2,
                 dropout=0., slice_num=64, front_blocks=2,
                 latent_ffn_ratio=2., cdpa_mode='entry', cdpa_source_chunk_size=0,
                 front_latent_mode='full'):
        super().__init__()
        self.config = architecture(n_hidden, n_layers, n_head, mlp_ratio, dropout,
                                   slice_num, front_blocks, latent_ffn_ratio, cdpa_mode, front_latent_mode)
        self.preprocess = nn.Sequential(nn.Linear(71, 2 * n_hidden), nn.GELU(),
                                        nn.Linear(2 * n_hidden, n_hidden))
        _init_linear(self.preprocess[0])
        _init_linear(self.preprocess[2])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        # NumPy linspace -> float32 and x-major/y-minor flattening match the
        # original get_grid; no global CUDA allocation or changed coordinate domain.
        gx = torch.tensor(np.linspace(-2, 4, 8), dtype=torch.float32)
        gy = torch.tensor(np.linspace(-1.5, 1.5, 8), dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1).reshape(64, 2)
        self.register_buffer('reference', grid, persistent=False)
        self.core = CDLNO(self.config, source_chunk_size=cdpa_source_chunk_size)
        # No recursive parent initialization of queries or CDPA.

    def adapter_architecture(self):
        return adapter_architecture()

    def forward(self, data):
        x, pos = getattr(data, 'x', None), getattr(data, 'pos', None)
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] != 7 or x.shape[0] < 1:
            raise ValueError('data.x must have shape [N,7] with N>0')
        if not x.is_floating_point():
            raise ValueError('data.x must be floating-point')
        if not isinstance(pos, Tensor) or pos.shape != (x.shape[0], 2):
            raise ValueError('data.pos must have shape [N,2]')
        if pos.device != x.device or pos.dtype != x.dtype:
            raise ValueError('data.pos and data.x must share floating dtype and device')
        _single_graph(data, x.shape[0])
        distances = (pos[:, None, :] - self.reference.to(dtype=pos.dtype)[None]).square().sum(-1).sqrt()
        features = torch.cat((x, distances), dim=-1).unsqueeze(0)
        h_0 = self.preprocess(features) + self.placeholder[None, None, :]
        return self.core(h_0).squeeze(0)
