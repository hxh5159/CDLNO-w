"""CDLNO adapter for the ShapeNet-Car PyG graph contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cdlno.config import CDLNOArchitectureConfig
from cdlno.core import CDLNO
from cdlno.modules import _init_linear


def architecture(n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2,
                 dropout=0.0, slice_num=64, front_blocks=2,
                 latent_ffn_ratio=2.0, cdpa_mode="entry", front_latent_mode="full"):
    """Describe the fixed task input contract without constructing weights."""
    if dropout != 0.0:
        raise ValueError('the confirmed CDLNO core requires dropout=0')
    return CDLNOArchitectureConfig(
        task_name="shapenet-car", L=n_layers, F=front_blocks, M=slice_num,
        d_model=n_hidden, num_heads=n_head, ffn_ratio=mlp_ratio,
        latent_ffn_ratio=latent_ffn_ratio, cdpa_mode=cdpa_mode, front_latent_mode=front_latent_mode,
        attention_dropout=dropout, structured=False, output_dim=4,
    ).validate()


def adapter_architecture():
    return {
        "version": "shapenet-car-v1", "input_channels": 7, "stem_width": 7,
        "input_order": ["xyz3", "sdf1", "normal3"],
        "output_order": ["velocity3", "pressure1"], "output_channels": 4,
        "placeholder": True, "Time_Input": False, "unified_pos": False,
        "geometry_encoder": False, "batch_policy": "single-graph",
    }


class Model(nn.Module):
    """Map one PyG CFD graph with seven input channels to four outputs.

    ``geom_data`` is intentionally accepted for compatibility and ignored, as
    in the baseline.  A multi-graph PyG Batch is rejected because concatenating
    its nodes would create a physically invalid single field.
    """

    def __init__(self, n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2,
                 dropout=0.0, slice_num=64, front_blocks=2,
                 latent_ffn_ratio=2.0, cdpa_mode="entry",
                 cdpa_source_chunk_size=0, front_latent_mode="full"):
        super().__init__()
        self.config = architecture(n_hidden, n_layers, n_head, mlp_ratio, dropout,
                                   slice_num, front_blocks, latent_ffn_ratio, cdpa_mode, front_latent_mode)
        self.preprocess = nn.Sequential(
            nn.Linear(7, 2 * n_hidden), nn.GELU(), nn.Linear(2 * n_hidden, n_hidden)
        )
        _init_linear(self.preprocess[0])
        _init_linear(self.preprocess[2])
        # The original task always has fx=None: preserve its active placeholder.
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.core = CDLNO(self.config, source_chunk_size=cdpa_source_chunk_size)
        # No parent apply/reset: core queries and CDPA retain their initialization.

    def adapter_architecture(self):
        return adapter_architecture()

    @staticmethod
    def _single_graph(cfd_data, n):
        error = ("CDLNO ShapeNet-Car accepts one physical graph per forward; "
                 "multi-graph PyG Batch is unsupported")
        integer_types = (torch.int32, torch.int64)
        ptr = getattr(cfd_data, "ptr", None)
        if ptr is not None:
            if not isinstance(ptr, Tensor) or ptr.ndim != 1 or ptr.dtype not in integer_types:
                raise ValueError("cfd_data.ptr must be a one-dimensional integer tensor")
            if ptr.numel() > 2:
                raise ValueError(error)
            if ptr.numel() != 2 or ptr[0].item() != 0 or ptr[1].item() != n:
                raise ValueError("single-graph ptr must be [0,N]")
        batch = getattr(cfd_data, "batch", None)
        if batch is not None:
            if (not isinstance(batch, Tensor) or batch.shape != (n,)
                    or batch.dtype not in integer_types):
                raise ValueError("cfd_data.batch must be an integer tensor of shape [N]")
            if torch.any(batch != 0).item():
                raise ValueError(error + "; a single-graph batch must be all zero")

    def forward(self, data):
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise TypeError("ShapeNet-Car input must be (cfd_data, geom_data)")
        cfd_data, _geom_data = data
        x = getattr(cfd_data, "x", None)
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[-1] != 7:
            raise ValueError("cfd_data.x must have shape [N,7]")
        if not x.is_floating_point() or x.shape[0] < 1:
            raise ValueError("cfd_data.x must be nonempty floating-point data")
        self._single_graph(cfd_data, x.shape[0])
        # No y, edge_index, surf, or geometry object is read or modified.
        h_0 = self.preprocess(x.unsqueeze(0)) + self.placeholder[None, None, :]
        return self.core(h_0).squeeze(0)


__all__ = ["Model"]
