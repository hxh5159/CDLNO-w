"""AirfRANS v2 wrapper preserving the native Data forward contract."""

import numpy as np
import torch
from torch import nn

from cdlno.linearno.airfrans import AirfRANSLinearNO, PointMLP, initialize_release_weights
from .construction import LoopForwardViewV2
from .core import LinearNOFFNLoopCore
from .factories import industrial_factories


class LoopedAirfRANSModelV2(LoopForwardViewV2, AirfRANSLinearNO):
    def __init__(self, *, space_dim, n_hidden, n_head, dropout, act, mlp_ratio,
                 fun_dim, out_dim, ref, unified_pos, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, core_ffn_mode, linearno_rank, latent_width,
                 feature_seed, linear):
        nn.Module.__init__(self)
        if (space_dim, fun_dim, out_dim) != (7, 0, 4) or act != "gelu" or n_hidden % n_head:
            raise ValueError("invalid AirfRANS v2 interface")
        self.__name__ = "LoopedLinearNOFFNV2"; self.ref = ref; self.unified_pos = unified_pos
        self.preprocess = PointMLP(space_dim + (ref * ref if unified_pos else 0), n_hidden * 2, n_hidden)
        self.n_hidden, self.space_dim = n_hidden, space_dim
        factories = industrial_factories("airfrans", hidden=n_hidden, heads=n_head,
            rank=linearno_rank, dropout=dropout, mlp_ratio=mlp_ratio, out_dim=out_dim)
        self.loop = LinearNOFFNLoopCore(prefix_blocks=prefix_blocks,
            recurrent_core_blocks=recurrent_core_blocks, loop_repeats=loop_repeats,
            suffix_blocks=suffix_blocks, residual_mode=residual_mode,
            core_ffn_mode=core_ffn_mode, block_factory=factories[0],
            operator_factory=factories[1], point_ffn_factory=factories[2])
        self.apply(initialize_release_weights)
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        if core_ffn_mode == "round_specific_latent":
            self.loop.install_latent_ffns(inner_width=latent_width, feature_seed=feature_seed)
        gx = torch.tensor(np.linspace(-2, 4, ref), dtype=torch.float)
        gy = torch.tensor(np.linspace(-1.5, 1.5, ref), dtype=torch.float)
        self.register_buffer("reference", torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(ref * ref, 2), persistent=False)

