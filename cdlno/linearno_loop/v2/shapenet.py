"""ShapeNet-Car v2 wrapper preserving the native tuple forward contract."""

import torch
from torch import nn

from cdlno.linearno.shapenet import ShapeNetLinearNO, PointMLP, initialize_release_weights
from .construction import LoopForwardViewV2
from .core import LinearNOFFNLoopCore
from .factories import industrial_factories


class LoopedShapeNetModelV2(LoopForwardViewV2, ShapeNetLinearNO):
    def __init__(self, *, space_dim, n_hidden, n_head, dropout, act, mlp_ratio,
                 fun_dim, out_dim, ref, unified_pos, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, core_ffn_mode, linearno_rank, latent_width,
                 feature_seed, Time_Input, H, W, isregular):
        nn.Module.__init__(self)
        if space_dim + fun_dim != 7 or out_dim != 4 or act != "gelu" or n_hidden % n_head:
            raise ValueError("invalid ShapeNet v2 interface")
        if linearno_rank % (n_hidden // n_head):
            raise ValueError("ShapeNet actual M must be an integer multiple of head_dim")
        self.H, self.W, self.ref, self.unified_pos, self.Time_Input = H, W, ref, unified_pos, Time_Input
        if unified_pos:
            self.register_buffer("pos", self.get_grid(), persistent=False)
        self.preprocess = PointMLP(ref * ref if unified_pos else space_dim + fun_dim, n_hidden * 2, n_hidden)
        self.n_hidden, self.space_dim = n_hidden, space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
        factories = industrial_factories("car", hidden=n_hidden, heads=n_head,
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

