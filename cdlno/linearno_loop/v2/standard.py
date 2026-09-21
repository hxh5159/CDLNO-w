"""Standard benchmark v2 wrapper with native input/output semantics."""

import importlib

import torch
from torch import nn

from .construction import LoopForwardViewV2
from .core import LinearNOFFNLoopCore
from .factories import standard_factories

native = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO")


class LoopedStandardModelV2(LoopForwardViewV2, native.Model):
    def __init__(self, *, space_dim, n_hidden, n_head, dropout, act, mlp_ratio,
                 fun_dim, out_dim, ref, unified_pos, prefix_blocks,
                 recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, core_ffn_mode, linearno_rank, latent_width,
                 feature_seed, Time_Input, H, W, linearno_variant):
        nn.Module.__init__(self)
        if act != "gelu" or n_hidden % n_head:
            raise ValueError("v2 Standard requires GELU and hidden divisible by heads")
        self.H, self.W, self.ref, self.unified_pos = H, W, ref, unified_pos
        self.fun_dim, self.out_dim = fun_dim, out_dim
        self.linearno_variant, self.linearno_rank = linearno_variant, linearno_rank
        if unified_pos:
            self.register_buffer("pos", native._unified_positions(H, W, ref), persistent=False)
        self.preprocess = native.PointwiseMLP(fun_dim + (ref * ref if unified_pos else space_dim), n_hidden * 2, n_hidden)
        self.Time_Input, self.n_hidden, self.space_dim = Time_Input, n_hidden, space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
        factories = standard_factories(hidden=n_hidden, heads=n_head, rank=linearno_rank,
            variant=linearno_variant, dropout=dropout, mlp_ratio=mlp_ratio, H=H, W=W, out_dim=out_dim)
        self.loop = LinearNOFFNLoopCore(prefix_blocks=prefix_blocks,
            recurrent_core_blocks=recurrent_core_blocks, loop_repeats=loop_repeats,
            suffix_blocks=suffix_blocks, residual_mode=residual_mode,
            core_ffn_mode=core_ffn_mode, block_factory=factories[0],
            operator_factory=factories[1], point_ffn_factory=factories[2])
        self.apply(native.initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(n_hidden, dtype=torch.float32) * (1 / n_hidden))
        if core_ffn_mode == "round_specific_latent":
            self.loop.install_latent_ffns(inner_width=latent_width, feature_seed=feature_seed)

