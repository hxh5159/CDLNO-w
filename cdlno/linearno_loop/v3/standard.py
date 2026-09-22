"""V3 Standard wrapper; native Model forward owns all task input semantics."""
import importlib

import torch
from torch import nn

from .construction import LoopForwardViewV3, constructor_config
from .core import LinearNOLoopCoreV3


native = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO")


class LoopedStandardModelV3(LoopForwardViewV3, native.Model):
    def __init__(self, *, dropout, activation, linearno_variant, heads, ffn_ratio,
                 space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, latent_width,
                 prefix_blocks, recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, latent_enabled, adapter_mode, adapter_rank,
                 adapter_alpha, latent_seed, adapter_seed):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        config = constructor_config(kwargs)
        nn.Module.__init__(self)
        self.H, self.W, self.ref = grid_height, grid_width, ref
        self.unified_pos = unified_pos
        self.fun_dim, self.out_dim = fun_dim, out_dim
        self.linearno_variant, self.linearno_rank = linearno_variant, actual_M
        if unified_pos:
            self.register_buffer("pos", native._unified_positions(grid_height, grid_width, ref),
                                 persistent=False)
        channels = fun_dim + (ref * ref if unified_pos else space_dim)
        self.preprocess = native.PointwiseMLP(channels, hidden_width * 2, hidden_width)
        self.Time_Input, self.n_hidden, self.space_dim = time_input, hidden_width, space_dim
        if time_input:
            self.time_fc = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                                         nn.Linear(hidden_width, hidden_width))

        def block_factory(*, last_layer):
            return native.LinearNOBlock(hidden=hidden_width, heads=heads, rank=actual_M,
                variant=linearno_variant, dropout=dropout, mlp_ratio=ffn_ratio,
                H=grid_height, W=grid_width, last_layer=last_layer, out_dim=out_dim)

        self.loop = LinearNOLoopCoreV3(config=config, block_factory=block_factory)
        self.apply(native.initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) /
                                        hidden_width)
        self.loop.install_features()

