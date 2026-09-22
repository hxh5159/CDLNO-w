"""V3 ShapeNet-Car wrapper with actual M independent of head dimension."""
import torch
from torch import nn

from cdlno.linearno.shapenet import (PointMLP, ShapeNetBlock, ShapeNetLinearNO,
                                     initialize_release_weights)
from .construction import LoopForwardViewV3, constructor_config
from .core import LinearNOLoopCoreV3


class LoopedShapeNetModelV3(LoopForwardViewV3, ShapeNetLinearNO):
    def __init__(self, *, dropout, activation, linearno_variant, heads, ffn_ratio,
                 space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, latent_width,
                 prefix_blocks, recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, latent_enabled, adapter_mode, adapter_rank,
                 adapter_alpha, latent_seed, adapter_seed):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        config = constructor_config(kwargs)
        nn.Module.__init__(self)
        if linearno_variant != "shapenet" or activation != "gelu":
            raise ValueError("invalid V3 ShapeNet wrapper contract")
        self.H, self.W, self.ref = grid_height, grid_width, ref
        self.unified_pos, self.Time_Input = unified_pos, time_input
        if unified_pos:
            self.register_buffer("pos", self.get_grid(), persistent=False)
        channels = ref * ref if unified_pos else space_dim + fun_dim
        self.preprocess = PointMLP(channels, hidden_width * 2, hidden_width)
        self.n_hidden, self.space_dim = hidden_width, space_dim
        if time_input:
            self.time_fc = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.SiLU(),
                                         nn.Linear(hidden_width, hidden_width))

        def block_factory(*, last_layer):
            # ShapeNetBlock already accepts arbitrary absolute rank. Only the old
            # full wrapper translated release key_ratio through M % head_dim.
            return ShapeNetBlock(hidden_width, heads, actual_M, ffn_ratio, dropout,
                                 out_dim, last_layer)

        self.loop = LinearNOLoopCoreV3(config=config, block_factory=block_factory)
        self.apply(initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) /
                                        hidden_width)
        self.loop.install_features()

