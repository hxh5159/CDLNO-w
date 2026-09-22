"""V3 AirfRANS wrapper preserving the approved native Data forward."""
import numpy as np
import torch
from torch import nn

from cdlno.linearno.airfrans import (AirfRANSBlock, AirfRANSLinearNO, PointMLP,
                                     initialize_release_weights)
from .construction import LoopForwardViewV3, constructor_config
from .core import LinearNOLoopCoreV3


class LoopedAirfRANSModelV3(LoopForwardViewV3, AirfRANSLinearNO):
    def __init__(self, *, dropout, activation, linearno_variant, heads, ffn_ratio,
                 space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, latent_width,
                 prefix_blocks, recurrent_core_blocks, loop_repeats, suffix_blocks,
                 residual_mode, latent_enabled, adapter_mode, adapter_rank,
                 adapter_alpha, latent_seed, adapter_seed):
        kwargs = {key: value for key, value in locals().items() if key != "self"}
        config = constructor_config(kwargs)
        nn.Module.__init__(self)
        if linearno_variant != "airfrans" or activation != "gelu" or time_input:
            raise ValueError("invalid V3 AirfRANS wrapper contract")
        self.__name__ = "LoopedLinearNOLatentAdapterV3"
        self.ref, self.unified_pos = ref, unified_pos
        channels = space_dim + (ref * ref if unified_pos else 0)
        self.preprocess = PointMLP(channels, hidden_width * 2, hidden_width)
        self.n_hidden, self.space_dim = hidden_width, space_dim

        def block_factory(*, last_layer):
            return AirfRANSBlock(hidden_width, heads, actual_M, ffn_ratio, dropout,
                                 out_dim, last_layer)

        self.loop = LinearNOLoopCoreV3(config=config, block_factory=block_factory)
        self.apply(initialize_release_weights)
        self.placeholder = nn.Parameter(torch.rand(hidden_width, dtype=torch.float32) /
                                        hidden_width)
        self.loop.install_features()
        gx = torch.tensor(np.linspace(-2, 4, ref), dtype=torch.float32)
        gy = torch.tensor(np.linspace(-1.5, 1.5, ref), dtype=torch.float32)
        reference = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(ref * ref, 2)
        self.register_buffer("reference", reference, persistent=False)
