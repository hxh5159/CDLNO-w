"""NS/Plasticity field and time adapters around the unchanged four-scale core.

Each call owns its latent/coverage graph; this wrapper stores no time history.
Stable class path: cdlno.msar_lno.temporal.TemporalModel.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules import _check_grid, _init_linear
from ..standard import TEMPORAL_TASKS, _grid
from .config import MSARArchitectureConfig, MSARTrainingConfig
from .core import MSARLNO


class TemporalModel(nn.Module):
    def __init__(self, *, config: MSARArchitectureConfig, task_name: str,
                 training_config: MSARTrainingConfig | None = None,
                 H=None, W=None, ref=8, unified_pos=None):
        super().__init__()
        if type(config) is not MSARArchitectureConfig:
            raise ValueError('MSARArchitectureConfig required')
        config.validate()
        if task_name not in TEMPORAL_TASKS:
            raise ValueError('MSAR temporal adapter supports only ns/plasticity')
        task = TEMPORAL_TASKS[task_name]
        _check_grid((H, W), H * W if type(H) is int and type(W) is int else 0)
        if (H, W) != task['grid']:
            raise ValueError(f"{task_name} requires its accepted spatial grid {task['grid']}")
        if type(ref) is not int or ref != 8:
            raise ValueError('temporal task reference contract requires ref=8')
        if unified_pos is None:
            unified_pos = task['unified_pos']
        if type(unified_pos) not in (int, bool) or unified_pos != task['unified_pos']:
            raise ValueError('unified_pos must preserve the original task coordinate contract')
        cfg = MSARTrainingConfig() if training_config is None else training_config
        if type(cfg) is not MSARTrainingConfig:
            raise ValueError('MSARTrainingConfig required')
        cfg.validate()
        self.config, self.training_config = config, cfg
        self.task_name, self.H, self.W, self.ref = task_name, H, W, ref
        self.structured = True  # Input layout only; never requests convolution.
        self.unified_pos, self.fun_dim = bool(unified_pos), task['fun_dim']
        self.out_dim, self.Time_Input = task['out_dim'], task['time_input']
        if self.Time_Input and config.d < 2:
            raise ValueError('Plasticity time embedding requires d>=2')
        stem = (ref * ref if self.unified_pos else 2) + self.fun_dim
        self.preprocess = nn.Sequential(nn.Linear(stem, 2*config.d), nn.GELU(), nn.Linear(2*config.d, config.d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        if self.unified_pos:
            pos = (_grid(H, W)[:, None] - _grid(ref, ref)[None]).square().sum(-1).sqrt()
            self.register_buffer('pos', pos.unsqueeze(0), persistent=False)
        if self.Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(config.d, config.d), nn.SiLU(), nn.Linear(config.d, config.d))
            _init_linear(self.time_fc[0]); _init_linear(self.time_fc[2])
        self.core = MSARLNO(config, output_dim=self.out_dim, training_config=cfg)
        # Each primitive initialized itself, and the core owns the only head.

    def adapter_architecture(self):
        return dict(version='msar-temporal-standard-v1', task=self.task_name, space_dim=2,
                    fun_dim=self.fun_dim, output_dim=self.out_dim, Time_Input=self.Time_Input,
                    ref=self.ref, unified_pos=self.unified_pos, placeholder=False,
                    stem_width=self.preprocess[0].in_features, grid_shape=[self.H, self.W],
                    position_rule='baseline-structured-index' if self.unified_pos else 'original-coordinates',
                    time_rule='baseline-sincos-silu-add-v1' if self.Time_Input else 'window10-no-time-embedding',
                    lift='pointwise-linear-gelu-linear-2d-v1', output_head='core-layernorm-linear-v1')

    def forward(self, x, fx, T=None, *, return_aux=False, training_config=None):
        if (not isinstance(x, Tensor) or not x.is_floating_point() or x.ndim != 3
                or x.shape[-1] != 2 or min(x.shape[:2]) < 1):
            raise ValueError('x must be nonempty floating [B,N,2]')
        if x.shape[1] != self.H * self.W:
            raise ValueError('N must preserve spatial H*W; time is not a spatial axis')
        if not isinstance(fx, Tensor) or fx.shape != (*x.shape[:2], self.fun_dim):
            raise ValueError(f'{self.task_name} requires fx[B,N,{self.fun_dim}]')
        if fx.device != x.device or fx.dtype != x.dtype:
            raise ValueError('fx and x must share floating dtype/device')
        if self.Time_Input:
            if not isinstance(T, Tensor) or T.shape != (x.shape[0], 1):
                raise ValueError('Plasticity requires T[B,1] for one time per sample')
            if not T.is_floating_point() or T.device != x.device:
                raise ValueError('T must be floating on x.device')
        elif T is not None:
            raise ValueError('NS has no explicit time condition; provide its ten-frame fx window')
        pos = self.pos.to(dtype=x.dtype).expand(x.shape[0], -1, -1) if self.unified_pos else x
        lifted = self.preprocess(torch.cat((pos, fx), -1))
        if self.Time_Input:
            # Preserve the accepted per-sample frequency, cos/sin order and injection.
            half = self.config.d // 2
            freqs = torch.exp(-torch.log(torch.tensor(10000., device=T.device)) *
                              torch.arange(half, device=T.device, dtype=T.dtype) / half)
            emb = T[:, :, None] * freqs[None, None, :]
            emb = torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)
            if self.config.d % 2:
                emb = torch.cat((emb, torch.zeros_like(emb[..., :1])), dim=-1)
            lifted = lifted + self.time_fc(emb).expand(-1, x.shape[1], -1)
        return self.core(lifted, return_aux=return_aux, training_config=training_config)
