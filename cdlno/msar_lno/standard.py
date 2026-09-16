"""Four static task field adapters; all MSAR lift/head computation is pointwise."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules import _check_grid, _init_linear
from ..standard import STATIC_TASKS, _grid
from .config import MSARArchitectureConfig, MSARTrainingConfig
from .core import MSARLNO


class StaticModel(nn.Module):
    """Original x/fx contract -> pointwise lift -> MSAR core (owns the head).

    structured describes the input layout only, never a convolution. Darcy's
    reference distances use the accepted index-grid order; other tasks use the
    original coordinates, including Pipe's already-normalized coordinates.
    """
    def __init__(self, *, config: MSARArchitectureConfig, task_name: str,
                 training_config: MSARTrainingConfig | None = None,
                 H=None, W=None, ref=8, unified_pos=None):
        super().__init__()
        if type(config) is not MSARArchitectureConfig:
            raise ValueError('MSARArchitectureConfig required')
        config.validate()
        if task_name not in STATIC_TASKS:
            raise ValueError('MSAR static adapter supports only darcy/elasticity/airfoil/pipe')
        task = STATIC_TASKS[task_name]
        self.structured = task['structured']
        if self.structured:
            _check_grid((H, W), H * W if type(H) is int and type(W) is int else 0)
        elif H is not None or W is not None:
            raise ValueError('Elasticity must omit H/W')
        if type(ref) is not int or ref != 8:
            raise ValueError('static task reference contract requires ref=8')
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
        self.unified_pos, self.fun_dim = bool(unified_pos), task['fun_dim']
        self.out_dim, self.Time_Input = 1, False
        stem = (ref * ref if self.unified_pos else 2) + self.fun_dim
        self.preprocess = nn.Sequential(nn.Linear(stem, 2*config.d), nn.GELU(), nn.Linear(2*config.d, config.d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        if not self.fun_dim:
            self.placeholder = nn.Parameter(torch.rand(config.d) / config.d)
        if self.unified_pos:
            pos = (_grid(H, W)[:, None] - _grid(ref, ref)[None]).square().sum(-1).sqrt()
            self.register_buffer('pos', pos.unsqueeze(0), persistent=False)
        self.core = MSARLNO(config, output_dim=1, training_config=cfg)
        # Core already owns the final LN+Linear. No extra head/recursive init.

    def adapter_architecture(self):
        return dict(version='msar-static-standard-v1', task=self.task_name, space_dim=2,
                    fun_dim=self.fun_dim, output_dim=1, Time_Input=False, ref=self.ref,
                    unified_pos=self.unified_pos, placeholder=not bool(self.fun_dim),
                    stem_width=self.preprocess[0].in_features,
                    grid_shape=[self.H, self.W] if self.structured else None,
                    position_rule='baseline-structured-index' if self.unified_pos else 'original-coordinates',
                    lift='pointwise-linear-gelu-linear-2d-v1', output_head='core-layernorm-linear-v1')

    def forward(self, x, fx, T=None, *, return_aux=False, training_config=None):
        if (not isinstance(x, Tensor) or not x.is_floating_point() or x.ndim != 3
                or x.shape[-1] != 2 or min(x.shape[:2]) < 1):
            raise ValueError('x must be nonempty floating [B,N,2]')
        if self.structured and x.shape[1] != self.H * self.W:
            raise ValueError('N must preserve H*W')
        if T is not None:
            raise ValueError('static tasks have no time condition')
        if self.fun_dim:
            if not isinstance(fx, Tensor) or fx.shape != (*x.shape[:2], 1):
                raise ValueError('Darcy requires fx[B,N,1]')
            if fx.device != x.device or fx.dtype != x.dtype:
                raise ValueError('fx and x must share floating dtype/device')
        elif fx is not None:
            raise ValueError('this static task requires fx=None')
        pos = self.pos.to(dtype=x.dtype).expand(x.shape[0], -1, -1) if self.unified_pos else x
        lifted = self.preprocess(torch.cat((pos, fx), -1) if fx is not None else pos)
        if not self.fun_dim:
            lifted = lifted + self.placeholder[None, None]
        return self.core(lifted, return_aux=return_aux, training_config=training_config)
