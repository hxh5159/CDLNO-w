"""Standard benchmark adapters using the existing coordinate/lift conventions."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules import _check_grid, _init_linear
from ..standard import STATIC_TASKS, TEMPORAL_TASKS, _grid
from .config import KCDNOArchitectureConfig
from .matched import make_core


class StandardModel(nn.Module):
    """One task lift/head around KCDNO; no data or training code."""

    def __init__(self, *, config: KCDNOArchitectureConfig, task_name: str,
                 H=None, W=None, ref=8, unified_pos=None):
        super().__init__()
        config.validate()
        tasks = STATIC_TASKS | TEMPORAL_TASKS
        if task_name not in tasks:
            raise ValueError('task is not a supported KCDNO standard benchmark')
        task = tasks[task_name]
        self.structured = task['structured']
        expected = 'conv_ffn' if self.structured else 'point_ffn'
        if config.point_module != expected:
            raise ValueError(f'{task_name} requires {expected}')
        if self.structured:
            _check_grid((H, W), H * W if type(H) is int and type(W) is int else 0)
        elif H is not None or W is not None:
            raise ValueError('point task must omit H/W')
        if task_name in TEMPORAL_TASKS and (H, W) != task['grid']:
            raise ValueError(f"{task_name} requires its original grid {task['grid']}")
        if type(ref) is not int or ref != 8:
            raise ValueError('KCDNO task contract requires ref=8')
        if unified_pos is None:
            unified_pos = task['unified_pos']
        if type(unified_pos) not in (int, bool) or unified_pos != task['unified_pos']:
            raise ValueError('unified_pos must preserve the task coordinate contract')
        self.config, self.task_name = config, task_name
        self.H, self.W, self.ref = H, W, ref
        self.unified_pos, self.fun_dim = bool(unified_pos), task['fun_dim']
        self.Time_Input = task.get('time_input', False)
        if self.Time_Input and config.d < 2:
            raise ValueError('Plasticity time embedding requires d>=2')
        self.out_dim, self.n_hidden = task.get('out_dim', 1), config.d
        stem = (ref * ref if self.unified_pos else 2) + self.fun_dim
        self.preprocess = nn.Sequential(nn.Linear(stem, 2*config.d), nn.GELU(), nn.Linear(2*config.d, config.d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        if not self.fun_dim:
            self.placeholder = nn.Parameter(torch.rand(config.d) / config.d)
        if self.unified_pos:
            pos = (_grid(H, W)[:, None] - _grid(ref, ref)[None]).square().sum(-1).sqrt()
            self.register_buffer('pos', pos.unsqueeze(0), persistent=False)
        if self.Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(config.d, config.d), nn.SiLU(), nn.Linear(config.d, config.d))
            _init_linear(self.time_fc[0]); _init_linear(self.time_fc[2])
        self.core = make_core(config)
        self.output_norm = nn.LayerNorm(config.d, eps=config.norm_eps)
        self.output = nn.Linear(config.d, self.out_dim)
        _init_linear(self.output)

    def adapter_architecture(self):
        return dict(version='kcdno-standard-v1', task=self.task_name, space_dim=2,
                    fun_dim=self.fun_dim, output_dim=self.out_dim, Time_Input=self.Time_Input,
                    ref=self.ref, unified_pos=self.unified_pos, placeholder=not bool(self.fun_dim),
                    stem_width=self.preprocess[0].in_features,
                    grid_shape=[self.H, self.W] if self.structured else None,
                    position_rule='baseline-structured-index' if self.unified_pos else 'original-coordinates',
                    output_head='layernorm-linear-v1')

    def forward(self, x, fx, T=None):
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[-1] != 2 or not x.is_floating_point():
            raise ValueError('x must be floating [B,N,2]')
        if min(x.shape[:2]) < 1:
            raise ValueError('empty batch/points')
        if self.structured and x.shape[1] != self.H*self.W:
            raise ValueError('N must preserve H*W')
        if self.Time_Input:
            if not isinstance(T, Tensor) or T.shape != (x.shape[0], 1):
                raise ValueError('Plasticity requires T[B,1]')
            if not T.is_floating_point() or T.device != x.device:
                raise ValueError('T must be floating on x.device')
        elif T is not None:
            raise ValueError('this task has no explicit time condition')
        if self.fun_dim:
            if not isinstance(fx, Tensor) or fx.shape != (*x.shape[:2], self.fun_dim):
                raise ValueError(f'fx must be [B,N,{self.fun_dim}]')
            if fx.device != x.device or fx.dtype != x.dtype:
                raise ValueError('fx and x must share floating dtype/device')
        elif fx is not None:
            raise ValueError('this task requires fx=None')
        pos = self.pos.to(dtype=x.dtype).expand(x.shape[0], -1, -1) if self.unified_pos else x
        lifted = self.preprocess(torch.cat((pos, fx), -1) if fx is not None else pos)
        if not self.fun_dim:
            lifted = lifted + self.placeholder[None, None]
        if self.Time_Input:
            # Identical frequency construction/injection to the accepted task
            # adapter; one spatial field per T, never flatten time into N.
            half = self.n_hidden // 2
            freqs = torch.exp(-torch.log(torch.tensor(10000., device=T.device)) *
                              torch.arange(half, device=T.device, dtype=T.dtype) / half)
            emb = T[:, :, None] * freqs[None, None, :]
            emb = torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)
            if self.n_hidden % 2:
                emb = torch.cat((emb, torch.zeros_like(emb[..., :1])), dim=-1)
            lifted = lifted + self.time_fc(emb).expand(-1, x.shape[1], -1)
        features = self.core(lifted, grid_shape=(self.H, self.W) if self.structured else None)
        return self.output(self.output_norm(features))
