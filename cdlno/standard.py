"""Input adapters for the four static standard benchmarks (no data loading)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .config import CDLNOArchitectureConfig
from .core import CDLNO
from .modules import _init_linear


STATIC_TASKS = {
    'darcy': dict(heads=8, latents=64, structured=True, fun_dim=1, unified_pos=True),
    'elasticity': dict(heads=8, latents=64, structured=False, fun_dim=0, unified_pos=False),
    'airfoil': dict(heads=4, latents=64, structured=True, fun_dim=0, unified_pos=False),
    'pipe': dict(heads=4, latents=32, structured=True, fun_dim=0, unified_pos=False),
}

TEMPORAL_TASKS = {
    'ns': dict(heads=8, latents=64, structured=True, fun_dim=10, out_dim=1,
               unified_pos=True, time_input=False, grid=(64, 64)),
    'plasticity': dict(heads=8, latents=64, structured=True, fun_dim=1, out_dim=4,
                       unified_pos=False, time_input=True, grid=(101, 31)),
}


def _axis(length: int) -> Tensor:
    # Match the baseline's NumPy linspace -> float32 conversion exactly.
    return torch.tensor(np.linspace(0, 1, length), dtype=torch.float32)


def _grid(height: int, width: int) -> Tensor:
    x, y = torch.meshgrid(_axis(height), _axis(width), indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


class StaticStandardModel(nn.Module):
    """Baseline input lifting followed by the shared CDLNO core.

    fx=None is the actual contract for elasticity/airfoil/pipe; Darcy requires
    one coefficient channel. Static tasks have no time-conditioned parameters.
    """

    structured_adapter: bool

    def __init__(
        self, space_dim=2, n_layers=8, n_hidden=128, dropout=0.0,
        n_head=None, Time_Input=False, act='gelu', mlp_ratio=2,
        fun_dim=None, out_dim=1, slice_num=None, ref=8, unified_pos=None,
        H=None, W=None, *, task_name, front_blocks=2, latent_ffn_ratio=2,
        cdpa_mode='entry', cdpa_source_chunk_size=0, front_latent_mode='full',
    ):
        super().__init__()
        if task_name not in STATIC_TASKS:
            raise ValueError('this adapter supports only darcy/elasticity/airfoil/pipe')
        task = STATIC_TASKS[task_name]
        if self.structured_adapter != task['structured']:
            raise ValueError('task does not match structured/irregular adapter')
        fun_dim = task['fun_dim'] if fun_dim is None else fun_dim
        unified_pos = task['unified_pos'] if unified_pos is None else unified_pos
        if type(space_dim) is not int or space_dim != 2 or type(fun_dim) is not int or fun_dim != task['fun_dim']:
            raise ValueError('static task space_dim/fun_dim violates its input contract')
        if type(out_dim) is not int or out_dim != 1:
            raise ValueError('these four tasks require out_dim=1')
        if Time_Input is not False or act != 'gelu':
            raise ValueError('static task adapters require Time_Input=False and act=gelu')
        if type(unified_pos) not in (bool, int) or unified_pos not in (0, 1):
            raise ValueError('unified_pos must be bool or 0/1')
        if type(ref) is not int or ref < 1:
            raise ValueError('ref must be a positive integer')
        if not task['structured'] and (H is not None or W is not None):
            raise ValueError('Elasticity is an irregular point set; omit H/W')
        self.config = CDLNOArchitectureConfig(
            task_name=task_name, L=n_layers, F=front_blocks,
            M=task['latents'] if slice_num is None else slice_num, d_model=n_hidden,
            num_heads=task['heads'] if n_head is None else n_head,
            ffn_ratio=mlp_ratio, latent_ffn_ratio=latent_ffn_ratio,
            cdpa_mode=cdpa_mode, attention_dropout=dropout, front_latent_mode=front_latent_mode,
            structured=task['structured'], grid_shape=(H, W) if task['structured'] else None,
            output_dim=out_dim,
        ).validate()
        self.task_name = task_name
        self.space_dim, self.fun_dim = space_dim, fun_dim
        self.Time_Input = False
        self.unified_pos, self.ref = bool(unified_pos), ref
        self.H, self.W = H, W
        self.n_hidden = n_hidden
        stem_width = (ref * ref if self.unified_pos else space_dim) + fun_dim
        self.preprocess = nn.Sequential(
            nn.Linear(stem_width, 2 * n_hidden), nn.GELU(), nn.Linear(2 * n_hidden, n_hidden),
        )
        _init_linear(self.preprocess[0])
        _init_linear(self.preprocess[2])
        if fun_dim == 0:
            self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        if self.unified_pos:
            reference = _grid(ref, ref)
            if task['structured']:
                # The baseline structured model uses a fixed index grid, not x.
                points = _grid(H, W)
                pos = ((points[:, None, :] - reference[None, :, :]).square().sum(-1)).sqrt()
                self.register_buffer('pos', pos.unsqueeze(0), persistent=False)
            else:
                self.register_buffer('reference', reference, persistent=False)
        self.core = CDLNO(self.config, source_chunk_size=cdpa_source_chunk_size)
        # No recursive apply/reset: the core owns its validated initialization.

    def adapter_architecture(self):
        """Semantic adapter fields checked in addition to core architecture."""
        return dict(version='static-standard-v1', space_dim=self.space_dim,
                    fun_dim=self.fun_dim, Time_Input=False, act='gelu',
                    unified_pos=self.unified_pos, ref=self.ref,
                    stem_width=self.preprocess[0].in_features,
                    placeholder=self.fun_dim == 0,
                    position_rule='baseline-structured-index' if self.config.structured
                    else 'baseline-irregular-coordinates')

    def forward(self, x, fx, T=None):
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError('x must be [B,N,2]')
        if not x.is_floating_point() or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError('x must be nonempty floating-point coordinates')
        if self.config.structured and x.shape[1] != self.H * self.W:
            raise ValueError('N must equal the original H*W grid layout')
        if T is not None:
            raise ValueError('static tasks do not accept a time condition')
        if self.fun_dim:
            if not isinstance(fx, Tensor) or fx.shape != (*x.shape[:2], self.fun_dim):
                raise ValueError('Darcy requires fx[B,N,1]')
            if not fx.is_floating_point() or fx.device != x.device or fx.dtype != x.dtype:
                raise ValueError('fx and x must share floating dtype and device')
        elif fx is not None:
            raise ValueError('this static task requires fx=None')
        if self.unified_pos:
            if self.config.structured:
                position = self.pos.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
            else:
                position = (x[:, :, None] - self.reference.to(dtype=x.dtype)[None, None]).square().sum(-1).sqrt()
        else:
            position = x
        lifted = self.preprocess(torch.cat((position, fx), -1) if fx is not None else position)
        if self.fun_dim == 0:
            lifted = lifted + self.placeholder[None, None, :]
        return self.core(lifted)


class TemporalStandardModel(nn.Module):
    """NS/Plasticity adapter preserving one-call temporal conditioning.

    The shared core is rebuilt on every call. Temporal rollout policy remains in
    the experiment loop: NS passes a rolling fx window; Plasticity passes one
    scalar T per call. No latent state crosses calls.
    """

    structured_adapter = True

    def __init__(
        self, space_dim=2, n_layers=8, n_hidden=128, dropout=0.0,
        n_head=None, Time_Input=False, act='gelu', mlp_ratio=2,
        fun_dim=None, out_dim=None, slice_num=None, ref=8, unified_pos=None,
        H=None, W=None, *, task_name, front_blocks=2, latent_ffn_ratio=2,
        cdpa_mode='entry', cdpa_source_chunk_size=0, front_latent_mode='full',
    ):
        super().__init__()
        if task_name not in TEMPORAL_TASKS:
            raise ValueError('this adapter supports only ns/plasticity')
        task = TEMPORAL_TASKS[task_name]
        fun_dim = task['fun_dim'] if fun_dim is None else fun_dim
        out_dim = task['out_dim'] if out_dim is None else out_dim
        unified_pos = task['unified_pos'] if unified_pos is None else unified_pos
        if type(space_dim) is not int or space_dim != 2 or fun_dim != task['fun_dim']:
            raise ValueError('temporal task space_dim/fun_dim violates its input contract')
        if out_dim != task['out_dim'] or Time_Input != task['time_input']:
            raise ValueError('temporal task output/time contract violated')
        if act != 'gelu' or type(ref) is not int or ref < 1:
            raise ValueError('temporal task activation/ref contract violated')
        if (H, W) != task['grid']:
            raise ValueError(f"{task_name} requires grid_shape={task['grid']}")
        self.config = CDLNOArchitectureConfig(
            task_name=task_name, L=n_layers, F=front_blocks,
            M=task['latents'] if slice_num is None else slice_num, d_model=n_hidden,
            num_heads=task['heads'] if n_head is None else n_head,
            ffn_ratio=mlp_ratio, latent_ffn_ratio=latent_ffn_ratio,
            cdpa_mode=cdpa_mode, attention_dropout=dropout, front_latent_mode=front_latent_mode,
            structured=True, grid_shape=(H, W), output_dim=out_dim,
        ).validate()
        self.task_name, self.space_dim, self.fun_dim = task_name, space_dim, fun_dim
        self.Time_Input, self.unified_pos, self.ref = Time_Input, bool(unified_pos), ref
        self.H, self.W, self.n_hidden = H, W, n_hidden
        stem_width = (ref * ref if self.unified_pos else space_dim) + fun_dim
        self.preprocess = nn.Sequential(
            nn.Linear(stem_width, 2 * n_hidden), nn.GELU(), nn.Linear(2 * n_hidden, n_hidden),
        )
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        if self.unified_pos:
            reference = _grid(ref, ref)
            points = _grid(H, W)
            pos = ((points[:, None, :] - reference[None, :, :]).square().sum(-1)).sqrt()
            self.register_buffer('pos', pos.unsqueeze(0), persistent=False)
        self.time_fc = None
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))
            _init_linear(self.time_fc[0]); _init_linear(self.time_fc[2])
        self.core = CDLNO(self.config, source_chunk_size=cdpa_source_chunk_size)

    def adapter_architecture(self):
        return dict(version='temporal-standard-v1', task_name=self.task_name,
                    space_dim=self.space_dim, fun_dim=self.fun_dim,
                    Time_Input=self.Time_Input, act='gelu', unified_pos=self.unified_pos,
                    ref=self.ref, stem_width=self.preprocess[0].in_features,
                    placeholder=False, position_rule='baseline-structured-index'
                    if self.unified_pos else 'baseline-structured-coordinates',
                    output_dim=self.config.output_dim, grid_shape=[self.H, self.W])

    def forward(self, x, fx, T=None):
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError('x must be [B,N,2]')
        if x.shape[1] != self.H * self.W or not x.is_floating_point():
            raise ValueError('x must preserve the original H*W layout')
        if not isinstance(fx, Tensor) or fx.shape != (*x.shape[:2], self.fun_dim):
            raise ValueError(f'expected fx[B,N,{self.fun_dim}]')
        if fx.device != x.device or fx.dtype != x.dtype:
            raise ValueError('x and fx must share device and floating dtype')
        if self.Time_Input:
            if not isinstance(T, Tensor) or T.shape != (x.shape[0], 1):
                raise ValueError('Plasticity requires T[B,1]')
            if T.device != x.device or not T.is_floating_point():
                raise ValueError('T must be floating-point on x device')
        elif T is not None:
            raise ValueError('NS does not use T; time is represented by rolling fx')
        position = self.pos.to(dtype=x.dtype).expand(x.shape[0], -1, -1) if self.unified_pos else x
        lifted = self.preprocess(torch.cat((position, fx), -1))
        if self.Time_Input:
            half = self.n_hidden // 2
            freqs = torch.exp(-torch.log(torch.tensor(10000., device=T.device)) *
                              torch.arange(half, device=T.device, dtype=T.dtype) / half)
            emb = T[:, :, None] * freqs[None, None, :]
            emb = torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)
            if self.n_hidden % 2:
                emb = torch.cat((emb, torch.zeros_like(emb[..., :1])), dim=-1)
            lifted = lifted + self.time_fc(emb).expand(-1, x.shape[1], -1)
        return self.core(lifted)
