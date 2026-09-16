"""Stable single-graph MSAR wrappers; no loader, graph construction or losses."""
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor, nn

from ..airfrans import _single_graph as sampled_single_graph
from ..modules import _init_linear
from .config import MSARArchitectureConfig, MSARTrainingConfig
from .core import MSARLNO, MSARAuxOutput


def adapter_architecture(task):
    common = dict(input_channels=7, output_channels=4, placeholder=True, Time_Input=False,
                  lift='pointwise-linear-gelu-linear-2d-v1', output_head='core-layernorm-linear-v1')
    if task == 'car':
        return common | dict(version='msar-shapenet-car-v1', stem_width=7,
            input_order=['xyz3', 'sdf1', 'normal3'], output_order=['velocity3', 'pressure1'],
            unified_pos=False, geometry_encoder=False, batch_policy='single-graph')
    if task == 'airfrans':
        return common | dict(version='msar-airfrans-v1', stem_width=71,
            input_order=['xy2', 'Uinf2', 'sdf1', 'normal2'], output_order=['vx', 'vy', 'p', 'nut'],
            ref=8, reference_domain=[[-2., 4.], [-1.5, 1.5]],
            reference_rule='append-distances-from-data.pos', batch_policy='single-graph-sampled-ptr-v1')
    raise ValueError('industrial MSAR task must be car or airfrans')


class _GraphModel(nn.Module):
    def __init__(self, *, config, training_config=None):
        super().__init__()
        if type(config) is not MSARArchitectureConfig:
            raise ValueError('MSARArchitectureConfig required')
        config.validate()
        cfg = MSARTrainingConfig() if training_config is None else training_config
        if type(cfg) is not MSARTrainingConfig:
            raise ValueError('MSARTrainingConfig required')
        cfg.validate()
        self.config, self.training_config = config, cfg
        d = config.d
        self.preprocess = nn.Sequential(nn.Linear(self.adapter_architecture()['stem_width'], 2*d),
                                        nn.GELU(), nn.Linear(2*d, d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        self.placeholder = nn.Parameter(torch.rand(d) / d)
        if self.task_name == 'airfrans':
            gx = torch.tensor(np.linspace(-2, 4, 8), dtype=torch.float32)
            gy = torch.tensor(np.linspace(-1.5, 1.5, 8), dtype=torch.float32)
            grid = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), -1).reshape(64, 2)
            self.register_buffer('reference', grid, persistent=False)
        self.core = MSARLNO(config, output_dim=4, training_config=cfg)

    def adapter_architecture(self):
        return adapter_architecture(self.task_name)

    def forward(self, data, *, return_aux=False, training_config=None):
        if self.task_name == 'car':
            if not isinstance(data, (tuple, list)) or len(data) != 2:
                raise TypeError('MSAR ShapeNet-Car requires (cfd_data, geom_data)')
            data, _ = data  # Preserve loader geometry work; baseline model ignores it.
        x = getattr(data, 'x', None)
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] != 7 or x.shape[0] < 1 or not x.is_floating_point():
            raise ValueError('MSAR industrial data.x must be nonempty floating [N,7]')
        # Existing accepted Air sampled-ptr rule; Car additionally requires exact N.
        sampled_single_graph(data, x.shape[0])
        ptr = getattr(data, 'ptr', None)
        if self.task_name == 'car' and ptr is not None and ptr[1].item() != x.shape[0]:
            raise ValueError('MSAR Car single-graph ptr must be [0,N]')
        features = x
        if self.task_name == 'airfrans':
            pos = getattr(data, 'pos', None)
            if not isinstance(pos, Tensor) or pos.shape != (x.shape[0], 2):
                raise ValueError('MSAR AirfRANS data.pos must be [N,2]')
            if pos.dtype != x.dtype or pos.device != x.device:
                raise ValueError('data.pos and data.x must share floating dtype/device')
            distances = (pos[:, None] - self.reference.to(dtype=pos.dtype)[None]).square().sum(-1).sqrt()
            features = torch.cat((x, distances), -1)
        lifted = self.preprocess(features.unsqueeze(0)) + self.placeholder[None, None]
        output = self.core(lifted, return_aux=return_aux, training_config=training_config)
        # Explicit per-call aux, keeping prediction identity within its record.
        if isinstance(output, MSARAuxOutput):
            return replace(output, prediction=output.prediction.squeeze(0))
        return output.squeeze(0)


class CarModel(_GraphModel):
    task_name = 'car'


class AirfRANSModel(_GraphModel):
    task_name = 'airfrans'
