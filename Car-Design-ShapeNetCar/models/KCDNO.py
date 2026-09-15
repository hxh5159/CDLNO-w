"""Stable ShapeNet-Car KCDNO class; reuse the accepted single-graph validator."""
import torch
from torch import Tensor, nn
from models.CDLNO import Model as CDLNOModel, adapter_architecture
from cdlno.kcdno.matched import make_core
from cdlno.modules import _init_linear


class Model(CDLNOModel):
    def __init__(self, *, config):
        nn.Module.__init__(self)  # Do not construct/reinitialize an old core.
        config.validate()
        if config.point_module != 'point_ffn':
            raise ValueError('ShapeNet-Car requires point_ffn')
        self.config = config
        self.preprocess = nn.Sequential(nn.Linear(7, 2*config.d), nn.GELU(), nn.Linear(2*config.d, config.d))
        _init_linear(self.preprocess[0]); _init_linear(self.preprocess[2])
        self.placeholder = nn.Parameter(torch.rand(config.d) / config.d)
        self.core = make_core(config)
        self.output_norm = nn.LayerNorm(config.d, eps=config.norm_eps)
        self.output = nn.Linear(config.d, 4)
        _init_linear(self.output)

    def adapter_architecture(self):
        return adapter_architecture() | {'version':'kcdno-shapenet-car-v1', 'output_head':'layernorm-linear-v1'}

    def forward(self, data):
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise TypeError('ShapeNet-Car requires (cfd_data, geom_data)')
        cfd, _ = data
        x = getattr(cfd, 'x', None)
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[-1] != 7 or x.shape[0] < 1 or not x.is_floating_point():
            raise ValueError('cfd_data.x must be nonempty floating [N,7]')
        self._single_graph(cfd, x.shape[0])
        h = self.preprocess(x.unsqueeze(0)) + self.placeholder[None, None]
        return self.output(self.output_norm(self.core(h))).squeeze(0)
