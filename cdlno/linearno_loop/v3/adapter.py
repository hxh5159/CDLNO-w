"""Head-shared, pointwise bilateral Q/K low-rank logit increments.

This primitive has no visit counter: the future attention controller decides
when to call it. It never applies a temperature, softmax, residual, or 1/R.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .initialization import FLOAT_DTYPES, feature_options


class BilateralQKLowRankAdapter(nn.Module):
    """Return two [B,h,N,M] delta logits from projected features [B,h,N,d_h].

Q/K have independent A[r,d_h], B[M,r]; neither matrix has a head dimension.
Construction uses a private CPU generator and then moves the parameters, so
it consumes no public CPU/CUDA/Python/NumPy RNG. All forward tensors are local.

Autocast follows the two F.linear operations just as for base Q/K logits;
outside autocast the input and parameters must have the same floating dtype.
Nonfinite tensor values propagate through ordinary algebra: no sanitization,
detach, zero shortcut, or per-forward GPU synchronization is introduced.
"""

    def __init__(self, head_dim, latent_tokens, rank=4, alpha=4., *,
                 feature_seed, device=None, dtype=None):
        super().__init__()
        for name, value in (('head_dim', head_dim), ('latent_tokens', latent_tokens), ('rank', rank)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer, not bool')
        if type(alpha) not in (int, float):
            raise TypeError('alpha must be a finite positive number, not bool')
        try:
            alpha = float(alpha)
        except OverflowError as error:
            raise ValueError('alpha must be finite') from error
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        target_device, target_dtype = feature_options(feature_seed, device, dtype)
        self.head_dim, self.latent_tokens, self.rank = head_dim, latent_tokens, rank
        self.alpha, self.scale, self.feature_seed = alpha, alpha / rank, feature_seed
        generator = torch.Generator(device='cpu').manual_seed(feature_seed)
        for side in ('q', 'k'):
            a = torch.empty(rank, head_dim, device='cpu', dtype=target_dtype)
            b = torch.zeros(latent_tokens, rank, device='cpu', dtype=target_dtype)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            self.register_parameter('A_' + side, nn.Parameter(a.to(target_device)))
            self.register_parameter('B_' + side, nn.Parameter(b.to(target_device)))

    def _validate(self, features):
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError('adapter features must have shape [B,h,N,d_h]')
        if min(features.shape) < 1 or features.shape[-1] != self.head_dim:
            raise ValueError(f'adapter requires positive B/h/N and d_h={self.head_dim}')
        if features.layout != torch.strided:
            raise ValueError('adapter features require dense strided layout')
        if features.dtype not in FLOAT_DTYPES:
            raise TypeError('adapter features must be float16/bfloat16/float32/float64')
        parameters = (self.A_q, self.B_q, self.A_k, self.B_k)
        if any(p.device != features.device for p in parameters):
            raise ValueError('adapter parameters and features must be on the same device')
        if self.A_q.dtype not in FLOAT_DTYPES or any(p.dtype != self.A_q.dtype for p in parameters):
            raise TypeError('all adapter parameters must have the same supported floating dtype')
        if features.dtype != self.A_q.dtype:
            # Float64 is never downcast by autocast. A mixed-double operation
            # cannot be made valid merely by enabling a low-precision context.
            if (not torch.is_autocast_enabled(features.device.type)
                    or torch.float64 in (features.dtype, self.A_q.dtype)):
                raise TypeError('adapter feature/parameter dtype mismatch outside compatible autocast')

    def forward(self, features):
        self._validate(features)
        delta_q = F.linear(F.linear(features, self.A_q), self.B_q) * self.scale
        delta_k = F.linear(F.linear(features, self.A_k), self.B_k) * self.scale
        return delta_q, delta_k
