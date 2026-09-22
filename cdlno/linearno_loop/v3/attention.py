"""V3-only LinearNO attention visits; no loop, task wrapper, or activation cache.

Inherited construction uses actual per-head M (``rank``), including ShapeNet.
The old task-local ShapeNet constructors/validation are not changed. Construct
the common backbone and finish its release initialization BEFORE installing
features. Never recursively initialize the augmented tree afterwards.
"""
import torch

from cdlno.linearno.attention import LinearNOAttention
from .adapter import BilateralQKLowRankAdapter
from .latent import build_latent_context_ffn


class V3LinearNOAttention(LinearNOAttention):
    """Native forward by default; explicit zero-based round_index for the core.

    A core caller passes round_index=0/1 on each visit of the same instance.
    The adapter executes only at index 1. With no adapter, arbitrary nonnegative
    indices support custom repeat counts. Prefix/suffix retain native attention;
    calling without round_index always delegates to the original forward.
    """

    def install_features(self, *, latent_enabled=False, latent_width=None,
                         adapter_mode='none', adapter_rank=4, adapter_alpha=4.,
                         latent_seed=None, adapter_seed=None):
        """Install once after common initialization, without advancing its RNG.

        Disabled features have no registered modules or state keys. Build active
        modules locally first so invalid options cannot leave a partial install.
        Feature seeds are explicit and supplied by the resolved V3 config.
        """
        if getattr(self, '_features_installed', False):
            raise RuntimeError('V3 attention features already installed')
        if type(latent_enabled) is not bool:
            raise TypeError('latent_enabled must be bool')
        if adapter_mode not in ('none', 'bilateral_qk_lowrank_second_visit'):
            raise ValueError('unknown V3 adapter_mode')
        options = dict(device=self.to_q.weight.device, dtype=self.to_q.weight.dtype)
        processor = (build_latent_context_ffn(self.heads * self.dim_head, latent_width,
                     feature_seed=latent_seed, **options) if latent_enabled else None)
        adapter = (BilateralQKLowRankAdapter(self.dim_head, self.rank, adapter_rank,
                   adapter_alpha, feature_seed=adapter_seed, **options)
                   if adapter_mode != 'none' else None)
        if processor is not None:
            self.latent_processor = processor
        if adapter is not None:
            self.adapter = adapter
        self._features_installed = True

    def forward(self, x, *, round_index=None):
        if round_index is None:
            return super().forward(x)
        if type(round_index) is not int or round_index < 0:
            raise ValueError('round_index must be a nonnegative integer, not bool')
        if 'adapter' in self._modules and round_index > 1:
            raise ValueError('adapter-on supports only round_index 0/1 (R=2)')
        use_adapter = round_index == 1 and 'adapter' in self._modules
        use_latent = 'latent_processor' in self._modules
        if not use_adapter and not use_latent:
            return super().forward(x)

        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError('LinearNO attention requires a tensor [B,N,dim]')
        B, N, channels = x.shape
        if B < 1 or N < 1 or channels != self.dim:
            raise ValueError(f'LinearNO attention expected B,N > 0 and dim={self.dim}; got {tuple(x.shape)}')
        if not x.is_floating_point():
            raise TypeError('LinearNO attention input must be floating point')
        if self.variant in ('conv', 'conv_temp'):
            if N != self.H * self.W:
                raise ValueError(f'LinearNO conv requires N = H * W = {self.H} * {self.W}; got N={N}')
            grid = x.transpose(1, 2).reshape(B, channels, self.H, self.W)
            projected = self.in_project_x(grid)
            features = projected.reshape(B, self.heads, self.dim_head, N).transpose(-1, -2)
        else:
            projected = self.in_project_x(x)
            features = projected.reshape(B, N, self.heads, self.dim_head).transpose(1, 2)
            if self.variant == 'airfrans':
                features = features.contiguous()
        q_logits, k_logits, values = self.to_q(features), self.to_k(features), self.to_v(features)
        if use_adapter:
            delta_q, delta_k = self.adapter(features)
            q_logits = q_logits + delta_q
            k_logits = k_logits + delta_k
        if self.variant in ('temp', 'conv_temp'):
            q_logits = q_logits / self.temperature_q.clamp(0.01, 1.)
            k_logits = k_logits / self.temperature_k.clamp(0.01, 1.)
        elif self.variant == 'shapenet':
            q_logits = q_logits / self.tempreature_q.clamp(0.1, 2.)
            k_logits = k_logits / self.tempreature_k.clamp(0.1, 2.)
        queries = q_logits.softmax(dim=-1)
        keys = k_logits.softmax(dim=-2)
        context = torch.einsum('bhnm,bhnd->bhmd', keys, values)
        if use_latent:
            processed = self.latent_processor(context)
            if not isinstance(processed, torch.Tensor) or processed.shape != context.shape:
                raise ValueError('latent processor must preserve [B,h,M,d_h]')
            if processed.device != context.device or processed.dtype != context.dtype:
                raise ValueError('latent processor must preserve context device/dtype')
            context = processed
        readout = torch.einsum('bhnm,bhmd->bhnd', queries, context)
        merged = readout.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(merged)
