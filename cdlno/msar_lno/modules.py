"""MSAR-LNO M2 primitives only: no core, task adapters, or training loop.

Attention uses the accepted LRSA bias/QK-norm/initialization rules. Coverage
weights are local to an explicitly requested training call, never module state.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..modules import PlainFFN, RMSNorm, _ProjectedAttention, _SelfAttention, _init_linear
from .config import MSARTrainingConfig, positive_int

NORM_EPS = 1e-6


def _dimensions(dim: int, heads: int) -> None:
    positive_int('dim', dim)
    positive_int('heads', heads)
    if dim % heads:
        raise ValueError('dim must be divisible by heads')


def _tokens(x: Tensor, dim: int, name: str) -> None:
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise ValueError(f'{name} must be a floating tensor')
    if x.ndim != 3 or x.shape[-1] != dim or min(x.shape[:2]) < 1:
        raise ValueError(f'{name} must be nonempty [B,N,{dim}]')


def _mask(mask: Tensor | None, batch: int, tokens: int, device) -> Tensor | None:
    if mask is None:
        return None
    if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
            or mask.shape != (batch, tokens) or mask.device != device):
        raise ValueError('valid_mask must be bool[B,Nsource] on the input device')
    # An empty source makes softmax undefined. Only an explicitly supplied mask
    # incurs this validation; the ordinary unmasked path has no host reduction.
    if not bool(mask.any(-1).all()):
        raise ValueError('every sample must have at least one valid source token')
    return mask


def _heads(x: Tensor, heads: int) -> Tensor:
    return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)


class LearnedQueryDown(nn.Module):
    """Direct learned Q[M,d], with no Wq and no learned-query residual.

    The default result is a Tensor. ``return_aux=True`` returns (output, A),
    where A is None unless training AND coverage.coverage_enabled. No weights
    are retained on self. In particular eval/off/weight=0 never computes A.
    Projections act on the supplied source; no hidden source pre-norm is added.
    """

    def __init__(self, dim: int, heads: int, num_latents: int) -> None:
        super().__init__()
        _dimensions(dim, heads)
        positive_int('num_latents', num_latents)
        self.dim, self.heads, self.num_latents = dim, heads, num_latents
        self.head_dim = dim // heads
        self.latent_queries = nn.Parameter(torch.empty(num_latents, dim))
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim, NORM_EPS)
        self.k_norm = RMSNorm(self.head_dim, NORM_EPS)
        for layer in (self.to_k, self.to_v, self.to_out):
            _init_linear(layer)
        nn.init.orthogonal_(self.latent_queries)

    def forward(self, source: Tensor, *, valid_mask: Tensor | None = None,
                return_aux: bool = False, coverage: MSARTrainingConfig | None = None):
        _tokens(source, self.dim, 'source')
        if type(return_aux) is not bool:
            raise ValueError('return_aux must be bool')
        if coverage is not None and type(coverage) is not MSARTrainingConfig:
            raise ValueError('coverage must be MSARTrainingConfig or None')
        if coverage is not None:
            coverage.validate()
        batch, n, _ = source.shape
        valid = _mask(valid_mask, batch, n, source.device)
        # Invalid padding does not supply keys or values (and gets zero gradient).
        work = source if valid is None else source.masked_fill(~valid[..., None], 0)
        k = self.k_norm(_heads(self.to_k(work), self.heads))
        v = _heads(self.to_v(work), self.heads)
        q = self.latent_queries.reshape(self.num_latents, self.heads, self.head_dim)
        q = self.q_norm(q.permute(1, 0, 2).unsqueeze(0)).to(k.dtype).expand(batch, -1, -1, -1)
        need_weights = self.training and return_aux and coverage is not None and coverage.coverage_enabled
        attention = None
        if need_weights:
            # Accumulate scores/softmax/AV in FP32 under AMP; preserve double for
            # mathematical verification. Coverage itself always computes FP32.
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=source.device.type, enabled=False):
                logits = (q.to(dtype) @ k.to(dtype).transpose(-1, -2)) * self.head_dim ** -0.5
                if valid is not None:
                    logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
                attention = logits.softmax(-1)
                y = (attention @ v.to(dtype)).to(v.dtype)
        else:
            y = F.scaled_dot_product_attention(q, k, v,
                attn_mask=None if valid is None else valid[:, None, None, :],
                dropout_p=0.0, is_causal=False, scale=self.head_dim ** -0.5)
        y = y.transpose(1, 2).reshape(batch, self.num_latents, self.dim)
        output = self.to_out(y)
        return (output, attention) if return_aux else output


class LatentFFNSAFFNBlock(nn.Module):
    """Three independent pre-RMSNorm residual branches: FFN1, SA, FFN2."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        _dimensions(dim, heads)
        self.dim = dim
        self.norm1 = RMSNorm(dim, NORM_EPS)
        self.ffn1 = PlainFFN(dim, 2)
        self.norm_sa = RMSNorm(dim, NORM_EPS)
        self.sa = _SelfAttention(dim, heads, qk_norm=True)
        self.norm2 = RMSNorm(dim, NORM_EPS)
        self.ffn2 = PlainFFN(dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        _tokens(x, self.dim, 'latent')
        x1 = x + self.ffn1(self.norm1(x))
        x2 = x1 + self.sa(self.norm_sa(x1))
        return x2 + self.ffn2(self.norm2(x2))


class QueryAlignedUpCross(_ProjectedAttention):
    """Pure receiver-head cross read; no query residual, pre-norm, or FFN.

    Independent Q/K/V/O projections and per-head QK RMSNorm are inherited from
    the unchanged, verified shared primitive. Q rows address the target slots.
    """

    def __init__(self, dim: int, heads: int) -> None:
        _dimensions(dim, heads)
        super().__init__(dim, heads, qk_norm=True)

    def forward(self, current_encoder: Tensor, deeper_decoder: Tensor) -> Tensor:
        _tokens(current_encoder, self.dim, 'current_encoder')
        _tokens(deeper_decoder, self.dim, 'deeper_decoder')
        if (current_encoder.dtype != deeper_decoder.dtype
                or current_encoder.device != deeper_decoder.device):
            raise ValueError('encoder/decoder must share dtype and device')
        return super().forward(current_encoder, deeper_decoder)


class PairwiseAttnResFusion(nn.Module):
    """Two aligned sources, raw values, fixed factor 2; w is the only parameter."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        positive_int('dim', dim)
        self.dim = dim
        self.w = nn.Parameter(torch.zeros(dim))

    def forward(self, encoder: Tensor, up: Tensor) -> Tensor:
        _tokens(encoder, self.dim, 'encoder')
        _tokens(up, self.dim, 'up')
        if encoder.shape != up.shape or encoder.dtype != up.dtype or encoder.device != up.device:
            raise ValueError('encoder/up must share shape, dtype and device (aligned slots)')
        dtype = torch.float64 if encoder.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=encoder.device.type, enabled=False):
            raw = torch.stack((encoder, up), dim=-2).to(dtype)  # [B,M,source=2,d]
            keys = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + NORM_EPS)
            score = (keys * self.w.to(dtype)).sum(-1)
            alpha = score.softmax(-1)
            fused = 2 * (alpha[..., None] * raw).sum(-2)
        return fused.to(encoder.dtype)


def _attention_shape(attention: Tensor) -> tuple[int, int]:
    if (not isinstance(attention, Tensor) or not attention.is_floating_point()
            or attention.ndim != 4 or min(attention.shape) < 1):
        raise ValueError('attention must be nonempty floating [B,H,M,Nsource]')
    return attention.shape[0], attention.shape[-1]


class CoverageFloorLoss(nn.Module):
    """Parameter-free raw coverage; source SUM, batch MEAN, then layer MEAN.

    The caller applies coverage_weight exactly once to this raw loss. The weight
    is used here only to select the true off path. No attention is needed off;
    passing None returns a CPU FP32 scalar zero, without constructing a graph.
    Explicit source_measure[B,N] is optional nonnegative integration mass; no
    dataset weights are inferred. It is masked and normalized independently per B.
    """

    def __init__(self, config: MSARTrainingConfig | None = None) -> None:
        super().__init__()
        self.config = MSARTrainingConfig() if config is None else config
        if type(self.config) is not MSARTrainingConfig:
            raise ValueError('config must be MSARTrainingConfig')
        self.config.validate()

    def forward(self, attention: Tensor | None, *, valid_mask: Tensor | None = None,
                source_measure: Tensor | None = None) -> Tensor:
        if not self.config.coverage_enabled:
            return torch.zeros((), dtype=torch.float32,
                               device=attention.device if isinstance(attention, Tensor) else None)
        batch, n = _attention_shape(attention)
        valid = _mask(valid_mask, batch, n, attention.device)
        if valid is None:
            valid = torch.ones((batch, n), device=attention.device, dtype=torch.bool)
        with torch.autocast(device_type=attention.device.type, enabled=False):
            p = attention.float().masked_fill(~valid[:, None, None], 0).mean(1).mean(1)
            if source_measure is None:
                mass = valid.float()
            else:
                if (not isinstance(source_measure, Tensor) or not source_measure.is_floating_point()
                        or source_measure.shape != (batch, n) or source_measure.device != attention.device):
                    raise ValueError('source_measure must be floating[B,Nsource] on attention device')
                mass = source_measure.float().masked_fill(~valid, 0)
                if not bool((torch.isfinite(mass) & (mass >= 0)).all()):
                    raise ValueError('valid source_measure entries must be finite and nonnegative')
                maximum = mass.amax(-1, keepdim=True)
                if not bool((maximum > 0).all()):
                    raise ValueError('every source_measure sample needs positive valid mass')
                # Same normalized measure; avoid overflow in a sum of large,
                # individually finite FP32 integration weights.
                mass = mass / maximum
            mu = mass / mass.sum(-1, keepdim=True)
            terms = F.relu(self.config.coverage_kappa * mu - p).square() / (mu + self.config.coverage_eps)
            return terms.masked_fill(~valid, 0).sum(-1).mean()

    def mean_layers(self, attentions: Sequence[Tensor], *,
                    valid_masks: Sequence[Tensor | None] | None = None,
                    source_measures: Sequence[Tensor | None] | None = None) -> Tensor:
        if not isinstance(attentions, (tuple, list)) or not attentions:
            raise ValueError('attentions must be a nonempty sequence of layer tensors')
        masks = [None] * len(attentions) if valid_masks is None else valid_masks
        measures = [None] * len(attentions) if source_measures is None else source_measures
        if len(masks) != len(attentions) or len(measures) != len(attentions):
            raise ValueError('layer masks/measures must match the number of attentions')
        return torch.stack([self(a, valid_mask=m, source_measure=mu)
                            for a, m, mu in zip(attentions, masks, measures)]).mean()


@torch.no_grad()
def coverage_diagnostics(attention: Tensor, *, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
    """Explicit no-grad diagnostics only; no invocation from default forwards.

    CoverageRatio=exp(H(p))/Nvalid. DiversityRatio=erank(Abar Abar^T)/M,
    using normalized nonnegative Gram eigenvalues to define effective rank.
    Returns per-sample device tensors; no CPU statistics or retained module state.
    """
    batch, n = _attention_shape(attention)
    valid = _mask(valid_mask, batch, n, attention.device)
    with torch.autocast(device_type=attention.device.type, enabled=False):
        a = attention.float().mean(1)
        if valid is not None:
            a = a.masked_fill(~valid[:, None], 0)
        p = a.mean(1)
        entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
        count = n if valid is None else valid.sum(-1)
        eigenvalues = torch.linalg.eigvalsh(a @ a.transpose(-1, -2)).clamp_min(0)
        spectral = eigenvalues / eigenvalues.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        spectral_entropy = -(spectral * spectral.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
        return dict(coverage_ratio=entropy.exp() / count,
                    diversity_ratio=spectral_entropy.exp() / a.shape[1])


__all__ = ['LearnedQueryDown', 'LatentFFNSAFFNBlock', 'QueryAlignedUpCross',
           'PairwiseAttnResFusion', 'CoverageFloorLoss', 'coverage_diagnostics']
