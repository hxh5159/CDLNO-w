"""Independent four-scale MSAR-LNO on already lifted point features.

Stable pickle path: cdlno.msar_lno.core.MSARLNO. Task lifts/coordinates/time and
factory integration remain outside this module. There is no cross-call state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..modules import _init_linear
from .config import MSARArchitectureConfig, MSARTrainingConfig, positive_int
from .diagnostics import observe_down, observe_fusion
from .modules import (LearnedQueryDown, LatentFFNSAFFNBlock, QueryAlignedUpCross,
                      PairwiseAttnResFusion, CoverageFloorLoss, _tokens)


@dataclass(frozen=True)
class MSARAuxOutput:
    """Explicit per-call output; no A/E/D tensors or module-owned loss cache.

    Raw coverage scalars are FP32, before multiplication by coverage_weight.
    Entries are ordered by encoder scale 1,2,3,4, regardless of decoder order.
    Off/eval zeros have no auxiliary graph. Diagnostics contain detached small
    per-sample reductions, with fusion entries ordered by scale 1,2,3.
    """
    prediction: Tensor
    coverage_per_level: Tensor  # [4]
    coverage_mean: Tensor       # scalar
    coverage_active: bool
    diagnostics: dict | None = None


class MSARLNO(nn.Module):
    """E0[B,N,d] -> prediction[B,N,output_dim], defaulting to Light / scalar output.

    The caller owns pointwise lifting and original task fields/time semantics.
    This core owns the final LayerNorm+Linear head. ``return_aux=True`` opts into
    training coverage and, when configured, detached diagnostics. Ordinary
    inference always returns a Tensor and never computes coverage/diagnostics.
    ``training_config`` overrides only this call; it never mutates saved config.
    """

    def __init__(self, config: MSARArchitectureConfig | None = None, *, output_dim: int = 1,
                 training_config: MSARTrainingConfig | None = None) -> None:
        super().__init__()
        self.config = MSARArchitectureConfig() if config is None else config
        if type(self.config) is not MSARArchitectureConfig:
            raise ValueError('config must be MSARArchitectureConfig')
        self.config.validate()
        self.training_config = MSARTrainingConfig() if training_config is None else training_config
        self._check_training(self.training_config)
        positive_int('output_dim', output_dim)
        self.output_dim = output_dim
        c = self.config
        self.downs = nn.ModuleList(LearnedQueryDown(c.d, h, m) for h, m in zip(c.heads, c.num_latents))
        self.encoders = nn.ModuleList(
            nn.Sequential(*(LatentFFNSAFFNBlock(c.d, h) for _ in range(depth)))
            for h, depth in zip(c.heads, c.encoder_depths))
        self.decoders = nn.ModuleList(
            nn.Sequential(*(LatentFFNSAFFNBlock(c.d, h) for _ in range(depth)))
            for h, depth in zip(c.heads, c.decoder_depths))
        # Index i addresses encoder scale i+1. Deepest scale has no Up/fusion.
        self.ups = nn.ModuleList(QueryAlignedUpCross(c.d, c.heads[i]) for i in range(3))
        self.fusions = nn.ModuleList(PairwiseAttnResFusion(c.d) for _ in range(3))
        self.final_up = QueryAlignedUpCross(c.d, c.heads[0])
        self.output_norm = nn.LayerNorm(c.d, eps=c.norm_eps)
        self.output = nn.Linear(c.d, output_dim)
        _init_linear(self.output)
        # Each primitive already initialized itself. Never recursively reset it.

    @staticmethod
    def _check_training(config: MSARTrainingConfig) -> None:
        if type(config) is not MSARTrainingConfig:
            raise ValueError('training_config must be MSARTrainingConfig')
        config.validate()

    def forward(self, e0: Tensor, *, return_aux: bool = False,
                training_config: MSARTrainingConfig | None = None,
                valid_mask: Tensor | None = None, source_measure: Tensor | None = None
                ) -> Tensor | MSARAuxOutput:
        _tokens(e0, self.config.d, 'e0 (already lifted point features)')
        if type(return_aux) is not bool:
            raise ValueError('return_aux must be bool')
        cfg = self.training_config if training_config is None else training_config
        self._check_training(cfg)
        need_coverage = bool(self.training and return_aux and cfg.coverage_enabled)
        need_diagnostics = bool(return_aux and cfg.diagnostics)
        loss = CoverageFloorLoss(cfg) if need_coverage else None
        per_level, down_stats = [], []
        fusion_stats = [None, None, None] if need_diagnostics else None
        encodings = []
        source = e0
        for level, (down, encoder) in enumerate(zip(self.downs, self.encoders)):
            # Only E0 has a task source mask/measure. All latent slots are valid.
            mask = valid_mask if level == 0 else None
            attention = None
            if need_coverage:
                compressed, attention = down(source, valid_mask=mask, return_aux=True, coverage=cfg)
                per_level.append(loss(attention, valid_mask=mask,
                                      source_measure=source_measure if level == 0 else None))
            else:
                compressed = down(source, valid_mask=mask)
            if need_diagnostics:
                down_stats.append(observe_down(down, source, valid_mask=mask, attention=attention))
            source = encoder(compressed)
            encodings.append(source)
        decoded = self.decoders[3](encodings[3])  # D4: no skip/fusion/extra processor.
        for level in (2, 1, 0):
            encoder = encodings[level]
            aligned = self.ups[level](encoder, decoded)
            fused = self.fusions[level](encoder, aligned)
            if need_diagnostics:
                fusion_stats[level] = observe_fusion(self.fusions[level], encoder, aligned)
            decoded = self.decoders[level](fused)
        # An already lifted FP32 E0 can enter an autocast region whose decoder
        # output is half precision. Match the projection input dtype, as the
        # caller's pointwise lift would under AMP; no learned transform/skip.
        u0 = self.final_up(e0.to(decoded.dtype), decoded)
        prediction = self.output(self.output_norm(u0))  # No E0 skip or point residual block.
        if not return_aux:
            return prediction
        coverage = torch.stack(per_level) if need_coverage else e0.new_zeros(4, dtype=torch.float32)
        diagnostics = dict(encoder=tuple(down_stats), fusion=tuple(fusion_stats)) if need_diagnostics else None
        return MSARAuxOutput(prediction, coverage, coverage.mean(), need_coverage, diagnostics)


__all__ = ['MSARLNO', 'MSARAuxOutput']
