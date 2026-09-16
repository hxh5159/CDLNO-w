"""Explicit, per-forward MSAR training objective; no trainer or mutable cache.

Call the original task loss on prediction (including its original decode/mask)
first, then combine it here. Task-specific rollout reductions stay in the task.
Logging returns detached scalar tensors, with no hidden host synchronization.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import MSARArchitectureConfig, MSARTrainingConfig
from .core import MSARAuxOutput


@dataclass(frozen=True)
class MSARTrainingForward:
    prediction: Tensor
    auxiliary: MSARAuxOutput | None
    training: MSARTrainingConfig


@dataclass(frozen=True)
class MSARLoss:
    pde: Tensor
    coverage_raw: Tensor
    coverage_weighted: Tensor
    total: Tensor
    coverage_active: bool

    def log_values(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).detach() for name in
                ('pde', 'coverage_raw', 'coverage_weighted', 'total')}


def training_forward(model, *inputs, training_config=None, **kwargs) -> MSARTrainingForward:
    """New-family only. Neither toggles train/eval nor mutates model config.

    Explicit aux/config arguments will also be the contract for future task
    wrappers; existing old-family forward calls must never call this adapter.
    """
    if type(getattr(model, 'config', None)) is not MSARArchitectureConfig:
        raise ValueError('MSAR training adapter requires an msar_lno model')
    if not model.training:
        raise ValueError('training_forward requires model.train(); eval uses model(...) directly')
    cfg = model.training_config if training_config is None else training_config
    if type(cfg) is not MSARTrainingConfig:
        raise ValueError('training_config must be MSARTrainingConfig')
    cfg.validate()
    if 'return_aux' in kwargs:
        raise ValueError('training_forward owns the explicit return_aux request')
    need_aux = cfg.coverage_enabled or cfg.diagnostics
    output = model(*inputs, return_aux=need_aux, training_config=cfg, **kwargs)
    if need_aux:
        if not isinstance(output, MSARAuxOutput) or output.coverage_active != cfg.coverage_enabled:
            raise ValueError('MSAR auxiliary output disagrees with the requested training objective')
        return MSARTrainingForward(output.prediction, output, cfg)
    if not isinstance(output, Tensor):
        raise ValueError('MSAR coverage-off forward must return prediction Tensor')
    return MSARTrainingForward(output, None, cfg)


def training_objective(pde_loss: Tensor, forward: MSARTrainingForward) -> MSARLoss:
    """LPDE + weight * mean_l(raw coverage_l); no rescaling of the task loss.

    Off/weight0 returns the EXACT pde_loss object, without an extra add/graph.
    The caller owns mean-over-time/member semantics; this is one forward only.
    """
    if not isinstance(forward, MSARTrainingForward):
        raise ValueError('explicit MSARTrainingForward required')
    if not isinstance(pde_loss, Tensor) or pde_loss.ndim != 0 or not pde_loss.is_floating_point():
        raise ValueError('original PDE loss must be a floating scalar tensor')
    if pde_loss.device != forward.prediction.device:
        raise ValueError('PDE loss and prediction must be on the same device')
    cfg = forward.training
    cfg.validate()
    if not cfg.coverage_enabled:
        zero = pde_loss.new_zeros((), dtype=torch.float32)
        return MSARLoss(pde_loss, zero, zero, pde_loss, False)
    aux = forward.auxiliary
    if (not isinstance(aux, MSARAuxOutput) or not aux.coverage_active
            or aux.prediction is not forward.prediction
            or aux.coverage_per_level.shape != (4,) or aux.coverage_mean.ndim != 0
            or aux.coverage_mean.dtype != torch.float32 or aux.coverage_mean.device != pde_loss.device):
        raise ValueError('active coverage requires this forward\'s four-level FP32 auxiliary output')
    raw = aux.coverage_mean
    weighted = cfg.coverage_weight * raw
    return MSARLoss(pde_loss, raw, weighted, pde_loss + weighted, True)


def rollout_objective(pde_loss: Tensor, forwards: tuple[MSARTrainingForward, ...]) -> MSARLoss:
    """One NS update: original SUM of PDE losses + weight * MEAN time coverage.

    Every item is an actual forward in this batch's teacher-forced loop. Reuse
    the single-call adapter's validation; only its raw coverage is aggregated.
    No time multiplier, PDE rescaling, detach, or model-owned accumulator.
    Off/weight0 returns the original PDE loss object exactly.
    """
    if not isinstance(forwards, tuple) or not forwards:
        raise ValueError('rollout requires a nonempty tuple of actual training forwards')
    if any(not isinstance(item, MSARTrainingForward) for item in forwards):
        raise ValueError('rollout requires MSARTrainingForward records')
    cfg = forwards[0].training
    if any(item.training != cfg for item in forwards):
        raise ValueError('all forwards in one rollout must use the same training objective')
    pieces = tuple(training_objective(pde_loss, item) for item in forwards)
    if not cfg.coverage_enabled:
        return pieces[0]
    raw = torch.stack([piece.coverage_raw for piece in pieces]).mean()
    weighted = cfg.coverage_weight * raw
    return MSARLoss(pde_loss, raw, weighted, pde_loss + weighted, True)


class ObjectiveMetrics:
    """One epoch's detached per-step objective means, never model-owned state.

    Original task train_loss keeps its original per-sample reduction separately.
    These four metrics use the same step denominator to preserve total=PDE+aux.
    """
    def __init__(self):
        self.totals = None
        self.steps = 0

    def add(self, loss: MSARLoss):
        values = torch.stack(tuple(loss.log_values().values()))
        self.totals = values if self.totals is None else self.totals + values
        self.steps += 1

    def values(self):
        if not self.steps:
            raise ValueError('cannot record an epoch with no training steps')
        means = (self.totals / self.steps).cpu().tolist()
        return dict(zip(('objective_pde', 'coverage_raw', 'coverage_weighted', 'objective_total'), means))
