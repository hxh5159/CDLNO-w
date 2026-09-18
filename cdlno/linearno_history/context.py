"""Immutable, graph-preserving history, owned by one model forward call."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RawHistoryContext:
    """Only pre-AttnRes summaries are authoritative; never a module attribute.

    Tuple snapshots share tensor references (no clone/detach/in-place update).
    Autograd may retain tensors needed by the prediction; this Python context
    is never part of model state or a checkpoint. Observers are test-only callers
    responsible for releasing any references they explicitly retain.
    """

    raw: tuple[torch.Tensor, ...] = ()

    def __post_init__(self):
        if type(self.raw) is not tuple:
            raise TypeError('raw history must be a tuple')
        first = None
        for value in self.raw:
            if not isinstance(value, torch.Tensor) or value.ndim != 4 or not value.is_floating_point():
                raise ValueError('raw summary must be floating [B,H,M,d_h]')
            if min(value.shape) < 1:
                raise ValueError('raw summary axes must be nonempty')
            if first is not None:
                if (value.shape[0], value.shape[1], value.shape[3], value.device, value.dtype) != first:
                    raise ValueError('history B/H/d_h/device/dtype must stay constant within one forward')
            first = (value.shape[0], value.shape[1], value.shape[3], value.device, value.dtype)

    def before_block(self, index: int) -> tuple[torch.Tensor, ...]:
        if type(index) is not int or index != len(self.raw):
            raise ValueError('block index must equal the number of completed raw summaries')
        return self.raw

    def after_block(self, index: int, raw: torch.Tensor) -> RawHistoryContext:
        self.before_block(index)
        return RawHistoryContext(self.raw + (raw,))

    def __reduce_ex__(self, protocol):
        raise TypeError('forward-local raw history cannot be serialized')
