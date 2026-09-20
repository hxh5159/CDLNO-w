"""Receiver-specific point-domain Attention Residuals, frozen formula v1."""
from collections.abc import Sequence

import torch
from torch import nn


class PointDepthAttnRes(nn.Module):
    """Mix raw [B,N,hidden] sources independently at each aligned point.

    Exactly two hidden-vectors: query (zero) and RMS scale (one). No source,
    channel or point projections, residual addition, dropout, or persistent
    history. FP64 stays FP64; low-precision sources accumulate in FP32 and
    return to their original dtype. source_weights is an explicit diagnostic
    recomputation, returning [S,B,N]; normal forward returns only the output.
    """
    eps = 1e-6

    def __init__(self, hidden):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError('hidden must be a positive integer, not bool')
        self.hidden = hidden
        self.query = nn.Parameter(torch.zeros(hidden))
        self.norm_scale = nn.Parameter(torch.ones(hidden))

    def _validate(self, sources):
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not len(sources):
            raise ValueError('sources must be a nonempty sequence of [B,N,hidden] tensors')
        first = sources[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 3:
            raise ValueError('source[0] must be a tensor [B,N,hidden]')
        if first.shape[0] < 1 or first.shape[1] < 1 or first.shape[2] != self.hidden:
            raise ValueError(f'source[0] requires B,N>0 and hidden={self.hidden}; got {tuple(first.shape)}')
        if first.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError('source dtype must be float16/bfloat16/float32/float64')
        for index, value in enumerate(sources):
            if not isinstance(value, torch.Tensor) or value.shape != first.shape:
                raise ValueError(f'source[{index}] shape must match [B,N,hidden]={tuple(first.shape)}')
            if value.device != first.device:
                raise ValueError(f'source[{index}] device must match {first.device}')
            if value.dtype != first.dtype:
                raise TypeError(f'source[{index}] dtype must match {first.dtype}')
        if self.query.device != first.device or self.norm_scale.device != first.device:
            raise ValueError('receiver parameters and sources must be on the same device')
        if self.query.dtype != self.norm_scale.dtype:
            raise TypeError('query and norm_scale dtype must match')
        if not (self.query.dtype == first.dtype or
                self.query.dtype == torch.float32 and first.dtype in (torch.float16, torch.bfloat16)):
            raise TypeError('receiver dtype must match sources, or be FP32 for low-precision sources')
        return first

    def _weights_and_values(self, sources):
        # Caller validates first. All tensors are local; no module-side cache.
        first = sources[0]
        dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
        values = torch.stack(tuple(sources), dim=0).to(dtype=dtype)  # [S,B,N,H]
        keys = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        scores = (keys * self.norm_scale.to(dtype) * self.query.to(dtype)).sum(dim=-1)
        return scores.softmax(dim=0), values

    def source_weights(self, sources):
        first = self._validate(sources)
        with torch.autocast(device_type=first.device.type, enabled=False):
            weights, _ = self._weights_and_values(sources)
        return weights

    def forward(self, sources):
        first = self._validate(sources)
        if len(sources) == 1:
            # Singleton softmax is exactly one, and both router gradients are
            # structurally zero. Preserve identity and the source's live graph.
            return first
        with torch.autocast(device_type=first.device.type, enabled=False):
            weights, values = self._weights_and_values(sources)
            output = (weights.unsqueeze(-1) * values).sum(dim=0)
        return output.to(dtype=first.dtype)
