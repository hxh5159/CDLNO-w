"""Prevent accidental full-tree reinitialization after feature installation."""
from torch import nn


class InitializedWrapper(nn.Module):
    def apply(self, fn):
        if getattr(self, '_initialization_complete', False):
            raise RuntimeError('V4 full-tree initialization is already complete')
        return super().apply(fn)

