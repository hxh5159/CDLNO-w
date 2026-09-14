"""Shared, data-free CDLNO engineering primitives.

Configuration imports remain independent of torch. Building blocks are exposed
from ``cdlno.modules``; ``CDLNO`` is loaded lazily from ``cdlno.core``. Task
input lifting and experiment adapters belong to later phases.
"""

from .config import CDLNOArchitectureConfig, CDLNORuntimeConfig

__all__ = [
    "CDLNOArchitectureConfig",
    "CDLNORuntimeConfig",
    "CDLNO",
]


def __getattr__(name):
    if name == "CDLNO":
        from .core import CDLNO
        return CDLNO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
