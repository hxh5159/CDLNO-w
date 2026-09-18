"""Optional LinearNO propagation-kernel monitoring.

The monitor is intentionally isolated from model and benchmark code.  It is
activated only by :mod:`monitor.run` (or the documented environment
variables), and records the low-rank point kernel
``P_l = Q_l @ K_l.T`` without materializing an ``N x N`` tensor.
"""

from .kernels import kernel_cosine, pairwise_kernel_similarity

__all__ = ["kernel_cosine", "pairwise_kernel_similarity"]
