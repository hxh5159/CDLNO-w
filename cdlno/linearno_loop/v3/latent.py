"""Seed-isolated factory for the unchanged, audited V2 latent FFN primitive.

The returned object is exactly LatentContextFFN, with identical state keys and
forward. No wrapper module, new math, dropout, or extra persistent state.
"""
import torch

from cdlno.linearno_loop.v2.latent import LatentContextFFN
from .initialization import feature_options


def build_latent_context_ffn(hidden, inner_width, *, feature_seed, device=None, dtype=None):
    """Initialize once, then cast/move; install only after public-tree init.

As in V2, release initialization is performed in the CPU default dtype before
the caller's requested conversion. Seeding only the CPU default generator in
a restoring scope also leaves every CUDA generator unchanged. CPU allocation
is explicit even if the caller has selected CUDA as the default device.
"""
    target_device, target_dtype = feature_options(feature_seed, device, dtype)
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(feature_seed)
        module = LatentContextFFN(hidden, inner_width)
        module.initialize_release_identity()
    return module.to(device=target_device, dtype=target_dtype)
