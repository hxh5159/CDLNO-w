"""Validation shared by isolated feature constructors, without global seeding."""
import torch


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def feature_options(feature_seed, device, dtype):
    if type(feature_seed) is not int or not 0 <= feature_seed < 2**63:
        raise ValueError('feature_seed must be an integer in [0, 2**63), not bool')
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if dtype not in FLOAT_DTYPES:
        raise TypeError('feature dtype must be float16/bfloat16/float32/float64')
    device = torch.device('cpu' if device is None else device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('feature device must be CPU or CUDA')
    return device, dtype
