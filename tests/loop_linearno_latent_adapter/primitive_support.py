"""Deterministic test tensors, tolerances and numerical-error evidence."""
import torch

ERRORS = []


def close(actual, expected, *, label, dtype):
    atol, rtol = (2e-12, 2e-11) if dtype == torch.float64 else (3e-6, 3e-5)
    ERRORS.append(dict(label=label, dtype=str(dtype), max_absolute=float((actual.detach()-expected.detach()).abs().max()),
                       atol=atol, rtol=rtol))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def values(shape, *, dtype=torch.float64, device='cpu', offset=0.):
    import math
    return (torch.arange(math.prod(shape), dtype=dtype, device=device).reshape(shape).sin() * .4 + offset)


def nonzero(module):
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            parameter.copy_(values(parameter.shape, dtype=parameter.dtype, device=parameter.device, offset=.07*(index+1)))
    return module


def params(module):
    return {name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()}
