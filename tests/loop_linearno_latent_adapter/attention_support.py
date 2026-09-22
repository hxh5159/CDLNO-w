"""LAA3 fixtures and recorded comparisons; expected math lives in the oracle."""
import torch
from cdlno.linearno.attention import LinearNOAttention, initialize_release_weights

VARIANTS = ('plain', 'temp', 'conv', 'conv_temp', 'airfrans', 'shapenet')
ERRORS, CUDA_ROWS = [], []


def kwargs(variant, dropout=0., *, dim=6, heads=2, rank=5):
    result = dict(dim=dim, heads=heads, dim_head=dim//heads, rank=rank,
                  variant=variant, dropout=dropout)
    if variant in ('conv', 'conv_temp'):
        result.update(H=2, W=3)
    return result


def make(variant, latent=False, adapter=False, *, dtype=torch.float64,
         device='cpu', dropout=0., rank=5, seed=71):
    from cdlno.linearno_loop.v3.attention import V3LinearNOAttention
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        m = V3LinearNOAttention(**kwargs(variant, dropout, rank=rank))
        m.apply(initialize_release_weights)
    m.to(device=device, dtype=dtype)
    m.install_features(latent_enabled=latent, latent_width=7,
                       adapter_mode='bilateral_qk_lowrank_second_visit' if adapter else 'none',
                       adapter_rank=2, adapter_alpha=3., latent_seed=172, adapter_seed=273)
    return m


def fill(module):
    # Moderate, asymmetric, nonzero weights exercise temperature and both softmaxes.
    with torch.no_grad():
        for i, (name, p) in enumerate(module.named_parameters()):
            if name.startswith(('temperature', 'tempreature')):
                p.copy_(torch.linspace(.27, .73, p.numel(), device=p.device, dtype=p.dtype).reshape_as(p))
            else:
                p.copy_((torch.sin(torch.arange(p.numel(), device=p.device, dtype=p.dtype)*.73+i)*.19).reshape_as(p))
                if name.endswith('norm.weight'): p.add_(1.)
    return module


def sample(dtype=torch.float64, device='cpu', batch=2, points=6):
    t = torch.arange(batch*points*6, dtype=dtype, device=device)
    return (torch.sin(t*.37)*1.3).reshape(batch, points, 6)


def compare(a, b, label, *, atol=None, rtol=None):
    if atol is None: atol, rtol = (3e-12, 3e-10) if a.dtype==torch.float64 else (4e-6, 5e-5)
    ERRORS.append(dict(label=label, dtype=str(a.dtype), max_abs=float((a.detach().double()-b.detach().double()).abs().max()), atol=atol, rtol=rtol))
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, check_dtype=False)


def clone_parameters(module, dtype=None):
    return {k: p.detach().to(dtype=dtype or p.dtype).clone().requires_grad_() for k,p in module.named_parameters()}
