"""Independent v4 attention oracles; no call to the tested forward."""
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from cdlno.linearno.attention import initialize_release_weights
from cdlno.linearno_loop.v4.attention import V4LinearNOAttention
from cdlno.linearno_loop.v4.temperature import multiplier


def _linear(x, layer):
    return F.linear(x, layer.weight, layer.bias)


def _oracle(module, x):
    batch, points, channels = x.shape
    if module.variant in ("conv", "conv_temp"):
        grid = x.transpose(1, 2).reshape(batch, channels, module.H, module.W)
        projected = F.conv2d(grid, module.in_project_x.weight, module.in_project_x.bias,
                             padding=module.in_project_x.padding)
        features = projected.reshape(batch, module.heads, module.dim_head, points).transpose(-1, -2)
    else:
        features = _linear(x, module.in_project_x).reshape(
            batch, points, module.heads, module.dim_head).transpose(1, 2)
        if module.variant == "airfrans":
            features = features.contiguous()
    q_logits = _linear(features, module.to_q)
    k_logits = _linear(features, module.to_k)
    values = _linear(features, module.to_v)
    q_delta = _linear(F.gelu(_linear(features, module.q_temperature.fc1), approximate="tanh"),
                      module.q_temperature.fc2)
    if module.temperature_mode == "latent_k_point_q":
        reduced = features.mean(dim=2)
        k_delta = _linear(F.gelu(_linear(reduced, module.k_temperature.fc1), approximate="tanh"),
                          module.k_temperature.fc2).unsqueeze(2)
    else:
        k_delta = _linear(F.gelu(_linear(features, module.k_temperature.fc1), approximate="tanh"),
                          module.k_temperature.fc2)
    if module.variant in ("temp", "conv_temp"):
        tau_q = module.temperature_q.clamp(.01, 1.)
        tau_k = module.temperature_k.clamp(.01, 1.)
    elif module.variant == "shapenet":
        tau_q = module.tempreature_q.clamp(.1, 2.)
        tau_k = module.tempreature_k.clamp(.1, 2.)
    else:
        tau_q = tau_k = 1.
    queries = (q_logits / (tau_q * torch.exp(math.log(2.) * torch.tanh(q_delta)))).softmax(-1)
    keys = (k_logits / (tau_k * torch.exp(math.log(2.) * torch.tanh(k_delta)))).softmax(-2)
    context = torch.einsum("bhnm,bhnd->bhmd", keys, values)
    merged = torch.einsum("bhnm,bhmd->bhnd", queries, context).transpose(1, 2).reshape(
        batch, points, module.heads * module.dim_head)
    first = _linear(merged, module.to_out[0])
    if len(module.to_out) == 4:
        return _linear(F.gelu(first), module.to_out[2])
    return first


@pytest.mark.parametrize("variant", ("plain", "temp", "conv", "conv_temp", "airfrans", "shapenet"))
@pytest.mark.parametrize("mode", ("latent_k_point_q", "point_k_point_q"))
def test_six_variant_nonzero_temperature_oracle(variant, mode):
    kwargs = dict(heads=2, dim_head=4, rank=3, variant=variant, dropout=0.)
    points = 6 if variant in ("conv", "conv_temp") else 7
    if variant in ("conv", "conv_temp"):
        kwargs.update(H=2, W=3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        module = V4LinearNOAttention(8, **kwargs).double()
        module.apply(initialize_release_weights)
    module._planned_temperature_mode = mode
    module.install_temperature_predictors(mode=mode, public_seed=17, logical_depth=3)
    module.double()
    with torch.no_grad():
        module.q_temperature.fc2.weight.copy_(torch.linspace(-.3, .2, module.q_temperature.fc2.weight.numel()).reshape_as(module.q_temperature.fc2.weight))
        module.q_temperature.fc2.bias.fill_(.07)
        module.k_temperature.fc2.weight.copy_(torch.linspace(.2, -.25, module.k_temperature.fc2.weight.numel()).reshape_as(module.k_temperature.fc2.weight))
        module.k_temperature.fc2.bias.fill_(-.03)
    x = torch.linspace(-.8, .9, 2 * points * 8, dtype=torch.float64).reshape(2, points, 8)
    torch.testing.assert_close(module(x, logical_depth=3), _oracle(module, x), atol=1e-12, rtol=1e-11)
    features = module._features(x)
    assert module.q_temperature(features).shape == (2, 2, points, 1)
    expected_k = (2, 2, 1, 3) if mode == "latent_k_point_q" else (2, 2, points, 1)
    assert module.k_temperature(features).shape == expected_k
    assert torch.all((multiplier(module.q_temperature(features)) > .5) &
                     (multiplier(module.q_temperature(features)) < 2.))


def test_temperature_axes_ordering_and_no_quadratic_attention():
    raw = torch.tensor([[[[3., -2.], [2., 2.], [1., 4.]]]])
    latent_tau = torch.tensor([[[[.5, 1.8]]]])
    assert torch.equal(raw.argsort(dim=2), (raw / latent_tau).argsort(dim=2))
    point_tau = torch.tensor([[[[2.], [.5], [.5]]]])
    assert not torch.equal(raw.argsort(dim=2), (raw / point_tau).argsort(dim=2))
    q = torch.tensor([[[[3., 1., -1.]]]])
    assert torch.equal(q.argsort(-1), (q / torch.tensor([[[[1.7]]]])).argsort(-1))

    module = V4LinearNOAttention(8, heads=2, dim_head=4, rank=3, variant="plain")
    module.apply(initialize_release_weights); module._planned_temperature_mode = "point_k_point_q"
    module.install_temperature_predictors(mode="point_k_point_q", public_seed=9, logical_depth=0)
    shapes = []
    original = torch.einsum
    def record(equation, *operands):
        result = original(equation, *operands); shapes.append(tuple(result.shape)); return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "einsum", record)
        module(torch.randn(2, 11, 8))
    assert shapes == [(2, 2, 3, 4), (2, 2, 11, 4)]
    assert not any(shape[-2:] in ((11, 11), (3, 3)) for shape in shapes)


def test_predictor_seed_isolation_and_independence():
    def built(mode):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            module = V4LinearNOAttention(8, heads=2, dim_head=4, rank=5, variant="plain")
            module.apply(initialize_release_weights)
        module._planned_temperature_mode = mode
        module.install_temperature_predictors(mode=mode, public_seed=29, logical_depth=4)
        return module
    before = torch.get_rng_state().clone(); latent = built("latent_k_point_q")
    point = built("point_k_point_q"); assert torch.equal(before, torch.get_rng_state())
    for key, value in latent.q_temperature.state_dict().items():
        assert torch.equal(value, point.q_temperature.state_dict()[key])
    assert not torch.equal(latent.k_temperature.fc1.weight, point.k_temperature.fc1.weight)
    assert id(latent.q_temperature) != id(latent.k_temperature)
    assert torch.count_nonzero(latent.q_temperature.fc2.weight) == 0
    assert torch.count_nonzero(latent.k_temperature.fc2.bias) == 0
    source = inspect.getsource(V4LinearNOAttention.forward)
    assert "gumbel" not in source.lower() and "N,N" not in source and "M,M" not in source
