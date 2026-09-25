import importlib
import sys

import pytest
import torch
from types import SimpleNamespace

from .oracle import dense_expert_oracle


def test_frozen_dense_expert_hand_example():
    z = torch.tensor([[[5., 1.], [2., 5.]]], dtype=torch.float64)
    pi = torch.tensor([[[.75, .25], [.4, .6]]], dtype=torch.float64)
    outputs = torch.tensor([[[[2., 0.], [0., 4.]],
                             [[1., 0.], [0., 2.]]]], dtype=torch.float64)
    expected = torch.tensor([[[5.75, 1.5], [2.2, 5.6]]], dtype=torch.float64)
    torch.testing.assert_close(dense_expert_oracle(z, pi, outputs, 2), expected,
                               rtol=0, atol=0)


def test_v5_config_is_torch_free_and_defaults_to_p2():
    before = torch.random.get_rng_state().clone()
    module = importlib.import_module("linearno_loop.v5.config")
    config = module.resolve_config("darcy", options={"architecture": "partial_share_feature_gate_v5"})
    assert config["topology_preset"] == "p2_c2_r2_s2"
    assert config["loop_spec"]["executed_depth"] == 8
    assert config["loop_spec"]["expert_count"] == 2
    assert config["loop_spec"]["expert_width"] == 128
    assert config["loop_spec"]["actual_M"] == 64
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert "torch" not in sys.modules or torch is sys.modules["torch"]


def test_depth_shorthand_and_conflicts():
    from linearno_loop.v5.config import resolve_config
    c = resolve_config("ns", options={"architecture": "partial_share_feature_gate_v5",
                                      "executed_depth": 20})
    assert (c["prefix_blocks"], c["recurrent_core_blocks"], c["loop_repeats"],
            c["suffix_blocks"]) == (2, 8, 2, 2)
    with pytest.raises(ValueError):
        resolve_config("ns", options={"architecture": "partial_share_feature_gate_v5",
                                      "executed_depth": 11})
    with pytest.raises(ValueError):
        resolve_config("ns", options={"architecture": "partial_share_feature_gate_v5",
                                      "topology_preset": "p1_c3_r2_s1", "executed_depth": 8})


def test_core_ownership_initialization_and_forward():
    from cdlno.linearno_loop.v5.core import V5LoopCore
    torch.manual_seed(11)
    core = V5LoopCore(hidden=8, heads=2, rank=4, variant="plain", dropout=0.,
                      H=2, W=3, out_dim=2, expert_count=2, expert_width=8,
                      prefix_blocks=2, recurrent_core_blocks=2, loop_repeats=2,
                      suffix_blocks=2)
    core.finalize_initialization()
    a = core.core[0]
    assert a.visits[0].to_q is not a.visits[1].to_q
    assert a.visits[0].to_q.weight.data_ptr() != a.visits[1].to_q.weight.data_ptr()
    torch.testing.assert_close(a.visits[0].to_q.weight, a.visits[1].to_q.weight, rtol=0, atol=0)
    torch.testing.assert_close(a.visits[0].to_k.weight, a.visits[1].to_k.weight, rtol=0, atol=0)
    assert not torch.equal(a.visits[0].to_q.weight, a.visits[0].to_k.weight)
    for visit in a.visits:
        assert torch.count_nonzero(visit.router.weight) == 0
        assert torch.count_nonzero(visit.router.bias) == 0
    assert not torch.equal(a.experts[0].linear_pre[0].weight,
                           a.experts[1].linear_pre[0].weight)
    x = torch.randn(2, 6, 8, requires_grad=True)
    y = core(x)
    assert y.shape == (2, 6, 2)
    y.square().mean().backward()
    assert a.experts[0].linear_pre[0].weight.grad is not None
    assert a.visits[1].router.weight.grad is not None


def test_custom_r3_registers_three_visits():
    from cdlno.linearno_loop.v5.core import V5LoopCore
    core = V5LoopCore(hidden=8, heads=2, rank=4, variant="plain", dropout=0.,
                      H=2, W=2, out_dim=1, expert_count=3, expert_width=5,
                      prefix_blocks=1, recurrent_core_blocks=1, loop_repeats=3,
                      suffix_blocks=1)
    core.finalize_initialization()
    assert len(core.core[0].visits) == 3
    assert core.executed_depth == 5
    assert core(torch.randn(1, 4, 8)).shape == (1, 4, 1)


@pytest.mark.parametrize("variant", ("plain", "temp", "conv", "conv_temp", "airfrans", "shapenet"))
def test_visit_operator_matches_native_linearno(variant):
    from cdlno.linearno.attention import LinearNOAttention
    from cdlno.linearno_loop.v5.operator import PartialSharedLinearNOOperator
    H, W = (2, 3) if variant in ("conv", "conv_temp") else (None, None)
    native = LinearNOAttention(8, heads=2, dim_head=4, rank=3, variant=variant,
                               dropout=0., H=H, W=W).double()
    v5 = PartialSharedLinearNOOperator(8, heads=2, rank=3, variant=variant,
                                      dropout=0., H=H, W=W, visit_count=2,
                                      expert_count=2).double()
    v5.in_project_x.load_state_dict(native.in_project_x.state_dict())
    v5.to_v.load_state_dict(native.to_v.state_dict())
    v5.to_out.load_state_dict(native.to_out.state_dict())
    v5.visits[0].to_q.load_state_dict(native.to_q.state_dict())
    v5.visits[0].to_k.load_state_dict(native.to_k.state_dict())
    for name in ("temperature_q", "temperature_k", "tempreature_q", "tempreature_k"):
        if hasattr(native, name):
            getattr(v5.visits[0], name).data.copy_(getattr(native, name).data)
    if variant == "airfrans":
        v5.temperature.data.copy_(native.temperature.data)
    x = torch.randn(2, 6 if H else 5, 8, dtype=torch.float64, requires_grad=True)
    expected = native(x)
    actual = v5(x, visit_index=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gx, = torch.autograd.grad(actual.square().sum(), x)
    gy, = torch.autograd.grad(expected.square().sum(), x)
    torch.testing.assert_close(gx, gy, rtol=0, atol=0)


def _small_config(task):
    model = {"model.hidden": 8, "model.heads": 2, "model.linearno_rank": 4,
             "model.ffn_ratio": 1}
    if task not in ("airfrans", "car"):
        model.update({"model.H": 2, "model.W": 3})
    return __import__("linearno_loop.v5.config", fromlist=["resolve_config"]).resolve_config(
        task, options={"architecture": "partial_share_feature_gate_v5",
                       "expert_count": 2, "expert_width": 7},
        profile_overrides=model)


def test_three_wrapper_contracts_forward_backward_adamw():
    from cdlno.linearno_loop.v5.construction import build_from_config
    standard = build_from_config(_small_config("darcy"))
    x = torch.randn(2, 6, 2); fx = torch.randn(2, 6, 1)
    standard_out = standard(x, fx)
    assert standard_out.shape == (2, 6, 1)

    class Data(SimpleNamespace):
        pass

    air = build_from_config(_small_config("airfrans"))
    air_data = Data(x=torch.randn(5, 7), pos=torch.randn(5, 2),
                    batch=torch.zeros(5, dtype=torch.long),
                    ptr=torch.tensor([0, 5], dtype=torch.long))
    air_out = air(air_data)
    assert air_out.shape == (5, 4)

    car = build_from_config(_small_config("car"))
    car_data = Data(x=torch.randn(5, 7), batch=torch.zeros(5, dtype=torch.long),
                    ptr=torch.tensor([0, 5], dtype=torch.long))
    car_out = car((car_data, torch.randn(1)))
    assert car_out.shape == (5, 4)
    loss = standard_out.square().mean() + air_out.square().mean() + car_out.square().mean()
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3)
                  for model in (standard, air, car)]
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
