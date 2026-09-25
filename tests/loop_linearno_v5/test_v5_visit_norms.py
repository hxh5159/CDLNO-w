import json

import pytest
import torch

from linearno_loop.v5.contracts import ARCHITECTURE
from linearno_loop.v5.config import resolve_config, run_directory_id, validate_config


def _core(mode, *, hidden=8, core_blocks=2, repeats=2):
    from cdlno.linearno_loop.v5.core import V5LoopCore

    model = V5LoopCore(
        hidden=hidden, heads=2, rank=4, variant="plain", dropout=0.0,
        H=2, W=3, out_dim=1, expert_count=2, expert_width=7,
        prefix_blocks=1, recurrent_core_blocks=core_blocks,
        loop_repeats=repeats, suffix_blocks=1, core_norm_mode=mode,
    )
    model.finalize_initialization()
    return model


def _config(task="darcy", mode="visit_independent", **options):
    selected = dict(
        architecture=ARCHITECTURE,
        core_norm_mode=mode,
        expert_count=2,
        expert_width=7,
    )
    selected.update(options)
    overrides = {
        "model.hidden": 8,
        "model.heads": 2,
        "model.linearno_rank": 4,
        "model.ffn_ratio": 1,
    }
    if task not in ("airfrans", "car"):
        overrides.update({"model.H": 2, "model.W": 3})
    return resolve_config(task, options=selected, profile_overrides=overrides)


def test_config_default_roundtrip_and_run_id_separate_norm_modes():
    default = resolve_config("darcy", options={"architecture": ARCHITECTURE})
    shared = resolve_config(
        "darcy", options={"architecture": ARCHITECTURE, "core_norm_mode": "shared"}
    )
    assert default["core_norm_mode"] == "visit_independent"
    assert default["loop_spec"]["core_norm_mode"] == "visit_independent"
    assert default["model_spec"]["constructor_kwargs"]["core_norm_mode"] == "visit_independent"
    assert shared["core_norm_mode"] == "shared"
    assert "norm-visit_independent" in run_directory_id(default)
    assert "norm-shared" in run_directory_id(shared)
    assert run_directory_id(default) != run_directory_id(shared)
    assert validate_config(json.loads(json.dumps(default))) == default
    with pytest.raises(ValueError, match="core_norm_mode"):
        resolve_config(
            "darcy", options={"architecture": ARCHITECTURE, "core_norm_mode": "other"}
        )


def test_launcher_routes_norm_mode_and_airfoil_reference_count():
    from cdlno.linearno_loop.v5.checkpoint import measure_parameters
    from cdlno.linearno_loop.v5.construction import build_from_config
    from tran_evaluate.linearno_loop_v5.launch import plan_v5
    from tools.linearno_loop_accounting import analytic_v5

    _, planned = plan_v5([
        "airfoil", "train", "--dry-run", "--topology", "p1_c3_r2_s1",
        "--expert-count", "3", "--core-norm-mode", "shared",
        "--seed", "0", "--gpu", "0",
    ])
    assert planned["config"]["core_norm_mode"] == "shared"
    assert "norm-shared" in planned["run"]
    assert "--linearno-loop-core-norm-mode" in planned["argv"]

    counts = {}
    flops = {}
    for mode in ("shared", "visit_independent"):
        config = resolve_config("airfoil", options={
            "architecture": ARCHITECTURE,
            "topology_preset": "p1_c3_r2_s1",
            "expert_count": 3,
            "core_norm_mode": mode,
        })
        counts[mode] = measure_parameters(build_from_config(config), config)["total"]
        flops[mode] = analytic_v5(config, B=1, N=11271)["matrix_flops"]
    assert counts == {"shared": 1_456_025, "visit_independent": 1_457_561}
    assert flops["shared"] == flops["visit_independent"]


@pytest.mark.parametrize(
    "core_blocks,repeats,expected_extra",
    ((3, 2, 12 * 8), (2, 2, 8 * 8), (2, 3, 16 * 8)),
)
def test_norm_ownership_storage_and_exact_parameter_delta(
    core_blocks, repeats, expected_extra
):
    torch.manual_seed(41)
    independent = _core(
        "visit_independent", core_blocks=core_blocks, repeats=repeats
    )
    torch.manual_seed(41)
    shared = _core("shared", core_blocks=core_blocks, repeats=repeats)

    assert sum(p.numel() for p in independent.parameters()) - sum(
        p.numel() for p in shared.parameters()
    ) == expected_extra == 4 * 8 * core_blocks * (repeats - 1)

    independent_block = independent.core[0]
    shared_block = shared.core[0]
    for which in (0, 1):
        first = independent_block.norms_for_visit(0)[which]
        second = independent_block.norms_for_visit(1)[which]
        assert first is not second
        assert first.weight.data_ptr() != second.weight.data_ptr()
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)
        assert shared_block.norms_for_visit(0)[which] is shared_block.norms_for_visit(1)[which]

    assert not independent.prefix[0].additional_ln_1
    assert not independent.prefix[0].additional_ln_2
    assert not independent.suffix[0].additional_ln_1
    assert not independent.suffix[0].additional_ln_2
    assert not any("additional_ln" in key for key in shared.state_dict())

    before = independent_block.norms_for_visit(0)[0].weight.detach().clone()
    with torch.no_grad():
        independent_block.norms_for_visit(1)[0].weight.add_(1)
    torch.testing.assert_close(
        independent_block.norms_for_visit(0)[0].weight, before, rtol=0, atol=0
    )


def test_modes_have_identical_common_initialization_rng_and_initial_forward():
    before = torch.random.get_rng_state().clone()
    torch.manual_seed(73)
    independent = _core("visit_independent")
    after_independent = torch.random.get_rng_state().clone()
    torch.manual_seed(73)
    shared = _core("shared")
    after_shared = torch.random.get_rng_state().clone()
    torch.testing.assert_close(after_independent, after_shared, rtol=0, atol=0)

    independent_state = independent.state_dict()
    shared_state = shared.state_dict()
    for key, value in shared_state.items():
        torch.testing.assert_close(independent_state[key], value, rtol=0, atol=0)

    x = torch.randn(2, 6, 8)
    torch.testing.assert_close(independent(x), shared(x), rtol=0, atol=0)
    torch.random.set_rng_state(before)


def test_every_executed_visit_norm_has_finite_gradient_and_one_optimizer_owner():
    model = _core("visit_independent", core_blocks=2, repeats=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == len({id(parameter) for parameter in owned})
    model(torch.randn(2, 6, 8)).square().mean().backward()
    for block in model.core:
        for visit in range(model.loop_repeats):
            for norm in block.norms_for_visit(visit):
                assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
                assert norm.bias.grad is not None and torch.isfinite(norm.bias.grad).all()


@pytest.mark.parametrize("task", ("darcy", "airfrans", "car"))
@pytest.mark.parametrize("mode", ("visit_independent", "shared"))
def test_three_wrapper_families_forward_backward(task, mode):
    from types import SimpleNamespace

    from cdlno.linearno_loop.v5.construction import build_from_config

    model = build_from_config(_config(task, mode))
    if task == "darcy":
        output = model(torch.randn(2, 6, 2), torch.randn(2, 6, 1))
    elif task == "airfrans":
        data = SimpleNamespace(
            x=torch.randn(5, 7), pos=torch.randn(5, 2),
            batch=torch.zeros(5, dtype=torch.long),
            ptr=torch.tensor([0, 5], dtype=torch.long),
        )
        output = model(data)
    else:
        data = SimpleNamespace(
            x=torch.randn(5, 7), batch=torch.zeros(5, dtype=torch.long),
            ptr=torch.tensor([0, 5], dtype=torch.long),
        )
        output = model((data, torch.randn(1)))
    output.square().mean().backward()
    assert output.shape[-1] in (1, 4)


@pytest.mark.parametrize("mode", ("visit_independent", "shared"))
def test_strict_checkpoint_roundtrip_and_cross_mode_rejected_before_tensor_load(
    tmp_path, monkeypatch, mode
):
    from cdlno.linearno_loop.v5.checkpoint import load_model, save_pair
    from cdlno.linearno_loop.v5.construction import build_from_config

    config = _config(mode=mode)
    directory = tmp_path / run_directory_id(config)
    model = build_from_config(config)
    save_pair(directory, model, config)
    loaded, metadata = load_model(directory, "final")
    assert metadata["resolved_config"]["core_norm_mode"] == mode
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("tensor loading must not happen after metadata conflict")

    monkeypatch.setattr(torch, "load", forbidden)
    other = "shared" if mode == "visit_independent" else "visit_independent"
    with pytest.raises(ValueError, match="core_norm_mode"):
        load_model(directory, "final", explicit={"core_norm_mode": other})
    assert not called


def test_previous_pair_revision_is_rejected_by_metadata_before_tensor_loading(
    tmp_path, monkeypatch
):
    from cdlno.linearno_loop.v5.checkpoint import save_pair
    from cdlno.linearno_loop.v5.construction import build_from_config
    from linearno_loop.v5.contracts import digest
    from linearno_loop.v5.schema import read_metadata

    config = _config(mode="shared")
    directory = tmp_path / run_directory_id(config)
    save_pair(directory, build_from_config(config), config)
    payload = json.loads((directory / "architecture.json").read_text())
    payload["checkpoint_version"] = 1
    body = dict(payload)
    body.pop("metadata_hash")
    payload["metadata_hash"] = digest(body)
    legacy = tmp_path / "pair-v1-architecture.json"
    legacy.write_text(json.dumps(payload))

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("old metadata must fail before tensor loading")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="checkpoint schema/version mismatch"):
        read_metadata(legacy)
    assert not called
