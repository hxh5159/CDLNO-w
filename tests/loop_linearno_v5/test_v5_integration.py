import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from linearno_loop.v5.config import resolve_config, validate_config
from linearno_loop.v5.contracts import ARCHITECTURE


def small_config(task="darcy", **options):
    base = dict(architecture=ARCHITECTURE, expert_count=2, expert_width=7)
    base.update(options)
    overrides = {"model.hidden": 8, "model.heads": 2,
                 "model.linearno_rank": 4, "model.ffn_ratio": 1}
    if task not in ("airfrans", "car"):
        overrides.update({"model.H": 2, "model.W": 3})
    return resolve_config(task, options=base, profile_overrides=overrides)


def test_config_json_roundtrip_hash_and_torch_free_subprocess(tmp_path):
    config = resolve_config("darcy", options={"architecture": ARCHITECTURE})
    assert validate_config(json.loads(json.dumps(config))) == config
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        "from linearno_loop.v5.config import resolve_config\n"
        f"c=resolve_config('darcy',options={{'architecture':'{ARCHITECTURE}'}})\n"
        "assert 'torch' not in sys.modules\n"
        "print(c['config_hash'])\n"
    )
    import os, subprocess, sys
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    first = subprocess.check_output([sys.executable, str(script)], env=env, text=True)
    second = subprocess.check_output([sys.executable, str(script)], env=env, text=True)
    assert first == second == config["config_hash"] + "\n"


@pytest.mark.parametrize("task,expected_f,expected_m", [
    ("airfoil", 128, 64), ("darcy", 128, 64), ("elasticity", 128, 64),
    ("pipe", 128, 64), ("plasticity", 128, 64), ("ns", 512, 32),
    ("airfrans", 512, 32), ("car", 512, 32),
])
def test_eight_task_default_expert_width_and_rank(task, expected_f, expected_m):
    config = resolve_config(task, options={"architecture": ARCHITECTURE})
    assert config["expert_width"] == expected_f
    assert config["actual_M"] == expected_m


def test_complete_initialization_lifecycle_and_independent_storage():
    from cdlno.linearno_loop.v5.construction import build_from_config
    model = build_from_config(small_config("elasticity"), initialization_seed=31)
    for block in model.loop.core:
        source = block.visits[0]
        for visit in block.visits[1:]:
            assert visit.to_q.weight.data_ptr() != source.to_q.weight.data_ptr()
            assert visit.to_k.weight.data_ptr() != source.to_k.weight.data_ptr()
            torch.testing.assert_close(visit.to_q.weight, source.to_q.weight, rtol=0, atol=0)
            torch.testing.assert_close(visit.to_k.weight, source.to_k.weight, rtol=0, atol=0)
            torch.testing.assert_close(visit.temperature_q, source.temperature_q, rtol=0, atol=0)
            torch.testing.assert_close(visit.temperature_k, source.temperature_k, rtol=0, atol=0)
        assert not torch.equal(source.to_q.weight, source.to_k.weight)
        assert all(torch.count_nonzero(visit.router.weight) == 0 and
                   torch.count_nonzero(visit.router.bias) == 0
                   for visit in block.visits)


def test_router_and_all_experts_receive_gradients_and_visits_diverge():
    from cdlno.linearno_loop.v5.construction import build_from_config
    model = build_from_config(small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(2, 6, 2)
    fx = torch.randn(2, 6, 1)
    loss = model(x, fx).square().mean()
    loss.backward()
    block = model.loop.core[0]
    assert all(expert.linear_pre[0].weight.grad is not None and
               torch.count_nonzero(expert.linear_pre[0].weight.grad)
               for expert in block.experts)
    assert all(visit.router.weight.grad is not None and
               torch.count_nonzero(visit.router.weight.grad)
               for visit in block.visits)
    optimizer.step()
    assert not torch.equal(block.visits[0].router.weight,
                           block.visits[1].router.weight)


def test_strict_checkpoint_roundtrip_and_pre_tensor_conflict(tmp_path, monkeypatch):
    from cdlno.linearno_loop.v5.checkpoint import load_model, save_pair
    from cdlno.linearno_loop.v5.construction import build_from_config
    config = small_config()
    model = build_from_config(config)
    manifest = save_pair(tmp_path, model, config)
    assert manifest.name == "epoch_0001.json"
    loaded, metadata = load_model(tmp_path, "final")
    assert metadata["resolved_config"] == config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)

    called = False
    original = torch.load

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="explicit expert_count conflicts"):
        load_model(tmp_path, "final", explicit={"expert_count": 3})
    assert not called


def test_saved_metadata_drives_real_resume_and_eval_parsers(tmp_path):
    from cdlno.linearno_loop.v5.checkpoint import save_pair
    from cdlno.linearno_loop.v5.construction import build_from_config
    from linearno_loop.v5.config import run_directory_id
    from tran_evaluate.linearno_loop_v5.launch import plan_v5
    config = small_config()
    directory = tmp_path / run_directory_id(config)
    save_pair(directory, build_from_config(config), config)
    for action, checkpoint in (("resume", "latest"), ("eval", "final")):
        _, planned = plan_v5(["darcy", action, "--experiment-dir", str(directory),
                              "--checkpoint", checkpoint, "--gpu", "0"])
        assert planned["config"] == config
        assert planned["run"] == str(directory)
    with pytest.raises(SystemExit):
        plan_v5(["darcy", "eval", "--experiment-dir", str(directory),
                 "--expert-count", "3"])


@pytest.mark.parametrize("task,relative_path", [
    ("airfoil", "fno/airfoil/naca"),
    ("darcy", "fno"),
    ("elasticity", "fno"),
    ("pipe", "fno/pipe"),
    ("ns", "fno"),
    ("plasticity", "fno/plas_N987_T20.mat"),
])
def test_standard_launcher_resolves_one_task_specific_data_path(
        task, relative_path, tmp_path):
    from tran_evaluate.linearno_loop_v5.launch import plan_v5

    data_root = tmp_path / "data root"
    _, planned = plan_v5([
        task, "train", "--dry-run", "--data-root", str(data_root),
        "--gpu", "0",
    ])
    paths = [planned["argv"][index + 1]
             for index, token in enumerate(planned["argv"])
             if token == "--data_path"]
    assert paths == [str(data_root / relative_path)]


def test_standard_launcher_preserves_one_explicit_data_path(tmp_path):
    from tran_evaluate.linearno_loop_v5.launch import plan_v5

    explicit = tmp_path / "explicit airfoil data"
    _, planned = plan_v5([
        "airfoil", "train", "--dry-run", "--data_path", str(explicit),
        "--gpu", "0",
    ])
    paths = [planned["argv"][index + 1]
             for index, token in enumerate(planned["argv"])
             if token == "--data_path"]
    assert paths == [str(explicit)]


@pytest.mark.parametrize("task,expected", [
    ("airfrans", {
        "--my_path": "AirfRANS/Dataset",
    }),
    ("car", {
        "--data_dir": "mlcfd_data/training_data",
        "--save_dir": "mlcfd_data/preprocessed_data",
    }),
])
def test_industrial_launcher_resolves_one_path_per_option(task, expected, tmp_path):
    from tran_evaluate.linearno_loop_v5.launch import plan_v5

    data_root = tmp_path / "industrial data root"
    _, planned = plan_v5([
        task, "train", "--dry-run", "--data-root", str(data_root),
        "--gpu", "0",
    ])
    for option, relative_path in expected.items():
        paths = [planned["argv"][index + 1]
                 for index, token in enumerate(planned["argv"])
                 if token == option]
        assert paths == [str(data_root / relative_path)]


@pytest.mark.parametrize("task", ("airfrans", "car"))
def test_industrial_real_parsers_both_presets_and_independent_k_f(task):
    from tran_evaluate.linearno_loop_v5.launch import plan_v5

    for topology in ("p1_c3_r2_s1", "p2_c2_r2_s2"):
        _, planned = plan_v5([
            task, "train", "--dry-run", "--topology", topology,
            "--expert-count", "3", "--expert-width", "11",
            "--seed", "17", "--gpu", "0",
        ])
        config = planned["config"]
        assert config["topology_preset"] == topology
        assert (config["expert_count"], config["expert_width"]) == (3, 11)


def test_train_eval_preserves_selected_gpu_for_evaluation(monkeypatch, tmp_path):
    from tran_evaluate.linearno_loop_v5 import launch

    args = SimpleNamespace(task="darcy", action="train_eval", then_eval=False, gpu=3,
                           print_run_dir=False, print_config=False, dry_run=False)
    training = {
        "run": str(tmp_path / "run"),
        "argv": ["exp_darcy.py", "--data_path", str(tmp_path / "data"), "--gpu", "3"],
        "environment": {"CUDA_VISIBLE_DEVICES": "3"},
    }
    executions = []
    evaluations = []
    monkeypatch.setattr(launch, "plan_v5", lambda argv=None: (args, training))
    monkeypatch.setattr(launch, "execute", lambda value: executions.append(value) or 0)

    def plan(task, action, tokens, environment):
        evaluations.append((task, action, tokens, environment))
        return {"phase": "eval", "argv": list(tokens)}

    monkeypatch.setattr(launch, "plan", plan)
    assert launch.main([]) == 0
    assert executions == [training, {"phase": "eval", "argv": [
        "--experiment-dir", training["run"], "--gpu", "3",
        "--data_path", str(tmp_path / "data"),
    ]}]
    assert evaluations == [("darcy", "eval", [
        "--experiment-dir", training["run"], "--gpu", "3",
        "--data_path", str(tmp_path / "data"),
    ], training["environment"])]


@pytest.mark.parametrize("task,selected_paths", [
    ("airfoil", {"--data_path": "/selected/airfoil"}),
    ("darcy", {"--data_path": "/selected/fno"}),
    ("elasticity", {"--data_path": "/selected/fno"}),
    ("pipe", {"--data_path": "/selected/pipe"}),
    ("ns", {"--data_path": "/selected/fno"}),
    ("plasticity", {"--data_path": "/selected/plasticity.mat"}),
    ("airfrans", {"--my_path": "/selected/AirfRANS/Dataset"}),
    ("car", {"--data_dir": "/selected/car/raw", "--save_dir": "/selected/car/cache"}),
])
def test_train_eval_removes_native_path_defaults_for_every_task(
        task, selected_paths, monkeypatch, tmp_path):
    from tran_evaluate.linearno_loop_v5 import launch

    args = SimpleNamespace(task=task, action="train_eval", then_eval=False, gpu=0,
                           print_run_dir=False, print_config=False, dry_run=False)
    training_argv = ["entry.py"]
    for option, path in selected_paths.items():
        training_argv.extend([option, path])
    training = {
        "run": str(tmp_path / "run"),
        "argv": training_argv,
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
    }
    executions = []
    monkeypatch.setattr(launch, "plan_v5", lambda argv=None: (args, training))
    monkeypatch.setattr(launch, "execute", lambda value: executions.append(value) or 0)

    def plan(task_name, action, tokens, environment):
        argv = ["evaluation.py"]
        for option in selected_paths:
            argv.extend([option, "/native/default"])
        argv.extend(tokens)
        return {"task": task_name, "action": action, "argv": argv}

    monkeypatch.setattr(launch, "plan", plan)
    assert launch.main([]) == 0
    evaluation = executions[-1]
    for option, path in selected_paths.items():
        values = [evaluation["argv"][index + 1]
                  for index, token in enumerate(evaluation["argv"])
                  if token == option]
        assert values == [path]


def test_v5_recorder_verifies_real_visit_and_expert_schedule(tmp_path):
    from cdlno.linearno_loop.v5.construction import build_from_config
    from cdlno.linearno_loop.v5.recording import observe, expected_schedule
    config = small_config()
    model = build_from_config(config)
    args = SimpleNamespace(_linearno_loop_config=config, linearno_run_dir=tmp_path,
                           eval=0, resume=False, seed=config["seed"],
                           linearno_task="darcy")
    observe(args, model)
    model(torch.randn(1, 6, 2), torch.randn(1, 6, 1))
    manifest = json.loads((tmp_path / "loop_run_manifest.json").read_text())
    row = manifest["members"]["member_000"]
    assert row["actual_call_schedule"] == expected_schedule(config)
    assert row["expert_calls"] == config["loop_spec"]["executed_depth"] * config["expert_count"]
    assert row["observation"] == "first_forward_verified"


def test_qk_route_shapes_are_n_by_m_only():
    from cdlno.linearno_loop.v5.construction import build_from_config
    model = build_from_config(small_config())
    shapes = []
    handles = []
    for block in (*model.loop.prefix, *model.loop.core, *model.loop.suffix):
        for visit in block.visits:
            handles.append(visit.to_q.register_forward_hook(
                lambda module, inputs, output: shapes.append(tuple(output.shape))))
            handles.append(visit.to_k.register_forward_hook(
                lambda module, inputs, output: shapes.append(tuple(output.shape))))
    model(torch.randn(1, 6, 2), torch.randn(1, 6, 1))
    for handle in handles:
        handle.remove()
    assert len(shapes) == 2 * model.loop.executed_depth
    assert all(shape == (1, 2, 6, 4) for shape in shapes)


@pytest.mark.parametrize("topology", ("p1_c3_r2_s1", "p2_c2_r2_s2"))
@pytest.mark.parametrize("experts", (2, 3, 4, 8))
def test_accounting_matches_live_parameters_and_shape_trace(topology, experts):
    from cdlno.linearno_loop.v5.checkpoint import measure_parameters
    from cdlno.linearno_loop.v5.construction import build_from_config
    from tools.linearno_loop_accounting import analytic_v5, audit
    config = small_config(topology_preset=topology, expert_count=experts)
    model = build_from_config(config)
    expected = analytic_v5(config, B=1, N=6)
    measured = measure_parameters(model, config)
    assert expected["parameters"] == measured["total"]
    assert expected["parameter_parts"] == measured["groups"]
    if experts == 2:
        traced = audit(model, (torch.randn(1, 6, 2), torch.randn(1, 6, 1)), B=1, N=6)
        assert traced["matrix_macs"] == expected["matrix_macs"]
        assert traced["parameter_parts"] == expected["parameter_parts"]
        assert traced["forbidden_attention"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_cuda_autocast_gate_and_experts_are_finite(dtype):
    from cdlno.linearno_loop.v5.construction import build_from_config
    model = build_from_config(small_config()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=dtype):
        output = model(torch.randn(1, 6, 2, device="cuda"),
                       torch.randn(1, 6, 1, device="cuda"))
        loss = output.square().mean()
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
