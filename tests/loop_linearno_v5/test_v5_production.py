import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / "PDE-Solving-StandardBenchmark"
if str(STANDARD) not in sys.path:
    sys.path.insert(0, str(STANDARD))

from cdlno.linearno.profiles import digest
from cdlno.linearno.standard_entry import finish, start
from cdlno.linearno_loop.standard_entry import (LoopStandardRun, model_kwargs,
                                                 model_module)
from linearno_loop.v5.config import resolve_config, run_directory_id
from linearno_loop.v5.contracts import ARCHITECTURE


def parse_standard(task, tokens):
    from linearno import static_worker, temporal_worker
    from cdlno_entry import parse_args
    parser = (temporal_worker.parser_for if task in ("ns", "plasticity")
              else static_worker.parser_for)(task)
    return parse_args(parser, task, list(map(str, tokens)))


def flags(task, topology="p2_c2_r2_s2", experts=2, width=None):
    model = ("LinearNO_Irregular_Mesh" if task == "elasticity"
             else "LinearNO_Structured_Mesh_2D")
    values = ["--model", model, "--linearno-loop", "1",
              "--linearno-loop-architecture", ARCHITECTURE,
              "--linearno-loop-topology", topology,
              "--linearno-loop-residual-mode", "operator_1_expert_1_over_r",
              "--linearno-loop-dense-expert-count", str(experts)]
    if width is not None:
        values += ["--linearno-loop-dense-expert-width", str(width)]
    return values


def test_pure_provenance_is_stable_after_v5_routing_is_committed(monkeypatch):
    from cdlno.linearno import standard_entry

    relative = "cdlno/linearno/standard_entry.py"
    committed_v5_source = (ROOT / relative).read_bytes()
    real_check_output = subprocess.check_output

    def committed_head(command, *args, **kwargs):
        if command[:2] == ["git", "show"] and command[2] == "HEAD:" + relative:
            return (committed_v5_source.decode() if kwargs.get("text") else
                    committed_v5_source)
        return real_check_output(command, *args, **kwargs)

    monkeypatch.setattr(standard_entry.subprocess, "check_output", committed_head)
    actual = standard_entry.provenance()
    expected = json.loads(
        (ROOT / "docs/loop_linearno_audit/ll6/pre-edit.json").read_text()
    )["pure"]
    for field in ("source_sha256", "normalized_patch_sha256"):
        assert actual[field] == expected[field]


def small_config(task, *, epochs=1, nmodel=None):
    overrides = {"model.hidden": 8, "model.heads": 2,
                 "model.ffn_ratio": 1,
                 "training.epochs": epochs, "runtime.seed": 17}
    if task not in ("airfrans", "car"):
        overrides.update({"training.batch_size": 2})
    if nmodel is not None:
        overrides["training.nmodel"] = nmodel
    return resolve_config(task, options={"architecture": ARCHITECTURE,
        "topology_preset": "p2_c2_r2_s2",
        "residual_mode": "operator_1_expert_1_over_r",
        "actual_M": 4, "expert_count": 2, "expert_width": 7},
        profile_overrides=overrides)


def test_real_six_task_parsers_presets_and_independent_k_f():
    for task in ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity"):
        for topology in ("p1_c3_r2_s1", "p2_c2_r2_s2"):
            args = parse_standard(task, flags(task, topology, experts=3, width=11))
            config = args._linearno_loop_config
            assert config["architecture"] == ARCHITECTURE
            assert (config["expert_count"], config["expert_width"]) == (3, 11)
            kwargs = model_kwargs(args)
            assert (kwargs["n_hidden"], kwargs["n_head"],
                    kwargs["linearno_rank"], kwargs["mlp_ratio"]) == (
                        args.n_hidden, args.n_heads, args.linearno_rank, args.mlp_ratio)
            assert model_module(args) is not None
    custom = parse_standard("darcy", [*flags("darcy", "custom"),
        "--linearno-loop-prefix-blocks", "1",
        "--linearno-loop-core-blocks", "2",
        "--linearno-loop-repeats", "3",
        "--linearno-loop-suffix-blocks", "1"])
    assert custom._linearno_loop_config["loop_spec"]["executed_depth"] == 8


def test_standard_epoch_checkpoint_resume_and_eval(tmp_path):
    task = "elasticity"
    config = small_config(task, epochs=2)
    directory = tmp_path / run_directory_id(config)
    tuning = ["--epochs", "2", "--n-hidden", "8", "--n-heads", "2",
              "--linearno-rank", "4", "--mlp_ratio", "1",
              "--batch-size", "2", "--seed", "17",
              "--linearno-loop-dense-expert-width", "7",
              "--experiment-dir", str(directory)]
    args = parse_standard(task, [*flags(task), *tuning])
    assert args._linearno_loop_config == config
    protocol = config["profile_spec"]["values"]["data"]
    args._linearno_data = dict(split=protocol["split"], sampling=protocol["sampling"],
        checksums={"SYNTHETIC": digest("v5-standard")}, scope="SYNTHETIC")
    x = torch.randn(4, 6, 2)

    def loaders():
        return (torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=2, shuffle=True),
                torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:2]), batch_size=2))

    start(args, task)
    try:
        model = model_module(args).Model(**model_kwargs(args))
        run = LoopStandardRun(args, model)
        train, test = loaders()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        run.prepare(optimizer, scheduler, train, test)
        for (batch,) in train:
            optimizer.zero_grad(); model(batch, None).square().mean().backward(); optimizer.step()
        scheduler.step(); run.complete_epoch(1); first = run.save(model)
    finally:
        finish(args)

    resume = parse_standard(task, ["--resume", "--experiment-dir", str(directory)])
    resume._linearno_data = copy.deepcopy(args._linearno_data)
    start(resume, task)
    try:
        resumed = model_module(resume).Model(**model_kwargs(resume))
        train2, test2 = loaders()
        optimizer2 = torch.optim.AdamW(resumed.parameters(), lr=resume.lr, weight_decay=resume.weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=2)
        run2 = LoopStandardRun(resume, resumed)
        run2.prepare(optimizer2, scheduler2, train2, test2)
        assert run2.start_epoch == 1
        for (batch,) in train2:
            optimizer2.zero_grad(); resumed(batch, None).square().mean().backward(); optimizer2.step()
        scheduler2.step(); run2.complete_epoch(2); final = run2.save(resumed)
    finally:
        finish(resume)
    assert first.name == "epoch_0001.json" and final.name == "epoch_0002.json"
    assert (directory / "checkpoints/final.json").is_file()

    evaluation = parse_standard(task, ["--eval", "1", "--experiment-dir", str(directory)])
    evaluation._linearno_data = copy.deepcopy(args._linearno_data)
    start(evaluation, task)
    try:
        evaluated = model_module(evaluation).Model(**model_kwargs(evaluation))
        LoopStandardRun(evaluation, evaluated).load(evaluated)
        assert torch.isfinite(evaluated(x[:1], None)).all()
    finally:
        finish(evaluation)


def _graph(points=9):
    data = Data(x=torch.randn(points, 7), pos=torch.randn(points, 2),
                y=torch.randn(points, 4), surf=torch.arange(points) % 2 == 0,
                batch=torch.zeros(points, dtype=torch.long),
                ptr=torch.tensor([0, points], dtype=torch.long))
    return data


def test_airfrans_native_weighted_loss_checkpoint_and_eval(tmp_path, monkeypatch):
    project = ROOT / "Airfoil-Design-AirfRANS"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    import train as air_train
    from cdlno.linearno_loop import air_entry
    from cdlno.linearno_loop.industrial_state import construct
    config = small_config("airfrans", epochs=1, nmodel=1)
    directory = tmp_path / "air-v5"; directory.mkdir()
    fake = dict(split="SYNTHETIC fixed", sampling="synthetic fixed",
                checksums={"manifest.json": "a" * 64}, task_variant="full",
                steps_per_epoch=1, scheduler_total_steps=2, manifest_hash="a" * 64,
                train_count=1, validation_count=0, manifest_keys=[])
    args = SimpleNamespace(_linearno_loop_config=config, _linearno_config=config["profile_spec"],
        _linearno_model_spec=config["model_spec"], linearno_task="airfrans", task="full",
        seed=17, nmodel=1, nb_epochs=1, resume=False, eval=0, checkpoint="final",
        device="cpu", weight=config["profile_spec"]["values"]["objective"]["surface_weight"],
        linearno_run_dir=directory)
    coef = (np.zeros(4), np.ones(4), np.zeros(4), np.ones(4))
    graph = _graph()
    monkeypatch.setattr(air_entry, "_data_spec", lambda *a, **k: fake)
    run = air_entry.AirRun(args, args._linearno_config, tmp_path, coef, None, manifest={}, evaluation=False)
    model = construct(args)
    hparams = dict(batch_size=1, nb_epochs=1, lr=1e-3, subsampling=9, r=.05,
                   max_neighbors=64, debug=0)
    air_train.main("cpu", [graph], [graph], model, hparams, str(directory),
                   criterion="MSE_weighted", reg=args.weight, val_iter=10,
                   name_mod="LinearNO", val_sample=True, linearno_run=run)
    run.finish_ensemble([model])
    args.eval = 1; args._linearno_metadata = __import__(
        "linearno_loop.versioning", fromlist=["read_metadata"]).read_metadata(
            directory / "architecture.json")
    loaded = air_entry.AirRun(args, args._linearno_config, tmp_path, coef, None,
                              manifest={}, evaluation=True).load_models()
    assert len(loaded) == 1 and tuple(loaded[0](graph).shape) == (9, 4)


def test_airfrans_member_interrupt_resume_final_and_eval(tmp_path, monkeypatch):
    from cdlno.linearno_loop import air_entry
    from cdlno.linearno_loop.industrial_state import construct
    from linearno_loop.versioning import read_metadata

    config = small_config("airfrans", epochs=2, nmodel=1)
    directory = tmp_path / "air-v5-resume"; directory.mkdir()
    fake = dict(split="SYNTHETIC fixed", sampling="synthetic fixed",
                checksums={"manifest.json": "c" * 64}, task_variant="full",
                steps_per_epoch=1, scheduler_total_steps=4,
                manifest_hash="c" * 64, train_count=1, validation_count=0,
                manifest_keys=[])
    common = dict(_linearno_loop_config=config, _linearno_config=config["profile_spec"],
        _linearno_model_spec=config["model_spec"], linearno_task="airfrans", task="full",
        seed=17, nmodel=1, nb_epochs=2, eval=0, checkpoint="latest", device="cpu",
        weight=config["profile_spec"]["values"]["objective"]["surface_weight"],
        linearno_run_dir=directory)
    coef = (np.zeros(4), np.ones(4), np.zeros(4), np.ones(4))
    graph = _graph()
    monkeypatch.setattr(air_entry, "_data_spec", lambda *a, **k: fake)

    def objects(args):
        model = construct(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=4)
        run = air_entry.AirRun(args, args._linearno_config, tmp_path, coef, None,
                               manifest={}, evaluation=False)
        return model, optimizer, scheduler, run

    def weighted_step(model, optimizer, scheduler):
        optimizer.zero_grad()
        prediction = model(graph)
        point_loss = (prediction - graph.y).square()
        loss = point_loss[~graph.surf].mean() + common["weight"] * point_loss[graph.surf].mean()
        loss.backward(); optimizer.step(); scheduler.step()

    args = SimpleNamespace(**common, resume=False)
    model, optimizer, scheduler, run = objects(args)
    assert run.prepare(model, optimizer, scheduler, [graph], [graph],
                       "MSE_weighted", args.weight, 10, True) == (0, None)
    weighted_step(model, optimizer, scheduler)
    history1 = dict(curves=[[0.0] for _ in range(8)], val_loss=0.0)
    run.complete_epoch(1, model, history1)
    committed = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resume_args = SimpleNamespace(**common, resume=True,
                                  _linearno_metadata=read_metadata(directory / "architecture.json"))
    resumed, optimizer2, scheduler2, run2 = objects(resume_args)
    epoch, restored_history = run2.prepare(
        resumed, optimizer2, scheduler2, [graph], [graph],
        "MSE_weighted", resume_args.weight, 10, True)
    assert epoch == 1 and restored_history == history1 and scheduler2.last_epoch == 1
    for name, value in committed.items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)

    weighted_step(resumed, optimizer2, scheduler2)
    history2 = dict(curves=[[0.0, 0.0] for _ in range(8)], val_loss=0.0)
    run2.complete_epoch(2, resumed, history2)
    run2.finish_ensemble([resumed])
    assert (directory / "member_000/checkpoints/final.json").is_file()

    eval_args = SimpleNamespace(**{**common, "resume": False, "eval": 1,
        "checkpoint": "final",
        "_linearno_metadata": read_metadata(directory / "architecture.json")})
    evaluated = air_entry.AirRun(eval_args, eval_args._linearno_config, tmp_path, coef,
                                 None, manifest={}, evaluation=True).load_models()
    assert len(evaluated) == 1 and tuple(evaluated[0](graph).shape) == (9, 4)


def test_car_native_graph_checkpoint_resume_and_eval(tmp_path):
    from cdlno.linearno_loop import car_entry
    from cdlno.linearno_loop.industrial_state import construct
    config = small_config("car", epochs=1)
    directory = tmp_path / "car-v5"; directory.mkdir()
    fake = dict(split="SYNTHETIC fixed", sampling="synthetic fixed",
        checksums={"raw": "b" * 64}, fold_id=3, train_samples=["a"],
        test_samples=["b"], cfd_mesh=False, r=.2, val_iter=1,
        preprocessed=1, ntrain=1, ntest=1)
    args = SimpleNamespace(_linearno_loop_config=config, _linearno_config=config["profile_spec"],
        _linearno_model_spec=config["model_spec"], linearno_task="car", seed=17,
        nb_epochs=1, lr=1e-3, batch_size=1, val_iter=1, weight=.5, device="cpu",
        linearno_run_dir=directory, eval=0, resume=False, checkpoint="final",
        fold_id=3, cfd_mesh=False, r=.2, data_dir=str(tmp_path))
    coef = (np.zeros(7), np.ones(7), np.zeros(4), np.ones(4))
    graph = _graph(); geom = torch.randn(1)
    run = car_entry.CarRun(args, fake, coef, None)
    model = construct(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                     total_steps=2, final_div_factor=1000.)
    run.data["scheduler"]["epochs"] = 1
    run.data["scheduler"]["total_steps"] = 2
    run.prepare(model, optimizer, scheduler)
    optimizer.zero_grad(); model((graph, geom)).square().mean().backward(); optimizer.step(); scheduler.step()
    run.complete_epoch(1, model, {"synthetic_loss": 0.0})
    run.export_final(model)
    args.resume = True; args._linearno_metadata = __import__(
        "linearno_loop.versioning", fromlist=["read_metadata"]).read_metadata(
            directory / "architecture.json")
    args._linearno_checkpoint = directory / "checkpoints/epoch_0001.json"
    resumed = construct(args); opt2 = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=1e-3, total_steps=2, final_div_factor=1000.)
    assert car_entry.CarRun(args, fake, coef, None).prepare(resumed, opt2, sch2) == 1
    args.eval = 1
    loaded = car_entry.CarRun(args, fake, coef, None).load()
    assert tuple(loaded((graph, geom)).shape) == (9, 4)
