"""LAA7 industrial V3 parser, PyG boundary and strict-pair coverage."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from .wrapper_support import construct, example, config, finite_step
from cdlno.linearno_loop.v3.checkpoint import load_model, save_pair
from linearno_loop.v3.schema import make_metadata, write_metadata
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR


MODES = ("sr_1_over_r", "rb_attnres", "lb_attnres_1_over_r")
ABLATIONS = ((False, False), (False, True), (True, False), (True, True))
ROOT = Path(__file__).resolve().parents[2]


def _air_parser(tokens):
    sys.path.insert(0, str(ROOT / "Airfoil-Design-AirfRANS"))
    from cdlno_entry import parse_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model"); parser.add_argument("-n", "--nmodel", default=1, type=int)
    parser.add_argument("-w", "--weight", default=1., type=float)
    parser.add_argument("-t", "--task", default="full"); parser.add_argument("-s", "--score", default=0, type=int)
    parser.add_argument("--my_path", default="/tmp"); parser.add_argument("--save_path", default="/tmp")
    return parse_args(parser, argv=list(tokens))


def _car_parser(tokens):
    sys.path.insert(0, str(ROOT / "Car-Design-ShapeNetCar"))
    from models.cdlno_run import parse_args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/tmp"); parser.add_argument("--save_dir", default="/tmp")
    parser.add_argument("--fold_id", default=3, type=int); parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--val_iter", default=10, type=int); parser.add_argument("--cfd_config_dir", default="x")
    parser.add_argument("--cfd_model"); parser.add_argument("--cfd_mesh", action="store_true")
    parser.add_argument("--r", default=.2, type=float); parser.add_argument("--weight", default=.5, type=float)
    parser.add_argument("--lr", default=.001, type=float); parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--nb_epochs", default=2, type=int); parser.add_argument("--preprocessed", default=1, type=int)
    return parse_args(parser, argv=list(tokens))


def _flags(task, cost, mode, latent, adapter):
    model_flag = "--model" if task == "airfrans" else "--cfd_model"
    values = [model_flag, "LinearNO", "--linearno-loop", "1",
              "--linearno-loop-architecture", ARCHITECTURE_SELECTOR,
              "--linearno-loop-cost-profile", cost,
              "--linearno-loop-topology", "d12",
              "--linearno-loop-residual-mode", mode,
              "--linearno-loop-latent", str(int(latent)),
              "--linearno-loop-adapter-mode",
              "bilateral_qk_lowrank_second_visit" if adapter else "none",
              "--linearno-loop-adapter-rank", "4", "--linearno-loop-adapter-alpha", "4",
              "--linearno-rank", "32", "--seed", "17"]
    if task == "airfrans": values += ["--gpu", "0"]
    return values


class LAA7IndustrialTests(unittest.TestCase):
    def test_parser_matrix_two_profiles_three_residuals_four_ablations(self):
        rows = []
        for task in ("airfrans", "car"):
            for cost in ("matched_v1", "efficient_v1"):
                for mode in MODES:
                    for latent, adapter in ABLATIONS:
                        args = (_air_parser if task == "airfrans" else _car_parser)(
                            _flags(task, cost, mode, latent, adapter))
                        cfg = args._linearno_loop_config
                        self.assertEqual(cfg["config_version"], 3)
                        self.assertEqual(cfg["architecture_extension"], "loop_linearno_latent_adapter_v3")
                        self.assertEqual(cfg["loop_spec"]["cost_profile"], cost)
                        self.assertEqual(cfg["loop_spec"]["residual_mode"], mode)
                        self.assertEqual(cfg["loop_spec"]["latent_enabled"], latent)
                        self.assertEqual(cfg["loop_spec"]["adapter_mode"] != "none", adapter)
                        if task == "car": self.assertEqual(args.fold_id, 3)
                        rows.append((task, cost, mode, latent, adapter))
        self.assertEqual(len(rows), 48)

    def test_industrial_profile_depth_constructor_matrix(self):
        torch.set_num_threads(1)
        for task in ("airfrans", "car"):
            for cost in ("matched_v1", "efficient_v1"):
                for depth in ("d12", "d20", "d28", "d60"):
                    with self.subTest(task=task, cost=cost, depth=depth):
                        c = config(task, cost=cost, formal_width=True)
                        # Replace only the versioned topology shorthand and
                        # re-resolve the saved base profile through the public
                        # V3 schema; this is a construction dry-run, not data.
                        from linearno_loop.v3.config import resolve_config
                        options = dict(c["request"]["options"])
                        options["topology_preset"] = depth
                        options.pop("topology_preset", None)
                        options["executed_depth"] = int(depth[1:])
                        options["latent_enabled"] = False
                        options["adapter_mode"] = "none"
                        c = resolve_config(task, c["request"]["profile"], options=options,
                                           profile_overrides=c["request"]["profile_overrides"])
                        model = construct(c)
                        self.assertEqual(model.loop.loop_repeats, 2)

    def test_native_pyg_contract_and_member_isolation(self):
        air = config("airfrans", cost="custom", latent=True, adapter=True)
        first, second = construct(air, member_seed=101), construct(air, member_seed=102)
        self.assertFalse({id(p) for p in first.parameters()} & {id(p) for p in second.parameters()})
        data = example(air, points=9)[0]
        self.assertEqual(tuple(first(data).shape), (9, 4))
        with self.assertRaises(ValueError):
            bad = data.clone(); bad.batch[-1] = 1; first(bad)
        finite_step(first, air)
        car = config("car", cost="custom", latent=True, adapter=True)
        model = construct(car).eval()
        graph, geom = example(car, points=9)[0]
        self.assertEqual(tuple(model((graph, geom)).shape), (9, 4))
        with self.assertRaises(ValueError): model(graph)
        finite_step(model, car)

    def test_v3_pair_roundtrip_and_fresh_process_load(self):
        c = config("car", cost="custom", latent=True, adapter=True)
        model = construct(c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        output = model(*example(c, points=7)); output.square().mean().backward(); optimizer.step(); scheduler.step()
        from .wrapper_support import checkpoint_metadata
        metadata = checkpoint_metadata(c, model, optimizer, scheduler, epoch=1, total=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); write_metadata(root / "architecture.json", metadata)
            manifest = save_pair(root, model, metadata)
            loaded, loaded_metadata = load_model(root, "epoch_0001")
            self.assertEqual(loaded_metadata["config_hash"], c["config_hash"])
            sample = example(c, points=7)
            torch.testing.assert_close(loaded(*sample), model(*sample))
            script = ("from cdlno.linearno_loop.v3.checkpoint import load_model; "
                      "import sys; m,meta=load_model(sys.argv[1], 'epoch_0001'); "
                      "print(meta['config_hash'], len(m.state_dict()))")
            fresh = subprocess.run([sys.executable, "-B", "-c", script, str(root)],
                                   cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)
            self.assertIn(c["config_hash"], fresh.stdout)

    def test_air_native_weighted_synthetic_train_checkpoint_resume_eval(self):
        """Run the native Air loop on one PyG graph without radius sampling."""
        c = config("airfrans", cost="custom", latent=True, adapter=True)
        from cdlno.linearno_loop import air_entry
        from cdlno.linearno_loop.industrial_state import construct
        import numpy as np
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "Airfoil-Design-AirfRANS"))
        import train as air_train
        fake = dict(split="SYNTHETIC fixed", sampling="synthetic fixed",
                    checksums={"manifest.json": "a" * 64}, task_variant="full",
                    steps_per_epoch=1, scheduler_total_steps=2,
                    manifest_hash="a" * 64, train_count=1, validation_count=0,
                    manifest_keys=[])
        with tempfile.TemporaryDirectory() as tmp, patch.object(air_entry, "_data_spec", return_value=fake):
            directory = Path(tmp) / "airfrans-v3-synthetic"; directory.mkdir()
            args = SimpleNamespace(_linearno_loop_config=c, _linearno_config=c["base_profile_spec"],
                _linearno_model_spec=c["model_spec"], linearno_task="airfrans", task="full",
                seed=17, nmodel=1, nb_epochs=1, resume=False, eval=0, checkpoint="final",
                device="cpu", weight=c["profile_spec"]["values"]["objective"]["surface_weight"],
                linearno_run_dir=directory)
            coef = (np.zeros(4), np.ones(4), np.zeros(4), np.ones(4))
            graph = example(c, points=9)[0]
            graph.y = torch.randn(9, 4); graph.surf = torch.tensor([True, False] * 4 + [True])
            run = air_entry.AirRun(args, args._linearno_config, Path(tmp), coef, None,
                                   manifest={}, evaluation=False)
            hparams = dict(batch_size=1, nb_epochs=1, lr=1e-3, subsampling=9, r=.05,
                           max_neighbors=64, debug=0)
            model = construct(args)
            air_train.main("cpu", [graph], [graph], model, hparams, str(directory),
                           criterion="MSE_weighted", reg=args.weight, val_iter=10,
                           name_mod="LinearNO", val_sample=True, linearno_run=run)
            run.finish_ensemble([model])
            self.assertTrue((directory / "ensemble.json").is_file())
            args_eval = copy.copy(args); args_eval.eval = 1; args_eval._linearno_metadata = __import__(
                "linearno_loop.versioning", fromlist=["read_metadata"]).read_metadata(directory / "architecture.json")
            eval_run = air_entry.AirRun(args_eval, args_eval._linearno_config, Path(tmp), coef, None,
                                        manifest={}, evaluation=True)
            loaded = eval_run.load_models()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(tuple(loaded[0](graph).shape), (9, 4))

    def test_car_native_synthetic_train_checkpoint_resume(self):
        c = config("car", cost="custom", latent=True, adapter=True)
        from cdlno.linearno_loop import car_entry
        from cdlno.linearno_loop.industrial_state import construct
        fake = dict(split="SYNTHETIC fixed", sampling="synthetic fixed", checksums={"raw": "b" * 64},
                    fold_id=3, train_samples=["a"], test_samples=["b"], cfd_mesh=False,
                    r=.2, val_iter=1, preprocessed=1, ntrain=1, ntest=1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "car-v3-synthetic"; directory.mkdir()
            common = dict(_linearno_loop_config=c, _linearno_config=c["base_profile_spec"],
                _linearno_model_spec=c["model_spec"], linearno_task="car", seed=17,
                nb_epochs=1, lr=1e-3, batch_size=1, val_iter=1, weight=.5,
                device="cpu", linearno_run_dir=directory, eval=0, resume=False,
                checkpoint="final", fold_id=3, cfd_mesh=False, r=.2,
                data_dir=tmp)
            args = SimpleNamespace(**common)
            coef = (torch.zeros(7).numpy(), torch.ones(7).numpy(),
                    torch.zeros(4).numpy(), torch.ones(4).numpy())
            graph, geom = example(c, points=9)[0]
            graph.y = torch.randn(9, 4); graph.surf = torch.tensor([True, False] * 4 + [True])
            run = car_entry.CarRun(args, fake, coef, None)
            # Keep the resolved profile immutable while constraining this
            # synthetic native loop to one epoch.
            run.data["scheduler"]["epochs"] = 1
            run.data["scheduler"]["total_steps"] = 2
            model = construct(args)
            run.train("cpu", [(graph, geom)], [(graph, geom)], model,
                      dict(lr=1e-3, batch_size=1, nb_epochs=1), str(directory),
                      reg=.5, val_iter=1, coef_norm=coef, record=None)
            manifest = directory / "checkpoints" / "epoch_0001.json"
            self.assertTrue(manifest.is_file())
            resume_args = args
            resume_args.resume = True
            resume_args._linearno_metadata = __import__("linearno_loop.versioning", fromlist=["read_metadata"]).read_metadata(
                directory / "architecture.json")
            resume_args._linearno_checkpoint = manifest
            resumed = run
            model2 = construct(resume_args)
            optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                total_steps=2, final_div_factor=1000.)
            self.assertEqual(resumed.prepare(model2, optimizer, scheduler), 1)
            # Exercise the native Car evaluation loader after the same strict
            # checkpoint boundary used for resume.
            loaded = run.load()
            prediction = loaded((graph, geom))
            self.assertEqual(tuple(prediction.shape), (9, 4))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_torch_cluster_boundary_is_recordable(self):
        try:
            import torch_cluster  # noqa: F401
        except Exception as error:
            self.skipTest("torch_cluster unavailable: " + repr(error))


if __name__ == "__main__":
    unittest.main()
