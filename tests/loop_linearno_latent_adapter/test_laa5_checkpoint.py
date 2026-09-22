import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from linearno_loop.v3.schema import write_metadata
from .wrapper_support import (CHECKPOINT_ROWS, checkpoint_metadata, config,
                              construct, example, invoke)


ROOT = Path(__file__).resolve().parents[2]


class CheckpointTests(unittest.TestCase):
    def _save(self, root, task, *, member=0):
        from cdlno.linearno_loop.v3 import checkpoint
        c = config(task, seed=31 + member)
        model = construct(c, member_seed=401 + member)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        output = invoke(model, c, example(c, batch=1)); output.square().mean().backward()
        optimizer.step(); scheduler.step()
        generators = {"train": torch.Generator().manual_seed(71 + member)}
        metadata = checkpoint_metadata(c, model, optimizer, scheduler,
                                       generators=generators)
        directory = root / task
        directory.mkdir(); write_metadata(directory / "architecture.json", metadata)
        manifest = checkpoint.save_pair(directory, model, metadata)
        return c, model, optimizer, scheduler, generators, metadata, manifest

    def test_three_wrappers_pair_hash_and_fresh_process_strict_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            for task in ("elasticity", "airfrans", "car"):
                c, model, optimizer, scheduler, generators, metadata, manifest = self._save(Path(temp), task)
                from cdlno.linearno_loop.v3 import checkpoint
                checked, state = checkpoint.read_pair(manifest, expected=c)
                self.assertEqual(checked, metadata)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(state[key], value.cpu(), atol=0, rtol=0)
                env = dict(os.environ, PYTHONPATH=str(ROOT / "tests") + os.pathsep + str(ROOT),
                           PYTHONDONTWRITEBYTECODE="1")
                explicit = json.dumps({"task": task, "residual_mode": c["loop_spec"]["residual_mode"]})
                result = subprocess.run([sys.executable, "-B", "-m",
                    "loop_linearno_latent_adapter.laa5_worker", str(manifest.parent.parent),
                    manifest.stem, explicit], cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                row = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(row["config_hash"], c["config_hash"])
                self.assertEqual(row["state_keys"], list(model.state_dict()))
                self.assertTrue(row["eval_finite"] and row["resume_finite"])
                self.assertEqual((row["restored_epoch"], row["continued_scheduler_epoch"]), (1, 2))
                CHECKPOINT_ROWS.append(dict(task=task, format=checkpoint.FORMAT,
                                            fresh_process=True, status="PASS"))

    def test_metadata_first_conflicts_precede_import_construction_and_tensor_load(self):
        with tempfile.TemporaryDirectory() as temp:
            c, _, _, _, _, _, manifest = self._save(Path(temp), "elasticity")
            from cdlno.linearno_loop.v3 import checkpoint
            conflicts = ({"task": "car"}, {"residual_mode": "rb_attnres"},
                         {"actual_M": 7}, {"hidden_width": 10},
                         {"latent_enabled": False}, {"adapter_rank": 3})
            for explicit in conflicts:
                with self.subTest(explicit=explicit), \
                     patch("torch.load", side_effect=AssertionError("tensor read before metadata conflict")), \
                     patch("importlib.import_module", side_effect=AssertionError("model import before metadata conflict")):
                    with self.assertRaises(ValueError):
                        checkpoint.load_model(manifest.parent.parent, manifest.stem,
                                              explicit=explicit)
            # A v2 manifest must be rejected by format before any tensor read.
            record = json.loads(manifest.read_text()); record["format"] = "linearno-loop-epoch-pair-v2"
            manifest.write_text(json.dumps(record))
            with patch("torch.load", side_effect=AssertionError("cross-version tensor read")):
                with self.assertRaises(ValueError): checkpoint.inspect_checkpoint(manifest.parent.parent, manifest.stem)

    def test_optimizer_scheduler_scaler_rng_normalizer_and_ensemble_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            c, model, optimizer, scheduler, generators, metadata, manifest = self._save(Path(temp), "airfrans")
            from cdlno.linearno.schema import unpack_state
            from cdlno.linearno_loop.v3 import checkpoint
            state = metadata["resume_state"]
            checkpoint.validate_optimizer_state(unpack_state(state["optimizer"]), optimizer)
            bad = copy.deepcopy(unpack_state(state["optimizer"]))
            bad["param_groups"][0]["params"][1] = bad["param_groups"][0]["params"][0]
            with self.assertRaises(ValueError): checkpoint.validate_optimizer_state(bad, optimizer)
            with self.assertRaises(ValueError):
                checkpoint.validate_scaler_state({"scale": torch.tensor(1.)}, None)
            tampered = copy.deepcopy(metadata); tampered["normalizer_spec"]["records"]["input"]["algorithm"] = "other"
            with self.assertRaises(ValueError): checkpoint.validate_resume_metadata(metadata, tampered)
            tampered = copy.deepcopy(metadata); tampered["ensemble_manifest"] = [{
                "member_id": "0", "order": 0, "path": "members/0.pt",
                "sha256": hashlib.sha256(b"member").hexdigest(), "format": "state_dict"}]
            with self.assertRaises(ValueError): checkpoint.validate_resume_metadata(metadata, tampered)
            fresh = construct(c)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
            fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=2)
            _, weights = checkpoint.read_pair(manifest, expected=c)
            checkpoint.restore_training_state(fresh, fresh_optimizer, fresh_scheduler, metadata,
                weights=weights, generators=generators, sampler={"scope": "synthetic"},
                normalizer_spec=metadata["normalizer_spec"], scaler=None)

    def test_rng_generator_contract_conflict_precedes_weight_application(self):
        with tempfile.TemporaryDirectory() as temp:
            c, _, _, _, generators, metadata, manifest = self._save(Path(temp), "elasticity")
            from cdlno.linearno_loop.v3 import checkpoint
            _, weights = checkpoint.read_pair(manifest, expected=c)
            fresh = construct(c)
            optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            before = {key: value.detach().clone() for key, value in fresh.state_dict().items()}
            with self.assertRaises(ValueError):
                checkpoint.restore_training_state(fresh, optimizer, scheduler, metadata,
                    weights=weights, generators={},
                    sampler={"scope": "synthetic"},
                    normalizer_spec=metadata["normalizer_spec"], scaler=None)
            for key, value in fresh.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_enabled_cpu_scaler_and_real_ensemble_manifest(self):
        from cdlno.linearno_loop.v3 import checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); c = config("airfrans"); model = construct(c)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            try:
                scaler = torch.amp.GradScaler("cpu")
            except TypeError:
                self.skipTest("installed Torch lacks CPU GradScaler")
            optimizer.zero_grad()
            loss = invoke(model, c, example(c, batch=1)).float().square().mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
            member = root / "members" / "member_0.pt"; member.parent.mkdir()
            torch.save(model.state_dict(), member)
            manifest = [dict(member_id="0", order=0,
                path=str(member.relative_to(root)), sha256=checkpoint.sha256(member),
                format="state_dict")]
            generators = {"train": torch.Generator().manual_seed(91)}
            metadata = checkpoint_metadata(c, model, optimizer, scheduler,
                generators=generators, scaler=scaler, ensemble=manifest)
            write_metadata(root / "architecture.json", metadata)
            pair = checkpoint.save_pair(root, model, metadata)
            checked, weights = checkpoint.read_pair(pair, expected=c)
            fresh = construct(c); opt2 = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=2)
            scaler2 = torch.amp.GradScaler("cpu")
            checkpoint.restore_training_state(fresh, opt2, sch2, checked,
                weights=weights, generators=generators, sampler={"scope": "synthetic"},
                normalizer_spec=checked["normalizer_spec"], scaler=scaler2)
            self.assertEqual(scaler2.state_dict(), scaler.state_dict())
            member.write_bytes(b"corrupt")
            with self.assertRaises(ValueError): checkpoint.inspect_checkpoint(root, pair.stem)

    def test_refuses_overwrite_and_v1_v2_readers_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            _, model, _, _, _, metadata, manifest = self._save(Path(temp), "car")
            before = {path: path.read_bytes() for path in manifest.parent.parent.rglob("*") if path.is_file()}
            from cdlno.linearno_loop.v3 import checkpoint
            self.assertEqual(checkpoint.save_pair(manifest.parent.parent, model, metadata), manifest)
            after = {path: path.read_bytes() for path in manifest.parent.parent.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            with torch.no_grad(): next(model.parameters()).add_(1)
            with self.assertRaises(ValueError): checkpoint.save_pair(manifest.parent.parent, model, metadata)
            import cdlno.linearno_loop.checkpoint as v1
            import cdlno.linearno_loop.v2.checkpoint as v2
            self.assertEqual(v1.FORMAT, "linearno-loop-epoch-pair-v1")
            self.assertEqual(v2.FORMAT, "linearno-loop-epoch-pair-v2")

    def test_v1_v2_pairs_fresh_process_replay_without_byte_changes(self):
        from linearno_loop.contracts import seal
        from cdlno.linearno_loop.construction import build_from_config as build_v1
        from cdlno.linearno_loop.v2.construction import build_from_config as build_v2
        from loop_linearno.support import config as config_v1, metadata as metadata_v1
        from loop_linearno_ffn.support import config as config_v2, metadata as metadata_v2
        cases = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for version, make_config, make_metadata, build in (
                    (1, config_v1, metadata_v1, build_v1),
                    (2, config_v2, metadata_v2, build_v2)):
                directory = root / f"v{version}"; directory.mkdir()
                c = make_config(); model = build(c); metadata = make_metadata(c)
                metadata["resume_state"]["epoch"] = 1
                metadata["resume_state"]["global_step"] = 1
                metadata = seal(metadata, "metadata_hash")
                if version == 1:
                    from linearno_loop.schema import write_metadata as write
                    from cdlno.linearno_loop import checkpoint as backend
                else:
                    from linearno_loop.v2.schema import write_metadata as write
                    from cdlno.linearno_loop.v2 import checkpoint as backend
                write(directory / "architecture.json", metadata)
                manifest = backend.save_pair(directory, model, metadata)
                before = {path.relative_to(directory): path.read_bytes()
                          for path in directory.rglob("*") if path.is_file()}
                env = dict(os.environ,
                    PYTHONPATH=str(ROOT / "tests") + os.pathsep + str(ROOT),
                    PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
                result = subprocess.run([sys.executable, "-B", "-m",
                    "loop_linearno_latent_adapter.legacy_pair_worker", str(version),
                    str(directory), manifest.stem], cwd=ROOT, env=env,
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                row = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(row["config_hash"], c["config_hash"])
                after = {path.relative_to(directory): path.read_bytes()
                         for path in directory.rglob("*") if path.is_file()}
                self.assertEqual(before, after)
                cases.append(row)
        self.assertEqual([row["format"] for row in cases],
                         ["linearno-loop-epoch-pair-v1", "linearno-loop-epoch-pair-v2"])


if __name__ == "__main__": unittest.main()
