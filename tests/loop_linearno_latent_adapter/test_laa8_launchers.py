"""LAA8 V3 launcher, saved-config and recording ownership checks."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from .wrapper_support import config, construct, example, checkpoint_metadata
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.schema import restore_config
from tran_evaluate.linearno_loop import recording


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "tran_evaluate" / "linearno_loop_v3"
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
PROFILES = ("matched_v1", "efficient_v1")


def run_script(profile, task, *args):
    return subprocess.run(
        ["bash", str(V3 / profile / f"{task}.sh"), *map(str, args)],
        cwd=ROOT, text=True, capture_output=True,
    )


class LAA8LauncherTests(unittest.TestCase):
    def test_shell_syntax_and_all_16_real_parser_previews(self):
        for profile in PROFILES:
            for task in TASKS:
                script = V3 / profile / f"{task}.sh"
                syntax = subprocess.run(["bash", "-n", str(script)], cwd=ROOT)
                self.assertEqual(syntax.returncode, 0, script)
                result = run_script(profile, task, "preview", "--gpu", 0, "--seed", 17)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                plan = json.loads(result.stdout)
                config = plan["config"]
                self.assertEqual(config["config_version"], 3)
                self.assertEqual(config["loop_spec"]["cost_profile"], profile)
                self.assertEqual(config["loop_spec"]["executed_depth"], 12)
                self.assertEqual(tuple(config["loop_spec"][key] for key in
                                      ("prefix_blocks", "recurrent_core_blocks", "loop_repeats", "suffix_blocks")),
                                 (2, 4, 2, 2))
                self.assertEqual(config["loop_spec"]["residual_mode"], "sr_1_over_r")
                self.assertTrue(config["loop_spec"]["latent_enabled"])
                self.assertEqual(config["loop_spec"]["adapter_mode"],
                                 "bilateral_qk_lowrank_second_visit")
                self.assertEqual(config["loop_spec"]["adapter_rank"], 4)
                self.assertEqual(config["loop_spec"]["adapter_alpha"], 4.0)
                expected_m = 32 if task in ("ns", "airfrans", "car") else 64
                self.assertEqual(config["loop_spec"]["actual_M"], expected_m)
                self.assertIn(config["config_hash"], plan["run"])

    def test_default_override_order_paths_and_unique_run_ids(self):
        first = run_script("matched_v1", "airfoil", "print-run-dir", "--seed", 4,
                           "--output-root", "/tmp/LAA8 output with spaces")
        second = run_script("matched_v1", "airfoil", "print-run-dir", "--seed", 4,
                            "--output-root", "/tmp/LAA8 output with spaces")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(first.stdout.strip(), second.stdout.strip())
        self.assertIn("/tmp/LAA8 output with spaces", first.stdout)
        self.assertRegex(first.stdout, r"__matched_v1__P2-C4-R2-S2__sr_1_over_r__")
        self.assertFalse(Path(first.stdout.strip()).exists())

        override = run_script("matched_v1", "elasticity", "dry-run", "--seed", 2,
                              "--gpu", 1, "--linearno-loop-topology", "d20",
                              "--linearno-loop-residual-mode", "rb_attnres",
                              "--linearno-loop-latent", 0,
                              "--linearno-loop-adapter-mode", "none")
        self.assertEqual(override.returncode, 0, override.stdout + override.stderr)
        self.assertIn("P2-C8-R2-S2", override.stdout)
        self.assertIn("rb_attnres", override.stdout)
        self.assertIn("__z0a0", override.stdout)
        ordered = run_script("matched_v1", "darcy", "train_eval", "--dry-run", "--gpu", 1)
        self.assertEqual(ordered.returncode, 0, ordered.stdout + ordered.stderr)
        self.assertIn("Then eval same RUN only after successful training", ordered.stdout)
        spaced_data = run_script("efficient_v1", "airfrans", "preview", "--gpu", 0,
                                 "--data-root", "/tmp/AirfRANS data with spaces")
        self.assertEqual(spaced_data.returncode, 0, spaced_data.stdout + spaced_data.stderr)
        self.assertIn("/tmp/AirfRANS data with spaces", spaced_data.stdout)

    def test_profile_conflict_custom_requirements_and_error_propagation(self):
        conflict = run_script("matched_v1", "darcy", "dry-run",
                              "--linearno-loop-cost-profile", "efficient_v1")
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("profile conflict", conflict.stderr)
        missing = run_script("custom", "darcy", "dry-run")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("custom requires explicit hidden_width", missing.stderr)
        custom = run_script("custom", "darcy", "dry-run",
                            "--linearno-loop-hidden-width", 64,
                            "--linearno-loop-latent-width", 128,
                            "--linearno-rank", 32,
                            "--linearno-loop-topology", "custom",
                            "--linearno-loop-prefix-blocks", 2,
                            "--linearno-loop-core-blocks", 2,
                            "--linearno-loop-repeats", 2,
                            "--linearno-loop-suffix-blocks", 2,
                            "--linearno-loop-residual-mode", "sr_1_over_r",
                            "--linearno-loop-latent", 0,
                            "--linearno-loop-adapter-mode", "none",
                            "--linearno-loop-adapter-rank", 4,
                            "--linearno-loop-adapter-alpha", 4)
        self.assertEqual(custom.returncode, 0, custom.stdout + custom.stderr)
        self.assertIn("__custom__P2-C2-R2-S2__", custom.stdout)

    def test_saved_config_restore_and_conflict_before_tensor_load(self):
        resolved = config("darcy", cost="custom", latent=True, adapter=True)
        model = construct(resolved)
        import torch
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        metadata = checkpoint_metadata(resolved, model, optimizer, scheduler, epoch=0, total=2)
        restored = restore_config(metadata)
        self.assertEqual(restored["config"]["config_hash"], resolved["config_hash"])
        explicit = {"architecture": "operator_latent_adapter_v3", "task": "darcy",
                    "cost_profile": "custom", "linearno_rank": 6}
        with self.assertRaises(ValueError):
            restore_config(metadata, explicit=explicit)

    def test_v3_recording_reports_shared_core_not_v2_round_ffn(self):
        import torch
        c = config("darcy", cost="custom", latent=True, adapter=True)
        model = construct(c)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(_linearno_loop_config=c, linearno_run_dir=Path(tmp),
                                   eval=0, resume=False, seed=17)
            recording.observe(args, model)
            model(*example(c, batch=1))
            manifest = json.loads((Path(tmp) / recording.MANIFEST).read_text())
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["ownership"], "shared_complete_core_block_across_rounds")
        self.assertNotIn("core_ffns", " ".join(manifest["expected_call_schedule"]))
        member = manifest["members"]["member_000"]
        self.assertEqual(member["actual_call_schedule"], manifest["expected_call_schedule"])
        self.assertEqual(member["parameter_parts"]["shared_core"] > 0, True)
        self.assertEqual(member["parameter_parts"]["adapters"] > 0, True)
        self.assertEqual(sum("latent_ffn" in event for event in member["actual_call_schedule"]), 8)
        self.assertEqual(sum("adapter" in event for event in member["actual_call_schedule"]), 4)


if __name__ == "__main__":
    unittest.main()
