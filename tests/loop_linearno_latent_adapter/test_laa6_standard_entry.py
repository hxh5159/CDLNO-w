import json
from pathlib import Path
import tempfile
import unittest

from cdlno_entry import parse_args
from linearno.static_worker import parser_for
from linearno.temporal_worker import parser_for as temporal_parser
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR


TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity")


def parse(task, tokens):
    parser = temporal_parser if task in ("ns", "plasticity") else parser_for
    return parse_args(parser(task), task, list(map(str, tokens)))


def base(task):
    model = "LinearNO_Irregular_Mesh" if task == "elasticity" else "LinearNO_Structured_Mesh_2D"
    return ["--model", model, "--linearno-loop", "1",
            "--linearno-loop-architecture", ARCHITECTURE_SELECTOR,
            "--linearno-loop-cost-profile", "efficient_v1",
            "--linearno-loop-topology", "d12",
            "--linearno-loop-residual-mode", "sr_1_over_r",
            "--linearno-loop-latent", "0",
            "--linearno-loop-adapter-mode", "none",
            "--linearno-rank", "32" if task == "ns" else "64", "--seed", "7"]


class LAA6StandardEntryTests(unittest.TestCase):
    def test_explicit_v3_all_six_tasks_and_constructor_bridge(self):
        for task in TASKS:
            args = parse(task, base(task))
            config = args._linearno_loop_config
            self.assertEqual(config["config_version"], 3)
            self.assertEqual(config["architecture_extension"], "loop_linearno_latent_adapter_v3")
            self.assertEqual(config["loop_spec"]["cost_profile"], "efficient_v1")
            self.assertFalse(config["loop_spec"]["latent_enabled"])
            self.assertEqual(config["loop_spec"]["adapter_mode"], "none")
            self.assertEqual(args.linearno_rank, 32 if task == "ns" else 64)
            self.assertIn(config["config_hash"], args.linearno_run_dir.name)
            from cdlno.linearno_loop.standard_entry import model_kwargs, model_module
            kwargs = model_kwargs(args, H=config["loop_spec"]["grid_height"],
                                  W=config["loop_spec"]["grid_width"])
            self.assertEqual(kwargs["n_hidden"], config["loop_spec"]["hidden_width"])
            self.assertEqual(kwargs["linearno_rank"], 32 if task == "ns" else 64)
            self.assertIsNotNone(model_module(args))

    def test_v1_without_architecture_is_unchanged(self):
        args = parse("darcy", ["--model", "LinearNO_Structured_Mesh_2D", "--linearno-loop", "1",
                                "--linearno-loop-topology", "p2_c2_r2_s2",
                                "--linearno-loop-residual-mode", "sr_1_over_r"])
        self.assertEqual(args._linearno_loop_config["config_version"], 1)

    def test_v3_saved_sidecar_is_metadata_selected(self):
        args = parse("darcy", base("darcy"))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "saved"
            directory.mkdir()
            (directory / "architecture.json").write_text(json.dumps({
                "family": "linearno_loop",
                "architecture_extension": "loop_linearno_latent_adapter_v3",
                "config_version": 3,
            }))
            from linearno_loop.versioning import is_v3, read_json
            self.assertTrue(is_v3(read_json(directory / "architecture.json")))
            self.assertEqual(args._linearno_loop_config["config_version"], 3)

    def test_cost_profile_without_architecture_does_not_select_v3(self):
        with self.assertRaises(SystemExit):
            parse("darcy", ["--model", "LinearNO_Structured_Mesh_2D", "--linearno-loop", "1",
                             "--linearno-loop-cost-profile", "efficient_v1"])


if __name__ == "__main__":
    unittest.main()
