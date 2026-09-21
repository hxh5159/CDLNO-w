import copy
import importlib
import json
from pathlib import Path
import unittest

from linearno_loop.config import resolve_config as resolve_v1
from linearno_loop.contracts import LoopSchemaError, canonical_json
from linearno_loop.v2.config import latent_width, resolve_config, run_directory_id, validate_config
from linearno_loop.v2.contracts import CORE_FFN_MODES
from linearno_loop.v2.matrix import configuration_matrix


def options(**extra):
    return {"topology_preset": "p1_c3_r2_s1", "residual_mode": "sr_1_over_r",
            "core_ffn_mode": "round_specific", **extra}


class LF1ConfigTests(unittest.TestCase):
    def test_v1_exact_replay_and_v2_default_rank(self):
        request = {"topology_preset": "p1_c3_r2_s1", "residual_mode": "sr_1_over_r"}
        before = canonical_json(resolve_v1("darcy", options=request))
        v2 = resolve_config("darcy", options=options())
        self.assertEqual(canonical_json(resolve_v1("darcy", options=request)), before)
        self.assertEqual(v2["loop_spec"]["resolved_rank"], v2["loop_spec"]["base_rank"])
        self.assertEqual(v2["loop_spec"]["rank_multiplier"], 1)
        self.assertEqual(v2["architecture_extension"], "loop_linearno_ffn_v2")
        self.assertEqual(v2["config_version"], 2)

    def test_modes_ownership_width_and_constructor(self):
        for task in ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car"):
            for mode in CORE_FFN_MODES:
                config = resolve_config(task, options=options(core_ffn_mode=mode))
                loop = config["loop_spec"]
                self.assertEqual(loop["point_ffn_instance_count"], 1 + 3 * 2 + 1)
                self.assertEqual(loop["latent_ffn"]["instance_count"], 3 if mode.endswith("latent") else 0)
                self.assertEqual(config["model_spec"]["constructor_kwargs"]["core_ffn_mode"], mode)
                self.assertIn("feature_seed", config["model_spec"]["constructor_kwargs"])
                self.assertEqual(validate_config(json.loads(canonical_json(config))), config)
        self.assertEqual(latent_width("conv_temp", 128), 572)
        self.assertEqual(latent_width("plain", 256), 256)

    def test_strict_invalid_types_unknown_and_rank_conflicts(self):
        bad = [
            {}, {"topology_preset": "p1_c3_r2_s1", "residual_mode": "sr_1_over_r"},
            options(core_ffn_mode="stacked"), options(core_ffn_mode=True),
            options(rank_multiplier=True), options(rank_multiplier=3),
            options(rank_multiplier=2, linearno_rank=64), options(unexpected=1),
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(LoopSchemaError):
                resolve_config("darcy", options=value)
        for value in ("64", 64.0, True, 0):
            with self.assertRaises(LoopSchemaError):
                resolve_config("darcy", options=options(linearno_rank=value))

    def test_run_ids_and_full_planned_matrix(self):
        ids = set()
        for mode in CORE_FFN_MODES:
            for multiplier in (1, 2):
                config = resolve_config("darcy", options=options(core_ffn_mode=mode, rank_multiplier=multiplier))
                identifier = run_directory_id(config)
                self.assertIn(mode, identifier); self.assertIn("__v2__", identifier)
                ids.add(identifier)
        self.assertEqual(len(ids), 4)
        matrix = configuration_matrix()
        self.assertEqual(len(matrix["runs"]), 8 * 2 * 3 * 2 * 3)
        self.assertEqual(len(matrix["controls"]), 16)
        self.assertEqual(len({row["run_id"] for row in matrix["runs"]}), len(matrix["runs"]))

    def test_schema_modules_do_not_import_torch_or_task_entries(self):
        for name in ("linearno_loop.v2.contracts", "linearno_loop.v2.config", "linearno_loop.v2.schema"):
            module = importlib.import_module(name)
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("import torch", source)
            self.assertNotIn("standard_entry", source)

    def test_tampering_is_not_resealed_into_a_valid_config(self):
        config = resolve_config("darcy", options=options())
        for path, value in (("core_ffn_mode", "round_specific_latent"),
                            ("point_ffn_instance_count", 9), ("resolved_rank", 128),
                            ("feature_timestep_encoding", True)):
            bad = copy.deepcopy(config); bad["loop_spec"][path] = value
            from linearno_loop.contracts import seal
            with self.assertRaisesRegex(LoopSchemaError, path):
                validate_config(seal(bad, "config_hash"))


if __name__ == "__main__":
    unittest.main()
