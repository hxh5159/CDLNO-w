"""LAA9 accounting checks; all inputs are synthetic and in-memory."""
import unittest

from linearno_loop.v3.costs import analytic_cost
from tools.linearno_loop_accounting import analytic, analytic_v2, analytic_v3, measured_parameters
from . import __path__  # noqa: F401  (keeps this directory a test package)
from tools.linearno_loop_laa9 import ABLATIONS, DEPTHS, MODES, PROFILES, TASKS, matrix_rows, small_config
from cdlno.linearno_loop.v3.construction import build_from_config


class LAA9AccountingTests(unittest.TestCase):
    def test_full_parse_matrix_has_all_axes(self):
        rows = matrix_rows()
        self.assertEqual(len(rows), 768)
        self.assertEqual({row["profile"] for row in rows}, set(PROFILES))
        self.assertEqual({row["depth"] for row in rows}, set(DEPTHS))
        self.assertEqual({row["residual_mode"] for row in rows}, set(MODES))
        self.assertEqual({(row["latent_enabled"], row["adapter_enabled"]) for row in rows}, set(ABLATIONS))

    def test_v3_dispatch_and_measured_d12_partition(self):
        config = small_config("elasticity", "matched_v1", "d12", "sr_1_over_r", True, True)
        expected = analytic_cost(config)
        self.assertEqual(analytic_v3(config), expected)
        self.assertEqual(analytic(config), expected)
        model = build_from_config(config)
        groups = measured_parameters(model)
        self.assertEqual(groups, expected["parameter_groups"])
        self.assertEqual(sum(groups.values()), expected["total_parameters"])

    def test_off_features_have_no_feature_keys_and_router_is_mode_local(self):
        for mode in MODES:
            config = small_config("darcy", "efficient_v1", "d12", mode, False, False)
            model = build_from_config(config)
            keys = list(model.state_dict())
            self.assertFalse(any("latent_processor" in key or ".adapter." in key for key in keys))
            if mode == "sr_1_over_r":
                self.assertFalse(any(key.startswith("loop.rb_") or key.startswith("loop.lb_") for key in keys))
            elif mode == "rb_attnres":
                self.assertTrue(any(key.startswith("loop.rb_") for key in keys))
                self.assertFalse(any(key.startswith("loop.lb_") for key in keys))
            else:
                self.assertTrue(any(key.startswith("loop.lb_") for key in keys))
                self.assertFalse(any(key.startswith("loop.rb_") for key in keys))

    def test_v2_dispatch_is_byte_contract_independent(self):
        from linearno_loop.config import resolve_config as v1_resolve
        from linearno_loop.v2.config import resolve_config as old_resolve
        v1 = v1_resolve("darcy", options={"topology_preset": "p1_c3_r2_s1",
                                           "residual_mode": "sr_1_over_r",
                                           "linearno_rank": 4},
                            profile_overrides={"model.hidden": 8, "model.heads": 2,
                                               "model.H": 3, "model.W": 5})
        # The generic dispatcher still returns the exact historical V1 shape.
        self.assertNotIn("cost_version", analytic(v1))
        self.assertEqual(analytic(v1)["parameters"], sum(analytic(v1)["parameter_parts"].values()))
        old = old_resolve("darcy", options={"topology_preset": "p1_c3_r2_s1",
                                              "residual_mode": "sr_1_over_r",
                                              "core_ffn_mode": "round_specific"})
        self.assertEqual(analytic(old), analytic_v2(old))


if __name__ == "__main__":
    unittest.main()
