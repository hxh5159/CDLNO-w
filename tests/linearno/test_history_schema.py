"""R1 schema tests; no model, factory, task entry, or checkpoint construction."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import unittest

from linearno_history.schema import (
    ARCHITECTURE_EXTENSION, HistorySchemaError, assert_structural_compatibility,
    derive_fair_seeds, feature_signature, resolve_feature_config,
    run_directory_id, validate_family_feature_config, validate_innovation_spec,
    validate_legacy_metadata, validate_research_metadata, validate_strict_load_policy,
)

FIXTURE = Path(__file__).parent / "fixtures" / "history_r1_schema.json"


class HistorySchemaContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_feature_truth_and_derived_signature(self):
        for sig, cfg in self.fixture["feature_configs"].items():
            resolved = resolve_feature_config(cfg)
            self.assertEqual(resolved["feature_signature"], sig)
            self.assertEqual(resolved["linearno_latent_attnres"], sig[1] == "1")
            self.assertEqual(resolved["linearno_history_k_conditioning"], sig[3] == "1")
        self.assertEqual(resolve_feature_config(self.fixture["feature_configs"]["A1K0"])["linearno_attnres_history_dropout_p"], .1)
        self.assertEqual(feature_signature(False, False), "A0K0")

    def test_feature_validation_rejects_invalid_types_and_families(self):
        cfg = self.fixture["feature_configs"]["A0K0"]
        for key, value in (("linearno_latent_attnres", "false"), ("linearno_history_k_conditioning", 0)):
            bad = dict(cfg); bad[key] = value
            with self.assertRaises(HistorySchemaError): resolve_feature_config(bad)
        bad = dict(cfg); bad["linearno_attnres_history_dropout_p"] = .1
        with self.assertRaises(HistorySchemaError): resolve_feature_config(bad)
        bad = dict(self.fixture["feature_configs"]["A1K0"]); bad["linearno_attnres_history_dropout_p"] = 0.
        with self.assertRaises(HistorySchemaError): resolve_feature_config(bad)
        with self.assertRaises(HistorySchemaError): validate_family_feature_config("transolver", self.fixture["feature_configs"]["A1K0"])
        self.assertEqual(validate_family_feature_config("linearno", cfg)["feature_signature"], "A0K0")

    def test_research_metadata_validates_without_constructing(self):
        metadata = self.fixture["valid_research_metadata"]
        result = validate_research_metadata(metadata, feature_config=self.fixture["feature_configs"]["A1K1"])
        self.assertEqual(result["architecture_extension"], ARCHITECTURE_EXTENSION)
        self.assertEqual(result["innovation_spec"]["features"]["feature_signature"], "A1K1")
        probe = subprocess.run([sys.executable, "-c", "import sys; import linearno_history.schema; assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)"], check=False)
        self.assertEqual(probe.returncode, 0)

    def test_each_research_signature_requires_matching_spec(self):
        metadata = self.fixture["valid_research_metadata"]
        for sig in ("A1K0", "A0K1"):
            bad = json.loads(json.dumps(metadata)); bad["feature_signature"] = sig
            with self.assertRaises(HistorySchemaError): validate_research_metadata(bad)
        bad = json.loads(json.dumps(metadata)); bad["innovation_spec"]["features"]["feature_signature"] = "A0K1"
        with self.assertRaises(HistorySchemaError): validate_research_metadata(bad)

    def test_legacy_checkpoint_is_never_guessed_as_research(self):
        legacy = self.fixture["legacy_metadata"]
        self.assertEqual(validate_legacy_metadata(legacy), "legacy_linearno")
        with self.assertRaises(HistorySchemaError): validate_legacy_metadata(legacy, self.fixture["feature_configs"]["A1K0"])
        bad = dict(legacy); bad["innovation_spec"] = {}
        with self.assertRaises(HistorySchemaError): validate_legacy_metadata(bad)

    def test_load_is_strict_and_structure_checked_before_weights(self):
        validate_strict_load_policy(True)
        with self.assertRaises(HistorySchemaError): validate_strict_load_policy(False)
        meta = self.fixture["valid_research_metadata"]
        assert_structural_compatibility(meta, meta)
        bad = json.loads(json.dumps(meta)); bad["innovation_spec"]["base_linearno"]["n_layers"] = 8
        with self.assertRaises(HistorySchemaError): assert_structural_compatibility(meta, bad)
        bad = json.loads(json.dumps(meta)); bad["feature_signature"] = "A0K1"
        with self.assertRaises(HistorySchemaError): assert_structural_compatibility(meta, bad)
        bad = json.loads(json.dumps(meta)); bad["model_spec"]["class_path"] = "cdlno.linearno_history.core.OtherCore"
        with self.assertRaises(HistorySchemaError): validate_research_metadata(bad)

    def test_run_directory_and_fair_seed_protocol(self):
        self.assertEqual(run_directory_id("darcy", "official_release", 8, True, False, 7), "darcy__official_release__L8__A1K0__seed7")
        with self.assertRaises(HistorySchemaError): run_directory_id("darcy", "official_release", 3, False, False, 7)
        a = derive_fair_seeds(7, task="darcy", split="train")
        b = derive_fair_seeds(7, task="darcy", split="train")
        c = derive_fair_seeds(7, task="darcy", split="test")
        self.assertEqual(a, b); self.assertNotEqual(a["dataloader_generator_seed"], c["dataloader_generator_seed"])
        self.assertEqual(a["public_backbone_seed"], 7)


if __name__ == "__main__": unittest.main()

class HistoryMatrixFixtures(unittest.TestCase):
    def test_core_and_wrapper_matrix_are_complete_without_model_construction(self):
        matrix = json.loads((Path(__file__).parents[2] / "docs/linearno_history_audit/r1/configuration_matrix.json").read_text())
        self.assertEqual(matrix["core_matrix_count"], 20)
        self.assertEqual(len(matrix["core_matrix"]), 20)
        self.assertEqual({row["feature_signature"] for row in matrix["core_matrix"]}, {"A0K0", "A1K0", "A0K1", "A1K1"})
        self.assertEqual({row["n_layers"] for row in matrix["core_matrix"]}, {4, 5, 6, 7, 8})
        self.assertEqual(matrix["wrapper_matrix_count"], 32)
        self.assertEqual(len(matrix["wrapper_matrix"]), 32)
        self.assertTrue(matrix["no_real_data"] and matrix["no_launcher_integration"])

    def test_negative_fixture_is_explicit(self):
        cases = json.loads((Path(__file__).parents[2] / "docs/linearno_history_audit/r1/negative_cases.json").read_text())["cases"]
        ids = {case["id"] for case in cases}
        for required in ("bool-string", "dropout-off", "feature-weight-mismatch", "strict-false", "all-history-mask", "cross-forward-cache", "disabled-extra-state", "depth-low", "depth-high"):
            self.assertIn(required, ids)

    def test_machine_schema_is_json_and_has_closed_top_level(self):
        schema = json.loads((Path(__file__).parents[2] / "docs/linearno_history_audit/r1/innovation_schema.json").read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["family"]["const"], "linearno_history")
