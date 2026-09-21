import copy
import json
from pathlib import Path
import tempfile
import unittest

from linearno_loop.contracts import LoopSchemaError, canonical_json, seal
from linearno_loop.v2.schema import read_metadata, restore_config, validate_metadata, write_metadata
from support import config, metadata


class LF1SchemaTests(unittest.TestCase):
    def test_roundtrip_restore_and_mode_assertion(self):
        for mode in ("round_specific", "round_specific_latent"):
            saved = metadata(config(mode=mode))
            self.assertEqual(validate_metadata(json.loads(canonical_json(saved))), saved)
            restored = restore_config(saved, explicit={"linearno_loop": True, "core_ffn_mode": mode,
                "linearno_rank": saved["loop_spec"]["resolved_rank"]})
            self.assertEqual(restored["config"], saved["resolved_config"])
            wrong = "round_specific_latent" if mode == "round_specific" else "round_specific"
            with self.assertRaisesRegex(LoopSchemaError, "core_ffn_mode"):
                restore_config(saved, explicit={"core_ffn_mode": wrong})

    def test_version_topology_rank_and_seed_tampering_rejected(self):
        saved = metadata()
        cases = (("architecture_extension", "loop_linearno_v1"), ("schema_version", 1),
                 ("config_version", 1))
        for field, value in cases:
            bad = copy.deepcopy(saved); bad[field] = value
            with self.assertRaisesRegex(LoopSchemaError, field):
                validate_metadata(seal(bad, "metadata_hash"))
        for field, value in (("residual_mode", "rb_attnres"), ("resolved_rank", 128),
                             ("core_ffn_mode", "round_specific_latent")):
            bad = copy.deepcopy(saved); bad["loop_spec"][field] = value
            with self.assertRaisesRegex(LoopSchemaError, field):
                validate_metadata(seal(bad, "metadata_hash"))

    def test_create_only_disk_roundtrip(self):
        saved = metadata()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "architecture.json"
            write_metadata(path, saved)
            self.assertEqual(read_metadata(path), saved)
            with self.assertRaises(FileExistsError):
                write_metadata(path, saved)


if __name__ == "__main__":
    unittest.main()
