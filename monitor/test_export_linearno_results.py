"""Synthetic artifact tests: no torch, models, checkpoints, datasets or GPU."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("export_linearno_results.py")
spec = importlib.util.spec_from_file_location("pure_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pure monitor export ")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "LinearNO-monitor"
        self.root = self.repo / "output"
        self.root.mkdir(parents=True)

    def fixture(self, name="baseline", family="linearno", flags=()):
        run = self.root / name / "darcy"
        monitor = self.root / name / "monitor-darcy-train"
        write_json(run / "config.json", dict(model=family, resolved_arguments=dict(seed=1)))
        write_json(run / "architecture.json", dict(family=family, model_spec=dict(
            class_path="model.LinearNO.Model", constructor_kwargs=dict(n_layers=2)),
            resume_state={"optimizer": "MUST NOT EXPORT"}))
        (run / "model.pt").write_bytes(b"MUST NOT EXPORT")
        command = ["bash", "tran_evaluate/linearno/darcy_train.sh", "--experiment-dir", str(run), *flags]
        write_json(monitor / "runtime_config.json", dict(capture_every_validations=10,
            max_samples_per_snapshot=8, command=command))
        write_json(monitor / "run_summary.json", {"returncode": 0})
        for step in range(1, 4):
            snap = monitor / "snapshots" / f"validation_{step:06d}"
            write_json(snap / "metadata.json", dict(validation_index=step * 10, captured_samples=2))
            with (snap / "similarity.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample", "head", "layer_i", "layer_j", "rho"])
                for sample in range(2):
                    for head in range(2):
                        for i in range(1, 3):
                            for j in range(1, 3):
                                writer.writerow([sample, head, i, j,
                                    1 if i == j else (1 + sample * 2 + head) / 5])
            # Exporter treats binary arrays/images as opaque byte-exact payloads.
            (snap / "similarity_values.npz").write_bytes(b"synthetic opaque npz payload")
            (snap / "similarity_heatmap.png").write_bytes(b"synthetic opaque plot payload")
        return monitor, run

    def quiet_export(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return exporter.export(self.root, self.repo / "monitor_exports", repo=self.repo, **kwargs)

    def test_mixed_tree_keeps_only_pure_and_never_exports_weights(self):
        good, _ = self.fixture()
        self.fixture("research-metadata", family="linearno_history")
        for a, k in ((1, 0), (0, 1), (1, 1)):
            self.fixture(f"flags-A{a}K{k}", flags=(
                f"--linearno-latent-attnres={a}", "--linearno_history_k_conditioning", str(k)))
        history = self.root / "history-run"
        write_json(history / "runtime_config.json", dict(max_snapshots=8, command=[]))
        (history / "diagnostics.jsonl").write_text('{"P_similarity": [[1]]}\n')
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        destination = self.quiet_export(plots="endpoints")
        manifest = json.loads((destination / "manifest.json").read_text())
        self.assertEqual(sum(r["included"] for r in manifest["monitors"]), 1)
        self.assertEqual(len(manifest["monitors"]), 6)
        with tarfile.open(destination / "pure_linearno_monitor.tar.gz") as bundle:
            names = bundle.getnames()
            self.assertEqual(sum(n.endswith("similarity.csv") for n in names), 3)
            self.assertEqual(sum(n.endswith(".npz") for n in names), 3)
            self.assertEqual(sum(n.endswith(".png") for n in names), 2)
            self.assertFalse(any(n.endswith(".pt") or "diagnostics.jsonl" in n or "flags-A" in n for n in names))
            sidecar = next(n for n in names if n.endswith("architecture_summary.json"))
            self.assertNotIn(b"MUST NOT EXPORT", bundle.extractfile(sidecar).read())
            for row in manifest["files"]:
                data = bundle.extractfile(row["path"]).read()
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
        self.assertEqual(exporter.csv_matrix(good / "snapshots/validation_000001/similarity.csv"),
                         [[1.0, 0.5], [0.5, 1.0]])
        self.assertIn("0.50000000", (destination / "summary.txt").read_text())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_metadata_veto_innovation_class_and_config_flag(self):
        changes = [dict(innovation_spec={}), dict(architecture_extension="linearno_history_v1"),
                   dict(model_spec={"class_path": "model.LinearNO_History.AttnResModel"}),
                   dict(features={"linearno_history_k_conditioning": True})]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                monitor, run = self.fixture(str(index))
                write_json(run / "architecture.json", {"family": "linearno", **change})
                self.assertFalse(exporter.inspect_monitor(monitor, self.repo)["included"])

    def test_unknown_corrupt_and_conflicting_provenance_are_skipped(self):
        monitor, run = self.fixture()
        write_json(monitor / "monitor_config.json", dict(capture_every_validations=10, command=["other"]))
        self.assertFalse(exporter.inspect_monitor(monitor, self.repo)["included"])
        (monitor / "monitor_config.json").unlink()
        (run / "config.json").write_text("not json")
        self.assertFalse(exporter.inspect_monitor(monitor, self.repo)["included"])
        unknown = self.root / "unknown"
        write_json(unknown / "runtime_config.json", dict(capture_every_validations=10, command=["unknown.sh"]))
        self.assertFalse(exporter.inspect_monitor(unknown, self.repo)["included"])
        with self.assertRaisesRegex(ValueError, "未找到"):
            self.quiet_export()
        self.assertFalse((self.repo / "monitor_exports").exists())

    def test_explicit_a0k0_and_command_only_evidence(self):
        monitor, run = self.fixture(flags=("--linearno_latent_attnres", "0",
                                          "--linearno_history_k_conditioning=0"))
        self.assertTrue(exporter.inspect_monitor(monitor, self.repo)["included"])
        shutil.rmtree(run)
        row = exporter.inspect_monitor(monitor, self.repo)
        self.assertTrue(row["included"])
        self.assertTrue(row["warnings"])
        self.quiet_export(list_only=True)
        self.assertFalse((self.repo / "monitor_exports").exists())

    def test_run_alias_last_wins_and_relocation_still_checks_metadata(self):
        monitor, _ = self.fixture()
        _, research = self.fixture("research", family="linearno_history")
        path = monitor / "runtime_config.json"
        config = json.loads(path.read_text())
        original = list(config["command"])
        config["command"] += ["--linearno-run-dir=" + str(research)]
        write_json(path, config)
        self.assertFalse(exporter.inspect_monitor(monitor, self.repo)["included"])
        # A copied checkout may retain absolute paths to the old checkout.
        old = self.repo / "old-missing-checkout" / research.relative_to(self.repo)
        config["command"] = original + ["--experiment-dir", str(old)]
        write_json(path, config)
        row = exporter.inspect_monitor(monitor, self.repo)
        self.assertFalse(row["included"])
        self.assertEqual(row["run_directory"], str(research))
        self.assertIn("Relocated", row["run_resolution"])

    def test_direct_cli_uses_its_checkout_from_another_cwd_and_unique_exports(self):
        self.fixture()
        script = self.repo / "monitor" / SCRIPT.name
        script.parent.mkdir()
        shutil.copyfile(SCRIPT, script)
        for _ in range(2):
            result = subprocess.run([sys.executable, "-I", "-B", str(script)], cwd=self.temporary.name,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("纯 LinearNO", result.stdout)
        self.assertEqual(len(list((self.repo / "monitor_exports").glob("*/pure_linearno_monitor.tar.gz"))), 2)

    def test_empty_capture_and_incomplete_csv_remain_explicit(self):
        monitor, _ = self.fixture()
        shutil.rmtree(monitor / "snapshots")
        destination = self.quiet_export()
        self.assertIn("no captured numerical results", (destination / "summary.txt").read_text())
        path = self.root / "incomplete.csv"
        path.write_text("layer_i,layer_j,rho\n1,2,0.8\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            exporter.csv_matrix(path)


if __name__ == "__main__":
    unittest.main()
