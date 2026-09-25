import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests/loop_linearno_v5/standard_worker.py"


@pytest.mark.parametrize("task", (
    "airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity"))
def test_native_standard_interrupt_resume_eval(task, tmp_path):
    directory = tmp_path / task
    environment = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tests"))
    reports = {}
    for action in ("interrupt", "resume", "eval"):
        report = tmp_path / f"{task}_{action}.json"
        subprocess.check_call(
            [sys.executable, str(WORKER), task, action, str(directory), str(report)],
            cwd=ROOT, env=environment)
        reports[action] = json.loads(report.read_text())
    assert reports["interrupt"]["completed_epoch"] == 1
    assert reports["resume"]["completed_epoch"] == 2
    assert reports["eval"]["metadata_unchanged"]
    expected = 10 if task == "ns" else 20 if task == "plasticity" else 1
    assert reports["interrupt"]["counts"]["train_forward"] == expected
    assert reports["resume"]["counts"]["train_forward"] == expected
    assert reports["eval"]["counts"]["eval_forward"] >= expected
    assert reports["resume"]["visualization_files"]
    assert "train_results.json" in reports["resume"]["metrics_files"]
    assert any(path.startswith("evaluations/") and path.endswith("/results.json")
               for path in reports["eval"]["metrics_files"])
