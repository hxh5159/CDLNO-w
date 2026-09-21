"""Capture the LF0 starting tree without changing any pre-existing file."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LF0_ALLOWED = {
    "docs/LOOP_LINEARNO_FFN_REFERENCE_AUDIT.md",
    "docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_STATUS.md",
}
LF0_PREFIX = "docs/loop_linearno_ffn_audit/lf0/"


def command(*args: str, check: bool = True) -> str:
    command_args = ("git", "-c", "core.quotePath=false", *args[1:]) if args and args[0] == "git" else args
    result = subprocess.run(command_args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"{args!r}: {result.stderr}")
    return result.stdout


def lines(value: str) -> set[str]:
    return {line for line in value.splitlines() if line}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def in_scope(relative: str) -> bool:
    prefixes = (
        "cdlno/linearno_loop/",
        "linearno_loop/",
        "tests/loop_linearno/",
        "cdlno/linearno/",
        "cdlno/linearno_history/",
        "tran_evaluate/linearno_loop/",
        "docs/loop_linearno_audit/",
    )
    exact = {
        "cdlno/checkpoint.py",
        "cdlno/experiment.py",
        "cdlno/training_state.py",
        "cdlno/training_observer.py",
        "cdlno/visualization.py",
        "PDE-Solving-StandardBenchmark/cdlno_entry.py",
        "PDE-Solving-StandardBenchmark/model_dict.py",
        "Airfoil-Design-AirfRANS/cdlno_entry.py",
        "Airfoil-Design-AirfRANS/main.py",
        "Airfoil-Design-AirfRANS/main_evaluation.py",
        "Airfoil-Design-AirfRANS/train.py",
        "Car-Design-ShapeNetCar/models/cdlno_run.py",
        "Car-Design-ShapeNetCar/main.py",
        "Car-Design-ShapeNetCar/main_evaluation.py",
        "Car-Design-ShapeNetCar/train.py",
    }
    if relative in exact or relative.startswith(prefixes):
        return True
    if relative.startswith("PDE-Solving-StandardBenchmark/"):
        name = Path(relative).name
        return name.startswith("exp_") or name.startswith("LinearNO")
    if relative.startswith(("Airfoil-Design-AirfRANS/models/", "Car-Design-ShapeNetCar/models/")):
        return "LinearNO" in Path(relative).name
    if relative.startswith("tools/"):
        return "linearno_loop" in Path(relative).name
    if relative.startswith("docs/"):
        return Path(relative).name.startswith("LOOP_LINEARNO")
    return False


def symbols(relative: str) -> list[dict[str, object]]:
    path = ROOT / relative
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result.append({
                "name": node.name,
                "kind": type(node).__name__,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
            })
    return result


def main() -> None:
    tracked = lines(command("git", "ls-files"))
    untracked = lines(command("git", "ls-files", "--others", "--exclude-standard"))
    ignored = lines(command("git", "ls-files", "--others", "--ignored", "--exclude-standard"))
    preexisting = sorted((tracked | untracked | ignored) - LF0_ALLOWED)
    preexisting = [path for path in preexisting if not path.startswith(LF0_PREFIX)]
    records = []
    for relative in preexisting:
        path = ROOT / relative
        if not path.is_file():
            continue
        classification = "tracked" if relative in tracked else "untracked" if relative in untracked else "ignored"
        records.append({
            "path": relative,
            "classification": classification,
            "size": path.stat().st_size,
            "sha256": digest(path),
            "lf0_scope": in_scope(relative),
        })
    status = command("git", "status", "--short", "--branch", "--untracked-files=all")
    ignored_status = command("git", "status", "--short", "--ignored", "--untracked-files=all")
    manifest = {
        "stage": "LF0",
        "capture_rule": "all pre-existing git tracked/untracked/ignored files; LF0 allowed outputs excluded",
        "repo_root": str(ROOT),
        "remote": command("git", "remote", "-v").splitlines(),
        "branch": command("git", "branch", "--show-current").strip(),
        "head": command("git", "rev-parse", "HEAD").strip(),
        "head_tree": command("git", "rev-parse", "HEAD^{tree}").strip(),
        "status_short_branch": status.splitlines(),
        "status_with_ignored": ignored_status.splitlines(),
        "counts": {
            "tracked": len(tracked),
            "untracked": len(untracked),
            "ignored": len(ignored),
            "files_recorded": len(records),
            "lf0_scope": sum(record["lf0_scope"] for record in records),
        },
        "files": records,
    }
    (OUT / "start-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUT / "start-status.txt").write_text(status)
    (OUT / "start-status-ignored.txt").write_text(ignored_status)
    (OUT / "start-tracked.diff").write_text(command("git", "diff", "--binary", "--no-ext-diff"))
    (OUT / "start-staged.diff").write_text(command("git", "diff", "--cached", "--binary", "--no-ext-diff"))

    scoped = [record for record in records if record["lf0_scope"]]
    (OUT / "scope-manifest.json").write_text(json.dumps({
        "count": len(scoped),
        "files": scoped,
    }, indent=2, ensure_ascii=False) + "\n")

    source_files = [record["path"] for record in scoped if record["path"].endswith(".py")]
    index = {relative: symbols(relative) for relative in source_files}
    (OUT / "source-symbols.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")

    priorities = [
        "docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md",
        "docs/CDLNO_EXPERIMENT_OUTPUTS.md",
        "docs/output_audit",
        "docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md",
        "docs/LOOP_LINEARNO_LL9R.md",
        "docs/LOOP_LINEARNO_IMPLEMENTATION_REPORT.md",
    ]
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "repo_root": str(ROOT),
        "priorities": [{"path": name, "exists": (ROOT / name).exists()} for name in priorities],
        "constraints": {
            "real_data": False,
            "long_training": False,
            "dependency_changes": False,
            "production_edits": False,
        },
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
