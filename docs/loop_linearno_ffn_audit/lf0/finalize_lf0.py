"""Finalize the LF0 freeze without writing outside the authorized evidence tree."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
START = json.loads((OUT / "start-manifest.json").read_text())
ALLOWED_EXACT = {
    "docs/LOOP_LINEARNO_FFN_REFERENCE_AUDIT.md",
    "docs/LOOP_LINEARNO_FFN_IMPLEMENTATION_STATUS.md",
}
ALLOWED_PREFIX = "docs/loop_linearno_ffn_audit/lf0/"


def run(*args: str) -> str:
    command_args = ("git", "-c", "core.quotePath=false", *args[1:]) if args and args[0] == "git" else args
    result = subprocess.run(command_args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.stderr}")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_status_paths() -> set[str]:
    """Recover paths omitted by the initial manifest's quoted Git output."""
    result = set()
    for line in START["status_with_ignored"]:
        if len(line) < 4:
            continue
        value = line[3:]
        if value.startswith('"'):
            value = ast.literal_eval(value)
            value = value.encode("latin1").decode("utf-8")
        result.add(value)
    return result


def allowed(path: str) -> bool:
    return path in ALLOWED_EXACT or path.startswith(ALLOWED_PREFIX)


def head_sha256(path: str) -> str | None:
    result = subprocess.run(
        ("git", "show", f"{START['head']}:{path}"), cwd=ROOT, capture_output=True
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def main() -> None:
    changed = []
    missing = []
    for record in START["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            missing.append(record["path"])
        elif sha256(path) != record["sha256"]:
            changed.append(record["path"])

    tracked = set(run("git", "ls-files").splitlines())
    untracked = set(run("git", "ls-files", "--others", "--exclude-standard").splitlines())
    ignored = set(run("git", "ls-files", "--others", "--ignored", "--exclude-standard").splitlines())
    original = {record["path"] for record in START["files"]}
    head_verified = sorted(tracked - original)
    head_mismatches = [
        path for path in head_verified
        if head_sha256(path) != sha256(ROOT / path)
    ]
    original.update(head_verified)
    status_only = sorted(
        path for path in decoded_status_paths() - original if not allowed(path)
    )
    original.update(status_only)
    new_paths = sorted((tracked | untracked | ignored) - original)
    unexpected = [
        path for path in new_paths if not allowed(path)
    ]
    tracked_diff = run("git", "diff", "--binary", "--no-ext-diff")
    staged_diff = run("git", "diff", "--cached", "--binary", "--no-ext-diff")
    status = run("git", "status", "--short", "--branch", "--untracked-files=all")
    result = {
        "stage": "LF0",
        "head": run("git", "rev-parse", "HEAD").strip(),
        "head_tree": run("git", "rev-parse", "HEAD^{tree}").strip(),
        "start_files_checked": len(START["files"]),
        "protected_scope_checked": START["counts"]["lf0_scope"],
        "changed_preexisting_files": changed,
        "missing_preexisting_files": missing,
        "classification_only_preexisting_paths": status_only,
        "classification_only_reason": "The initial collector used Git's quoted paths; these paths were present in start-status-ignored.txt but lack a starting hash.",
        "head_verified_preexisting_paths": head_verified,
        "head_verified_mismatches": head_mismatches,
        "new_paths": new_paths,
        "unexpected_new_paths": unexpected,
        "tracked_diff_empty": not bool(tracked_diff),
        "staged_diff_empty": not bool(staged_diff),
        "pass": not changed and not missing and not head_mismatches and not unexpected and not tracked_diff and not staged_diff,
    }
    (OUT / "end-freeze.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "end-status.txt").write_text(status)
    (OUT / "end-tracked.diff").write_text(tracked_diff)
    (OUT / "end-staged.diff").write_text(staged_diff)
    if not result["pass"]:
        raise SystemExit("LF0 freeze failed; inspect end-freeze.json")


if __name__ == "__main__":
    main()
