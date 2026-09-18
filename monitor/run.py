#!/usr/bin/env python3
"""Run an existing benchmark command with optional LinearNO monitoring.

The command after ``--`` is executed unchanged in a child process.  The only
runtime additions are ``PYTHONPATH`` (for the opt-in ``sitecustomize``) and a
JSON config path.  This keeps existing launchers, model selection, losses,
checkpoints, and output directories untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="monitor artifact directory")
    parser.add_argument("--capture-every-validations", type=int, default=1)
    parser.add_argument("--max-samples-per-snapshot", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--eval-only", action="store_true", default=True,
                        help="capture only evaluation-mode forwards (default)")
    parser.add_argument("--", dest="separator", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a command after --")
    if args.capture_every_validations < 1 or args.max_samples_per_snapshot < 1:
        parser.error("capture interval and sample count must be positive")
    return args


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "runtime_config.json"
    config = {
        "output": str(output),
        "capture_every_validations": args.capture_every_validations,
        "max_samples_per_snapshot": args.max_samples_per_snapshot,
        "epsilon": args.epsilon,
        "no_plots": args.no_plots,
        "eval_only": args.eval_only,
        "command": args.command,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    monitor_dir = str(Path(__file__).resolve().parent)
    repository_root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join((repository_root, monitor_dir))
    env["PYTHONPATH"] = prefix + (os.pathsep + old_pythonpath if old_pythonpath else "")
    env["LINEARNO_MONITOR_CONFIG"] = str(config_path)
    result = subprocess.run(args.command, env=env, check=False)
    summary = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": args.command,
        "output": str(output),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
