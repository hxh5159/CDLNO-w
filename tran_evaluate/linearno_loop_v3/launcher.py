#!/usr/bin/env python3
"""Shared V3 launcher for the two cost-profile wrapper trees.

This process owns only action/profile argument normalization. The existing
loop launcher still performs the native parser call and dispatches the native
task entry, so this file never reads a dataset or loads a tensor by itself.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "tran_evaluate" / "linearno_loop" / "launch.py"
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
PROFILES = ("matched_v1", "efficient_v1", "custom")
ARCHITECTURE = "operator_latent_adapter_v3"
ADAPTER = "bilateral_qk_lowrank_second_visit"


def _value(tokens, index, flag):
    token = tokens[index]
    if token == flag:
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise ValueError(f"{flag} requires a value")
        return tokens[index + 1], index + 2
    return token.split("=", 1)[1], index + 1


def _has(tokens, *flags):
    return any(token.split("=", 1)[0] in flags for token in tokens)


def _profile_assertions(tokens, profile):
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--linearno-loop-cost-profile" or token.startswith("--linearno-loop-cost-profile="):
            value, index = _value(tokens, index, "--linearno-loop-cost-profile")
            if value != profile:
                raise ValueError(
                    f"profile conflict: wrapper owns cost_profile={profile}, got {value}; "
                    "use the matching profile directory"
                )
            continue
        if token == "--linearno-loop-architecture" or token.startswith("--linearno-loop-architecture="):
            value, index = _value(tokens, index, "--linearno-loop-architecture")
            if value != ARCHITECTURE:
                raise ValueError(f"V3 launcher requires architecture={ARCHITECTURE}")
            continue
        index += 1
    if profile == "custom":
        required = {
            "--linearno-loop-hidden-width": "custom requires explicit hidden_width",
            "--linearno-loop-latent-width": "custom requires explicit latent_width",
            "--linearno-rank": "custom requires explicit actual M via --linearno-rank",
            "--linearno-loop-topology": "custom requires explicit topology",
            "--linearno-loop-adapter-mode": "custom requires explicit adapter mode",
            "--linearno-loop-adapter-rank": "custom requires explicit adapter rank",
            "--linearno-loop-adapter-alpha": "custom requires explicit adapter alpha",
        }
        for flag, message in required.items():
            if not _has(tokens, flag):
                raise ValueError(message)


def _normalize_paths(task, tokens, environment):
    """Translate launcher conveniences to the native task flags once."""
    result = []
    index = 0
    env = dict(environment)
    while index < len(tokens):
        token = tokens[index]
        flag = token.split("=", 1)[0]
        if flag in ("--data-root", "--output-root"):
            value, index = _value(tokens, index, flag)
            if flag == "--output-root":
                env["CDLNO_RUNS_ROOT"] = str(Path(value).expanduser().resolve())
                continue
            native = {"airfrans": "--my_path", "car": "--data_dir"}.get(task, "--data_path")
            result.extend((native, value))
            continue
        if flag == "--run-dir":
            value, index = _value(tokens, index, flag)
            result.extend(("--experiment-dir", value))
            continue
        result.append(token)
        index += 1
    return result, env


def _defaults(profile):
    return [
        "--linearno-loop", "1",
        "--linearno-loop-architecture", ARCHITECTURE,
        "--linearno-loop-cost-profile", profile,
        "--linearno-loop-topology", "d12",
        "--linearno-loop-residual-mode", "sr_1_over_r",
        "--linearno-loop-latent", "1",
        "--linearno-loop-adapter-mode", ADAPTER,
        "--linearno-loop-adapter-rank", "4",
        "--linearno-loop-adapter-alpha", "4",
        "--linearno-profile", "paper_table8_on_release_model",
        "--seed", "0",
        "--gpu", "0",
    ]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        raise SystemExit("usage: launcher.py PROFILE TASK ACTION [options]")
    profile, task, action = argv[:3]
    user = argv[3:]
    if profile not in PROFILES:
        raise SystemExit(f"unknown cost profile: {profile}")
    if task not in TASKS:
        raise SystemExit(f"unknown task: {task}")
    allowed = {"train", "resume", "eval", "train_eval", "dry-run", "preview", "print-run-dir"}
    if action not in allowed:
        raise SystemExit(f"action must be one of: {' '.join(sorted(allowed))}")
    _profile_assertions(user, profile)
    normalized, environment = _normalize_paths(task, user, os.environ)
    actual = action
    suffix = []
    if action == "train_eval":
        actual = "train"
        suffix.append("--then-eval")
    elif action == "dry-run":
        actual = "train"
        suffix.append("--dry-run")
    elif action == "preview":
        actual = "train"
        suffix.append("--plan-json")
    elif action == "print-run-dir":
        actual = "train"
        suffix.append("--print-run-dir")
    # The defaults precede user values intentionally. The parser sees the
    # user's structural override last; profile conflicts are rejected above.
    command = [sys.executable, "-B", str(NATIVE), task, actual,
               *_defaults(profile), *normalized, *suffix]
    environment["CDLNO_REPO_ROOT"] = str(ROOT)
    environment.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"linearno_loop_v3: {error}", file=sys.stderr)
        raise SystemExit(2)
