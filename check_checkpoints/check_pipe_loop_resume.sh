#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
default_run="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin/output/pipe/linearno_loop/pipe__linearno_loop__paper_table8_on_release_model__P2-C2-R2-S2__sr_1_over_r__M64__seed2__cfg33afe724354d__20260920T172046055959Z_1bd9136e"

run="${1:-$default_run}"
selector="${2:-epoch_0401}"
python_bin="${CDLNO_PYTHON:-python}"

cd "$repo_root"
export CDLNO_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export RUN="$run"
export CHECKPOINT_SELECTOR="$selector"

PYTHONDONTWRITEBYTECODE=1 "$python_bin" -B - <<'PY'
import json
import math
import os
import sys
import traceback
from pathlib import Path

import torch

from cdlno.linearno.schema import unpack_state
try:
    from cdlno.linearno_loop.versioning import (
        checkpoint_api as resolve_checkpoint_api,
        construction_api as resolve_construction_api,
        provenance as resolve_provenance,
    )

    def checkpoint_api(config):
        return resolve_checkpoint_api(config)

    def build_model(config):
        return resolve_construction_api(config).build_from_config(config)

    def current_provenance(config):
        return resolve_provenance(config)

    repository_layout = "versioned-v1-v2"
except ModuleNotFoundError as error:
    # Runs created before the RoundFFN/latent-FFN extension have no versioning
    # dispatcher. Their authoritative V1 modules expose the same strict pair
    # checks directly.
    if error.name != "cdlno.linearno_loop.versioning":
        raise
    from cdlno.linearno_loop import checkpoint as v1_checkpoint
    from cdlno.linearno_loop.construction import build_from_config
    from cdlno.linearno_loop.provenance import provenance as v1_provenance

    def checkpoint_api(config):
        if config.get("config_version") != 1:
            raise ValueError("legacy repository layout supports only Looped LinearNO V1")
        return v1_checkpoint

    def build_model(config):
        return build_from_config(config)

    def current_provenance(config):
        return v1_provenance()

    repository_layout = "legacy-v1-direct"


def audit():
    run = Path(os.environ["RUN"]).resolve()
    selector = os.environ["CHECKPOINT_SELECTOR"]
    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("run =", run)
    print("selector =", selector)
    print("repository_layout =", repository_layout)
    check("run directory exists", run.is_dir())
    check("architecture.json exists", (run / "architecture.json").is_file())
    if not run.is_dir() or not (run / "architecture.json").is_file():
        raise FileNotFoundError("run directory or architecture.json is missing")

    architecture = json.loads((run / "architecture.json").read_text())
    config = architecture["resolved_config"]
    spec = config["loop_spec"]
    training = config["profile_spec"]["values"]["training"]

    check("family", architecture["family"] == "linearno_loop",
          architecture["family"])
    check("V1 architecture",
          architecture["architecture_extension"] == "loop_linearno_v1",
          architecture["architecture_extension"])
    check("config version", config["config_version"] == 1,
          str(config["config_version"]))
    check("task", spec["task"] == "pipe", spec["task"])
    topology = (
        spec["prefix_blocks"],
        spec["recurrent_core_blocks"],
        spec["loop_repeats"],
        spec["suffix_blocks"],
    )
    check("topology", topology == (2, 2, 2, 2), str(topology))
    check("residual", spec["residual_mode"] == "sr_1_over_r",
          spec["residual_mode"])
    check("actual M", spec["resolved_rank"] == 64,
          str(spec["resolved_rank"]))
    check("V2 FFN mode absent", "core_ffn_mode" not in spec)
    check("configured epochs", training["epochs"] == 500,
          str(training["epochs"]))

    api = checkpoint_api(config)

    # These calls verify architecture/metadata equality, the pair manifest,
    # referenced paths, file hashes, and checkpoint/weights tensor equality.
    metadata, manifest_path = api.inspect_checkpoint(
        run, selector, expected=config
    )
    metadata2, weights = api.read_pair(
        manifest_path, expected=config
    )
    check("metadata roundtrip", metadata == metadata2)

    manifest = json.loads(manifest_path.read_text())
    latest_path = run / "checkpoints/latest.json"
    check("latest pointer exists", latest_path.is_file())
    latest = json.loads(latest_path.read_text()) if latest_path.is_file() else {}

    committed = sorted(
        int(path.stem.split("_")[1])
        for path in (run / "checkpoints").glob("epoch_*.json")
        if ".metadata." not in path.name
    )
    selected_epoch = manifest["epoch"]
    check("committed checkpoints exist", bool(committed), str(committed))
    check("selected checkpoint is newest committed",
          bool(committed) and committed[-1] == selected_epoch,
          f"selected={selected_epoch}, newest={committed[-1] if committed else None}")
    check("latest points to selected checkpoint",
          latest.get("manifest") == manifest_path.name,
          str(latest))

    state = metadata["resume_state"]
    completed = state["epoch"]
    global_step = state["global_step"]
    remaining = training["epochs"] - completed

    check("metadata/manifest epoch", completed == selected_epoch,
          f"metadata={completed}, manifest={selected_epoch}")
    check("checkpoint is non-final",
          state["checkpoint_role"] == "epoch",
          state["checkpoint_role"])
    check("remaining epochs are positive", remaining > 0, str(remaining))

    plan = metadata["data_spec"]["scheduler"]
    steps_per_epoch = plan["steps_per_epoch"]
    expected_step = completed * steps_per_epoch

    check("scheduler type", plan["type"] == "OneCycleLR", plan["type"])
    check("global step", global_step == expected_step,
          f"saved={global_step}, expected={expected_step}")

    required_state = {
        "optimizer",
        "scheduler",
        "rng",
        "dataloader_generators",
        "sampler_state",
    }
    check("complete resume fields", required_state <= set(state),
          str(sorted(set(state))))

    current_provenance_value = current_provenance(config)
    saved_source = metadata["provenance_spec"]["source_sha256"]
    current_source = current_provenance_value["source_sha256"]
    check("current source provenance", current_source == saved_source,
          f"saved={saved_source}, current={current_source}")

    model = build_model(config)
    api.validate_model(model, metadata)
    check("model structure and state schema", True)

    model_params = sum(parameter.numel() for parameter in model.parameters())
    weight_params = sum(tensor.numel() for tensor in weights.values())
    check("model/weight tensor count", model_params == weight_params,
          f"model={model_params}, checkpoint={weight_params}")
    check("all saved weights finite",
          all(torch.isfinite(tensor).all().item() for tensor in weights.values()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["lr"],
        weight_decay=training["weight_decay"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training["lr"],
        epochs=training["epochs"],
        steps_per_epoch=steps_per_epoch,
    )

    optimizer_state = unpack_state(state["optimizer"])
    scheduler_state = unpack_state(state["scheduler"])
    rng_state = unpack_state(state["rng"])
    loader_state = unpack_state(state["dataloader_generators"])
    sampler_state = unpack_state(state["sampler_state"])

    api.validate_optimizer_state(optimizer_state, optimizer)
    check("optimizer state structure", True)
    check("scheduler last_epoch",
          scheduler_state["last_epoch"] == expected_step,
          f"saved={scheduler_state['last_epoch']}, expected={expected_step}")
    check("scheduler _step_count",
          scheduler_state["_step_count"] == expected_step + 1,
          f"saved={scheduler_state['_step_count']}, expected={expected_step + 1}")
    expected_total_steps = training["epochs"] * steps_per_epoch
    check("scheduler total_steps",
          scheduler_state["total_steps"] == expected_total_steps,
          f"saved={scheduler_state['total_steps']}, expected={expected_total_steps}")

    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    current_lr = optimizer.param_groups[0]["lr"]

    check("restored learning rate finite",
          math.isfinite(current_lr) and current_lr >= 0,
          f"{current_lr:.12g}")
    check("RNG fields",
          {"python", "numpy", "torch_cpu", "torch_cuda"} <= set(rng_state),
          str(sorted(rng_state)))
    check("DataLoader generator fields",
          {"train", "test"} <= set(loader_state),
          str(sorted(loader_state)))
    check("sampler state present",
          isinstance(sampler_state, dict) and bool(sampler_state),
          str(sampler_state))

    print()
    print("checkpoint_manifest =", manifest_path)
    print("completed_epochs    =", completed)
    print("remaining_epochs    =", remaining)
    print("steps_per_epoch     =", steps_per_epoch)
    print("global_step         =", global_step)
    print("restored_lr         =", f"{current_lr:.12g}")
    print("model_parameters    =", model_params)
    print("state_dict_keys     =", len(weights))
    print("committed_epochs    =", committed)

    if failures:
        print()
        print("AUDIT_RESULT=FAIL")
        print("failed_checks=" + ",".join(failures))
        return 1

    print()
    print("AUDIT_RESULT=PASS")
    print("This checkpoint contains the state required for exact epoch-boundary resume.")
    return 0


try:
    raise SystemExit(audit())
except SystemExit:
    raise
except Exception as error:
    print()
    print(f"AUDIT_RESULT=ERROR ({type(error).__name__}: {error})")
    traceback.print_exc()
    raise SystemExit(2)
PY
