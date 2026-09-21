"""Version-2 strict epoch pairs, isolated from the historical v1 format."""

import json
from pathlib import Path

import torch

from cdlno.training_state import _atomic, _model_state_check, _same
from cdlno.linearno.checkpoint import sha256, strict_load, resume_state, restore_random_state
from linearno_loop.contracts import require_equal
from linearno_loop.v2.contracts import CHECKPOINT_FORMAT
from linearno_loop.v2.schema import read_metadata, validate_constructor, validate_metadata

FORMAT = CHECKPOINT_FORMAT
IMMUTABLE = ("family", "architecture_extension", "schema_version", "config_version", "config_hash",
             "resolved_config", "load_policy", "loop_spec", "model_spec", "profile_spec", "data_spec",
             "objective_spec", "evaluation_spec", "provenance_spec", "normalizer_spec", "ensemble_manifest")


def immutable_equal(initial, metadata):
    for key in IMMUTABLE:
        require_equal(initial[key], metadata[key], "immutable." + key)


def validate_model(model, metadata):
    validate_metadata(metadata, constructor=type(model))
    if getattr(model, "config_hash", None) != metadata["config_hash"]:
        raise ValueError("model resolved v2 config differs from checkpoint")
    model.loop.validate_structure(require_latent=model.loop.core_ffn_mode == "round_specific_latent")
    from .construction import build_from_config
    template = build_from_config(metadata["resolved_config"]).to(dtype=next(model.parameters()).dtype)
    _model_state_check(model.state_dict(), template)
    for key in ("prefix_blocks", "recurrent_core_blocks", "loop_repeats", "suffix_blocks",
                "residual_mode", "core_ffn_mode"):
        require_equal(metadata["loop_spec"][key], getattr(model.loop, key), "model.loop." + key)


def validate_optimizer_state(saved, optimizer):
    from cdlno.linearno_loop.checkpoint import validate_optimizer_state as validate_v1_optimizer
    return validate_v1_optimizer(saved, optimizer)


def inspect_checkpoint(directory, selector, *, expected=None):
    directory = Path(directory).resolve()
    if selector in ("latest", "final"):
        pointer = json.loads((directory / "checkpoints" / f"{selector}.json").read_text())
        name = pointer["manifest"]
        if Path(name).name != name or sha256(directory / "checkpoints" / name) != pointer["sha256"]:
            raise ValueError("checkpoint pointer checksum/path mismatch")
    else:
        import re
        if not re.fullmatch(r"epoch_[0-9]{4,}", selector or ""):
            raise ValueError("checkpoint selector must be final, latest or epoch_XXXX")
        name = selector + ".json"
    manifest_path = directory / "checkpoints" / name
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("checkpoint family/version/format mismatch")
    for key, subdir, suffix in (("checkpoint", "checkpoints", ".pt"), ("weights", "weights", ".pt"),
                                ("metadata", "checkpoints", ".metadata.json")):
        path = (directory / manifest[key]["path"]).resolve()
        if path.parent != directory / subdir or path.name != manifest_path.stem + suffix:
            raise ValueError("checkpoint pair path mismatch")
        if sha256(path) != manifest[key]["sha256"]:
            raise ValueError(f"checkpoint {key} checksum mismatch")
    metadata = read_metadata(directory / manifest["metadata"]["path"])
    if expected is not None:
        require_equal(expected, metadata["resolved_config"], "checkpoint.resolved_config")
    immutable_equal(read_metadata(directory / "architecture.json"), metadata)
    if metadata["resume_state"]["epoch"] != manifest["epoch"]:
        raise ValueError("metadata/manifest epoch mismatch")
    if selector == "final" and metadata["resume_state"]["checkpoint_role"] != "final":
        raise ValueError("final evaluation requires final checkpoint")
    return metadata, manifest_path


def read_pair(manifest_path, *, expected=None):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent.parent
    metadata, _ = inspect_checkpoint(directory, manifest_path.stem, expected=expected)
    manifest = json.loads(manifest_path.read_text())
    saved = torch.load(directory / manifest["checkpoint"]["path"], map_location="cpu", weights_only=True)
    weights = torch.load(directory / manifest["weights"]["path"], map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"metadata", "model"} or saved["metadata"] != metadata:
        raise ValueError("checkpoint metadata/payload mismatch")
    if not _same(saved["model"], weights):
        raise ValueError("checkpoint and independent weights differ")
    return metadata, weights


def save_pair(directory, model, metadata):
    directory = Path(directory)
    validate_model(model, metadata)
    epoch = metadata["resume_state"]["epoch"]
    if epoch < 1:
        raise ValueError("only completed epochs can be committed")
    checkpoint_dir, weights_dir = directory / "checkpoints", directory / "weights"
    checkpoint_dir.mkdir(exist_ok=True); weights_dir.mkdir(exist_ok=True)
    name = f"epoch_{epoch:04d}"; manifest_path = checkpoint_dir / (name + ".json")
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if manifest_path.exists():
        old_metadata, old_state = read_pair(manifest_path, expected=metadata["resolved_config"])
        if old_metadata != metadata or not _same(state, old_state):
            raise ValueError("refusing to overwrite a committed epoch")
        return manifest_path
    import fcntl
    with (directory / ".checkpoint.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        committed = [int(path.stem.split("_")[1]) for path in checkpoint_dir.glob("epoch_*.json")
                     if ".metadata." not in path.name]
        if committed and epoch <= max(committed):
            raise ValueError("refusing to roll back committed checkpoint progress")
        paths = dict(checkpoint=checkpoint_dir / (name + ".pt"), weights=weights_dir / (name + ".pt"),
                     metadata=checkpoint_dir / (name + ".metadata.json"))
        if any(path.exists() for path in paths.values()):
            raise ValueError("uncommitted v2 checkpoint files require inspection")
        _atomic(paths["checkpoint"], dict(model=state, metadata=metadata))
        _atomic(paths["weights"], state); _atomic(paths["metadata"], metadata, json_file=True)
        record = dict(format=FORMAT, epoch=epoch,
                      **{key: dict(path=str(path.relative_to(directory)), sha256=sha256(path))
                         for key, path in paths.items()})
        _atomic(manifest_path, record, json_file=True)
        pointer = dict(manifest=manifest_path.name, sha256=sha256(manifest_path))
        _atomic(checkpoint_dir / "latest.json", pointer, json_file=True)
        if metadata["resume_state"]["checkpoint_role"] == "final":
            _atomic(checkpoint_dir / "final.json", pointer, json_file=True)
        _atomic(directory / "model.pt", state)
    return manifest_path

