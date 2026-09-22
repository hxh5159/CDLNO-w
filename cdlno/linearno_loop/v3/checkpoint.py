"""Metadata-first V3 epoch pairs and strict training-state restoration."""
import json
from pathlib import Path, PurePosixPath
from collections.abc import Mapping

import torch

from cdlno.training_state import _atomic, _model_state_check, _same
from cdlno.linearno.checkpoint import (restore_random_state, resume_state as base_resume_state,
                                       sha256, strict_load)
from cdlno.linearno.schema import pack_state, unpack_state
from linearno_loop.v3.config import validate_config
from linearno_loop.v3.contracts import CHECKPOINT_FORMAT, require_equal
from linearno_loop.v3.costs import analytic_cost
from linearno_loop.v3.schema import read_metadata, restore_config, validate_metadata


FORMAT = CHECKPOINT_FORMAT
IMMUTABLE = ("family", "architecture_extension", "schema_version", "config_version",
             "checkpoint_version", "checkpoint_format", "config_hash", "resolved_config",
             "load_policy", "loop_spec", "model_spec", "profile_spec", "data_spec",
             "objective_spec", "evaluation_spec", "provenance_spec", "normalizer_spec",
             "ensemble_manifest", "cost_spec", "parameter_measurement")


def immutable_equal(initial, metadata):
    for key in IMMUTABLE:
        require_equal(initial[key], metadata[key], "immutable." + key)


def _parameter_groups(model):
    groups = {name: 0 for name in ("stem", "time", "prefix", "shared_core",
                                   "suffix", "head", "latent", "adapter", "router")}
    for name, parameter in model.named_parameters():
        if ".latent_processor." in name:
            group = "latent"
        elif ".adapter." in name:
            group = "adapter"
        elif name.startswith(("loop.rb_", "loop.lb_")):
            group = "router"
        elif name.startswith("loop.prefix."):
            group = "prefix"
        elif name.startswith("loop.core."):
            group = "shared_core"
        elif name.startswith("loop.suffix."):
            group = "head" if any(part in name for part in (".ln_3.", ".mlp2.")) else "suffix"
        elif name.startswith("time_fc."):
            group = "time"
        elif name == "placeholder" or name.startswith("preprocess."):
            group = "stem"
        else:
            raise ValueError("unclassified V3 parameter: " + name)
        groups[group] += parameter.numel()
    return groups


def measure_parameters(model, config):
    checked = validate_config(config)
    actual = _parameter_groups(model)
    expected = analytic_cost(checked)["parameter_groups"]
    require_equal(expected, actual, "actual.parameter_groups")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    require_equal(sum(actual.values()), total, "actual.total")
    require_equal(total, trainable, "actual.trainable")
    return dict(status="measured", total=total, trainable=trainable, groups=actual)


def validate_model(model, metadata):
    validate_metadata(metadata, constructor=type(model))
    if getattr(model, "config_hash", None) != metadata["config_hash"]:
        raise ValueError("model resolved V3 config differs from checkpoint")
    model.loop.validate_structure(require_features=True)
    require_equal(metadata["parameter_measurement"],
                  measure_parameters(model, metadata["resolved_config"]),
                  "model.parameter_measurement")
    for key in ("prefix_blocks", "recurrent_core_blocks", "loop_repeats",
                "suffix_blocks", "residual_mode"):
        require_equal(metadata["loop_spec"][key], getattr(model.loop, key),
                      "model.loop." + key)


def validate_optimizer_state(saved, optimizer, *, model=None):
    from cdlno.linearno_loop.checkpoint import validate_optimizer_state as validate_v1
    validate_v1(saved, optimizer)
    if model is not None:
        expected = {id(parameter) for parameter in model.parameters()}
        actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(actual) != len({id(parameter) for parameter in actual}) or {id(p) for p in actual} != expected:
            raise ValueError("optimizer groups do not own exactly the V3 model parameters")


def _finite_state(value, path):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(path + " contains nonfinite tensor")
    elif isinstance(value, Mapping):
        for key, item in value.items(): _finite_state(item, path + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value): _finite_state(item, f"{path}[{index}]")
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError(path + " contains unsupported value")


def validate_scaler_state(saved, scaler):
    if saved is None:
        if scaler is not None:
            raise ValueError("checkpoint has no scaler but a scaler was supplied")
        return
    if scaler is None or not isinstance(saved, dict) or not saved:
        raise ValueError("checkpoint scaler state/current scaler mismatch")
    current = scaler.state_dict()
    if not isinstance(current, dict) or set(current) != set(saved):
        raise ValueError("scaler state fields differ")
    _finite_state(saved, "scaler")


def resume_state(optimizer, scheduler, epoch, steps_per_epoch, total_epochs,
                 generators, sampler, *, scaler=None):
    state = base_resume_state(optimizer, scheduler, epoch, steps_per_epoch,
                              total_epochs, generators, sampler)
    state["scaler"] = pack_state(None if scaler is None else scaler.state_dict())
    return state


def _manifest_path(directory, selector):
    if selector in ("latest", "final"):
        pointer_path = directory / "checkpoints" / f"{selector}.json"
        pointer = json.loads(pointer_path.read_text())
        name = pointer.get("manifest")
        if type(name) is not str or Path(name).name != name:
            raise ValueError("checkpoint pointer path mismatch")
        path = directory / "checkpoints" / name
        if sha256(path) != pointer.get("sha256"):
            raise ValueError("checkpoint pointer checksum mismatch")
        return path
    import re
    if not re.fullmatch(r"epoch_[0-9]{4,}", selector or ""):
        raise ValueError("checkpoint selector must be final, latest or epoch_XXXX")
    return directory / "checkpoints" / (selector + ".json")


def _validate_ensemble(directory, metadata):
    for member in metadata["ensemble_manifest"]:
        relative = PurePosixPath(member["path"])
        path = (directory / Path(*relative.parts)).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != member["sha256"]:
            raise ValueError("ensemble member path/checksum mismatch: " + member["member_id"])


def inspect_checkpoint(directory, selector, *, expected=None):
    directory = Path(directory).resolve()
    # The immutable run sidecar is the first parsed artifact. No model module or
    # tensor payload is imported/read before it and explicit config are valid.
    initial = read_metadata(directory / "architecture.json")
    if expected is not None:
        require_equal(validate_config(expected), initial["resolved_config"],
                      "checkpoint.resolved_config")
    manifest_path = _manifest_path(directory, selector)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("checkpoint family/version/format mismatch")
    for key, subdir, suffix in (("checkpoint", "checkpoints", ".pt"),
                                ("weights", "weights", ".pt"),
                                ("metadata", "checkpoints", ".metadata.json")):
        record = manifest.get(key, {})
        path = (directory / record.get("path", "")).resolve()
        if path.parent != directory / subdir or path.name != manifest_path.stem + suffix:
            raise ValueError("checkpoint pair path mismatch")
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError("checkpoint " + key + " checksum mismatch")
    metadata = read_metadata(directory / manifest["metadata"]["path"])
    immutable_equal(initial, metadata)
    if metadata["resume_state"]["epoch"] != manifest.get("epoch"):
        raise ValueError("metadata/manifest epoch mismatch")
    if selector == "final" and metadata["resume_state"]["checkpoint_role"] != "final":
        raise ValueError("final evaluation requires final checkpoint")
    _validate_ensemble(directory, metadata)
    return metadata, manifest_path


def read_pair(manifest_path, *, expected=None):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent.parent
    metadata, checked_path = inspect_checkpoint(directory, manifest_path.stem,
                                                 expected=expected)
    manifest = json.loads(checked_path.read_text())
    saved = torch.load(directory / manifest["checkpoint"]["path"],
                       map_location="cpu", weights_only=True)
    weights = torch.load(directory / manifest["weights"]["path"],
                         map_location="cpu", weights_only=True)
    if (not isinstance(saved, dict) or set(saved) != {"metadata", "model"} or
            saved["metadata"] != metadata):
        raise ValueError("checkpoint metadata/payload mismatch")
    if not _same(saved["model"], weights):
        raise ValueError("checkpoint and independent weights differ")
    return metadata, weights


def save_pair(directory, model, metadata):
    directory = Path(directory).resolve()
    validate_model(model, metadata)
    initial = read_metadata(directory / "architecture.json")
    immutable_equal(initial, metadata)
    _validate_ensemble(directory, metadata)
    epoch = metadata["resume_state"]["epoch"]
    if epoch < 1:
        raise ValueError("only completed epochs can be committed")
    checkpoint_dir, weights_dir = directory / "checkpoints", directory / "weights"
    checkpoint_dir.mkdir(exist_ok=True); weights_dir.mkdir(exist_ok=True)
    name = f"epoch_{epoch:04d}"; manifest_path = checkpoint_dir / (name + ".json")
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _model_state_check(state, model)
    if manifest_path.exists():
        old_metadata, old_state = read_pair(manifest_path,
                                            expected=metadata["resolved_config"])
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
        paths = dict(checkpoint=checkpoint_dir / (name + ".pt"),
                     weights=weights_dir / (name + ".pt"),
                     metadata=checkpoint_dir / (name + ".metadata.json"))
        if any(path.exists() for path in paths.values()):
            raise ValueError("uncommitted V3 checkpoint files require inspection")
        _atomic(paths["checkpoint"], dict(model=state, metadata=metadata))
        _atomic(paths["weights"], state)
        _atomic(paths["metadata"], metadata, json_file=True)
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


def validate_resume_metadata(initial, resumed):
    validate_metadata(initial); validate_metadata(resumed)
    immutable_equal(initial, resumed)


def _validate_random_state(state, generators):
    from cdlno.training_state import _validate_rng
    saved = unpack_state(state["rng"]); numpy_rng = saved["numpy"]
    native = dict(python=saved["python"], numpy=dict(algorithm=numpy_rng[0],
        keys=torch.tensor(numpy_rng[1].astype("int64")), position=numpy_rng[2],
        has_gauss=numpy_rng[3], cached_gaussian=numpy_rng[4]),
        cpu=saved["torch_cpu"], cuda=saved["torch_cuda"],
        generators=unpack_state(state["dataloader_generators"]))
    _validate_rng(native, dict(generators))


def restore_training_state(model, optimizer, scheduler, metadata, *, weights,
                           generators, sampler, normalizer_spec, scaler=None):
    validate_model(model, metadata)
    _model_state_check(weights, model)
    state = metadata["resume_state"]
    optimizer_state = unpack_state(state["optimizer"])
    scheduler_state = unpack_state(state["scheduler"])
    scaler_state = unpack_state(state["scaler"])
    validate_optimizer_state(optimizer_state, optimizer, model=model)
    if (not isinstance(scheduler_state, dict) or
            set(scheduler_state) != set(scheduler.state_dict()) or
            scheduler.optimizer is not optimizer):
        raise ValueError("resume scheduler state/optimizer mismatch")
    _finite_state(scheduler_state, "scheduler")
    validate_scaler_state(scaler_state, scaler)
    require_equal(metadata["normalizer_spec"], normalizer_spec,
                  "resume.normalizer_spec")
    require_equal(unpack_state(state["sampler_state"]), sampler,
                  "resume.sampler_state")
    _validate_random_state(state, generators)
    strict_load(model, weights)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler.load_state_dict(scaler_state)
    restore_random_state(state, generators)


def load_model(directory, selector, *, explicit=None, runtime=None, map_location="cpu"):
    # inspect + explicit restore complete before importing any V3 wrapper.
    metadata, manifest_path = inspect_checkpoint(directory, selector)
    restored = restore_config(metadata, explicit=explicit, runtime=runtime,
                              strict=True)
    from .construction import build_from_config
    model = build_from_config(restored["config"])
    checked, weights = read_pair(manifest_path, expected=restored["config"])
    strict_load(model, weights)
    return model.to(map_location), checked
