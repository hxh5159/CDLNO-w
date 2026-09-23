"""Metadata-first, atomic and strict v4 checkpoint pairs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch

from cdlno.training_state import _atomic, _model_state_check, _same
from cdlno.linearno.checkpoint import strict_load
from linearno_loop.v4.schema import make_metadata, write_metadata, read_metadata, validate_metadata
from linearno_loop.v4.config import validate_config

FORMAT = "resmlp-dual-temp-v4-pair-v1"
IMMUTABLE = ("family", "architecture", "architecture_family", "architecture_extension",
             "architecture_version", "checkpoint_schema", "checkpoint_version", "config_hash",
             "load_policy", "resolved_config", "model_spec", "profile_spec", "loop_spec",
             "rmlp_spec", "temperature_spec", "objective_spec", "evaluation_spec", "data_spec",
             "normalizer_spec", "provenance_spec", "ensemble_manifest", "parameter_measurement")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _parameter_groups(model):
    groups = Counter({key: 0 for key in ("stem", "time", "operators", "rmlp", "head",
                                         "temperature_predictors")})
    for name, parameter in model.named_parameters():
        if name.startswith("time_fc."):
            group = "time"
        elif name.startswith("loop.blocks."):
            group = ("temperature_predictors" if ".q_temperature." in name or ".k_temperature." in name
                     else "operators")
        elif name.startswith("loop.rmlp."):
            group = "rmlp"
        elif name.startswith(("final_norm.", "head.")):
            group = "head"
        elif name == "placeholder" or name.startswith("preprocess."):
            group = "stem"
        else:
            raise ValueError("unclassified V4 parameter: " + name)
        groups[group] += parameter.numel()
    return dict(groups)


def measure_parameters(model, config):
    checked = validate_config(config)
    if getattr(model, "v4_config", checked) != checked:
        raise ValueError("V4 model/config mismatch")
    groups = _parameter_groups(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if sum(groups.values()) != total or total != trainable:
        raise ValueError("V4 parameter ownership mismatch")
    return dict(status="measured", total=total, trainable=trainable, groups=groups)


def validate_model(model, metadata):
    checked = validate_metadata(metadata)
    if getattr(model, "v4_config", None) != checked["resolved_config"]:
        raise ValueError("V4 model resolved config differs from checkpoint")
    if measure_parameters(model, checked["resolved_config"]) != checked["parameter_measurement"]:
        raise ValueError("V4 model parameter measurement differs")
    if tuple(model.loop.schedule) != tuple(checked["rmlp_spec"]["route"]):
        raise ValueError("V4 model RMLP route differs")


def validate_optimizer_state(saved, optimizer, *, model=None):
    from cdlno.linearno_loop.checkpoint import validate_optimizer_state as validate
    validate(saved, optimizer)
    if model is not None:
        actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (len(actual) != len({id(parameter) for parameter in actual}) or
                {id(parameter) for parameter in actual} != {id(parameter) for parameter in model.parameters()}):
            raise ValueError("optimizer does not own exactly the V4 parameters")


def resume_state(*args, **kwargs):
    from cdlno.linearno.checkpoint import resume_state as base
    return base(*args, **kwargs)


def immutable_equal(initial, metadata):
    for key in IMMUTABLE:
        if initial.get(key) != metadata.get(key):
            raise ValueError("immutable V4 metadata mismatch: " + key)


def _manifest_path(directory, selector):
    checkpoint_dir = directory / "checkpoints"
    if selector in ("latest", "final"):
        pointer = json.loads((checkpoint_dir / (selector + ".json")).read_text())
        name = pointer.get("manifest")
        if type(name) is not str or Path(name).name != name:
            raise ValueError("V4 checkpoint pointer path mismatch")
        path = checkpoint_dir / name
        if not path.is_file() or sha256(path) != pointer.get("sha256"):
            raise ValueError("V4 checkpoint pointer checksum mismatch")
        return path
    if not re.fullmatch(r"epoch_[0-9]{4,}", selector or ""):
        raise ValueError("V4 checkpoint selector must be final, latest or epoch_XXXX")
    return checkpoint_dir / (selector + ".json")


def inspect_checkpoint(directory, selector, *, expected=None):
    directory = Path(directory).resolve()
    initial = read_metadata(directory / "architecture.json")
    if expected is not None and initial["resolved_config"] != validate_config(expected):
        raise ValueError("V4 expected config conflict")
    path = _manifest_path(directory, selector)
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("V4 checkpoint format mismatch")
    for key, subdir, suffix in (("checkpoint", "checkpoints", ".pt"),
                                ("weights", "weights", ".pt"),
                                ("metadata", "checkpoints", ".metadata.json")):
        record = manifest.get(key, {})
        target = (directory / record.get("path", "")).resolve()
        if target.parent != directory / subdir or target.name != path.stem + suffix:
            raise ValueError("V4 checkpoint pair path mismatch")
        if not target.is_file() or sha256(target) != record.get("sha256"):
            raise ValueError("V4 checkpoint " + key + " checksum mismatch")
    metadata = read_metadata(directory / manifest["metadata"]["path"])
    immutable_equal(initial, metadata)
    if metadata["resume_state"]["epoch"] != manifest.get("epoch"):
        raise ValueError("V4 checkpoint epoch mismatch")
    if selector == "final" and metadata["resume_state"]["checkpoint_role"] != "final":
        raise ValueError("V4 final evaluation requires a final checkpoint")
    return metadata, path


def read_pair(manifest_path, *, expected=None):
    path = Path(manifest_path); directory = path.parent.parent
    metadata, checked = inspect_checkpoint(directory, path.stem, expected=expected)
    manifest = json.loads(checked.read_text())
    payload = torch.load(directory / manifest["checkpoint"]["path"], map_location="cpu", weights_only=True)
    weights = torch.load(directory / manifest["weights"]["path"], map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"metadata", "model"} or
            payload["metadata"] != metadata or not _same(payload["model"], weights)):
        raise ValueError("V4 checkpoint/weights/metadata pair mismatch")
    return metadata, weights


def _synthetic_metadata(config, model):
    checksum = hashlib.sha256(b"V4 model-state synthetic fixture").hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    state = resume_state(optimizer, scheduler, 1, 0, 1, {}, dict(model_state_only=True))
    return make_metadata(config,
        data_spec=dict(protocol=config["profile_spec"]["values"]["data"], split="synthetic model-state",
                       sampling="synthetic model-state", checksums={"synthetic": checksum},
                       scope="synthetic", runtime={}),
        normalizer_spec=dict(policy="none", records={}),
        provenance_spec=dict(target_sha="0" * 40, source_sha256="0" * 64,
                             code_version="resmlp_dual_temp_v4-synthetic"),
        resume_state=state, ensemble_manifest=[],
        parameter_measurement=measure_parameters(model, config))


def save_pair(directory, model, metadata):
    directory = Path(directory).resolve()
    if "resolved_config" not in metadata and metadata.get("architecture") == "resmlp_dual_temp_v4":
        config = validate_config(metadata); metadata = _synthetic_metadata(config, model)
        directory.mkdir(parents=True, exist_ok=True)
        write_metadata(directory / "architecture.json", metadata)
    validate_model(model, metadata)
    initial = read_metadata(directory / "architecture.json"); immutable_equal(initial, metadata)
    epoch = metadata["resume_state"]["epoch"]
    if epoch < 1:
        raise ValueError("only completed V4 epochs can be saved")
    checkpoint_dir, weights_dir = directory / "checkpoints", directory / "weights"
    checkpoint_dir.mkdir(exist_ok=True); weights_dir.mkdir(exist_ok=True)
    name = f"epoch_{epoch:04d}"; manifest_path = checkpoint_dir / (name + ".json")
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _model_state_check(state, model)
    if manifest_path.exists():
        old_metadata, old_state = read_pair(manifest_path, expected=metadata["resolved_config"])
        if old_metadata != metadata or not _same(old_state, state):
            raise ValueError("refusing to overwrite a committed V4 epoch")
        return manifest_path
    import fcntl
    with (directory / ".checkpoint.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        committed = [int(path.stem.split("_")[1]) for path in checkpoint_dir.glob("epoch_*.json")
                     if ".metadata." not in path.name]
        if committed and epoch <= max(committed):
            raise ValueError("refusing to roll back V4 checkpoint progress")
        paths = dict(checkpoint=checkpoint_dir / (name + ".pt"),
                     weights=weights_dir / (name + ".pt"),
                     metadata=checkpoint_dir / (name + ".metadata.json"))
        if any(path.exists() for path in paths.values()):
            raise ValueError("uncommitted V4 checkpoint files require inspection")
        _atomic(paths["checkpoint"], dict(metadata=metadata, model=state))
        _atomic(paths["weights"], state)
        _atomic(paths["metadata"], metadata, json_file=True)
        manifest = dict(format=FORMAT, epoch=epoch,
                        **{key: dict(path=str(path.relative_to(directory)), sha256=sha256(path))
                           for key, path in paths.items()})
        _atomic(manifest_path, manifest, json_file=True)
        pointer = dict(manifest=manifest_path.name, sha256=sha256(manifest_path))
        _atomic(checkpoint_dir / "latest.json", pointer, json_file=True)
        if metadata["resume_state"]["checkpoint_role"] == "final":
            _atomic(checkpoint_dir / "final.json", pointer, json_file=True)
        _atomic(directory / "model.pt", state)
    return manifest_path


def load_pair(directory, model, config, *, optimizer=None):
    metadata, path = inspect_checkpoint(directory, "final", expected=config)
    _, state = read_pair(path, expected=config)
    validate_model(model, metadata); strict_load(model, state)
    return metadata


def load_model(directory, selector="final", *, explicit=None, map_location="cpu"):
    metadata, path = inspect_checkpoint(directory, selector)
    from linearno_loop.v4.schema import restore_config
    config = restore_config(metadata, explicit=explicit, strict=True)["config"]
    from .construction import build_from_config
    model = build_from_config(config)
    checked, state = read_pair(path, expected=config)
    validate_model(model, checked); strict_load(model, state)
    return model.to(map_location), checked
