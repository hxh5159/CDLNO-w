"""Strict JSON metadata contract for ``resmlp_dual_temp_v4``."""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .config import validate_config
from .contracts import CHECKPOINT_SCHEMA, CHECKPOINT_VERSION, V4SchemaError, digest

LOAD_POLICY = dict(metadata_first=True, strict=True, legacy_fallback=False,
                   architecture_migration=False)
EXTERNAL = {"data_spec", "normalizer_spec", "provenance_spec", "resume_state",
            "ensemble_manifest", "parameter_measurement"}


def _exact(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise V4SchemaError(f"{name} fields mismatch")


def _hash(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise V4SchemaError(f"{name} must be a SHA-256 digest")


def make_metadata(config, **sections):
    cfg = validate_config(config)
    _exact(sections, EXTERNAL, "metadata sections")
    meta = dict(
        family=cfg["family"], architecture=cfg["architecture"],
        architecture_family=cfg["architecture_family"],
        architecture_extension=cfg["architecture_extension"],
        architecture_version=cfg["architecture_version"],
        checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_version=CHECKPOINT_VERSION,
        config_hash=cfg["config_hash"], load_policy=deepcopy(LOAD_POLICY),
        resolved_config=deepcopy(cfg), model_spec=deepcopy(cfg["model_spec"]),
        profile_spec=deepcopy(cfg["profile_spec"]), loop_spec=deepcopy(cfg["loop_spec"]),
        rmlp_spec=deepcopy(cfg["rmlp_spec"]), temperature_spec=deepcopy(cfg["temperature_spec"]),
        objective_spec=deepcopy(cfg["profile_spec"]["values"]["objective"]),
        evaluation_spec=deepcopy(cfg["profile_spec"]["values"]["evaluation"]),
        **deepcopy(sections),
    )
    meta["metadata_hash"] = digest(meta)
    return validate_metadata(meta)


def validate_metadata(meta):
    keys = {"family", "architecture", "architecture_family", "architecture_extension",
            "architecture_version", "checkpoint_schema", "checkpoint_version", "config_hash",
            "load_policy", "resolved_config", "model_spec", "profile_spec", "loop_spec",
            "rmlp_spec", "temperature_spec", "objective_spec", "evaluation_spec", *EXTERNAL,
            "metadata_hash"}
    _exact(meta, keys, "metadata")
    try:
        json.dumps(meta, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise V4SchemaError("metadata must be finite JSON") from error
    if meta["checkpoint_schema"] != CHECKPOINT_SCHEMA or meta["checkpoint_version"] != CHECKPOINT_VERSION:
        raise V4SchemaError("checkpoint schema/version mismatch")
    cfg = validate_config(meta["resolved_config"])
    expected = dict(family=cfg["family"], architecture=cfg["architecture"],
                    architecture_family=cfg["architecture_family"],
                    architecture_extension=cfg["architecture_extension"],
                    architecture_version=cfg["architecture_version"],
                    config_hash=cfg["config_hash"], load_policy=LOAD_POLICY,
                    model_spec=cfg["model_spec"], profile_spec=cfg["profile_spec"],
                    loop_spec=cfg["loop_spec"], rmlp_spec=cfg["rmlp_spec"],
                    temperature_spec=cfg["temperature_spec"],
                    objective_spec=cfg["profile_spec"]["values"]["objective"],
                    evaluation_spec=cfg["profile_spec"]["values"]["evaluation"])
    for name, value in expected.items():
        if meta[name] != value:
            raise V4SchemaError(name + " differs from resolved config")
    data = meta["data_spec"]
    _exact(data, {"protocol", "split", "checksums", "sampling", "scope", "runtime"}, "data_spec")
    if data["protocol"] != cfg["profile_spec"]["values"]["data"] or data["scope"] not in ("synthetic", "real"):
        raise V4SchemaError("data protocol/scope mismatch")
    if not isinstance(data["split"], str) or not isinstance(data["sampling"], str):
        raise V4SchemaError("data split/sampling must be strings")
    if type(data["checksums"]) is not dict or not data["checksums"] or type(data["runtime"]) is not dict:
        raise V4SchemaError("data checksums/runtime are required")
    for name, value in data["checksums"].items():
        if not isinstance(name, str):
            raise V4SchemaError("data checksum names must be strings")
        _hash(value, "data checksum")
    normalizers = meta["normalizer_spec"]
    if (type(normalizers) is not dict or set(normalizers) != {"policy", "records"} or
            type(normalizers["policy"]) is not str or type(normalizers["records"]) is not dict):
        raise V4SchemaError("normalizer_spec mismatch")
    provenance = meta["provenance_spec"]
    if type(provenance) is not dict or not provenance:
        raise V4SchemaError("provenance_spec is required")
    for field in ("target_sha", "source_sha256", "code_version"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise V4SchemaError("provenance_spec." + field + " is required")
    state = meta["resume_state"]
    if (type(state) is not dict or type(state.get("epoch")) is not int or state["epoch"] < 0 or
            state.get("checkpoint_role") not in ("epoch", "final")):
        raise V4SchemaError("resume_state epoch/role mismatch")
    from linearno_loop.state import validate_resume, validate_normalizers
    validate_resume(state)
    validate_normalizers(normalizers, data['checksums'])
    measurement = meta["parameter_measurement"]
    _exact(measurement, {"status", "total", "trainable", "groups"}, "parameter_measurement")
    if (measurement["status"] != "measured" or type(measurement["total"]) is not int or
            type(measurement["trainable"]) is not int or type(measurement["groups"]) is not dict or
            sum(measurement["groups"].values()) != measurement["total"] or
            measurement["trainable"] != measurement["total"]):
        raise V4SchemaError("parameter measurement mismatch")
    members = meta["ensemble_manifest"]
    if type(members) is not list:
        raise V4SchemaError("ensemble_manifest must be a list")
    for index, member in enumerate(members):
        _exact(member, {"member_id", "order", "path", "sha256", "format"}, "ensemble member")
        relative = PurePosixPath(member["path"])
        if (member["order"] != index or member["format"] != "state_dict" or relative.is_absolute()
                or ".." in relative.parts or "\\" in member["path"]):
            raise V4SchemaError("ensemble member mismatch")
        _hash(member["sha256"], "ensemble sha256")
    saved = meta["metadata_hash"]
    body = deepcopy(meta); body.pop("metadata_hash")
    if saved != digest(body):
        raise V4SchemaError("metadata hash mismatch")
    return deepcopy(meta)


def write_metadata(path, meta):
    checked = validate_metadata(meta)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(checked, sort_keys=True, separators=(",", ":")) + "\n")


def read_metadata(path):
    return validate_metadata(json.loads(Path(path).read_text(encoding="utf-8")))


def restore_config(meta, *, explicit=None, runtime=None, strict=True):
    if strict is not True:
        raise V4SchemaError("strict=True is required")
    checked = validate_metadata(meta); cfg = checked["resolved_config"]
    expected = dict(task=cfg["task"], profile=cfg["profile"], architecture=cfg["architecture"],
                    temperature_mode=cfg["temperature_mode"], residual_mode=cfg["residual_mode"],
                    linearno_loop=True, seed=cfg["seed"])
    for key, value in (explicit or {}).items():
        if key not in expected or expected[key] != value:
            raise V4SchemaError("explicit " + key + " conflicts with V4 metadata")
    if set(runtime or {}) - {"device", "experiment_dir"}:
        raise V4SchemaError("unsupported V4 runtime override")
    return {"config": deepcopy(cfg), "runtime_changes": deepcopy(runtime or {}),
            "load_policy": deepcopy(LOAD_POLICY)}
