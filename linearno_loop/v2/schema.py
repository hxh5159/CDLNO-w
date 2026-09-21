"""Metadata-first v2 schema. Validation has no tensor or task imports."""

from copy import deepcopy
import inspect
from pathlib import Path, PurePosixPath

from linearno_loop.schema import LOAD_POLICY, PROVENANCE_FIELDS, REFERENCE_PINS
from linearno_loop.state import hash_string, text, validate_normalizers, validate_resume
from .config import validate_config
from .contracts import (
    ARCHITECTURE_EXTENSION, CONFIG_VERSION, FAMILY, HISTORY_FLAGS, SCHEMA_VERSION,
    LoopSchemaError, exact, integer, json_value, read_json, require_equal, seal, canonical_json,
)

SECTIONS = ("loop_spec", "model_spec", "profile_spec", "data_spec", "objective_spec",
            "evaluation_spec", "provenance_spec", "normalizer_spec", "resume_state", "ensemble_manifest")


def validate_constructor(model_spec, constructor):
    if model_spec["class_path"] != constructor.__module__ + "." + constructor.__qualname__:
        raise LoopSchemaError("model_spec.class_path: real constructor mismatch")
    signature = inspect.signature(constructor)
    if any(parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
           for parameter in signature.parameters.values()):
        raise LoopSchemaError("constructor must expose explicit fields, not *args/**kwargs")
    fields = {key for key, parameter in signature.parameters.items()
              if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}
    require_equal(sorted(fields), sorted(model_spec["constructor_kwargs"]), "constructor_kwargs.fields")
    try:
        signature.bind(**model_spec["constructor_kwargs"])
    except TypeError as error:
        raise LoopSchemaError(str(error)) from error


def validate_metadata(metadata, *, constructor=None):
    json_value(metadata)
    if type(metadata) is not dict or metadata.get("family") != FAMILY:
        raise LoopSchemaError("family: v2 loop loader only")
    keys = {"family", "architecture_extension", "schema_version", "config_version", "config_hash",
            "resolved_config", "load_policy", "metadata_hash", *SECTIONS}
    exact(metadata, keys, "metadata")
    config = validate_config(metadata["resolved_config"])
    for name, expected in (("architecture_extension", ARCHITECTURE_EXTENSION),
                           ("schema_version", SCHEMA_VERSION), ("config_version", CONFIG_VERSION),
                           ("config_hash", config["config_hash"]), ("load_policy", LOAD_POLICY)):
        require_equal(expected, metadata[name], name)
    for name in ("loop_spec", "model_spec", "profile_spec"):
        require_equal(config[name], metadata[name], name)
    if constructor is not None:
        validate_constructor(metadata["model_spec"], constructor)
    for name in ("objective", "evaluation"):
        require_equal(config["profile_spec"]["values"][name], metadata[name + "_spec"], name + "_spec")
    data = metadata["data_spec"]
    exact(data, {"protocol", "split", "checksums", "sampling", "scope", "runtime"}, "data_spec")
    require_equal(config["profile_spec"]["values"]["data"], data["protocol"], "data_spec.protocol")
    if data["scope"] not in ("synthetic", "real"):
        raise LoopSchemaError("data_spec.scope must label synthetic or real")
    text(data["split"], "data_spec.split"); text(data["sampling"], "data_spec.sampling")
    if type(data["checksums"]) is not dict or not data["checksums"] or type(data["runtime"]) is not dict:
        raise LoopSchemaError("data_spec: actual named checksums/runtime mapping required")
    for name, checksum in data["checksums"].items():
        text(name, "data checksum name"); hash_string(checksum, "data checksum")
    provenance = metadata["provenance_spec"]
    exact(provenance, PROVENANCE_FIELDS, "provenance_spec")
    for key in ("target_sha", "base_commit", "transolver_sha", "linearno_sha", "attnres_sha", "kimi_k3_sha"):
        hash_string(provenance[key], "provenance." + key, 40)
    for key in ("paper_sha256", "source_sha256", "normalized_patch_sha256"):
        hash_string(provenance[key], "provenance." + key)
    for key, expected in REFERENCE_PINS.items():
        require_equal(expected, provenance[key], "provenance." + key)
    if type(provenance["dirty"]) is not bool:
        raise LoopSchemaError("provenance.dirty: bool required")
    for key, expected in (("paper_version", "2511.06294v3"),
                          ("residual_scaling_version", "2606.18524v1"),
                          ("config_schema_version", CONFIG_VERSION),
                          ("metadata_schema_version", SCHEMA_VERSION)):
        require_equal(expected, provenance[key], "provenance." + key)
    text(provenance["code_version"], "provenance.code_version")
    if type(provenance["command"]) is not list or not provenance["command"] or type(provenance["environment"]) is not dict or not provenance["environment"]:
        raise LoopSchemaError("provenance: command/environment required")
    if any(type(argument) is not str for argument in provenance["command"]):
        raise LoopSchemaError("provenance.command: string argv required")
    validate_normalizers(metadata["normalizer_spec"], data["checksums"])
    validate_resume(metadata["resume_state"])
    members = metadata["ensemble_manifest"]
    if type(members) is not list:
        raise LoopSchemaError("ensemble_manifest: ordered list required")
    identifiers = set()
    for index, member in enumerate(members):
        exact(member, {"member_id", "order", "path", "sha256", "format"}, "ensemble member")
        text(member["member_id"], "member_id"); integer(member["order"], "member order")
        if member["member_id"] in identifiers or member["order"] != index:
            raise LoopSchemaError("ensemble id/order conflict")
        identifiers.add(member["member_id"]); text(member["path"], "ensemble path")
        path = PurePosixPath(member["path"])
        if path.is_absolute() or ".." in path.parts or not path.name or "\\" in member["path"]:
            raise LoopSchemaError("ensemble path must be relative descendant")
        if member["format"] != "state_dict":
            raise LoopSchemaError("loop ensemble requires state_dict")
        hash_string(member["sha256"], "ensemble sha256")
    require_equal(seal(metadata, "metadata_hash")["metadata_hash"], metadata["metadata_hash"], "metadata_hash")
    return deepcopy(metadata)


def make_metadata(config, **sections):
    checked = validate_config(config)
    external = set(SECTIONS) - {"loop_spec", "model_spec", "profile_spec", "objective_spec", "evaluation_spec"}
    exact(sections, external, "metadata sections")
    result = dict(
        family=FAMILY, architecture_extension=ARCHITECTURE_EXTENSION,
        schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION,
        config_hash=checked["config_hash"], resolved_config=checked,
        load_policy=deepcopy(LOAD_POLICY), **deepcopy(sections),
    )
    for name in ("loop_spec", "model_spec", "profile_spec"):
        result[name] = deepcopy(checked[name])
    for name in ("objective", "evaluation"):
        result[name + "_spec"] = deepcopy(checked["profile_spec"]["values"][name])
    return validate_metadata(seal(result, "metadata_hash"))


def read_metadata(path):
    return validate_metadata(read_json(path))


def write_metadata(path, metadata):
    checked = validate_metadata(metadata)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(checked) + "\n")


def restore_config(metadata, *, explicit=None, runtime=None, strict=True):
    if strict is not True:
        raise LoopSchemaError("strict must be True")
    checked = validate_metadata(metadata)
    config, loop = checked["resolved_config"], checked["loop_spec"]
    explicit = {} if explicit is None else explicit
    runtime = {} if runtime is None else runtime
    json_value(explicit, "explicit"); json_value(runtime, "runtime")
    if type(explicit) is not dict or type(runtime) is not dict:
        raise LoopSchemaError("explicit/runtime require objects")
    if set(explicit) & set(HISTORY_FLAGS):
        raise LoopSchemaError("loop metadata cannot mix history flags")
    if "linearno_rank" in explicit and "rank_multiplier" in explicit:
        raise LoopSchemaError("explicit actual rank/multiplier conflict")
    expected = {**loop, "family": FAMILY, "architecture_extension": ARCHITECTURE_EXTENSION,
                "linearno_loop": True, "linearno_rank": loop["resolved_rank"],
                "model_spec": config["model_spec"],
                "seed": config["profile_spec"]["values"]["runtime"]["seed"]}
    if set(explicit) - expected.keys():
        raise LoopSchemaError("unknown explicit checkpoint fields: " + str(sorted(set(explicit) - expected.keys())))
    errors = []
    from linearno_loop.contracts import differences
    for key, value in explicit.items():
        errors.extend(differences(expected[key], value, "explicit." + key))
    if errors:
        raise LoopSchemaError("\n".join(errors))
    if set(runtime) - {"device", "experiment_dir"}:
        raise LoopSchemaError("only device/experiment_dir are runtime changes")
    for key, value in runtime.items():
        text(value, "runtime." + key)
    saved_runtime = config["profile_spec"]["values"]["runtime"]
    changes = {key: dict(saved=saved_runtime.get(key), requested=value) for key, value in runtime.items()
               if saved_runtime.get(key) != value}
    return dict(config=deepcopy(config), runtime_changes=changes, load_policy=deepcopy(LOAD_POLICY))

