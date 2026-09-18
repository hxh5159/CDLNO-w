"""Pure config/schema contracts for the LinearNO history family.

This module deliberately does not import torch, task entry points, factories, or
model implementations. The R1 metadata fixture remains a historical contract;
complete runnable R5 metadata is validated by cdlno.linearno_history.checkpoint.
A/K math is not here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping

SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
RESEARCH_FAMILY = "linearno_history"
BASE_FAMILY = "linearno"
ARCHITECTURE_EXTENSION = "linearno_history_v1"
FEATURE_FIELDS = (
    "linearno_latent_attnres",
    "linearno_history_k_conditioning",
    "linearno_attnres_history_dropout_p",
)
BASE_VARIANTS = ("plain", "temp", "conv", "conv_temp", "airfrans", "shapenet")
TEMPERATURE_SEMANTICS = {
    "plain": {"used": False, "initial": None, "clamp": None},
    "conv": {"used": False, "initial": None, "clamp": None},
    "temp": {"used": True, "initial": 0.5, "clamp": [0.01, 1.0]},
    "conv_temp": {"used": True, "initial": 0.5, "clamp": [0.01, 1.0]},
    "airfrans": {"used": False, "initial": 0.5, "clamp": None, "dead_parameter": True},
    "shapenet": {"used": True, "initial": 0.5, "clamp": [0.1, 2.0], "key_spelling": "tempreature_q/k"},
}
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
PROFILES = ("paper_table8_on_release_model", "official_release", "transolver_matched")
DEPTHS = (4, 5, 6, 7, 8)
_SIGNATURE_RE = re.compile(r"^A[01]K[01]$")
_CLASS_PATH_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


class HistorySchemaError(ValueError):
    """A structural/configuration mismatch that must happen before construction."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistorySchemaError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], required: set[str], name: str, *, allow: set[str] = set()) -> None:
    missing = required - set(value)
    unknown = set(value) - required - allow
    if missing or unknown:
        raise HistorySchemaError(f"{name}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise HistorySchemaError(f"{name} must be a JSON boolean")
    return value


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HistorySchemaError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HistorySchemaError(f"{name} must be finite numeric")
    if value < minimum or value > maximum:
        raise HistorySchemaError(f"{name} must be in [{minimum}, {maximum}]")
    return float(value)


def _class_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _CLASS_PATH_RE.fullmatch(value):
        raise HistorySchemaError(f"{name} must be a dotted class path")
    lowered = value.lower()
    if any(token in lowered for token in ("placeholder", "dummy", "todo", "identity")):
        raise HistorySchemaError(f"{name} cannot identify a placeholder model")
    return value


def _clone(value: Any) -> Any:
    return deepcopy(value)


def feature_signature(attnres: bool, history_k: bool) -> str:
    return f"A{int(attnres)}K{int(history_k)}"


def resolve_feature_config(config: Mapping[str, Any], *, production: bool = True) -> dict[str, Any]:
    """Resolve the only two feature truths without touching a model or RNG."""
    config = _mapping(config, "feature_config")
    _exact(config, set(FEATURE_FIELDS), "feature_config")
    attnres = _bool(config[FEATURE_FIELDS[0]], FEATURE_FIELDS[0])
    history_k = _bool(config[FEATURE_FIELDS[1]], FEATURE_FIELDS[1])
    raw_dropout = config[FEATURE_FIELDS[2]]
    if raw_dropout is None:
        dropout = 0.1 if attnres else None
        source = "attnres_default" if attnres else "disabled"
    else:
        dropout = _finite(raw_dropout, FEATURE_FIELDS[2])
        if not attnres:
            raise HistorySchemaError("history dropout is only valid when latent AttnRes is enabled")
        # The released research configuration remains p=.1. A production p=0
        # exception is deliberately narrow: it is an explicit A1K0 ablation.
        if production and dropout not in (0.0, 0.1):
            raise HistorySchemaError("production AttnRes history dropout must be 0.1, or 0 only for A1K0")
        if production and dropout == 0.0 and history_k:
            raise HistorySchemaError("p=0 history dropout is only supported for the A1K0 ablation")
        source = "explicit"
    return {
        FEATURE_FIELDS[0]: attnres,
        FEATURE_FIELDS[1]: history_k,
        FEATURE_FIELDS[2]: dropout,
        "feature_signature": feature_signature(attnres, history_k),
        "dropout_source": source,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
    }


def validate_family_feature_config(family: str, config: Mapping[str, Any], *, production: bool = True) -> dict[str, Any]:
    if not isinstance(family, str):
        raise HistorySchemaError("family must be a string")
    resolved = resolve_feature_config(config, production=production)
    if family != RESEARCH_FAMILY and resolved["feature_signature"] != "A0K0":
        raise HistorySchemaError("A/K innovations are only valid for family linearno_history")
    if family == BASE_FAMILY and resolved["feature_signature"] != "A0K0":
        raise HistorySchemaError("base LinearNO cannot receive innovation flags")
    return resolved


def validate_base_descriptor(base: Mapping[str, Any]) -> dict[str, Any]:
    base = _mapping(base, "base_linearno")
    required = {"variant", "task_variant", "n_layers", "heads", "d_h", "actual_M", "temperature_semantics"}
    _exact(base, required, "base_linearno")
    variant = base["variant"]
    if variant not in BASE_VARIANTS:
        raise HistorySchemaError(f"unknown base LinearNO variant: {variant!r}")
    if not isinstance(base["task_variant"], str) or not base["task_variant"]:
        raise HistorySchemaError("base_linearno.task_variant is required")
    for key in ("n_layers", "heads", "d_h", "actual_M"):
        _int(base[key], f"base_linearno.{key}", 1)
    if base["n_layers"] not in DEPTHS:
        raise HistorySchemaError("research n_layers must be one of 4,5,6,7,8")
    if base["d_h"] * base["heads"] <= 0:
        raise HistorySchemaError("invalid base width")
    expected_temperature = TEMPERATURE_SEMANTICS[variant]
    if base["temperature_semantics"] != expected_temperature:
        raise HistorySchemaError("temperature semantics do not match the base variant")
    return _clone(base)


def _validate_attnres(spec: Mapping[str, Any], enabled: bool, dropout: float | None,
                      history_k: bool = False) -> dict[str, Any]:
    required = {"enabled", "version", "per_head_cross", "d_m_equals_d_h", "history_only_source_softmax",
                "current_in_source_softmax", "null_source", "gamma", "dropout"}
    _exact(spec, required, "innovation_spec.attnres")
    if _bool(spec["enabled"], "attnres.enabled") != enabled:
        raise HistorySchemaError("AttnRes enabled flag disagrees with feature config")
    if _int(spec["version"], "attnres.version", 1) != 1:
        raise HistorySchemaError("unsupported AttnRes version")
    for key in ("per_head_cross", "d_m_equals_d_h", "history_only_source_softmax"):
        if not _bool(spec[key], f"attnres.{key}"):
            raise HistorySchemaError(f"attnres.{key} must be true in v1")
    if _bool(spec["current_in_source_softmax"], "attnres.current_in_source_softmax"):
        raise HistorySchemaError("current summary cannot be an AttnRes source")
    null = _mapping(spec["null_source"], "attnres.null_source")
    _exact(null, {"present", "value", "always_in_softmax", "parameterized", "kept_when_all_history_masked"}, "attnres.null_source")
    if null != {"present": True, "value": "zero", "always_in_softmax": True, "parameterized": False, "kept_when_all_history_masked": True}:
        raise HistorySchemaError("invalid AttnRes null-source contract")
    gamma = _mapping(spec["gamma"], "attnres.gamma")
    _exact(gamma, {"shape", "initialization", "scope"}, "attnres.gamma")
    if gamma != {"shape": "per_receiver_scalar", "initialization": 0.0, "scope": "receiver_layer"}:
        raise HistorySchemaError("invalid AttnRes gamma contract")
    drop = _mapping(spec["dropout"], "attnres.dropout")
    _exact(drop, {"p", "train_only", "mask_granularity", "singleton_history_kept", "null_never_dropped", "inverted_scaling", "all_masked_fallback"}, "attnres.dropout")
    if drop["p"] != dropout:
        raise HistorySchemaError("AttnRes dropout p disagrees with resolved feature config")
    if (drop["p"] is not None and enabled and
            (not isinstance(drop["p"], float) or drop["p"] not in (0.0, 0.1)
             or (drop["p"] == 0.0 and history_k))):
        raise HistorySchemaError("research AttnRes dropout must be 0.1, or 0 only for A1K0")
    expected_drop = {"p": dropout, "train_only": True, "mask_granularity": "sample_by_real_source", "singleton_history_kept": True, "null_never_dropped": True, "inverted_scaling": False, "all_masked_fallback": "null_weight_one_finite_zero"}
    if drop != expected_drop:
        raise HistorySchemaError("invalid AttnRes history-dropout contract")
    return _clone(spec)


def _validate_history_k(spec: Mapping[str, Any], enabled: bool) -> dict[str, Any]:
    required = {"enabled", "version", "base_to_k_row_query", "all_history_token_bank", "point_slot_dot", "point_centering", "gate", "changes_q", "changes_v", "changes_reconstruction_q"}
    _exact(spec, required, "innovation_spec.history_k")
    if _bool(spec["enabled"], "history_k.enabled") != enabled:
        raise HistorySchemaError("history K enabled flag disagrees with feature config")
    if _int(spec["version"], "history_k.version", 1) != 1:
        raise HistorySchemaError("unsupported history K version")
    for key in ("base_to_k_row_query", "all_history_token_bank", "point_slot_dot", "point_centering"):
        if not _bool(spec[key], f"history_k.{key}"):
            raise HistorySchemaError(f"history_k.{key} must be true in v1")
    for key in ("changes_q", "changes_v", "changes_reconstruction_q"):
        if _bool(spec[key], f"history_k.{key}"):
            raise HistorySchemaError(f"history_k.{key} must be false in v1")
    gate = _mapping(spec["gate"], "history_k.gate")
    _exact(gate, {"scope", "function", "initialization", "shape"}, "history_k.gate")
    expected = {"scope": "receiver_layer_and_head", "function": "tanh", "initialization": 0.0, "shape": "[1,H,1,1]"}
    if gate != expected:
        raise HistorySchemaError("invalid history K gate contract")
    return _clone(spec)


def validate_innovation_spec(spec: Mapping[str, Any], *, feature_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = _mapping(spec, "innovation_spec")
    required = {"schema_version", "class_path", "base_linearno", "features", "attnres", "history_k", "raw_cache", "constructor_hyperparameters", "code_schema"}
    _exact(spec, required, "innovation_spec")
    if _int(spec["schema_version"], "innovation_spec.schema_version", 1) != 1:
        raise HistorySchemaError("unsupported innovation schema version")
    class_path = _class_path(spec["class_path"], "innovation_spec.class_path")
    base = validate_base_descriptor(spec["base_linearno"])
    features = _mapping(spec["features"], "innovation_spec.features")
    _exact(features, {"linearno_latent_attnres", "linearno_history_k_conditioning", "feature_signature"}, "innovation_spec.features")
    attnres = _bool(features["linearno_latent_attnres"], "innovation_spec.features.linearno_latent_attnres")
    history_k = _bool(features["linearno_history_k_conditioning"], "innovation_spec.features.linearno_history_k_conditioning")
    signature = feature_signature(attnres, history_k)
    if features["feature_signature"] != signature or signature == "A0K0":
        raise HistorySchemaError("research innovation_spec must describe A1K0, A0K1, or A1K1")
    if feature_config is not None:
        resolved = resolve_feature_config(feature_config)
        if resolved["feature_signature"] != signature:
            raise HistorySchemaError("innovation feature signature disagrees with feature config")
        dropout = resolved["linearno_attnres_history_dropout_p"]
    else:
        dropout = 0.1 if attnres else None
    validated_attnres = _validate_attnres(spec["attnres"], attnres, dropout, history_k)
    validated_k = _validate_history_k(spec["history_k"], history_k)
    raw = _mapping(spec["raw_cache"], "innovation_spec.raw_cache")
    _exact(raw, {"authoritative_value", "timing", "storage", "detach", "cross_forward", "cross_sample", "cross_time"}, "innovation_spec.raw_cache")
    expected_raw = {"authoritative_value": "C_raw_pre_attnres", "timing": "after_current_KtV_before_A_and_Q_reconstruction", "storage": "forward_local_tuple", "detach": False, "cross_forward": False, "cross_sample": False, "cross_time": False}
    if raw != expected_raw:
        raise HistorySchemaError("invalid raw-cache contract")
    hp = _mapping(spec["constructor_hyperparameters"], "innovation_spec.constructor_hyperparameters")
    if not hp or any(not isinstance(k, str) for k in hp):
        raise HistorySchemaError("constructor_hyperparameters must record all resolved structural fields")
    code = _mapping(spec["code_schema"], "innovation_spec.code_schema")
    _exact(code, {"config_schema_version", "implementation_version", "architecture_extension"}, "innovation_spec.code_schema")
    if (_int(code["config_schema_version"], "code_schema.config_schema_version", 1) != CONFIG_SCHEMA_VERSION
            or code["implementation_version"] not in ("r1-contract-only", "r5-internal-v1")
            or code["architecture_extension"] != ARCHITECTURE_EXTENSION):
        raise HistorySchemaError("invalid research code/schema descriptor")
    return {"schema_version": 1, "class_path": class_path, "base_linearno": base, "features": _clone(features), "attnres": validated_attnres, "history_k": validated_k, "raw_cache": raw, "constructor_hyperparameters": _clone(hp), "code_schema": code}


def _get(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        value = value[key]
    return value


STRUCTURAL_PATHS = (
    "schema_version", "family", "architecture_extension", "feature_signature",
    "innovation_spec.schema_version", "innovation_spec.class_path",
    "innovation_spec.features", "innovation_spec.base_linearno",
    "innovation_spec.attnres", "innovation_spec.history_k", "innovation_spec.raw_cache",
    "innovation_spec.constructor_hyperparameters", "innovation_spec.code_schema",
    "model_spec.class_path", "model_spec.constructor_kwargs",
)


def validate_research_metadata(metadata: Mapping[str, Any], *, feature_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = _mapping(metadata, "research checkpoint metadata")
    required = {"schema_version", "family", "architecture_extension", "feature_signature", "innovation_spec", "model_spec", "profile_spec", "provenance_spec"}
    _exact(metadata, required, "research checkpoint metadata")
    if _int(metadata["schema_version"], "metadata.schema_version", 1) != 1:
        raise HistorySchemaError("unsupported research metadata schema version")
    if metadata["family"] != RESEARCH_FAMILY or metadata["architecture_extension"] != ARCHITECTURE_EXTENSION:
        raise HistorySchemaError("metadata is not a LinearNO history checkpoint")
    if not _SIGNATURE_RE.fullmatch(metadata["feature_signature"]) or metadata["feature_signature"] == "A0K0":
        raise HistorySchemaError("research checkpoint must be A1K0, A0K1, or A1K1")
    if feature_config is None:
        # Standalone metadata validation has no resolved config object. Use the
        # recorded feature flags and dropout value to reconstruct that small
        # validation input; resolve_feature_config still rejects unsupported
        # values and A1K1/A0K* p=0 combinations.
        recorded = metadata["innovation_spec"]
        recorded_features = recorded.get("features", {})
        recorded_dropout = recorded.get("attnres", {}).get("dropout", {}).get("p")
        feature_config = {
            FEATURE_FIELDS[0]: recorded_features.get("linearno_latent_attnres"),
            FEATURE_FIELDS[1]: recorded_features.get("linearno_history_k_conditioning"),
            FEATURE_FIELDS[2]: recorded_dropout,
        }
    spec = validate_innovation_spec(metadata["innovation_spec"], feature_config=feature_config)
    if metadata["feature_signature"] != spec["features"]["feature_signature"]:
        raise HistorySchemaError("metadata and innovation_spec signatures differ")
    model_spec = _mapping(metadata["model_spec"], "model_spec")
    _exact(model_spec, {"class_path", "constructor_kwargs"}, "model_spec")
    _class_path(model_spec["class_path"], "model_spec.class_path")
    if not isinstance(model_spec["constructor_kwargs"], Mapping):
        raise HistorySchemaError("model_spec.constructor_kwargs must be an object")
    if model_spec["class_path"] != spec["class_path"]:
        raise HistorySchemaError("model_spec.class_path and innovation_spec.class_path differ")
    for name in ("profile_spec", "provenance_spec"):
        if not isinstance(metadata[name], Mapping):
            raise HistorySchemaError(f"{name} must be an object")
    return _clone(metadata)


def validate_legacy_metadata(metadata: Mapping[str, Any], feature_config: Mapping[str, Any] | None = None) -> str:
    """Return legacy_linearno only; never infer research from missing fields."""
    metadata = _mapping(metadata, "checkpoint metadata")
    resolved = resolve_feature_config(feature_config or {FEATURE_FIELDS[0]: False, FEATURE_FIELDS[1]: False, FEATURE_FIELDS[2]: None})
    if resolved["feature_signature"] != "A0K0":
        raise HistorySchemaError("A/K config cannot load a legacy checkpoint")
    if "innovation_spec" in metadata or metadata.get("family") == RESEARCH_FAMILY or metadata.get("architecture_extension"):
        raise HistorySchemaError("legacy loader cannot accept research metadata")
    return "legacy_linearno"


def validate_strict_load_policy(strict: bool) -> None:
    if strict is not True:
        raise HistorySchemaError("research checkpoint loading requires strict=True")


def assert_structural_compatibility(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Compare structural metadata before a constructor or weight load."""
    for path in STRUCTURAL_PATHS:
        try:
            left, right = _get(saved, path), _get(expected, path)
        except (KeyError, TypeError) as exc:
            raise HistorySchemaError(f"missing structural field before load: {path}") from exc
        if left != right:
            raise HistorySchemaError(f"structural metadata mismatch at {path}")


def run_directory_id(task: str, profile: str, layers: int, attnres: bool, history_k: bool,
                     seed: int, *, dropout_p: float | None = None) -> str:
    if task not in TASKS:
        raise HistorySchemaError(f"unknown task: {task}")
    if profile not in PROFILES:
        raise HistorySchemaError(f"unknown protocol profile: {profile}")
    if layers not in DEPTHS:
        raise HistorySchemaError("research depth must be 4..8")
    _int(seed, "seed", 0)
    suffix = "__nodrop" if (attnres and not history_k and dropout_p == 0.0) else ""
    return f"{task}__{profile}__L{layers}__{feature_signature(attnres, history_k)}{suffix}__seed{seed}"


def derive_fair_seeds(seed: int, *, task: str, split: str = "train") -> dict[str, Any]:
    _int(seed, "seed", 0)
    if task not in TASKS or not isinstance(split, str) or not split:
        raise HistorySchemaError("invalid fairness seed namespace")
    def derive(label: str) -> int:
        digest = hashlib.sha256(f"linearno_history_v1|{seed}|{task}|{split}|{label}".encode()).hexdigest()
        return int(digest[:8], 16)
    return {"public_backbone_seed": seed, "innovation_feature_seed": derive("innovation"), "dataloader_generator_seed": derive("dataloader"), "seed_namespace": f"linearno_history_v1|{task}|{split}"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
