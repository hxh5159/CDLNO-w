"""V3 names and strict JSON helpers (standard library only)."""

from copy import deepcopy
import hashlib
import json
import math

FAMILY = "linearno_loop"
ARCHITECTURE_SELECTOR = "operator_latent_adapter_v3"
ARCHITECTURE_SELECTOR_FIELD = "architecture"
ARCHITECTURE_EXTENSION = "loop_linearno_latent_adapter_v3"
SCHEMA_VERSION = CONFIG_VERSION = CHECKPOINT_VERSION = 3
CHECKPOINT_FORMAT = "linearno-loop-epoch-pair-v3"
FORMULA_VERSION = "loop-linearno-latent-adapter-v3-formula-v1"

TASKS = ("airfoil", "darcy", "elasticity", "pipe", "plasticity", "ns", "airfrans", "car")
COST_PROFILES = ("matched_v1", "efficient_v1", "custom")
RESIDUAL_MODES = ("sr_1_over_r", "rb_attnres", "lb_attnres_1_over_r")
ADAPTER_MODES = ("none", "bilateral_qk_lowrank_second_visit")
TOPOLOGY_FIELDS = ("prefix_blocks", "recurrent_core_blocks", "loop_repeats", "suffix_blocks")
DEPTH_TOPOLOGIES = {
    "d12": (2, 4, 2, 2),
    "d20": (2, 8, 2, 2),
    "d28": (2, 12, 2, 2),
    "d60": (2, 28, 2, 2),
}
DEPTHS = tuple(DEPTH_TOPOLOGIES)
ABLATIONS = ((False, False), (False, True), (True, False), (True, True))
CLASS_PATHS = {
    "standard": "cdlno.linearno_loop.v3.standard.LoopedStandardModelV3",
    "airfrans": "cdlno.linearno_loop.v3.airfrans.LoopedAirfRANSModelV3",
    "car": "cdlno.linearno_loop.v3.shapenet.LoopedShapeNetModelV3",
}
CLI_CONTRACT = {
    "--linearno-loop-architecture": {"field": ARCHITECTURE_SELECTOR_FIELD, "wire_value": ARCHITECTURE_SELECTOR},
    "--linearno-loop-cost-profile": {"field": "cost_profile", "wire_value": "|".join(COST_PROFILES)},
    "--linearno-loop-topology": {"field": "topology_preset", "wire_value": "d12|d20|d28|d60|preset|custom"},
    "--linearno-loop-executed-depth": {"field": "executed_depth", "wire_value": "int>=6,even"},
    "--linearno-loop-prefix-blocks": {"field": "prefix_blocks", "wire_value": "int>=0"},
    "--linearno-loop-core-blocks": {"field": "recurrent_core_blocks", "wire_value": "int>=1"},
    "--linearno-loop-repeats": {"field": "loop_repeats", "wire_value": "int>=1"},
    "--linearno-loop-suffix-blocks": {"field": "suffix_blocks", "wire_value": "int>=1"},
    "--linearno-loop-residual-mode": {"field": "residual_mode", "wire_value": "|".join(RESIDUAL_MODES)},
    "--linearno-loop-latent": {"field": "latent_enabled", "wire_value": "0|1"},
    "--linearno-loop-adapter-mode": {"field": "adapter_mode", "wire_value": "|".join(ADAPTER_MODES)},
    "--linearno-loop-adapter-rank": {"field": "adapter_rank", "wire_value": "int>=1"},
    "--linearno-loop-adapter-alpha": {"field": "adapter_alpha", "wire_value": "finite>0"},
    "--linearno-loop-hidden-width": {"field": "hidden_width", "wire_value": "int>=1"},
    "--linearno-loop-latent-width": {"field": "latent_width", "wire_value": "int>=1"},
    "--linearno-rank": {"field": "actual_M", "wire_value": "int>=1"},
    "--linearno-loop-heads": {"field": "heads", "wire_value": "int>=1"},
}
HISTORY_FLAGS = ("linearno_latent_attnres", "linearno_history_k_conditioning", "linearno_attnres_history_dropout_p")


class V3SchemaError(ValueError):
    """Raised before any model construction or tensor/weight loading."""


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise V3SchemaError(f"{name}: expected integer >= {minimum}, got {value!r}")
    return value


def finite(value, name, minimum=None, exclusive=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise V3SchemaError(f"{name}: expected finite number, got {value!r}")
    if minimum is not None and (value < minimum or (exclusive and value == minimum)):
        raise V3SchemaError(f"{name}: value is out of range")
    return value


def exact(value, keys, name):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise V3SchemaError(f"{name}: expected JSON object with string keys")
    missing, extra = set(keys) - value.keys(), value.keys() - set(keys)
    if missing or extra:
        raise V3SchemaError(f"{name}: missing={sorted(missing)}, unknown={sorted(extra)}")


def json_value(value, path="root"):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            json_value(item, f"{path}[{i}]")
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for key, item in value.items():
            json_value(item, f"{path}.{key}")
        return
    raise V3SchemaError(f"{path}: expected finite JSON value, got {type(value).__name__}")


def canonical_json(value):
    json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def differences(expected, actual, path="root"):
    if type(expected) is not type(actual):
        return [f"{path}: expected {expected!r}, got {actual!r} (type mismatch)"]
    if isinstance(expected, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                result.append(f"{path}.{key}: missing or unknown field")
            else:
                result.extend(differences(expected[key], actual[key], f"{path}.{key}"))
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: list length mismatch"]
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in differences(a, b, f"{path}[{i}]")]
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


def require_equal(expected, actual, path):
    errors = differences(expected, actual, path)
    if errors:
        raise V3SchemaError("\n".join(errors))


def seal(value, field):
    result = deepcopy(value)
    result[field] = digest({k: v for k, v in result.items() if k != field})
    return result


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise V3SchemaError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    import json as _json
    with open(path, encoding="utf-8") as stream:
        result = _json.load(stream, object_pairs_hook=pairs)
    json_value(result)
    return result
