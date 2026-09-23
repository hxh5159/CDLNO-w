"""Versioned, JSON-only contracts for ``resmlp_dual_temp_v4``."""
from copy import deepcopy
import hashlib, json, math

FAMILY = "linearno_loop"
ARCHITECTURE = "resmlp_dual_temp_v4"
ARCHITECTURE_EXTENSION = "resmlp_dual_temp"
ARCHITECTURE_VERSION = 4
CHECKPOINT_SCHEMA = "resmlp_dual_temp_v4"
CHECKPOINT_VERSION = 1
CONFIG_VERSION = 4
RESIDUAL_MODE = "sr_1_over_sqrt_r"
TEMPERATURE_MODES = ("base", "latent_k_point_q", "point_k_point_q")
TASKS = ("airfoil", "darcy", "elasticity", "pipe", "ns", "plasticity", "airfrans", "car")
TOPOLOGY = ("first", "A1", "B1", "C1", "A2", "B2", "C2", "last")
RMLP_OWNERS = ("first", "A", "B", "C", "last")
RMLP_DEPTHS = {"first": 2, "A": 3, "B": 3, "C": 3, "last": 2}

class V4SchemaError(ValueError):
    pass

def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise V4SchemaError(f"{name} must be integer >= {minimum}")

def finite(value, name, positive=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise V4SchemaError(f"{name} must be finite numeric")
    if positive and value <= 0:
        raise V4SchemaError(f"{name} must be positive")

def seal(value, field="config_hash"):
    body = deepcopy(value); body.pop(field, None); body[field] = digest(body); return body

def require_equal(expected, actual, path):
    if expected != actual:
        raise V4SchemaError(f"{path} mismatch: expected {expected!r}, got {actual!r}")

