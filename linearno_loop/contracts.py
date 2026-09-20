"""Frozen LL1 names and strict JSON helpers; standard library only."""
from copy import deepcopy
import hashlib
import json
import math

FAMILY = 'linearno_loop'
ARCHITECTURE_EXTENSION = 'loop_linearno_v1'
SCHEMA_VERSION = CONFIG_VERSION = 1
FORMULA_VERSION = 'point-loop-residual-v1'
TOPOLOGY_FIELDS = ('prefix_blocks', 'recurrent_core_blocks', 'loop_repeats', 'suffix_blocks')
PRESETS = {'p1_c3_r2_s1': (1, 3, 2, 1), 'p2_c2_r2_s2': (2, 2, 2, 2)}
RESIDUAL_MODES = ('sr_1_over_r', 'rb_attnres', 'lb_attnres_1_over_r')
CLASS_PATHS = {
    'standard': 'cdlno.linearno_loop.standard.LoopedStandardModel',
    'airfrans': 'cdlno.linearno_loop.airfrans.LoopedAirfRANSModel',
    'car': 'cdlno.linearno_loop.shapenet.LoopedShapeNetModel',
}
# Descriptors for FUTURE explicit constructors, not importable placeholder models.
CLI_CONTRACT = {
    '--linearno-loop': {'field': 'linearno_loop', 'wire_value': '0|1', 'typed_value': 'bool'},
    '--linearno-loop-topology': {'field': 'topology_preset', 'wire_value': 'preset|custom'},
    '--linearno-loop-prefix-blocks': {'field': 'prefix_blocks', 'wire_value': 'int>=0'},
    '--linearno-loop-core-blocks': {'field': 'recurrent_core_blocks', 'wire_value': 'int>=1'},
    '--linearno-loop-repeats': {'field': 'loop_repeats', 'wire_value': 'int>=1'},
    '--linearno-loop-suffix-blocks': {'field': 'suffix_blocks', 'wire_value': 'int>=1'},
    '--linearno-loop-residual-mode': {'field': 'residual_mode', 'wire_value': '|'.join(RESIDUAL_MODES)},
    '--linearno-loop-rank-multiplier': {'field': 'rank_multiplier', 'wire_value': '1|2'},
    '--linearno-rank': {'field': 'linearno_rank', 'wire_value': 'actual M int>=1'},
}
HISTORY_FLAGS = ('linearno_latent_attnres', 'linearno_history_k_conditioning',
                 'linearno_attnres_history_dropout_p')
OPTIONS = {'topology_preset', 'residual_mode', *TOPOLOGY_FIELDS, 'rank_multiplier', 'linearno_rank'}
SHARING = dict(core='same_physical_module_each_round', core_includes=['operator', 'mlp', 'ln_1', 'ln_2'],
               prefix_core_suffix='disjoint_physical_modules', core_registration='once',
               recompute_qkv_context_each_visit=True, output_head='last_suffix_only_once',
               routers='independent_per_logical_receiver', history='forward_local_raw_points',
               detach=False, cross_forward=False, cross_physical_time=False, cross_member=False)


class LoopSchemaError(ValueError):
    """Invalid loop contract; raised before any model/weight loading."""


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise LoopSchemaError(f'{name}: expected integer >= {minimum}, got {value!r}')
    return value


def exact(value, keys, name):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise LoopSchemaError(f'{name}: expected JSON object with string keys')
    missing, extra = set(keys) - value.keys(), value.keys() - set(keys)
    if missing or extra:
        raise LoopSchemaError(f'{name}: missing={sorted(missing)}, unknown={sorted(extra)}')


def json_value(value, path='root'):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for i, v in enumerate(value):
            json_value(v, f'{path}[{i}]')
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for k, v in value.items():
            json_value(v, f'{path}.{k}')
        return
    raise LoopSchemaError(f'{path}: expected finite JSON value, got {type(value).__name__}')


def canonical_json(value):
    json_value(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def differences(expected, actual, path='root'):
    """Field-level diagnostics; bool is never numerically equal to int."""
    if type(expected) is not type(actual):
        return [f'{path}: expected {expected!r}, got {actual!r} (type mismatch)']
    if isinstance(expected, dict):
        result = []
        for k in sorted(expected.keys() | actual.keys()):
            if k not in expected or k not in actual:
                result.append(f'{path}.{k}: missing or unknown field')
            else:
                result.extend(differences(expected[k], actual[k], f'{path}.{k}'))
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f'{path}: list length mismatch']
        return [d for i, (a, b) in enumerate(zip(expected, actual))
                for d in differences(a, b, f'{path}[{i}]')]
    return [] if expected == actual else [f'{path}: expected {expected!r}, got {actual!r}']


def require_equal(expected, actual, path):
    errors = differences(expected, actual, path)
    if errors:
        raise LoopSchemaError('\n'.join(errors))


def seal(value, field):
    result = deepcopy(value)
    result[field] = digest({k: v for k, v in result.items() if k != field})
    return result


def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise LoopSchemaError(f'duplicate JSON field: {k}')
            result[k] = v
        return result
    with open(path, encoding='utf-8') as stream:
        result = json.load(stream, object_pairs_hook=pairs)
    json_value(result)
    return result
