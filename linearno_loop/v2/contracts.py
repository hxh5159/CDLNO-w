"""Frozen v2 names. This module deliberately has no tensor dependencies."""

from linearno_loop.contracts import (
    FAMILY,
    FORMULA_VERSION,
    HISTORY_FLAGS,
    PRESETS,
    RESIDUAL_MODES,
    TOPOLOGY_FIELDS,
    LoopSchemaError,
    canonical_json,
    differences,
    digest,
    exact,
    integer,
    json_value,
    read_json,
    require_equal,
    seal,
)

ARCHITECTURE_EXTENSION = "loop_linearno_ffn_v2"
SCHEMA_VERSION = CONFIG_VERSION = 2
CHECKPOINT_FORMAT = "linearno-loop-epoch-pair-v2"
CORE_FFN_MODES = ("round_specific", "round_specific_latent")
CLASS_PATHS = {
    "standard": "cdlno.linearno_loop.v2.standard.LoopedStandardModelV2",
    "airfrans": "cdlno.linearno_loop.v2.airfrans.LoopedAirfRANSModelV2",
    "car": "cdlno.linearno_loop.v2.shapenet.LoopedShapeNetModelV2",
}
OPTIONS = {
    "topology_preset",
    "residual_mode",
    "core_ffn_mode",
    *TOPOLOGY_FIELDS,
    "rank_multiplier",
    "linearno_rank",
}
CLI_CONTRACT = {
    "--linearno-loop": {"field": "linearno_loop", "wire_value": "0|1"},
    "--linearno-loop-topology": {"field": "topology_preset", "wire_value": "preset|custom"},
    "--linearno-loop-prefix-blocks": {"field": "prefix_blocks", "wire_value": "int>=0"},
    "--linearno-loop-core-blocks": {"field": "recurrent_core_blocks", "wire_value": "int>=1"},
    "--linearno-loop-repeats": {"field": "loop_repeats", "wire_value": "int>=1"},
    "--linearno-loop-suffix-blocks": {"field": "suffix_blocks", "wire_value": "int>=1"},
    "--linearno-loop-residual-mode": {"field": "residual_mode", "wire_value": "|".join(RESIDUAL_MODES)},
    "--linearno-loop-core-ffn-mode": {"field": "core_ffn_mode", "wire_value": "|".join(CORE_FFN_MODES)},
    "--linearno-loop-rank-multiplier": {"field": "rank_multiplier", "wire_value": "1|2"},
    "--linearno-rank": {"field": "linearno_rank", "wire_value": "actual M int>=1"},
}

