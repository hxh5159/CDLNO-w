"""Version dispatch for loop configuration and metadata before tensor imports."""

from linearno_loop.contracts import ARCHITECTURE_EXTENSION as V1_EXTENSION, OPTIONS as V1_OPTIONS, read_json
from linearno_loop import config as v1_config
from linearno_loop import schema as v1_schema
from linearno_loop.v2.contracts import ARCHITECTURE_EXTENSION as V2_EXTENSION, OPTIONS as V2_OPTIONS
from linearno_loop.v2 import config as v2_config
from linearno_loop.v2 import schema as v2_schema

V3_EXTENSION = "loop_linearno_latent_adapter_v3"
V4_EXTENSION = "resmlp_dual_temp"
V5_EXTENSION = "partial_share_feature_gate"
# Keep the legacy parser importable without the optional V3 package. These are
# wire field names only; the real schema is imported inside V3-selected calls.
V3_OPTIONS = {'architecture', 'cost_profile', 'topology_preset', 'executed_depth',
              'prefix_blocks', 'recurrent_core_blocks', 'loop_repeats', 'suffix_blocks',
              'residual_mode', 'hidden_width', 'latent_width', 'actual_M', 'heads',
              'latent_enabled', 'adapter_mode', 'adapter_rank', 'adapter_alpha'}
V5_OPTIONS = {'architecture', 'topology_preset', 'executed_depth',
              'prefix_blocks', 'recurrent_core_blocks', 'loop_repeats', 'suffix_blocks',
              'residual_mode', 'expert_count', 'expert_width', 'actual_M'}
OPTIONS = V1_OPTIONS | V2_OPTIONS | V3_OPTIONS | V5_OPTIONS


def is_v2(value):
    return value.get("architecture_extension") == V2_EXTENSION or value.get("config_version") == 2


def is_v3(value):
    return value.get("architecture_extension") == V3_EXTENSION or value.get("config_version") == 3

def is_v4(value):
    return value.get("architecture") == "resmlp_dual_temp_v4" and value.get("architecture_extension") == V4_EXTENSION

def is_v5(value):
    return (value.get("architecture") == "partial_share_feature_gate_v5" and
            value.get("architecture_extension") == V5_EXTENSION and
            value.get("architecture_version") == 5)


def api(value):
    if is_v5(value):
        from linearno_loop.v5 import schema as v5_schema
        return v5_schema
    if is_v4(value):
        from linearno_loop.v4 import schema as v4_schema
        return v4_schema
    if is_v3(value):
        from linearno_loop.v3 import schema as v3_schema
        if value.get("architecture_extension") != V3_EXTENSION or value.get("config_version") != 3:
            raise ValueError("inconsistent v3 architecture/config version")
        return v3_schema
    if is_v2(value):
        if value.get("architecture_extension") != V2_EXTENSION or value.get("config_version") != 2:
            raise ValueError("inconsistent v2 architecture/config version")
        return v2_schema
    if value.get("architecture_extension") != V1_EXTENSION or value.get("config_version") != 1:
        raise ValueError("unknown loop architecture/config version")
    return v1_schema


def resolve_config(task, profile, *, options, profile_overrides):
    if options.get("architecture") == "partial_share_feature_gate_v5":
        from linearno_loop.v5.config import resolve_config as resolve_v5
        return resolve_v5(task, profile=profile, options=options,
                          profile_overrides=profile_overrides)
    if options.get("architecture") == "resmlp_dual_temp_v4":
        from linearno_loop.v4.config import resolve_config as resolve_v4
        return resolve_v4(task, profile=profile, options=options, profile_overrides=profile_overrides)
    if options.get("architecture") == "operator_latent_adapter_v3":
        from linearno_loop.v3 import config as v3_config
        return v3_config.resolve_config(task, profile, options=options,
                                        profile_overrides=profile_overrides)
    resolver = v2_config if "core_ffn_mode" in options else v1_config
    return resolver.resolve_config(task, profile, options=options, profile_overrides=profile_overrides)


def run_directory_id(config):
    if is_v5(config):
        from linearno_loop.v5.config import run_directory_id as run
        return run(config)
    if is_v4(config):
        from linearno_loop.v4.config import run_directory_id as run
        return run(config)
    if is_v3(config):
        from linearno_loop.v3 import config as v3_config
        return v3_config.run_directory_id(config)
    return (v2_config if is_v2(config) else v1_config).run_directory_id(config)


def read_metadata(path):
    header = read_json(path)
    return api(header).validate_metadata(header)


def validate_metadata(metadata, **kwargs):
    return api(metadata).validate_metadata(metadata, **kwargs)


def restore_config(metadata, **kwargs):
    return api(metadata).restore_config(metadata, **kwargs)


def make_metadata(config, **sections):
    if is_v5(config):
        from linearno_loop.v5.schema import make_metadata as make
        return make(config, **sections)
    if is_v4(config):
        from linearno_loop.v4.schema import make_metadata as make
        return make(config,**sections)
    v3_schema = None
    if is_v3(config):
        from linearno_loop.v3 import schema as v3_schema
    schema = v3_schema if is_v3(config) else v2_schema if is_v2(config) else v1_schema
    return schema.make_metadata(config, **sections)


def write_metadata(path, metadata):
    return api(metadata).write_metadata(path, metadata)


def validate_constructor(model_spec, constructor, *, version):
    if version == 5:
        import inspect
        return inspect.signature(constructor).bind(**model_spec['constructor_kwargs'])
    if version == 3:
        from linearno_loop.v3 import schema as schema
    else:
        schema = v2_schema if version == 2 else v1_schema
    return schema.validate_constructor(model_spec, constructor)
