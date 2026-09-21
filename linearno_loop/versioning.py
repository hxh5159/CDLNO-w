"""Version dispatch for loop configuration and metadata before tensor imports."""

from linearno_loop.contracts import ARCHITECTURE_EXTENSION as V1_EXTENSION, OPTIONS as V1_OPTIONS, read_json
from linearno_loop import config as v1_config
from linearno_loop import schema as v1_schema
from linearno_loop.v2.contracts import ARCHITECTURE_EXTENSION as V2_EXTENSION, OPTIONS as V2_OPTIONS
from linearno_loop.v2 import config as v2_config
from linearno_loop.v2 import schema as v2_schema

OPTIONS = V1_OPTIONS | V2_OPTIONS


def is_v2(value):
    return value.get("architecture_extension") == V2_EXTENSION or value.get("config_version") == 2


def api(value):
    if is_v2(value):
        if value.get("architecture_extension") != V2_EXTENSION or value.get("config_version") != 2:
            raise ValueError("inconsistent v2 architecture/config version")
        return v2_schema
    if value.get("architecture_extension") != V1_EXTENSION or value.get("config_version") != 1:
        raise ValueError("unknown loop architecture/config version")
    return v1_schema


def resolve_config(task, profile, *, options, profile_overrides):
    resolver = v2_config if "core_ffn_mode" in options else v1_config
    return resolver.resolve_config(task, profile, options=options, profile_overrides=profile_overrides)


def run_directory_id(config):
    return (v2_config if is_v2(config) else v1_config).run_directory_id(config)


def read_metadata(path):
    header = read_json(path)
    return api(header).validate_metadata(header)


def validate_metadata(metadata, **kwargs):
    return api(metadata).validate_metadata(metadata, **kwargs)


def restore_config(metadata, **kwargs):
    return api(metadata).restore_config(metadata, **kwargs)


def make_metadata(config, **sections):
    schema = v2_schema if is_v2(config) else v1_schema
    return schema.make_metadata(config, **sections)


def write_metadata(path, metadata):
    return api(metadata).write_metadata(path, metadata)


def validate_constructor(model_spec, constructor, *, version):
    return (v2_schema if version == 2 else v1_schema).validate_constructor(model_spec, constructor)

