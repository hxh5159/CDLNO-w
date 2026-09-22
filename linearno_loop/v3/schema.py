"""Metadata-first V3 contract. No tensor loading or construction."""

from copy import deepcopy
import inspect
from pathlib import Path, PurePosixPath

from linearno_loop.schema import LOAD_POLICY, PROVENANCE_FIELDS, REFERENCE_PINS
from linearno_loop.state import hash_string, text, validate_normalizers, validate_resume
from .config import validate_config
from .costs import analytic_cost, COST_VERSION
from linearno_loop.state import inspect_state
from .contracts import (
    ARCHITECTURE_EXTENSION, CONFIG_VERSION, CHECKPOINT_VERSION, FAMILY, HISTORY_FLAGS, SCHEMA_VERSION,
    V3SchemaError as LoopSchemaError, CHECKPOINT_FORMAT, ARCHITECTURE_SELECTOR, exact, integer, json_value, read_json, require_equal, seal, canonical_json,
)

SECTIONS = ("loop_spec", "model_spec", "profile_spec", "data_spec", "objective_spec",
            "evaluation_spec", "provenance_spec", "normalizer_spec", "resume_state", "ensemble_manifest", "cost_spec", "parameter_measurement")


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
        raise LoopSchemaError("family: v3 loop loader only")
    keys = {"family", "architecture_extension", "schema_version", "config_version", "config_hash",
            "resolved_config", "load_policy", "metadata_hash", "checkpoint_format", "checkpoint_version", *SECTIONS}
    exact(metadata, keys, "metadata")
    config = validate_config(metadata["resolved_config"])
    for name, expected in (("architecture_extension", ARCHITECTURE_EXTENSION),
                           ("checkpoint_format", CHECKPOINT_FORMAT),
                           ("checkpoint_version", CHECKPOINT_VERSION),
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
    state = metadata["resume_state"]
    if type(state) is not dict or 'scaler' not in state:
        raise LoopSchemaError('resume_state.scaler: required, encoded None when disabled')
    validate_resume({k:v for k,v in state.items() if k!='scaler'})
    scaler = inspect_state(state['scaler'], 'scaler')
    if scaler is not None and (type(scaler) is not dict or not scaler):
        raise LoopSchemaError('scaler must be encoded None or nonempty state dictionary')
    expected_cost = analytic_cost(config)
    require_equal(expected_cost, metadata['cost_spec'], 'cost_spec')
    measurement = metadata['parameter_measurement']
    exact(measurement, {'status','total','trainable','groups'}, 'parameter_measurement')
    if measurement['status']=='pending_model_construction':
        require_equal(dict(status='pending_model_construction',total=None,trainable=None,groups=None), measurement, 'parameter_measurement')
        if state['epoch'] != 0 or state['global_step'] != 0:
            raise LoopSchemaError('trained checkpoint requires actual parameter measurement')
    elif measurement['status']=='measured':
        integer(measurement['total'],'measured.total',1);integer(measurement['trainable'],'measured.trainable',1)
        require_equal(expected_cost['total_parameters'],measurement['total'],'measured.total')
        require_equal(expected_cost['total_parameters'],measurement['trainable'],'measured.trainable')
        require_equal(expected_cost['parameter_groups'],measurement['groups'],'measured.groups')
    else:
        raise LoopSchemaError('unknown parameter measurement status')
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
    external = set(SECTIONS) - {"loop_spec", "model_spec", "profile_spec", "objective_spec", "evaluation_spec", "cost_spec"}
    exact(sections, external, "metadata sections")
    result = dict(
        family=FAMILY, architecture_extension=ARCHITECTURE_EXTENSION,
        schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION, checkpoint_version=CHECKPOINT_VERSION,
        config_hash=checked["config_hash"], resolved_config=checked, checkpoint_format=CHECKPOINT_FORMAT,
        load_policy=deepcopy(LOAD_POLICY), **deepcopy(sections),
    )
    for name in ("loop_spec", "model_spec", "profile_spec"):
        result[name] = deepcopy(checked[name])
    for name in ("objective", "evaluation"):
        result[name + "_spec"] = deepcopy(checked["profile_spec"]["values"][name])
    result["cost_spec"] = analytic_cost(checked)
    return validate_metadata(seal(result, "metadata_hash"))


def read_metadata(path):
    return validate_metadata(read_json(path))


def write_metadata(path, metadata):
    checked = validate_metadata(metadata)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(checked) + "\n")


def restore_config(metadata, *, explicit=None, runtime=None, strict=True):
    if strict is not True:
        raise LoopSchemaError('strict must be True')
    checked=validate_metadata(metadata);config=checked['resolved_config'];loop=checked['loop_spec']
    explicit={} if explicit is None else explicit;runtime={} if runtime is None else runtime
    json_value(explicit,'explicit');json_value(runtime,'runtime')
    if type(explicit) is not dict or type(runtime) is not dict:
        raise LoopSchemaError('explicit/runtime require objects')
    if set(explicit)&set(HISTORY_FLAGS):raise LoopSchemaError('V3 cannot mix any history flags')
    expected={**config['resolution'], **{k:loop[k] for k in ('task','profile','cost_profile','topology_preset','unique_depth','executed_depth','comparator_depth','head_dim','variant','grid_height','grid_width')},
              'architecture':ARCHITECTURE_SELECTOR,'family':FAMILY,'architecture_extension':ARCHITECTURE_EXTENSION,
              'schema_version':SCHEMA_VERSION,'config_version':CONFIG_VERSION,'checkpoint_version':CHECKPOINT_VERSION,'checkpoint_format':CHECKPOINT_FORMAT,
              'linearno_loop':True,'linearno_rank':loop['actual_M'],'model_spec':config['model_spec'],
              'seed':config['profile_spec']['values']['runtime']['seed']}
    if set(explicit)-expected.keys():raise LoopSchemaError('unknown explicit checkpoint fields: '+str(sorted(set(explicit)-expected.keys())))
    from .contracts import differences, finite
    if 'adapter_alpha' in explicit:
        explicit=dict(explicit);finite(explicit['adapter_alpha'],'adapter_alpha',0,True)
        explicit['adapter_alpha']=float(explicit['adapter_alpha'])
    errors=[]
    for key,value in explicit.items():errors.extend(differences(expected[key],value,'explicit.'+key))
    if errors:raise LoopSchemaError('\n'.join(errors))
    if set(runtime)-{'device','experiment_dir'}:raise LoopSchemaError('only device/experiment_dir runtime changes allowed')
    for key,value in runtime.items():text(value,'runtime.'+key)
    saved_runtime=config['profile_spec']['values']['runtime']
    changes={k:dict(saved=saved_runtime.get(k),requested=v) for k,v in runtime.items() if saved_runtime.get(k)!=v}
    return dict(config=deepcopy(config),runtime_changes=changes,load_policy=deepcopy(LOAD_POLICY))
