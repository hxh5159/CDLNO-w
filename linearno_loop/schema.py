"""Metadata-first contract and compatibility checks. No tensor/model loader.

LL1 class paths are planned explicit constructors, not implemented models.
At LL3–LL7 the actual loader must verify the real callable signature, then use
strict=True and validate state/optimizer/RNG against actual backend objects.
"""
from copy import deepcopy
import inspect
from pathlib import Path, PurePosixPath

from .contracts import (FAMILY, ARCHITECTURE_EXTENSION, SCHEMA_VERSION, CONFIG_VERSION,
    HISTORY_FLAGS, LoopSchemaError, exact, integer, json_value,
    require_equal, canonical_json, seal, read_json)
from .config import validate_config
from .state import hash_string, text, validate_resume, validate_normalizers

SECTIONS = ('loop_spec','model_spec','profile_spec','data_spec','objective_spec','evaluation_spec',
            'provenance_spec','normalizer_spec','resume_state','ensemble_manifest')
LOAD_POLICY = dict(metadata_first=True, strict=True, constructor_validation='full_explicit_signature',
                   legacy_fallback=False, baseline_as_resume=False, missing_random_weights=False)
PROVENANCE_FIELDS = ('target_sha','base_commit','dirty','transolver_sha','linearno_sha',
    'paper_version','paper_sha256','attnres_sha','kimi_k3_sha','residual_scaling_version',
    'source_sha256','normalized_patch_sha256','code_version','config_schema_version',
    'metadata_schema_version','command','environment')
REFERENCE_PINS = {
    'transolver_sha':'75e0f67643806a81cd1d3f6adc88dd8c02416fe7',
    'linearno_sha':'3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
    'attnres_sha':'85e22310fe5ee860b4a023de312d791de8a5a5e6',
    'kimi_k3_sha':'3cb39dfd32e51c3328e2e4b4af21341247d06c43',
    'paper_sha256':'637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd',
}


def validate_constructor(model_spec, constructor):
    """Inspect a supplied future real class, never import/construct it here."""
    if model_spec['class_path'] != constructor.__module__+'.'+constructor.__qualname__:
        raise LoopSchemaError('model_spec.class_path: real constructor mismatch')
    signature = inspect.signature(constructor)
    if any(p.kind in (p.VAR_KEYWORD,p.VAR_POSITIONAL) for p in signature.parameters.values()):
        raise LoopSchemaError('constructor must expose explicit fields, not *args/**kwargs')
    fields = {k for k,p in signature.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD,p.KEYWORD_ONLY)}
    require_equal(sorted(fields), sorted(model_spec['constructor_kwargs']), 'constructor_kwargs.fields')
    try:
        signature.bind(**model_spec['constructor_kwargs'])
    except TypeError as error:
        raise LoopSchemaError(str(error)) from error


def validate_metadata(metadata, *, constructor=None):
    json_value(metadata)
    if type(metadata) is not dict or metadata.get('family') != FAMILY:
        raise LoopSchemaError('family: loop loader only; legacy/pure/history use their own unchanged loaders')
    keys = {'family','architecture_extension','schema_version','config_version','config_hash',
            'resolved_config','load_policy','metadata_hash',*SECTIONS}
    exact(metadata, keys, 'metadata')
    c = validate_config(metadata['resolved_config'])
    for name, expected in (('architecture_extension',ARCHITECTURE_EXTENSION), ('schema_version',SCHEMA_VERSION),
        ('config_version',CONFIG_VERSION), ('config_hash',c['config_hash']), ('load_policy',LOAD_POLICY)):
        require_equal(expected, metadata[name], name)
    for name in ('loop_spec','model_spec','profile_spec'):
        require_equal(c[name], metadata[name], name)
    if constructor is not None:
        validate_constructor(metadata['model_spec'], constructor)
    for name in ('objective','evaluation'):
        require_equal(c['profile_spec']['values'][name], metadata[name+'_spec'], name+'_spec')
    data = metadata['data_spec']
    exact(data, {'protocol','split','checksums','sampling','scope','runtime'}, 'data_spec')
    require_equal(c['profile_spec']['values']['data'], data['protocol'], 'data_spec.protocol')
    if data['scope'] not in ('synthetic','real'):
        raise LoopSchemaError('data_spec.scope must label synthetic or real')
    text(data['split'], 'data_spec.split'); text(data['sampling'], 'data_spec.sampling')
    if type(data['checksums']) is not dict or not data['checksums'] or type(data['runtime']) is not dict:
        raise LoopSchemaError('data_spec: actual named checksums/runtime mapping required')
    for name, checksum in data['checksums'].items():
        text(name, 'data checksum name'); hash_string(checksum, 'data checksum')
    prov = metadata['provenance_spec']; exact(prov, PROVENANCE_FIELDS, 'provenance_spec')
    for key in ('target_sha','base_commit','transolver_sha','linearno_sha','attnres_sha','kimi_k3_sha'):
        hash_string(prov[key], 'provenance.'+key, 40)
    for key in ('paper_sha256','source_sha256','normalized_patch_sha256'):
        hash_string(prov[key], 'provenance.'+key)
    for key, expected in REFERENCE_PINS.items():
        require_equal(expected, prov[key], 'provenance.'+key)
    if type(prov['dirty']) is not bool:
        raise LoopSchemaError('provenance.dirty: bool required')
    for key, expected in (('paper_version','2511.06294v3'),('residual_scaling_version','2606.18524v1'),
                          ('config_schema_version',CONFIG_VERSION),('metadata_schema_version',SCHEMA_VERSION)):
        require_equal(expected, prov[key], 'provenance.'+key)
    text(prov['code_version'], 'provenance.code_version')
    if type(prov['command']) is not list or not prov['command'] or type(prov['environment']) is not dict or not prov['environment']:
        raise LoopSchemaError('provenance: command/environment required')
    for arg in prov['command']:
        if type(arg) is not str:
            raise LoopSchemaError('provenance.command: string argv required')
    validate_normalizers(metadata['normalizer_spec'], data['checksums'])
    validate_resume(metadata['resume_state'])
    members = metadata['ensemble_manifest']
    if type(members) is not list:
        raise LoopSchemaError('ensemble_manifest: ordered list required')
    ids = set()
    for index, member in enumerate(members):
        exact(member, {'member_id','order','path','sha256','format'}, 'ensemble member')
        text(member['member_id'], 'member_id'); integer(member['order'], 'member order')
        if member['member_id'] in ids or member['order'] != index:
            raise LoopSchemaError('ensemble id/order conflict')
        ids.add(member['member_id']); text(member['path'], 'ensemble path')
        path = PurePosixPath(member['path'])
        if path.is_absolute() or '..' in path.parts or not path.name or '\\' in member['path']:
            raise LoopSchemaError('ensemble path must be relative descendant')
        if member['format'] != 'state_dict':
            raise LoopSchemaError('loop ensemble requires state_dict, no whole-object loading')
        hash_string(member['sha256'], 'ensemble sha256')
    require_equal(seal(metadata,'metadata_hash')['metadata_hash'], metadata['metadata_hash'], 'metadata_hash')
    return deepcopy(metadata)


def make_metadata(config, **sections):
    c = validate_config(config)
    external = set(SECTIONS) - {'loop_spec','model_spec','profile_spec','objective_spec','evaluation_spec'}
    exact(sections, external, 'metadata sections')
    result = dict(family=FAMILY, architecture_extension=ARCHITECTURE_EXTENSION,
        schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION, config_hash=c['config_hash'],
        resolved_config=c, load_policy=deepcopy(LOAD_POLICY), **deepcopy(sections))
    for name in ('loop_spec','model_spec','profile_spec'):
        result[name] = deepcopy(c[name])
    for name in ('objective','evaluation'):
        result[name+'_spec'] = deepcopy(c['profile_spec']['values'][name])
    return validate_metadata(seal(result, 'metadata_hash'))


def read_metadata(path):
    return validate_metadata(read_json(path))


def write_metadata(path, metadata):
    checked = validate_metadata(metadata)
    # Creation only: no training sidecar overwrite, no mkdir, no tensor loading.
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(checked)+'\n')


def restore_config(metadata, *, explicit=None, runtime=None, strict=True):
    """Plan eval/resume from SAVED metadata. Supplied fields assert, never override.

    Runtime device/output path changes are separately reported. This function has
    no torch.load/model/factory callback and performs no filesystem mutation.
    """
    if strict is not True:
        raise LoopSchemaError('strict must be True')
    m = validate_metadata(metadata); c = m['resolved_config']; s = c['loop_spec']
    explicit = {} if explicit is None else explicit
    runtime = {} if runtime is None else runtime
    json_value(explicit, 'explicit'); json_value(runtime, 'runtime')
    if type(explicit) is not dict or type(runtime) is not dict:
        raise LoopSchemaError('explicit/runtime require objects')
    if set(explicit) & set(HISTORY_FLAGS):
        raise LoopSchemaError('loop metadata cannot mix any history flags')
    if 'linearno_rank' in explicit and 'rank_multiplier' in explicit:
        raise LoopSchemaError('explicit actual rank/multiplier conflict')
    expected = {**s, 'family':FAMILY, 'architecture_extension':ARCHITECTURE_EXTENSION,
                'linearno_loop':True, 'linearno_rank':s['resolved_rank'], 'model_spec':c['model_spec'],
                'seed':c['profile_spec']['values']['runtime']['seed']}
    if set(explicit) - expected.keys():
        raise LoopSchemaError('unknown explicit checkpoint fields: '+str(sorted(set(explicit)-expected.keys())))
    errors = []
    from .contracts import differences
    for k,v in explicit.items():
        errors.extend(differences(expected[k], v, 'explicit.'+k))
    if errors:
        raise LoopSchemaError('\n'.join(errors))
    if set(runtime)-{'device','experiment_dir'}:
        raise LoopSchemaError('only device/experiment_dir are runtime changes; protocol changes require a new run')
    for k,v in runtime.items():
        text(v, 'runtime.'+k)
    saved_runtime = c['profile_spec']['values']['runtime']
    changes = {k:dict(saved=saved_runtime.get(k),requested=v) for k,v in runtime.items()
               if k not in saved_runtime or saved_runtime[k] != v}
    return dict(config=deepcopy(c), runtime_changes=changes, load_policy=deepcopy(LOAD_POLICY))
