"""Internal R5 config, separate from all benchmark parsers and registries.

Protocol profiles retain their existing meaning. The two booleans alone select
architecture; all other structural fields are derived and revalidated on read.
This module performs no model construction and does not import torch.
"""
from copy import deepcopy

from cdlno.linearno.profiles import validate_resolved, digest
from linearno_history.schema import (
    FEATURE_FIELDS, RESEARCH_FAMILY, ARCHITECTURE_EXTENSION, TEMPERATURE_SEMANTICS,
    HistorySchemaError, validate_family_feature_config, validate_base_descriptor,
    validate_innovation_spec, derive_fair_seeds, run_directory_id,
)

IMPLEMENTATION_VERSION = 'r5-internal-v1'
DEFAULT_FEATURES = dict(zip(FEATURE_FIELDS, (False, False, None)))
PATHS = {
    'standard': {
        'A0K0': 'model.LinearNO.Model',
        'A1K0': 'model.LinearNO_History.AttnResModel',
        'A0K1': 'model.LinearNO_History.HistoryKModel',
        'A1K1': 'model.LinearNO_History.JointHistoryModel'},
    'airfrans': {
        'A0K0': 'cdlno.linearno.airfrans.AirfRANSLinearNO',
        'A1K0': 'cdlno.linearno_history.models.AirfRANSAttnResModel',
        'A0K1': 'cdlno.linearno_history.models.AirfRANSHistoryKModel',
        'A1K1': 'cdlno.linearno_history.models.AirfRANSJointHistoryModel'},
    'car': {
        'A0K0': 'cdlno.linearno.shapenet.ShapeNetLinearNO',
        'A1K0': 'cdlno.linearno_history.models.ShapeNetAttnResModel',
        'A0K1': 'cdlno.linearno_history.models.ShapeNetHistoryKModel',
        'A1K1': 'cdlno.linearno_history.models.ShapeNetJointHistoryModel'},
}


def task_kind(task):
    return task if task in ('airfrans', 'car') else 'standard'


def baseline_model_spec(profile):
    m = profile['values']['model']
    fields = dict(space_dim='space_dim', n_layers='layers', n_hidden='hidden',
                  dropout='dropout', n_head='heads', act='activation', mlp_ratio='ffn_ratio',
                  fun_dim='fun_dim', out_dim='out_dim', linearno_rank='linearno_rank',
                  ref='ref', unified_pos='unified_pos')
    if profile['task'] != 'airfrans':
        fields.update(Time_Input='time_input', H='H', W='W')
    kwargs = {k: m[v] for k, v in fields.items()}
    if profile['task'] == 'airfrans':
        kwargs['linear'] = True
    elif profile['task'] == 'car':
        kwargs['isregular'] = False
    else:
        kwargs['linearno_variant'] = m['linearno_variant']
    # Inactive grids in historical profiles are None. Existing task adapters use
    # the release constructor's 85x85 default for these unused fields.
    for field in ('H', 'W'):
        if field in kwargs and kwargs[field] is None:
            kwargs[field] = 85
    return dict(class_path=PATHS[task_kind(profile['task'])]['A0K0'], constructor_kwargs=kwargs)


def resolve_config(profile, *, family='linearno', features=None, feature_seed=None):
    supplied = {} if features is None else dict(features)
    flags = validate_family_feature_config(family, {**DEFAULT_FEATURES, **supplied})
    if family not in ('linearno', RESEARCH_FAMILY):
        raise HistorySchemaError('internal LinearNO factory does not construct other model families')
    profile = validate_resolved(profile)
    task, m = profile['task'], profile['values']['model']
    sig = flags['feature_signature']
    seed = profile['values']['runtime']['seed']
    fair_seeds = derive_fair_seeds(seed, task=task)
    if feature_seed is not None and (type(feature_seed) is not int or not 0 <= feature_seed < 2**63):
        raise HistorySchemaError('feature_seed must be an integer in [0, 2**63)')
    if sig == 'A0K0' and feature_seed is not None:
        raise HistorySchemaError('A0K0 has no feature seed or innovation parameters')
    base = validate_base_descriptor(dict(variant=m['linearno_variant'],
        task_variant=f'standard_{task}' if task_kind(task) == 'standard' else task,
        n_layers=m['layers'], heads=m['heads'], d_h=m['hidden']//m['heads'],
        actual_M=m['linearno_rank'], temperature_semantics=TEMPERATURE_SEMANTICS[m['linearno_variant']]))
    spec = baseline_model_spec(profile)
    if sig != 'A0K0':
        spec['class_path'] = PATHS[task_kind(task)][sig]
        spec['constructor_kwargs']['feature_seed'] = (fair_seeds['innovation_feature_seed']
                                                     if feature_seed is None else feature_seed)
        # Keep the historical constructor/metadata shape for normal p=.1
        # research runs. Only the explicit A1K0 p=0 ablation needs a real
        # constructor field; old A1K0 checkpoints still use the class default.
        if (flags[FEATURE_FIELDS[0]] and not flags[FEATURE_FIELDS[1]]
                and flags[FEATURE_FIELDS[2]] == 0.0):
            spec['constructor_kwargs']['attnres_history_dropout_p'] = 0.0
    # Canonical effective dropout: explicit .1 and omitted/null resolve identically.
    flags.pop('dropout_source')
    result = dict(config_schema_version=1, family='linearno' if sig == 'A0K0' else RESEARCH_FAMILY,
                  profile_spec=profile, features=flags, base_linearno=base, model_spec=spec,
                  initialization=dict(public_backbone_seed=seed,
                      feature_seed=spec['constructor_kwargs'].get('feature_seed'),
                      feature_rng='isolated_cpu_fork_rng_per_operator',
                      dataloader_generator_seed=fair_seeds['dataloader_generator_seed']),
                  run_signature=run_directory_id(task, profile['profile'], m['layers'],
                      flags[FEATURE_FIELDS[0]], flags[FEATURE_FIELDS[1]], seed,
                      dropout_p=flags[FEATURE_FIELDS[2]]))
    result['resolved_hash'] = digest(result)
    return result


def validate_config(config):
    if not isinstance(config, dict):
        raise HistorySchemaError('resolved config must be an object')
    if config.get('resolved_hash') != digest({k: v for k, v in config.items() if k != 'resolved_hash'}):
        raise HistorySchemaError('resolved config checksum mismatch')
    try:
        expected = resolve_config(config['profile_spec'], family=config['family'],
            features={k: config['features'][k] for k in FEATURE_FIELDS},
            feature_seed=config['initialization']['feature_seed'])
    except (KeyError, TypeError) as error:
        raise HistorySchemaError('incomplete resolved config') from error
    if config != expected:
        raise HistorySchemaError('resolved config differs from its flags/profile/constructor/hash')
    return deepcopy(config)


def innovation_spec(config):
    c = validate_config(config)
    flags = c['features']
    if flags['feature_signature'] == 'A0K0':
        raise HistorySchemaError('A0K0 must not have innovation_spec')
    a, k = flags[FEATURE_FIELDS[0]], flags[FEATURE_FIELDS[1]]
    return validate_innovation_spec(dict(schema_version=1,
        class_path=c['model_spec']['class_path'], base_linearno=c['base_linearno'],
        features={FEATURE_FIELDS[0]: a, FEATURE_FIELDS[1]: k,
                  'feature_signature': flags['feature_signature']},
        attnres=dict(enabled=a, version=1, per_head_cross=True, d_m_equals_d_h=True,
            history_only_source_softmax=True, current_in_source_softmax=False,
            null_source=dict(present=True, value='zero', always_in_softmax=True,
                             parameterized=False, kept_when_all_history_masked=True),
            gamma=dict(shape='per_receiver_scalar', initialization=0., scope='receiver_layer'),
            dropout=dict(p=flags[FEATURE_FIELDS[2]], train_only=True,
                mask_granularity='sample_by_real_source', singleton_history_kept=True,
                null_never_dropped=True, inverted_scaling=False,
                all_masked_fallback='null_weight_one_finite_zero')),
        history_k=dict(enabled=k, version=1, base_to_k_row_query=True,
            all_history_token_bank=True, point_slot_dot=True, point_centering=True,
            gate=dict(scope='receiver_layer_and_head', function='tanh', initialization=0., shape='[1,H,1,1]'),
            changes_q=False, changes_v=False, changes_reconstruction_q=False),
        raw_cache=dict(authoritative_value='C_raw_pre_attnres',
            timing='after_current_KtV_before_A_and_Q_reconstruction', storage='forward_local_tuple',
            detach=False, cross_forward=False, cross_sample=False, cross_time=False),
        constructor_hyperparameters=c['model_spec']['constructor_kwargs'],
        code_schema=dict(config_schema_version=1, implementation_version=IMPLEMENTATION_VERSION,
                         architecture_extension=ARCHITECTURE_EXTENSION)),
        feature_config={key: flags[key] for key in FEATURE_FIELDS})


def structural_differences(saved, expected):
    """Field-level diagnostics; no weight access or constructor call."""
    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                yield from visit(a.get(key), b.get(key), f'{path}.{key}')
        elif a != b or type(a) is not type(b):
            yield f'{path}: saved={a!r}, requested={b!r}'
    keys = ('config_schema_version', 'family', 'features', 'base_linearno', 'model_spec')
    differences = [s for key in keys for s in visit(saved[key], expected[key], key)]
    for key in ('task', 'profile'):
        differences.extend(visit(saved['profile_spec'][key], expected['profile_spec'][key], 'profile_spec.'+key))
    return differences
