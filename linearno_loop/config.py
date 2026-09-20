"""Pure loop resolution on immutable existing profiles; no production parser."""
from copy import deepcopy
import hashlib

from cdlno.linearno.profiles import (
    DEFAULT_PROFILE, TASKS, PROFILES, resolve_config as resolve_base,
    derived_model, validate_model,
)
from .contracts import (
    FAMILY, ARCHITECTURE_EXTENSION, SCHEMA_VERSION, CONFIG_VERSION, FORMULA_VERSION,
    PRESETS, RESIDUAL_MODES, TOPOLOGY_FIELDS, OPTIONS, SHARING, CLASS_PATHS,
    HISTORY_FLAGS, LoopSchemaError, integer, exact, json_value, require_equal, digest, seal,
)


def route_intent(family, explicit):
    """Contract guard on TYPED explicit fields, NOT an argv parser or dispatcher.

    None means leave the old parser/metadata untouched. Existing actual-M alone
    is not a loop flag. Presence of old A/K flags (even false) is a conflict.
    """
    if type(explicit) is not dict:
        raise LoopSchemaError('explicit: expected mapping of supplied fields')
    active = set(explicit) & (OPTIONS - {'linearno_rank'} | {'linearno_loop'})
    if not active:
        return None
    if set(explicit) & set(HISTORY_FLAGS):
        raise LoopSchemaError('loop flags cannot mix with any linearno_history flags')
    if type(explicit.get('linearno_loop')) is not bool:
        raise LoopSchemaError('linearno_loop: explicit typed bool required with loop fields')
    if not explicit['linearno_loop']:
        if active != {'linearno_loop'}:
            raise LoopSchemaError('disabled loop cannot receive topology/residual/rank policy')
        return None
    if family not in ('linearno', FAMILY):
        raise LoopSchemaError('loop requires a LinearNO selection, never another model family')
    return FAMILY


def _profile(task, profile, overrides):
    json_value(overrides, 'profile_overrides')
    if type(overrides) is not dict:
        raise LoopSchemaError('profile_overrides: expected object')
    prohibited = {'model.layers', 'model.linearno_rank', 'model.qk_dim'} & overrides.keys()
    if prohibited:
        raise LoopSchemaError('loop topology/rank have sole truth; prohibited profile overrides: '+str(sorted(prohibited)))
    if any(k.startswith(('data.', 'objective.', 'evaluation.')) for k in overrides):
        raise LoopSchemaError('data/objective/evaluation protocols remain those of the selected profile')
    options = {} if task in ('airfrans', 'car') else {
        'contract': 'standard_temporal_l5' if task in ('ns', 'plasticity') else 'standard_static_l4'}
    try:
        result = resolve_base(task, profile, explicit=overrides, **options)
    except (ValueError, KeyError, TypeError) as error:
        raise LoopSchemaError(f'profile_spec: {error}') from error
    if result['unresolved']:
        raise LoopSchemaError('profile has unresolved fields: '+str(result['unresolved']))
    m = result['values']['model']
    if m['time_input'] and m['hidden'] % 2:
        raise LoopSchemaError('Time_Input requires even hidden for the native time embedding')
    if task == 'airfrans' and (m['space_dim'], m['fun_dim'], m['out_dim']) != (7, 0, 4):
        raise LoopSchemaError('AirfRANS native interface requires space_dim=7, fun_dim=0, out_dim=4')
    if task == 'car':
        if m['space_dim'] + m['fun_dim'] != 7 or m['out_dim'] != 4:
            raise LoopSchemaError('ShapeNet native interface requires space_dim+fun_dim=7, out_dim=4')
        if m['unified_pos'] and m['fun_dim']:
            raise LoopSchemaError('ShapeNet unified position replaces x and requires fun_dim=0')
    return result


def _topology(options):
    preset = options.get('topology_preset')
    if type(preset) is not str or preset not in (*PRESETS, 'custom'):
        raise LoopSchemaError('topology_preset: explicitly select a named preset or custom')
    supplied = set(options) & set(TOPOLOGY_FIELDS)
    if preset == 'custom':
        if supplied != set(TOPOLOGY_FIELDS):
            raise LoopSchemaError('custom topology requires all four P/C/R/S fields')
        values = [options[k] for k in TOPOLOGY_FIELDS]
    else:
        if supplied:
            raise LoopSchemaError('preset and explicit custom P/C/R/S fields are mutually exclusive')
        values = PRESETS[preset]
    for i, (k, v) in enumerate(zip(TOPOLOGY_FIELDS, values)):
        integer(v, k, 0 if i == 0 else 1)
    return dict(zip(TOPOLOGY_FIELDS, values))


def _attnres(mode, C, R, hidden):
    rb, lb = mode == 'rb_attnres', mode == 'lb_attnres_1_over_r'
    receivers = 2*C*R+1 if rb else R if lb else 0
    return dict(enabled=rb or lb, domain='aligned_points_B_N_hidden', eps=1e-6,
        query_initialization=0.0, norm_scale_initialization=1.0,
        norm='last_axis_RMS_keepdim_scale_only', source_softmax=True,
        value='raw', value_projection=False, bias=False, sqrt_hidden_scaling=False,
        source_count_scaling=False, output_1_over_r=False, depth_multihead=False,
        receiver_scope='logical_execution_position', receiver_count=receivers,
        router_parameter_count=2*hidden*receivers,
        source_visits=(2*C)*R*(R+3)//2+1 if rb else R*(R+3)//2 if lb else 0,
        singleton_receiver='identity_zero_effective_router_gradient' if rb else 'not_registered',
        zero_query_norm_gradient='zero_expected', formula_version=FORMULA_VERSION)


def _model_spec(profile, loop):
    m = profile['values']['model']
    fields = dict(space_dim='space_dim', n_hidden='hidden', n_head='heads',
        dropout='dropout', act='activation', mlp_ratio='ffn_ratio', fun_dim='fun_dim',
        out_dim='out_dim', ref='ref', unified_pos='unified_pos')
    kw = {k: m[v] for k, v in fields.items()}
    kw.update({k: loop[k] for k in TOPOLOGY_FIELDS})
    kw.update(residual_mode=loop['residual_mode'], linearno_rank=loop['resolved_rank'])
    task = profile['task']
    if task == 'airfrans':
        kw['linear'] = True
    else:
        kw.update(Time_Input=m['time_input'], H=m['H'] if m['H'] is not None else 85,
                  W=m['W'] if m['W'] is not None else 85)
        if task == 'car':
            kw['isregular'] = False
        else:
            kw['linearno_variant'] = m['linearno_variant']
    return dict(class_path=CLASS_PATHS.get(task, CLASS_PATHS['standard']), constructor_kwargs=kw)


def fair_seeds(config):
    """Pure hashes, no RNG calls. Mode omitted so paired runs share initialization."""
    p, s = config['profile_spec'], config['loop_spec']
    seed = p['values']['runtime']['seed']
    def derive(label, payload):
        raw = (label + ':' + digest(payload)).encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63)
    backbone = dict(task=p['task'], profile=p['profile'], seed=seed,
                    topology={k: s[k] for k in TOPOLOGY_FIELDS}, model=p['values']['model'],
                    actual_M=s['resolved_rank'])
    data = dict(task=p['task'], profile=p['profile'], seed=seed)
    return dict(protocol='paired-backbone-loader-v1', public_backbone_seed=seed,
        backbone_pair_id=digest(backbone), router_seed=derive('point-router', backbone),
        dataloader_generators={split: derive('loader-'+split, data) for split in ('train', 'test')},
        rule='initialize_unique_backbone_once_then_copy_across_modes;router_RNG_isolated',
        topology_pairs_are_parameter_matched=False)


def resolve_config(task, profile=DEFAULT_PROFILE, *, options=None, profile_overrides=None):
    """Typed train contract. Topology/mode required; M multiplier defaults to two.

    This returns JSON only. Future class_path strings are not imported. Original
    profile_spec is retained byte-for-byte; actual topology/rank live in loop_spec.
    """
    if type(task) is not str or task not in TASKS or type(profile) is not str or profile not in PROFILES:
        raise LoopSchemaError('unknown task/profile')
    options = deepcopy({} if options is None else options)
    json_value(options, 'options')
    if type(options) is not dict or set(options) - OPTIONS:
        raise LoopSchemaError('options: unknown fields or not an object')
    topology = _topology(options)
    mode = options.get('residual_mode')
    if type(mode) is not str or mode not in RESIDUAL_MODES:
        raise LoopSchemaError('residual_mode: explicitly select one of the three modes')
    overrides = deepcopy({} if profile_overrides is None else profile_overrides)
    base = _profile(task, profile, overrides)
    m = base['values']['model']
    M = m['linearno_rank']
    if 'linearno_rank' in options and 'rank_multiplier' in options:
        raise LoopSchemaError('explicit actual linearno_rank and rank_multiplier are mutually exclusive')
    if 'linearno_rank' in options:
        rank = integer(options['linearno_rank'], 'linearno_rank', 1)
        multiplier, policy = None, 'explicit_actual'
    else:
        multiplier = integer(options.get('rank_multiplier', 2), 'rank_multiplier', 1)
        if multiplier not in (1, 2):
            raise LoopSchemaError('rank_multiplier: only 1 or 2')
        rank, policy = M*multiplier, 'profile_multiplier'
    actual_model = {**m, 'linearno_rank': rank}
    try:
        validate_model(task, actual_model)
    except ValueError as error:
        raise LoopSchemaError(str(error)) from error
    P, C, R, S = (topology[k] for k in TOPOLOGY_FIELDS)
    loop = dict(schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION,
        formula_version=FORMULA_VERSION, task=task, profile=profile,
        topology_preset=options['topology_preset'], **topology,
        unique_depth=P+C+S, executed_depth=P+C*R+S, residual_mode=mode,
        base_rank=M, rank_multiplier=multiplier, resolved_rank=rank, rank_policy=policy,
        hidden=m['hidden'], heads=m['heads'], head_dim=m['hidden']//m['heads'],
        variant=m['linearno_variant'], point_domain_attnres=True, feature_timestep_encoding=False,
        operator_contract=derived_model(task, actual_model), sharing=deepcopy(SHARING),
        residual_contract=dict(prefix_suffix='native_unscaled', identity='unscaled',
            core_branch_scale='none_no_additive_residual' if mode=='rb_attnres' else '1/loop_repeats',
            sources={'sr_1_over_r':'none', 'rb_attnres':'anchor_completed_raw_round_sums_and_current_raw_partial',
                     'lb_attnres_1_over_r':'anchor_and_actual_Y_minus_H'}[mode],
            receiver_before_original_layernorm=True, head_calls=1),
        attnres=_attnres(mode, C, R, m['hidden']))
    sources = {k: 'derived' for k in loop}
    sources.update({k: 'cli_explicit' if k in options else 'preset' for k in TOPOLOGY_FIELDS})
    sources.update(topology_preset='cli_explicit', residual_mode='cli_explicit',
        base_rank='profile', rank_multiplier='not_applicable' if policy=='explicit_actual' else
        ('cli_explicit' if 'rank_multiplier' in options else 'family_default'),
        resolved_rank='cli_explicit' if policy=='explicit_actual' else 'derived',
        hidden=base['field_sources']['model.hidden'], heads=base['field_sources']['model.heads'],
        variant=base['field_sources']['model.linearno_variant'],
        point_domain_attnres='frozen_contract', feature_timestep_encoding='frozen_contract')
    result = dict(family=FAMILY, architecture_extension=ARCHITECTURE_EXTENSION,
        config_version=CONFIG_VERSION, request=dict(task=task, profile=profile, options=options,
        profile_overrides=overrides), loop_spec=loop, model_spec=_model_spec(base, loop),
        profile_spec=base, field_sources=dict(loop_spec=sources, profile_spec=deepcopy(base['field_sources'])))
    result['fair_comparison'] = fair_seeds(result)
    return seal(result, 'config_hash')


def validate_config(config):
    json_value(config)
    exact(config, {'family', 'architecture_extension', 'config_version', 'request', 'loop_spec',
                  'model_spec', 'profile_spec', 'field_sources', 'fair_comparison', 'config_hash'}, 'config')
    exact(config['request'], {'task', 'profile', 'options', 'profile_overrides'}, 'request')
    r = config['request']
    expected = resolve_config(r['task'], r['profile'], options=r['options'], profile_overrides=r['profile_overrides'])
    require_equal(expected, config, 'config')
    return deepcopy(config)


def run_directory_id(config):
    c = validate_config(config); s = c['loop_spec']; seed = c['profile_spec']['values']['runtime']['seed']
    P, C, R, S = (s[k] for k in TOPOLOGY_FIELDS)
    return (f"{s['task']}__{FAMILY}__{s['profile']}__P{P}-C{C}-R{R}-S{S}__"
            f"{s['residual_mode']}__M{s['resolved_rank']}__seed{seed}__cfg{c['config_hash'][:12]}")
