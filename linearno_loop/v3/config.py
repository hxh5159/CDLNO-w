"""Pure V3 config. Saved resolution is validated without re-resolving profiles."""
from copy import deepcopy
import hashlib

from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES
from linearno_loop.config import _profile, _attnres
from linearno_loop.contracts import PRESETS
from .profiles import width_for
from .contracts import (ARCHITECTURE_EXTENSION, ARCHITECTURE_SELECTOR, CLASS_PATHS,
    CONFIG_VERSION, COST_PROFILES, FAMILY, FORMULA_VERSION, RESIDUAL_MODES,
    ADAPTER_MODES, SCHEMA_VERSION, CHECKPOINT_VERSION, CHECKPOINT_FORMAT, TASKS, TOPOLOGY_FIELDS,
    V3SchemaError, exact, finite, integer, json_value, require_equal, seal, digest)

OPTIONS = {'architecture', 'cost_profile', 'topology_preset', 'executed_depth',
           *TOPOLOGY_FIELDS, 'residual_mode', 'hidden_width', 'latent_width',
           'actual_M', 'heads', 'latent_enabled', 'adapter_mode', 'adapter_rank', 'adapter_alpha'}
CONFIG_KEYS = {'family','architecture_extension','schema_version','config_version','checkpoint_version','checkpoint_format',
               'request','base_profile_spec','resolution','loop_spec','profile_spec','model_spec',
               'field_sources','fair_comparison','config_hash'}


def _seed(label, payload):
    return int.from_bytes(hashlib.sha256((label+':'+digest(payload)).encode()).digest()[:8], 'big') % 2**63


def _request(task, profile, options, overrides):
    if type(task) is not str or task not in TASKS or type(profile) is not str or profile not in PROFILES:
        raise V3SchemaError('unknown task/profile')
    json_value(options, 'options');json_value(overrides,'profile_overrides')
    if type(options) is not dict or set(options)-OPTIONS:
        raise V3SchemaError('options: unknown fields or not an object')
    if options.get('architecture') != ARCHITECTURE_SELECTOR:
        raise V3SchemaError('explicit architecture=operator_latent_adapter_v3 is required; cost_profile cannot select V3')
    if type(overrides) is not dict or any(not k.startswith(('training.','runtime.')) for k in overrides):
        raise V3SchemaError('profile_overrides allow training/runtime only; V3 architecture fields have a single source')
    cost = options.get('cost_profile','matched_v1')
    if type(cost) is not str or cost not in COST_PROFILES:
        raise V3SchemaError('unknown cost_profile')
    for field in ('hidden_width','latent_width','actual_M','heads','adapter_rank','executed_depth'):
        if field in options:integer(options[field],field,1)
    for i,field in enumerate(TOPOLOGY_FIELDS):
        if field in options:integer(options[field],field,0 if i==0 else 1)
    if 'adapter_alpha' in options:finite(options['adapter_alpha'],'adapter_alpha',0,True)
    if 'latent_enabled' in options and type(options['latent_enabled']) is not bool:
        raise V3SchemaError('latent_enabled must be bool')
    for field,allowed in [('residual_mode',RESIDUAL_MODES),('adapter_mode',ADAPTER_MODES)]:
        if field in options and (type(options[field]) is not str or options[field] not in allowed):
            raise V3SchemaError('unknown '+field)
    return cost


def _topology(options, cost):
    supplied=set(options)&set(TOPOLOGY_FIELDS)
    preset=options.get('topology_preset')
    depth=options.get('executed_depth')
    if 'topology_preset' in options and (type(preset) is not str or preset not in (*PRESETS,'custom','d12','d20','d28','d60')):
        raise V3SchemaError('unknown V3 topology_preset')
    if preset in PRESETS:
        if supplied or depth is not None:raise V3SchemaError('preset and custom/depth fields are mutually exclusive')
        values=PRESETS[preset]
    elif supplied:
        if preset!='custom' or supplied!=set(TOPOLOGY_FIELDS) or depth is not None:
            raise V3SchemaError('custom requires all four P/C/R/S and no executed-depth shorthand')
        values=tuple(options[k] for k in TOPOLOGY_FIELDS)
    else:
        if preset=='custom' and depth is None:raise V3SchemaError('custom topology needs four fields or executed_depth')
        if cost=='custom' and preset is None and depth is None:
            raise V3SchemaError('custom cost requires explicit topology or executed_depth')
        if preset and preset.startswith('d'):
            if depth is not None:raise V3SchemaError('choose depth shorthand or topology name, not both')
            depth=int(preset[1:])
        depth=12 if depth is None else depth
        if depth<6 or depth%2:raise V3SchemaError('executed_depth must be even >=6 for P=S=2,R=2')
        values=(2,(depth-4)//2,2,2)
    P,C,R,S=values
    if cost!='custom' and (P!=2 or S!=2 or R!=2 or P+C*R+S not in (12,20,28,60)):
        raise V3SchemaError('tabulated profiles require canonical D12/20/28/60 with P=S=2,R=2')
    return dict(zip(TOPOLOGY_FIELDS,values))


def _resolve_facts(task, options, base):
    cost=options.get('cost_profile','matched_v1');top=_topology(options,cost)
    m=base['values']['model'];D=top['prefix_blocks']+top['recurrent_core_blocks']*top['loop_repeats']+top['suffix_blocks']
    if cost=='custom':
        if not {'hidden_width','latent_width'}<=options.keys():raise V3SchemaError('custom requires explicit hidden_width and latent_width')
        H,Dz=options['hidden_width'],options['latent_width']
        heads,M=options.get('heads',m['heads']),options.get('actual_M',m['linearno_rank'])
    else:
        widths=width_for(cost,'d'+str(D),task);H,Dz=widths['hidden_width'],widths['latent_width']
        heads,M=8,32 if task in ('ns','airfrans','car') else 64
        for field,value in [('hidden_width',H),('latent_width',Dz),('heads',heads),('actual_M',M)]:
            if field in options:require_equal(value,options[field],'profile conflict: '+field+' (use custom to override)')
    mode=options.get('adapter_mode',ADAPTER_MODES[1])
    if H%heads:raise V3SchemaError('hidden_width must be divisible by heads')
    if m['time_input'] and H%2:raise V3SchemaError('time embedding requires even hidden_width')
    if mode!='none' and top['loop_repeats']!=2:raise V3SchemaError('adapter-on requires loop_repeats=2')
    return dict(**top,hidden_width=H,latent_width=Dz,heads=heads,actual_M=M,
                residual_mode=options.get('residual_mode',RESIDUAL_MODES[0]),
                latent_enabled=options.get('latent_enabled',True),adapter_mode=mode,
                adapter_rank=options.get('adapter_rank',4),adapter_alpha=float(options.get('adapter_alpha',4)))


def _assemble(request,base,facts):
    """Use already-resolved saved facts. Never look up a current profile table."""
    task,profile,options=request['task'],request['profile'],request['options']
    cost=options.get('cost_profile','matched_v1');b=base['values']['model']
    P,C,R,S=(facts[k] for k in TOPOLOGY_FIELDS);H=facts['hidden_width'];Dz=facts['latent_width']
    M,h=facts['actual_M'],facts['heads'];mode=facts['residual_mode']
    latent=facts['latent_enabled'];adapter=facts['adapter_mode']!='none'
    U,D=P+C+S,P+C*R+S;seed=base['values']['runtime']['seed']
    integer(seed,'seed',0)
    if seed>=2**64:raise V3SchemaError('seed must fit torch uint64')
    effective={k:deepcopy(v) for k,v in b.items() if k not in ('hidden','H','W','layers','linearno_rank')}
    effective.update(hidden_width=H,grid_height=b['H'],grid_width=b['W'],actual_M=M,heads=h,
                     unique_depth=U,executed_depth=D,latent_width=Dz)
    sources={k:'derived' for k in effective}
    for k in effective:
        if k in b:sources[k]=base['field_sources']['model.'+k]
    for k,old in [('grid_height','H'),('grid_width','W')]:sources[k]=base['field_sources']['model.'+old]
    for k in ('hidden_width','latent_width','heads','actual_M'):
        sources[k]='custom_explicit' if cost=='custom' and k in options else 'cost_profile_table' if cost!='custom' else 'task_profile'
    fields={k:('cli_explicit' if k in options else 'v3_default') for k in facts}
    fields.update({k:sources[k] for k in ('hidden_width','latent_width','heads','actual_M')})
    fields.update({k:'cli_explicit' if k in options else 'topology_shorthand' for k in TOPOLOGY_FIELDS})
    # Feature flags, residual, Dz and cost-profile label do not alter public initialization.
    common=dict(task=task,profile=profile,seed=seed,topology={k:facts[k] for k in TOPOLOGY_FIELDS},
                model={k:v for k,v in effective.items() if k!='latent_width'})
    latent_seed=_seed('v3-latent',dict(common=common,width=Dz))
    adapter_seed=_seed('v3-adapter',dict(common=common,rank=facts['adapter_rank']))
    loaders={s:_seed('loader-'+s,dict(task=task,profile=profile,seed=seed)) for s in ('train','test')}
    init=dict(protocol='v3-isolated-feature-init-v1',public_seed=seed,latent_seed=latent_seed,
              adapter_seed=adapter_seed,feature_rng_isolated=True,public_rng_advanced_by_features=False,
              zero_init_after_tree=True,dataloader_generators=loaders)
    ar=_attnres(mode,C,R,H)
    latent_spec=dict(enabled=latent,width=Dz,instance_count=C if latent else 0,
                     parameter_count=C*(2*H*Dz+Dz+3*H) if latent else 0,
                     position='after_KtV_before_Q_readout',head_merge='B_h_M_dh_to_B_M_H',
                     norm='LayerNorm_affine',norm_eps=1e-5,activation='GELU',dropout=0.0,
                     linear_bias=True,token_axis_mixing=False,second_weight_and_bias_zero=True,
                     sharing='one_per_core_position_across_rounds',registration='enabled_only')
    rank,alpha=facts['adapter_rank'],facts['adapter_alpha']
    adapt_spec=dict(mode=facts['adapter_mode'],enabled=adapter,rank=rank,alpha=alpha,scale=alpha/rank,
                    apply_visit=2 if adapter else None,round_index=1 if adapter else None,
                    instance_count=C if adapter else 0,parameter_count=C*2*rank*(H//h+M) if adapter else 0,
                    A_shape=[rank,H//h],B_shape=[M,rank],qk_independent=True,shared_across_heads=True,
                    value_unchanged=True,position='base_logits_plus_delta_before_temperature_softmax',
                    A_initialization='kaiming_uniform_isolated_seed',B_initialization='zero',
                    bias=False,gate=False,registration='enabled_only')
    loop=dict(schema_version=3,formula_version=FORMULA_VERSION,task=task,profile=profile,cost_profile=cost,
              topology_preset=options.get('topology_preset','executed_depth'),**deepcopy(facts),
              unique_depth=U,executed_depth=D,comparator_depth=U if cost!='custom' else None,
              base_M=b['linearno_rank'],head_dim=H//h,variant=b['linearno_variant'],ffn_ratio=b['ffn_ratio'],
              grid_height=b['H'],grid_width=b['W'],point_domain_attnres=True,feature_timestep_encoding=False,
              rank_policy='absolute_M_independent_of_head_dim',latent_ffn=latent_spec,adapter=adapt_spec,
              sharing=dict(core=['ln_1','operator','ln_2','point_ffn'],core_scope='physical_position_across_rounds',
                           prefix_core_suffix='disjoint_physical_modules',core_registration='once',
                           recompute_qkv_context_each_visit=True,head='last_suffix_only_once',
                           routers='independent_per_logical_receiver',history='forward_local_points',
                           detach=False,cross_forward=False,cross_physical_time=False,cross_member=False),
              residual_contract=dict(prefix_suffix='native_unscaled',identity='unscaled',
                 core_branch_scale='none_no_additive_residual' if mode=='rb_attnres' else '1/loop_repeats',
                 sources={'sr_1_over_r':'none','rb_attnres':'anchor_completed_raw_round_sums_and_current_raw_partial',
                          'lb_attnres_1_over_r':'anchor_and_actual_Y_minus_H'}[mode],
                 rb_amp_boundary='local_source_views_cast_to_anchor_dtype_no_detach',head_calls=1),
              attnres=ar,initialization=init,state_partition=dict(stem='preprocess,placeholder,time_fc',
                 prefix='loop.prefix',shared_core='loop.core',suffix='loop.suffix',head='last_suffix.ln_3,mlp2',
                 latent='per_core.latent_ffn' if latent else None,adapter='per_core.adapter' if adapter else None,
                 routers=mode if ar['enabled'] else None))
    fields={**{key:'derived' for key in loop},**fields,
            'cost_profile':'cli_explicit' if 'cost_profile' in options else 'v3_default',
            'topology_preset':'cli_explicit' if 'topology_preset' in options else 'executed_depth_shorthand',
            'grid_height':sources['grid_height'],'grid_width':sources['grid_width'],
            'base_M':'task_profile','task':'request','profile':'request',
            'sharing':'frozen_v3_contract','residual_contract':'frozen_residual_v1_contract'}
    kwargs={k:deepcopy(v) for k,v in effective.items() if k not in ('unique_depth','executed_depth')}
    kwargs.update({k:deepcopy(v) for k,v in facts.items()})
    kwargs.update(latent_seed=latent_seed,adapter_seed=adapter_seed)
    values=deepcopy(base['values']);values['model']=effective
    resolved_profile=dict(family=FAMILY,config_version=3,task=task,profile=profile,cost_profile=cost,
                          values=values,field_sources={**{k:v for k,v in base['field_sources'].items() if not k.startswith('model.')},
                          **{'model.'+k:v for k,v in sources.items()}},base_profile_hash=base['config_hash'])
    fair=dict(protocol='paired-v3-backbone-loader-v1',public_backbone_seed=seed,backbone_pair_id=digest(common),
              latent_seed=latent_seed,adapter_seed=adapter_seed,dataloader_generators=loaders,
              rule='full_core_shared;feature_installation_after_public_tree_in_isolated_rng',
              mathematical_identity_at_zero_features='same_width_same_topology_baseline_only')
    return seal(dict(family=FAMILY,architecture_extension=ARCHITECTURE_EXTENSION,schema_version=3,
        config_version=3,checkpoint_version=CHECKPOINT_VERSION,checkpoint_format=CHECKPOINT_FORMAT,request=deepcopy(request),
        base_profile_spec=deepcopy(base),resolution=deepcopy(facts),loop_spec=loop,
        model_spec=dict(class_path=CLASS_PATHS.get(task,CLASS_PATHS['standard']),constructor_kwargs=kwargs),
        profile_spec=resolved_profile,field_sources=dict(loop_spec=fields,profile_spec=resolved_profile['field_sources'],
            explicit_profile_assertions={k:options[k] for k in ('hidden_width','latent_width','heads','actual_M')
                                         if cost!='custom' and k in options}),
        fair_comparison=fair),'config_hash')


def resolve_config(task, profile=DEFAULT_PROFILE, *, options=None, profile_overrides=None):
    opts=deepcopy({} if options is None else options);overrides=deepcopy({} if profile_overrides is None else profile_overrides)
    _request(task,profile,opts,overrides)
    try:base=_profile(task,profile,overrides)
    except ValueError as e:raise V3SchemaError(str(e)) from e
    return _assemble(dict(task=task,profile=profile,options=opts,profile_overrides=overrides),base,_resolve_facts(task,opts,base))


def validate_config(config):
    """Replay saved facts, not current task/profile tables; reject structural forgeries.

    Hashes establish integrity, not authenticity. The future pair loader also
    verifies sidecar/manifest hashes and provenance before any tensor load.
    """
    json_value(config);exact(config,CONFIG_KEYS,'config')
    request=config['request'];exact(request,{'task','profile','options','profile_overrides'},'request')
    cost=_request(request['task'],request['profile'],request['options'],request['profile_overrides'])
    opts=request['options'];base=config['base_profile_spec'];facts=config['resolution']
    # Saved original catalog snapshot has its own checksum and closed structure.
    base_keys={'family','config_version','task','profile','values','field_sources','derived_model',
               'unresolved','ignored_legacy_defaults','config_hash','integration_contract'}
    exact(base,base_keys,'base_profile_spec')
    require_equal(seal(base,'config_hash')['config_hash'],base['config_hash'],'base_profile_hash')
    require_equal(request['task'],base['task'],'base_profile.task');require_equal(request['profile'],base['profile'],'base_profile.profile')
    exact(base['values'],{'model','training','runtime','data','objective','evaluation'},'base.values')
    exact(facts,{*TOPOLOGY_FIELDS,'hidden_width','latent_width','heads','actual_M','residual_mode',
                 'latent_enabled','adapter_mode','adapter_rank','adapter_alpha'},'resolution')
    top=_topology(opts,cost)
    for k,v in top.items():require_equal(v,facts[k],'resolution.'+k)
    for k in ('hidden_width','latent_width','heads','actual_M','adapter_rank'):integer(facts[k],k,1)
    finite(facts['adapter_alpha'],'adapter_alpha',0,True)
    if type(facts['adapter_alpha']) is not float:raise V3SchemaError('resolved adapter_alpha requires canonical float')
    for k in ('hidden_width','latent_width','heads','actual_M'):
        if k in opts:require_equal(opts[k],facts[k],'resolution.'+k)
    if cost=='custom':
        if not {'hidden_width','latent_width'}<=opts.keys():raise V3SchemaError('custom requires widths')
        for k,old in [('heads','heads'),('actual_M','linearno_rank')]:
            require_equal(opts.get(k,base['values']['model'][old]),facts[k],'resolution.'+k)
    else:
        require_equal(8,facts['heads'],'profile heads')
        require_equal(32 if request['task'] in ('ns','airfrans','car') else 64,facts['actual_M'],'profile actual_M')
    for k,default in [('residual_mode',RESIDUAL_MODES[0]),('latent_enabled',True),('adapter_mode',ADAPTER_MODES[1]),('adapter_rank',4)]:
        require_equal(opts.get(k,default),facts[k],'resolution.'+k)
    require_equal(float(opts.get('adapter_alpha',4)),facts['adapter_alpha'],'resolution.adapter_alpha')
    if facts['hidden_width']%facts['heads']:raise V3SchemaError('hidden_width must be divisible by heads')
    if base['values']['model']['time_input'] and facts['hidden_width']%2:raise V3SchemaError('time embedding requires even hidden')
    if facts['adapter_mode']!='none' and facts['loop_repeats']!=2:raise V3SchemaError('adapter-on requires R=2')
    expected=_assemble(request,base,facts)
    require_equal(expected,config,'config')
    return deepcopy(config)


def run_directory_id(config):
    c=validate_config(config);s=c['loop_spec'];P,C,R,S=(s[k] for k in TOPOLOGY_FIELDS)
    # Readable fields plus complete hash (all math, ownership, seeds, protocol,
    # and field-source facts). No timestamp; reservation is a later-stage job.
    return (f"{s['task']}__linearno_loop__v3__{s['cost_profile']}__P{P}-C{C}-R{R}-S{S}__"
            f"{s['residual_mode']}__H{s['hidden_width']}h{s['heads']}M{s['actual_M']}Dz{s['latent_width']}__"
            f"z{int(s['latent_enabled'])}a{int(s['adapter_mode']!='none')}r{s['adapter_rank']}x{s['adapter_alpha']:g}__"
            f"seed{c['profile_spec']['values']['runtime']['seed']}__{c['config_hash']}")
