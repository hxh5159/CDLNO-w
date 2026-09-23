"""Pure v4 configuration resolver; no torch import and no task data access."""
from copy import deepcopy
from cdlno.linearno.profiles import (resolve_config as resolve_base, validate_resolved,
                                    DEFAULT_PROFILE)
from .contracts import *
import hashlib

def _derived_seed(seed,label):
    return int.from_bytes(hashlib.sha256(f"v4:{seed}:{label}".encode()).digest()[:8],'big')%(2**63)

def _base(task, profile, explicit=None):
    contract = "car_transolver_mse_v1" if task == "car" else "airfrans_transolver_mse_v1" if task == "airfrans" else "standard_temporal_l5" if task in ("ns","plasticity") else "standard_static_l4"
    return resolve_base(task, profile, explicit=explicit or {}, contract=contract)

def resolve_config(task, *, profile=DEFAULT_PROFILE, options=None, profile_overrides=None):
    options = deepcopy(options or {})
    if task not in TASKS: raise V4SchemaError(f"unknown task {task}")
    if options.get("architecture") != ARCHITECTURE:
        raise V4SchemaError(f"explicit architecture={ARCHITECTURE} is required")
    unknown = set(options) - {"architecture","temperature_mode","residual_mode","seed","ffn_ratio"}
    if unknown: raise V4SchemaError(f"unknown v4 options: {sorted(unknown)}")
    mode = options.get("temperature_mode", "base")
    if mode not in TEMPERATURE_MODES: raise V4SchemaError("unknown temperature_mode")
    residual = options.get("residual_mode", RESIDUAL_MODE)
    if residual != RESIDUAL_MODE: raise V4SchemaError("v4 only supports sr_1_over_sqrt_r")
    overrides=deepcopy(profile_overrides or {})
    allowed_overrides={
        'training.lr','training.epochs','training.weight_decay','training.batch_size',
        'training.gradient_clip','runtime.save_name',
    }
    unknown_overrides=set(overrides)-allowed_overrides
    if unknown_overrides:raise V4SchemaError(f"v4 profile overrides are not allowed: {sorted(unknown_overrides)}")
    if 'seed' in options:overrides['runtime.seed']=options['seed']
    base = _base(task, profile, overrides)
    m = base["values"]["model"]
    ffn_ratio = m["ffn_ratio"] if "ffn_ratio" not in options else options["ffn_ratio"]
    integer(ffn_ratio, "ffn_ratio")
    if ffn_ratio != m['ffn_ratio']:raise V4SchemaError('v4 ResMLP ratio must equal the pure LinearNO task profile')
    seed = base["values"]["runtime"]["seed"]
    integer(seed, "seed", 0)
    model = dict(task=task, profile=profile, hidden_width=m["hidden"], heads=m["heads"], actual_M=m["linearno_rank"],
                 head_dim=m["hidden"]//m["heads"], variant=m["linearno_variant"], ffn_ratio=ffn_ratio,
                 grid_height=m["H"], grid_width=m["W"], space_dim=m["space_dim"], fun_dim=m["fun_dim"],
                 out_dim=m["out_dim"], ref=m["ref"], unified_pos=m["unified_pos"], time_input=m["time_input"],
                 dropout=m["dropout"], activation=m["activation"])
    class_path = ('cdlno.linearno_loop.v4.standard.LoopedStandardModelV4' if task not in ('airfrans','car') else
                  'cdlno.linearno_loop.v4.airfrans.LoopedAirfRANSModelV4' if task=='airfrans' else
                  'cdlno.linearno_loop.v4.shapenet.LoopedShapeNetModelV4')
    constructor=dict(space_dim=m['space_dim'],fun_dim=m['fun_dim'],out_dim=m['out_dim'],time_input=m['time_input'],
      ref=m['ref'],unified_pos=m['unified_pos'],hidden_width=m['hidden'],grid_height=m['H'],grid_width=m['W'],
      actual_M=m['linearno_rank'],heads=m['heads'],variant=m['linearno_variant'],dropout=m['dropout'],
      activation=m['activation'],ffn_ratio=ffn_ratio,temperature_mode=mode,public_seed=seed,architecture=ARCHITECTURE)
    result = dict(family=FAMILY, architecture=ARCHITECTURE, architecture_family=FAMILY,
        architecture_extension=ARCHITECTURE_EXTENSION, architecture_version=ARCHITECTURE_VERSION,
        checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_version=CHECKPOINT_VERSION,
        task=task, profile=profile, temperature_mode=mode, residual_mode=residual,
        topology=list(TOPOLOGY), prefix_blocks=1, recurrent_core_blocks=3, loop_repeats=2, suffix_blocks=1,
        rmlp_owners=list(RMLP_OWNERS), rmlp_depths=RMLP_DEPTHS,
        rmlp_execution_depths=[2,3,3,3,3,3,3,2],
        rmlp_spec=dict(route=['first','A','B','C','A','B','C','last'],
          owner_depths=RMLP_DEPTHS,input_width=m['hidden'],hidden_width=m['hidden']*ffn_ratio,
          output_width=m['hidden'],hidden_ratio=ffn_ratio,activation='GELU_tanh',
          input_residual_requested=True,output_residual_requested=True,
          input_residual_effective=ffn_ratio==1,output_residual_effective=ffn_ratio==1,
          hidden_residual_always=True,layer_norm=False,dropout=0.0,final_activation=False),
        model=model, seed=seed, predictor_hidden_width=m["linearno_rank"],
        temperature_spec=dict(
          routing_feature='post_in_project_per_head_pre_qkv',
          point_predictor=['Linear(d_h,M)','GELU_tanh','Linear(M,1)'],
          latent_k_predictor=['mean_N','Linear(d_h,M)','GELU_tanh','Linear(M,M)'],
          q_shape='[B,Hd,N,1]',
          k_shape='none' if mode=='base' else ('[B,Hd,1,M]' if mode=='latent_k_point_q' else '[B,Hd,N,1]'),
          multiplier='exp(log(2)*tanh(delta))',multiplier_open_interval=[0.5,2.0],
          base_policy=dict(plain=1.0,conv=1.0,temp='temperature_q/k.clamp(0.01,1.0)',
                           conv_temp='temperature_q/k.clamp(0.01,1.0)',
                           shapenet='tempreature_q/k.clamp(0.1,2.0)',airfrans=1.0),
          softmax_axes=dict(K='N',Q='M'),noise=None),
        public_contract=dict(operator_ownership="eight_independent", rmlp_schedule=["first","A","B","C","A","B","C","last"],
                             middle_scale="1/sqrt(2)", prefix_suffix_scale=1.0,
                             temperature_formula="tau_base*exp(log(2)*tanh(delta))"),
        model_spec=dict(class_path=class_path, constructor_kwargs=constructor),
        profile_spec=base,loop_spec=dict(task=task,profile=profile,unique_depth=8,executed_depth=8,
          prefix_blocks=1,recurrent_core_blocks=3,loop_repeats=2,suffix_blocks=1,residual_mode=RESIDUAL_MODE,
          temperature_mode=mode,hidden_width=m['hidden'],heads=m['heads'],actual_M=m['linearno_rank'],variant=m['linearno_variant']),
        fair_comparison=dict(public_seed=seed,dataloader_generators={k:_derived_seed(seed,k) for k in ('train','test')}))
    result["config_hash"] = digest(result)
    return result

def validate_config(config):
    if not isinstance(config, dict) or config.get("architecture") != ARCHITECTURE:
        raise V4SchemaError("not a v4 config")
    # The hash authenticates the values of a contract; it must not turn an
    # otherwise undeclared extension field into a silently accepted schema.
    # Keep this allow-list local to v4 so legacy v1/v2/v3 metadata remains
    # governed by its own validators.
    expected_fields = {
        "family", "architecture", "architecture_family", "architecture_extension",
        "architecture_version", "checkpoint_schema", "checkpoint_version",
        "task", "profile", "temperature_mode", "residual_mode", "topology",
        "prefix_blocks", "recurrent_core_blocks", "loop_repeats", "suffix_blocks",
        "rmlp_owners", "rmlp_depths", "rmlp_execution_depths", "rmlp_spec", "model",
        "seed", "predictor_hidden_width", "temperature_spec", "public_contract",
        "model_spec", "profile_spec", "loop_spec", "fair_comparison", "config_hash",
    }
    unknown_fields = set(config) - expected_fields
    if unknown_fields:
        raise V4SchemaError(f"undeclared v4 config fields: {sorted(unknown_fields)}")
    saved = config.get("config_hash"); body = deepcopy(config); body.pop("config_hash", None)
    if saved != digest(body): raise V4SchemaError("config hash mismatch")
    for key in ("task","temperature_mode","residual_mode","model","topology"): 
        if key not in config: raise V4SchemaError("missing "+key)
    if tuple(config["topology"]) != TOPOLOGY: raise V4SchemaError("v4 topology is fixed eight-block schedule")
    expected={'family':FAMILY,'architecture':ARCHITECTURE,'architecture_family':FAMILY,
      'architecture_extension':ARCHITECTURE_EXTENSION,'architecture_version':ARCHITECTURE_VERSION,
      'checkpoint_schema':CHECKPOINT_SCHEMA,'checkpoint_version':CHECKPOINT_VERSION,
      'residual_mode':RESIDUAL_MODE,'prefix_blocks':1,'recurrent_core_blocks':3,'loop_repeats':2,'suffix_blocks':1}
    for key,value in expected.items():require_equal(value,config.get(key),key)
    require_equal(list(RMLP_OWNERS),config.get('rmlp_owners'),'rmlp_owners')
    require_equal(RMLP_DEPTHS,config.get('rmlp_depths'),'rmlp_depths')
    require_equal([2,3,3,3,3,3,3,2],config.get('rmlp_execution_depths'),'rmlp_execution_depths')
    if config['temperature_mode'] not in TEMPERATURE_MODES:raise V4SchemaError('unknown temperature mode')
    profile=validate_resolved(config['profile_spec'])
    # A rehashed legacy profile must not open structural override channels.
    native = _base(config['task'], config['profile'])
    require_equal(native['values']['model'], profile['values']['model'], 'frozen pure model profile')
    require_equal(native['values']['data'], profile['values']['data'], 'frozen pure data profile')
    require_equal(native['values']['objective'], profile['values']['objective'], 'frozen pure objective')
    require_equal(native['values']['evaluation'], profile['values']['evaluation'], 'frozen pure evaluation')
    require_equal(config['task'],profile['task'],'profile_spec.task')
    require_equal(config['profile'],profile['profile'],'profile_spec.profile')
    pm=profile['values']['model'];m=config['model'];integer(m['hidden_width'],'hidden_width');integer(m['heads'],'heads');integer(m['actual_M'],'actual_M')
    if m['hidden_width']%m['heads']:raise V4SchemaError('hidden width must be divisible by heads')
    expected_model=dict(task=config['task'],profile=config['profile'],hidden_width=pm['hidden'],heads=pm['heads'],actual_M=pm['linearno_rank'],
      head_dim=pm['hidden']//pm['heads'],variant=pm['linearno_variant'],ffn_ratio=pm['ffn_ratio'],grid_height=pm['H'],grid_width=pm['W'],
      space_dim=pm['space_dim'],fun_dim=pm['fun_dim'],out_dim=pm['out_dim'],ref=pm['ref'],unified_pos=pm['unified_pos'],
      time_input=pm['time_input'],dropout=pm['dropout'],activation=pm['activation'])
    require_equal(expected_model,m,'model')
    require_equal(m['actual_M'],config.get('predictor_hidden_width'),'predictor_hidden_width')
    expected_temperature=dict(
      routing_feature='post_in_project_per_head_pre_qkv',
      point_predictor=['Linear(d_h,M)','GELU_tanh','Linear(M,1)'],
      latent_k_predictor=['mean_N','Linear(d_h,M)','GELU_tanh','Linear(M,M)'],
      q_shape='[B,Hd,N,1]',
      k_shape='none' if config['temperature_mode']=='base' else ('[B,Hd,1,M]' if config['temperature_mode']=='latent_k_point_q' else '[B,Hd,N,1]'),
      multiplier='exp(log(2)*tanh(delta))',multiplier_open_interval=[0.5,2.0],
      base_policy=dict(plain=1.0,conv=1.0,temp='temperature_q/k.clamp(0.01,1.0)',
                       conv_temp='temperature_q/k.clamp(0.01,1.0)',
                       shapenet='tempreature_q/k.clamp(0.1,2.0)',airfrans=1.0),
      softmax_axes=dict(K='N',Q='M'),noise=None)
    require_equal(expected_temperature,config.get('temperature_spec'),'temperature_spec')
    expected_public=dict(operator_ownership='eight_independent',rmlp_schedule=['first','A','B','C','A','B','C','last'],
      middle_scale='1/sqrt(2)',prefix_suffix_scale=1.0,temperature_formula='tau_base*exp(log(2)*tanh(delta))')
    require_equal(expected_public,config.get('public_contract'),'public_contract')
    expected_rmlp=dict(route=expected_public['rmlp_schedule'],owner_depths=RMLP_DEPTHS,input_width=m['hidden_width'],
      hidden_width=m['hidden_width']*m['ffn_ratio'],output_width=m['hidden_width'],hidden_ratio=m['ffn_ratio'],activation='GELU_tanh',
      input_residual_requested=True,output_residual_requested=True,input_residual_effective=m['ffn_ratio']==1,
      output_residual_effective=m['ffn_ratio']==1,hidden_residual_always=True,layer_norm=False,dropout=0.0,final_activation=False)
    require_equal(expected_rmlp,config.get('rmlp_spec'),'rmlp_spec')
    require_equal(config['temperature_mode'],config['loop_spec'].get('temperature_mode'),'loop_spec.temperature_mode')
    require_equal(config['seed'],profile['values']['runtime']['seed'],'seed')
    expected_class=('cdlno.linearno_loop.v4.standard.LoopedStandardModelV4' if config['task'] not in ('airfrans','car') else
                    'cdlno.linearno_loop.v4.airfrans.LoopedAirfRANSModelV4' if config['task']=='airfrans' else
                    'cdlno.linearno_loop.v4.shapenet.LoopedShapeNetModelV4')
    require_equal(expected_class,config['model_spec'].get('class_path'),'model_spec.class_path')
    expected_constructor=dict(space_dim=m['space_dim'],fun_dim=m['fun_dim'],out_dim=m['out_dim'],time_input=m['time_input'],ref=m['ref'],
      unified_pos=m['unified_pos'],hidden_width=m['hidden_width'],grid_height=m['grid_height'],grid_width=m['grid_width'],actual_M=m['actual_M'],
      heads=m['heads'],variant=m['variant'],dropout=m['dropout'],activation=m['activation'],ffn_ratio=m['ffn_ratio'],
      temperature_mode=config['temperature_mode'],public_seed=config['seed'],architecture=ARCHITECTURE)
    require_equal(expected_constructor,config['model_spec'].get('constructor_kwargs'),'model_spec.constructor_kwargs')
    expected_loop=dict(task=config['task'],profile=config['profile'],unique_depth=8,executed_depth=8,prefix_blocks=1,
      recurrent_core_blocks=3,loop_repeats=2,suffix_blocks=1,residual_mode=RESIDUAL_MODE,
      temperature_mode=config['temperature_mode'],hidden_width=m['hidden_width'],heads=m['heads'],actual_M=m['actual_M'],variant=m['variant'])
    require_equal(expected_loop,config.get('loop_spec'),'loop_spec')
    require_equal(dict(public_seed=config['seed'],dataloader_generators={k:_derived_seed(config['seed'],k) for k in ('train','test')}),
                  config.get('fair_comparison'),'fair_comparison')
    require_equal(config['config_hash'],digest(body),'config_hash')
    return deepcopy(config)

def run_directory_id(config):
    c=validate_config(config)
    return f"{c['task']}_resmlp_dual_temp_v4_{c['profile']}_{c['temperature_mode']}_sr_s{c['seed']}_{c['config_hash'][:12]}"
