"""V4 parameter owners and actual call schedule on the existing run manifest."""
import json
from pathlib import Path

from linearno_loop.v4.config import validate_config
from .checkpoint import measure_parameters


def observe(args, model, member=0):
    from cdlno.training_state import _atomic
    from cdlno.linearno_loop.industrial_state import member_seed
    from tran_evaluate.linearno_loop.recording import state_hash
    config = validate_config(args._linearno_loop_config)
    path = Path(args.linearno_run_dir) / 'loop_run_manifest.json'
    route = config['rmlp_spec']['route']
    schedule = []
    for index, owner in enumerate(route):
        schedule.append(f'loop.blocks.{index}.operator')
        if config['temperature_mode'] != 'base':
            schedule.extend([f'loop.blocks.{index}.Q_temperature', f'loop.blocks.{index}.K_temperature'])
        schedule.append('loop.rmlp.' + owner)
    schedule.extend(['final_norm', 'head'])
    base = dict(schema_version=4, architecture=config['architecture'], family='linearno_loop',
                config_hash=config['config_hash'], task=config['task'], profile=config['profile'],
                temperature_mode=config['temperature_mode'], residual_mode=config['residual_mode'],
                unique_depth=8, executed_depth=8, ownership='eight_operators_five_rmlp_owners',
                expected_call_schedule=schedule, fair_comparison=config['fair_comparison'])
    key = f'member_{member:03d}'
    if path.exists():
        saved = json.loads(path.read_text())
        for field, value in base.items():
            if saved.get(field) != value:
                raise ValueError('V4 run manifest mismatch: ' + field)
        if key in saved['members']:
            if not (args.eval or args.resume):
                raise ValueError('V4 initial member already recorded')
            return model
    elif args.eval or args.resume:
        # Direct constructor runs have no launcher initialization record.
        return model
    else:
        saved = {**base, 'members': {}}
    if args.eval or not path.parent.is_dir():
        raise ValueError('V4 recorder requires a reserved training run')
    state = model.state_dict()
    public = {name: value for name, value in state.items()
              if '.q_temperature.' not in name and '.k_temperature.' not in name}
    measurement = measure_parameters(model, config)
    saved['members'][key] = dict(initial_state_sha256=state_hash(state),
        public_backbone_initial_sha256=state_hash(public), parameters=measurement['total'],
        parameter_parts=measurement['groups'], operator_instances=8, point_ffn_instances=5,
        logical_operator_visits=8, point_ffn_visits=8,
        initialization_seed=member_seed(args, member) if config['task'] in ('car', 'airfrans') else args.seed,
        dataloader_generator_seeds=config['fair_comparison']['dataloader_generators'],
        actual_call_schedule=None, observation='pending_first_successful_forward')
    _atomic(path, saved, json_file=True)
    events, handles = [], []
    def begin(*unused):
        events.clear()
    def event(name):
        def record(*unused):
            events.append(name)
        return record
    def end(module, inputs, output):
        if output is None:
            events.clear()
            return
        try:
            if events != schedule:
                raise ValueError('V4 actual call schedule differs from frozen ownership')
            current = json.loads(path.read_text())
            current['members'][key].update(actual_call_schedule=list(events), observation='first_forward_verified')
            _atomic(path, current, json_file=True)
        finally:
            for handle in handles:
                handle.remove()
            handles.clear(); events.clear()
    handles.append(model.register_forward_pre_hook(begin))
    for index, block in enumerate(model.loop.blocks):
        handles.append(block.Attn.register_forward_pre_hook(event(f'loop.blocks.{index}.operator')))
        if config['temperature_mode'] != 'base':
            handles.append(block.Attn.q_temperature.register_forward_pre_hook(event(f'loop.blocks.{index}.Q_temperature')))
            handles.append(block.Attn.k_temperature.register_forward_pre_hook(event(f'loop.blocks.{index}.K_temperature')))
    for owner, module in model.loop.rmlp.items():
        handles.append(module.register_forward_pre_hook(event('loop.rmlp.' + owner)))
    handles.append(model.final_norm.register_forward_pre_hook(event('final_norm')))
    handles.append(model.head.register_forward_pre_hook(event('head')))
    handles.append(model.register_forward_hook(end, always_call=True))
    return model
