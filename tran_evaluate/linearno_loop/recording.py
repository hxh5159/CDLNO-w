"""Launcher-only observation of actual initial models; no training operations."""
import hashlib
import json
from pathlib import Path

MANIFEST='loop_run_manifest.json'


def schedule(spec):
    P,C,R,S=(spec[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
    mode=spec['residual_mode'];events=[]
    def body(group,index):
        events.extend([f'loop.{group}.{index}.operator',f'loop.{group}.{index}.mlp'])
    for i in range(P):body('prefix',i)
    for r in range(R):
        for c in range(C):
            for j,branch in enumerate(('operator','mlp')):
                if mode=='rb_attnres':events.append(f'loop.rb_receivers.{r}.{2*c+j}')
                events.append(f'loop.core.{c}.{branch}')
        if mode=='lb_attnres_1_over_r':
            events.append(f'loop.lb_boundaries.{r}' if r<R-1 else 'loop.lb_output')
    if mode=='rb_attnres':events.append('loop.rb_output')
    for i in range(S):body('suffix',i)
    events.extend([f'loop.suffix.{S-1}.final_norm',f'loop.suffix.{S-1}.head'])
    return events


def is_router(name):
    return name.startswith(('loop.rb_receivers.','loop.rb_output.','loop.lb_boundaries.','loop.lb_output.'))


def state_hash(state):
    import torch
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        x=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(x.dtype),list(x.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header)
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def observe(args,model,member=0):
    """Record before the first optimizer step; then capture one actual forward."""
    import torch
    from cdlno.training_state import _atomic
    from cdlno.linearno_loop.attnres import PointDepthAttnRes
    from cdlno.linearno_loop.industrial_state import member_seed
    cfg=args._linearno_loop_config;spec=cfg['loop_spec'];path=Path(args.linearno_run_dir)/MANIFEST
    base=dict(schema_version=1,family='linearno_loop',config_hash=cfg['config_hash'],task=spec['task'],
        profile=spec['profile'],topology=spec['topology_preset'],residual_mode=spec['residual_mode'],
        unique_depth=spec['unique_depth'],executed_depth=spec['executed_depth'],resolved_rank=spec['resolved_rank'],
        expected_call_schedule=schedule(spec),fair_comparison=cfg['fair_comparison'],
        selection_policy='predeclared paired seeds; final checkpoint; no test-based seed/checkpoint selection',
        launcher_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    key=f'member_{member:03d}'
    if path.exists():
        saved=json.loads(path.read_text())
        for field,value in base.items():
            if field!='launcher_sources' and saved[field]!=value:raise ValueError('loop run manifest mismatch: '+field)
        if key in saved['members']:
            if not (args.eval or args.resume):raise ValueError('initial member record already exists')
            return model
    elif args.eval or args.resume:
        # LL6/LL7 archives created without this launcher stay valid and immutable.
        print('LL8 manifest absent in older/direct run; original initialization is not fabricated.',flush=True)
        return model
    else:saved={**base,'members':{}}
    if args.eval:raise ValueError('incomplete ensemble initialization manifest')
    if not path.parent.is_dir():raise ValueError('native recorder must reserve run before initialization recording')
    parameters=dict(model.named_parameters());state=model.state_dict()
    backbone={k:v for k,v in state.items() if not is_router(k)}
    total=sum(p.numel() for p in parameters.values());routers=sum(p.numel() for k,p in parameters.items() if is_router(k))
    if routers!=spec['attnres']['router_parameter_count']:raise ValueError('measured router count differs from schema')
    row=dict(public_backbone_initial_sha256=state_hash(backbone),initial_state_sha256=state_hash(state),
        initialization_seed=member_seed(args,member) if spec['task'] in ('car','airfrans') else args.seed,
        dataloader_generator_seeds=cfg['fair_comparison']['dataloader_generators'],
        python_numpy_torch_seed=args.seed,router_initialization='query=zeros; norm=ones; consumes no RNG',
        parameters=total,backbone_parameters=total-routers,router_parameters=routers,
        actual_call_schedule=None,observation='pending_first_successful_forward')
    saved['members'][key]=row;_atomic(path,saved,json_file=True)
    events=[];handles=[]
    def begin(*unused):events.clear()
    def event(name):
        def record(*unused):events.append(name)
        return record
    def end(module,inputs,output):
        if output is None:return  # Exception: next forward starts a clean list.
        try:
            if events!=base['expected_call_schedule']:raise ValueError('actual loop call schedule differs from topology')
            current=json.loads(path.read_text())
            current['members'][key].update(actual_call_schedule=list(events),observation='first_forward_verified')
            _atomic(path,current,json_file=True)
        finally:
            for handle in handles:handle.remove()
            handles.clear();events.clear()
    handles.append(model.register_forward_pre_hook(begin))
    for group in ('prefix','core','suffix'):
        for i,physical in enumerate(getattr(model.loop,group)):
            block=physical.block;prefix=f'loop.{group}.{i}'
            handles.append(block.Attn.register_forward_pre_hook(event(prefix+'.operator')))
            handles.append(block.mlp.register_forward_pre_hook(event(prefix+'.mlp')))
            if block.last_layer:
                handles.append(block.ln_3.register_forward_pre_hook(event(prefix+'.final_norm')))
                handles.append(block.mlp2.register_forward_pre_hook(event(prefix+'.head')))
    for name,module in model.named_modules():
        if isinstance(module,PointDepthAttnRes):handles.append(module.register_forward_pre_hook(event(name)))
    handles.append(model.register_forward_hook(end,always_call=True))
    return model


def install():
    """Only new launchers install these observers; old invocation paths are inert."""
    from types import SimpleNamespace
    from cdlno.linearno_loop import standard_entry,industrial_state
    original_module=standard_entry.model_module
    original_construct=industrial_state.construct
    def module(args):
        original=original_module(args)
        def construct(**kwargs):return observe(args,original.Model(**kwargs))
        return SimpleNamespace(Model=construct)
    def construct(args,member=0):return observe(args,original_construct(args,member),member)
    standard_entry.model_module=module
    industrial_state.construct=construct
    # Handle already imported modules too (normal startup imports them later).
    import sys
    for name in ('cdlno.linearno_loop.air_entry','cdlno.linearno_loop.car_entry'):
        if name in sys.modules:sys.modules[name].construct=construct
