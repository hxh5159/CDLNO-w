"""Industrial routing and strict metadata adapters; no dataset imports."""
import importlib
import json
from pathlib import Path

from . import checkpoint as ck
from .config import DEFAULT_FEATURES, PATHS, resolve_config, task_kind
from .cli import take_features, run_family
from .fair_run import take_fair, attach
from linearno_history.schema import FEATURE_FIELDS, resolve_feature_config, derive_fair_seeds


def intercept(parser, tokens, *, task, evaluation):
    fair, remaining = take_fair(tokens)
    explicit, remaining = take_features(remaining)
    saved = run_family(remaining) == 'linearno_history'
    if not explicit and fair is None and not saved:
        return None
    flags = resolve_feature_config({**DEFAULT_FEATURES, **explicit})
    enabled = flags['feature_signature'] != 'A0K0'
    if not saved and not enabled and fair is not True:
        pure = importlib.import_module('cdlno.linearno.'+('air_entry' if task=='airfrans' else 'car_entry'))
        return pure.parse_args(parser, remaining, evaluation=evaluation)
    if (saved or enabled) and fair is False:
        parser.error('research runs require independent fair-run data generators')
    module = importlib.import_module('cdlno.linearno_history.'+('air_entry' if task=='airfrans' else 'car_entry'))
    return module.parse_args(parser, remaining, evaluation=evaluation, history_features=explicit)


def finish_args(args, explicit, tokens):
    saved = getattr(args, '_linearno_metadata', None)
    config = ck.config_from_metadata(saved) if saved is not None else resolve_config(args._linearno_config, family='linearno_history', features=explicit)
    if saved is not None:
        for key,value in explicit.items():
            if value != config['features'][key]:
                raise ValueError('explicit '+key+' conflicts with checkpoint')
    args._linearno_history_adapter = True
    if config['family'] == 'linearno_history':
        args._linearno_history_config = config
    args.linearno_family = config['family']
    args._linearno_model_spec = config['model_spec']
    for key in FEATURE_FIELDS:setattr(args,key,config['features'][key])
    explicit_dir = any(t.split('=')[0] in ('--experiment-dir','--linearno-run-dir') for t in tokens)
    if not explicit_dir and not (args.eval or args.resume):
        from cdlno.experiment import timestamp
        args.linearno_run_dir = args.linearno_run_dir.parent/(config['run_signature']+'__'+timestamp())
    result = attach(args,True)
    if args.eval or args.resume:
        from .fair_run import record
        record(args,None)
    return result


def check_structure(metadata):
    ck.validate_metadata(metadata)


def read_metadata(path, *, constructor=None):
    return ck.validate_metadata(ck.read_json(path))


def write_metadata(path, metadata, *, constructor=None):
    checked=ck.validate_metadata(metadata)
    with Path(path).open('x') as f:json.dump(checked,f,sort_keys=True)


def make_metadata(*, profile_spec, model_spec, constructor=None, **sections):
    choices=PATHS[task_kind(profile_spec['task'])]
    signatures=[s for s,p in choices.items() if p==model_spec['class_path']]
    if len(signatures)!=1:raise ValueError('unknown industrial class_path')
    sig=signatures[0]
    constructor_kwargs = model_spec['constructor_kwargs']
    # The no-history-dropout A1K0 ablation is encoded in the research
    # constructor spec. Recover that explicit value when validating an
    # industrial metadata write; omitted/.1 keeps the historical schema.
    dropout = constructor_kwargs.get('attnres_history_dropout_p') if sig == 'A1K0' else None
    c=resolve_config(profile_spec,family='linearno_history',features={
        FEATURE_FIELDS[0]:sig[1]=='1', FEATURE_FIELDS[1]:sig[3]=='1',
        FEATURE_FIELDS[2]:dropout}, feature_seed=constructor_kwargs.get('feature_seed'))
    if c['model_spec'] != model_spec:raise ValueError('industrial constructor/profile mismatch')
    return ck.make_metadata(c,**sections)


def inspect_checkpoint(directory, selector, constructor=None):
    return ck.inspect_checkpoint(directory,selector)


def read_pair(path, constructor=None):
    path=Path(path);metadata,_=ck.inspect_checkpoint(path.parent.parent,path.stem)
    return metadata,ck._read_pair(path,metadata)


def save_pair(directory, model, metadata, constructor=None):
    return ck.save_checkpoint(directory,model,metadata)


def construct(args, member=0):
    """Each ensemble backbone is paired across modes independent of prior dropout."""
    import torch
    from .factory import build_fair_model
    from .fair_run import record
    config=args._linearno_fair_config
    seed=config['initialization']['public_backbone_seed']
    if member:
        seed=derive_fair_seeds(seed,task=args.linearno_task,split=f'member_{member}')['public_backbone_seed']+member
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=build_fair_model(config)
    if member==0:record(args,model)
    if not args.eval:
        from .fair_run import MANIFEST
        import hashlib
        values={k:v for k,v in model.state_dict().items() if not k.startswith(('latent_attnres.','history_k.'))}
        h=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in values.items())).hexdigest()
        p=args.linearno_run_dir/f'initial_member_{member:03d}.json'
        initial=dict(member=member,public_seed=seed,feature_seed=config['initialization']['feature_seed'],backbone_sha256=h)
        if p.exists():
            if json.loads(p.read_text()) != initial:raise ValueError('ensemble public initialization mismatch')
        else:
            with p.open('x') as f:json.dump(initial,f,sort_keys=True)
    return model


def generators(args):
    import torch
    return {name:torch.Generator().manual_seed(derive_fair_seeds(args.seed,task=args.linearno_task,split=name)['dataloader_generator_seed']) for name in ('train','test')}


def verify_resume_source(saved, current):
    if saved['provenance_spec']['source_sha256'] != current['source_sha256']:
        raise ValueError('resume source code differs from saved run')
