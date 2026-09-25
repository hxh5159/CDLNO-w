"""Six Standard entries: loop construction/archives only, native task loops untouched."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from uuid import uuid4

from cdlno.linearno.standard_entry import (
    ROOT, KEYS, TASKS, ARG_FIELDS, StandardRun as PureStandardRun,
    model_kwargs as _legacy_model_kwargs, normalizer, verify_data, start, finish,
)
from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES
from cdlno.linearno.schema import normalizer_record, unpack_state
from linearno_loop.contracts import CLI_CONTRACT, HISTORY_FLAGS, OPTIONS, PRESETS, RESIDUAL_MODES, TOPOLOGY_FIELDS
from linearno_loop.v2.contracts import CORE_FFN_MODES
from linearno_loop.versioning import (OPTIONS as VERSIONED_OPTIONS, make_metadata, read_metadata,
                                      resolve_config, restore_config, run_directory_id, write_metadata,
                                      is_v3, is_v4, is_v5)

# Wire constants are kept local so the old parser has no import dependency on
# the tensor-bearing V3 package. The selected resolver revalidates them in its
# own schema after explicit architecture dispatch.
ARCHITECTURE_SELECTOR = 'operator_latent_adapter_v3'
V4_ARCHITECTURE_SELECTOR = 'resmlp_dual_temp_v4'
V5_ARCHITECTURE_SELECTOR = 'partial_share_feature_gate_v5'
V3_COST_PROFILES = ('matched_v1', 'efficient_v1', 'custom')
V3_ADAPTER_MODES = ('none', 'bilateral_qk_lowrank_second_visit')
V3_CLI_FIELDS = {'architecture', 'cost_profile', 'topology_preset', 'executed_depth',
                 'prefix_blocks', 'recurrent_core_blocks', 'loop_repeats', 'suffix_blocks',
                 'residual_mode', 'latent_enabled', 'adapter_mode', 'adapter_rank',
                 'adapter_alpha', 'hidden_width', 'latent_width', 'heads', 'actual_M'}


def _loop_parser():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for flag,spec in CLI_CONTRACT.items():
        if flag=='--linearno-rank':continue
        kw=dict(dest=spec['field'],default=argparse.SUPPRESS)
        if spec['field']=='linearno_loop':kw['choices']=('0','1')
        elif spec['field']=='topology_preset':kw['choices']=(*PRESETS,'custom','d12','d20','d28','d60')
        elif spec['field']=='residual_mode':
            kw['choices']=(*RESIDUAL_MODES,'operator_1_expert_1_over_r')
        else:kw['type']=int
        parser.add_argument(flag,**kw)
    parser.add_argument('--linearno-loop-core-ffn-mode',dest='core_ffn_mode',
                        choices=CORE_FFN_MODES,default=argparse.SUPPRESS)
    # V3 fields live in this loop-only parser. They are accepted for every
    # loop invocation, but the resolver rejects them unless the explicit
    # architecture selector is present, preserving v1/v2 behavior.
    parser.add_argument('--linearno-loop-architecture', dest='architecture',
                        choices=(ARCHITECTURE_SELECTOR,V4_ARCHITECTURE_SELECTOR,
                                 V5_ARCHITECTURE_SELECTOR), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-temperature-mode', dest='temperature_mode',
                        choices=('base','latent_k_point_q','point_k_point_q'), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-cost-profile', dest='cost_profile',
                        choices=V3_COST_PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-executed-depth', dest='executed_depth',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-latent', dest='latent_enabled',
                        type=int, choices=(0, 1), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-adapter-mode', dest='adapter_mode',
                        choices=V3_ADAPTER_MODES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-adapter-rank', dest='adapter_rank',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-adapter-alpha', dest='adapter_alpha',
                        type=float, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-hidden-width', dest='hidden_width',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-latent-width', dest='latent_width',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-heads', dest='heads',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-dense-expert-count', dest='expert_count',
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-loop-dense-expert-width', dest='expert_width',
                        type=int, default=argparse.SUPPRESS)
    return parser


def intercept(parser,task,tokens,selected_model):
    """Only explicit loop train or saved loop eval/resume selects this adapter."""
    probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    probe.add_argument('--eval',type=int,default=0);probe.add_argument('--resume',action='store_true')
    probe.add_argument('--experiment-dir','--linearno-run-dir',dest='directory',type=Path)
    intent,_=probe.parse_known_args(tokens)
    flags,remaining=_loop_parser().parse_known_args(tokens);explicit=vars(flags)
    path=intent.directory/'architecture.json' if intent.directory else None
    try:
        saved_loop=False
        if (intent.eval or intent.resume) and path is not None and path.is_file():
            from linearno_loop.contracts import read_json
            saved_loop=read_json(path).get('family')=='linearno_loop'
        if not explicit and not saved_loop:return None
        history={'--'+k for k in HISTORY_FLAGS}|{'--'+k.replace('_','-') for k in HISTORY_FLAGS}
        if any(t.split('=')[0] in history or t.split('=')[0]=='--linearno-fair-run' for t in tokens):
            raise ValueError('loop cannot mix history/A/K/fair-run flags, including explicit disabled values')
        expected=KEYS[1] if task=='elasticity' else KEYS[0]
        if selected_model is not None and selected_model!=expected:
            raise ValueError(f'{task} loop requires --model {expected}')
        if selected_model is None:
            if not saved_loop:raise ValueError(f'new loop training requires --model {expected}')
            remaining=['--model',expected,*remaining]
        if explicit.get('linearno_loop')=='0':
            if saved_loop or set(explicit)!={'linearno_loop'}:
                raise ValueError('disabled loop conflicts with loop configuration/checkpoint')
            from cdlno.linearno.standard_entry import parse_args
            return parse_args(parser,task,remaining)
        if 'linearno_loop' in explicit:explicit['linearno_loop']=True
        if not (intent.eval or intent.resume) and explicit.get('linearno_loop') is not True:
            raise ValueError('new loop train requires --linearno-loop 1')
        return parse_loop_args(parser,task,remaining,explicit)
    except (ValueError,OSError,KeyError,TypeError) as error:parser.error(str(error))


def parse_loop_args(parser,task,tokens,loop_explicit):
    if task not in TASKS:parser.error('LL6 integrates only six Standard tasks')
    parser.add_argument('--linearno-profile',choices=PROFILES,default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant',choices=('plain','temp','conv','conv_temp'),default=argparse.SUPPRESS)
    parser.add_argument('--linearno-rank',type=int,default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir','--linearno-run-dir',dest='linearno_run_dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',default=None)
    args=parser.parse_args(tokens)
    probe=copy.deepcopy(parser);probe._defaults.clear()
    for a in probe._actions:a.default=argparse.SUPPRESS
    explicit=vars(probe.parse_args(tokens))
    try:
        if args.eval not in (0,1) or args.eval and args.resume:raise ValueError('choose train, resume or eval')
        if args.checkpoint and not (args.eval or args.resume):raise ValueError('--checkpoint requires eval/resume')
        if {'n_layers','slice_num'} & explicit.keys():raise ValueError('loop topology/rank require PCRS and --linearno-rank; no --n-layers/--slice_num')
        for name,value in dict(downsample=1 if task=='ns' else 5,downsamplex=1,downsampley=1).items():
            if hasattr(args,name) and getattr(args,name)!=value:raise ValueError(f'loop task grid requires {name}={value}')
        options={k:v for k,v in loop_explicit.items() if k in VERSIONED_OPTIONS}
        # V3's CLI uses the same topology/residual names but has its own
        # versioned option namespace.  Normalize wire values before the pure
        # stdlib resolver sees them.
        v3_requested = loop_explicit.get('architecture') == ARCHITECTURE_SELECTOR
        v5_requested = loop_explicit.get('architecture') == V5_ARCHITECTURE_SELECTOR
        if loop_explicit.get('architecture') == V4_ARCHITECTURE_SELECTOR:
            return _parse_v4(parser,task,args,explicit,loop_explicit)
        if v3_requested:
            v3_fields=V3_CLI_FIELDS
            options={k:v for k,v in loop_explicit.items()
                    if k in VERSIONED_OPTIONS or k in v3_fields}
            if 'latent_enabled' in options:
                options['latent_enabled']=bool(options['latent_enabled'])
        if 'linearno_rank' in explicit:
            if v3_requested or v5_requested:
                options['actual_M']=explicit['linearno_rank']
            else:
                options['linearno_rank']=explicit['linearno_rank']
        if 'linearno_rank' in options and 'rank_multiplier' in options:raise ValueError('explicit actual rank/multiplier conflict')
        if options.get('topology_preset') in PRESETS and set(options)&set(TOPOLOGY_FIELDS):
            raise ValueError('preset and custom P/C/R/S fields are mutually exclusive')
        overrides={path:explicit[name] for name,path in ARG_FIELDS.items()
                   if name in explicit and name not in ('n_layers','linearno_rank')}
        if 'model.unified_pos' in overrides:
            if overrides['model.unified_pos'] not in (0,1):raise ValueError('unified_pos must be 0 or 1')
            overrides['model.unified_pos']=bool(overrides['model.unified_pos'])
        if args.eval or args.resume:
            if args.linearno_run_dir is None:raise ValueError('loop eval/resume requires explicit --experiment-dir')
            initial=read_metadata(args.linearno_run_dir/'architecture.json')
            if is_v4(initial):
                if loop_explicit.get('architecture', V4_ARCHITECTURE_SELECTOR) != V4_ARCHITECTURE_SELECTOR:
                    raise ValueError('explicit architecture conflicts with saved V4 metadata')
                return _parse_v4(parser,task,args,explicit,{**loop_explicit,'architecture':V4_ARCHITECTURE_SELECTOR})
            saved_v5=is_v5(initial)
            if saved_v5:
                v5_requested=True
            if saved_v5 != v5_requested and (saved_v5 or v5_requested):
                raise ValueError('checkpoint architecture version conflicts with explicit V5 architecture')
            saved_v3=is_v3(initial)
            if saved_v3:
                v3_requested=True
            if saved_v3 != v3_requested and (saved_v3 or v3_requested):
                raise ValueError('checkpoint architecture version conflicts with explicit loop architecture')
            if saved_v3 and not options:
                # A V3 eval/resume with no repeated structure fields must use
                # the saved resolved config; restore_config enforces all
                # immutable fields before any tensor payload is touched.
                options={}
            asserted=dict(options,task=task)
            if 'linearno_loop' in loop_explicit:asserted['linearno_loop']=loop_explicit['linearno_loop']
            if 'linearno_profile' in explicit:asserted['profile']=explicit['linearno_profile']
            config=restore_config(initial,explicit=asserted)['config']
            from .versioning import checkpoint_api
            checkpoint_module=checkpoint_api(config)
            for path,value in overrides.items():
                section,field=path.split('.')
                if config['profile_spec']['values'][section][field]!=value:
                    raise ValueError(f'explicit {path} conflicts with checkpoint')
            metadata,checkpoint=checkpoint_module.inspect_checkpoint(args.linearno_run_dir,args.checkpoint or ('latest' if args.resume else 'final'),expected=config)
            checkpoint_module.immutable_equal(initial,metadata)
            if args.resume and metadata['resume_state']['epoch']>=config['profile_spec']['values']['training']['epochs']:
                raise ValueError('run has completed its resolved epochs')
            args._linearno_metadata=metadata;args._linearno_checkpoint=checkpoint
        else:
            config=resolve_config(task,getattr(args,'linearno_profile',DEFAULT_PROFILE),options=options,profile_overrides=overrides)
        base=config['profile_spec'];s=config['loop_spec']
        if task=='darcy' and base['profile']=='official_release' and base['values']['training']['epochs']!=500:
            raise ValueError('Darcy official_release fixes scheduler epochs=500; non-500 training rejected')
        args.linearno_family='linearno_loop';args.linearno_task=task
        args._linearno_loop_config=config;args._linearno_config=base
        args._linearno_model_spec=config['model_spec'];args._linearno_normalizers={}
        if is_v3(config) or is_v5(config):
            # Keep task parser attributes compatible with the unchanged exp
            # bodies while V3's resolved model fields remain authoritative.
            m=base['values']['model']; l=config['loop_spec']
            legacy=dict(n_hidden=l['hidden_width'], n_layers=l['unique_depth'],
                        n_heads=l['heads'], mlp_ratio=m['ffn_ratio'], dropout=m['dropout'],
                        ref=m['ref'], unified_pos=m['unified_pos'],
                        linearno_variant=l['variant'], linearno_rank=l['actual_M'])
            args._linearno_legacy_model_spec=legacy
            args.n_hidden=legacy['n_hidden'];args.n_layers=legacy['n_layers']
            args.n_heads=legacy['n_heads'];args.mlp_ratio=legacy['mlp_ratio']
            args.dropout=legacy['dropout'];args.ref=legacy['ref'];args.unified_pos=legacy['unified_pos']
            args.linearno_variant=legacy['linearno_variant'];args.linearno_rank=legacy['linearno_rank']
            for name,path in ARG_FIELDS.items():
                if name in ('n_hidden','n_layers','n_heads','mlp_ratio','dropout','ref','unified_pos','linearno_variant','linearno_rank'):
                    continue
                section,field=path.split('.')
                setattr(args,name,base['values'][section][field])
        else:
            for name,path in ARG_FIELDS.items():
                section,field=path.split('.');setattr(args,name,base['values'][section][field])
            args.n_layers=s['unique_depth'];args.linearno_rank=s['resolved_rank']
        args.linearno_profile=base['profile']
        identifier=run_directory_id(config)
        args.save_name=args.save_name or identifier
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.save_name) or args.save_name in ('.','..'):
            raise ValueError('save_name must be a filename stem')
        if args.linearno_run_dir is None:
            from cdlno.experiment import timestamp
            family_dir = V5_ARCHITECTURE_SELECTOR if is_v5(config) else 'linearno_loop'
            args.linearno_run_dir=Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output'))/task/family_dir/(identifier+'__'+timestamp()+'_'+uuid4().hex[:8])
        if identifier not in args.linearno_run_dir.name:raise ValueError('loop run directory must contain '+identifier)
        args.linearno_run_dir=args.linearno_run_dir.resolve()
        if not (args.eval or args.resume) and args.linearno_run_dir.exists():raise ValueError('new loop run directory already exists')
        return args
    except (ValueError,OSError,KeyError,TypeError) as error:parser.error(str(error))


def _parse_v4(parser,task,args,explicit,loop_explicit):
    from linearno_loop.v4.config import resolve_config,run_directory_id
    from linearno_loop.v4.schema import read_metadata as read_v4_metadata
    if args.eval and args.resume:raise ValueError('choose train, resume or eval')
    unsupported=set(loop_explicit)-{'linearno_loop','architecture','temperature_mode'}
    if unsupported:raise ValueError('V4 does not accept legacy loop fields: '+str(sorted(unsupported)))
    structural={name for name,path in ARG_FIELDS.items() if path.startswith('model.')}
    supplied_structural=structural&set(explicit)
    if supplied_structural:raise ValueError('V4 model fields come from the pure LinearNO profile: '+str(sorted(supplied_structural)))
    profile=getattr(args,'linearno_profile',DEFAULT_PROFILE)
    if args.eval or args.resume:
        if args.linearno_run_dir is None:raise ValueError('V4 eval/resume requires --experiment-dir')
        metadata=read_v4_metadata(args.linearno_run_dir/'architecture.json')
        config=metadata['resolved_config']
        if config['task']!=task:raise ValueError('V4 checkpoint task mismatch')
        if 'linearno_profile' in explicit and explicit['linearno_profile']!=config['profile']:
            raise ValueError('V4 checkpoint profile conflict')
        if 'temperature_mode' in loop_explicit and loop_explicit['temperature_mode']!=config['temperature_mode']:
            raise ValueError('V4 checkpoint temperature mode conflict')
        for name,path in ARG_FIELDS.items():
            if name not in explicit or path.startswith('model.'):continue
            section,field=path.split('.')
            expected=config['profile_spec']['values'][section][field]
            if explicit[name]!=expected:raise ValueError('explicit '+path+' conflicts with V4 metadata')
        args._linearno_metadata=metadata
        args._linearno_checkpoint=args.linearno_run_dir/'checkpoints'/((args.checkpoint or ('latest' if args.resume else 'final'))+'.json')
    else:
        profile_overrides={path:explicit[name] for name,path in ARG_FIELDS.items()
                           if name in explicit and not path.startswith('model.') and name!='seed'}
        config=resolve_config(task,profile=profile,options=dict(architecture=V4_ARCHITECTURE_SELECTOR,
            temperature_mode=loop_explicit.get('temperature_mode','latent_k_point_q'),seed=explicit.get('seed',0)),
            profile_overrides=profile_overrides)
    base=config['profile_spec'];m=config['model']
    args.linearno_family='linearno_loop';args.linearno_task=task;args._linearno_loop_config=config;args._linearno_config=base
    args._linearno_model_spec=dict(class_path=config['model_spec']['class_path'],constructor_kwargs=dict(config['model_spec']['constructor_kwargs']))
    args._linearno_normalizers={};args.linearno_profile=config['profile']
    for name,path in ARG_FIELDS.items():
        section,field=path.split('.')
        if section!='model':setattr(args,name,base['values'][section][field])
    args.n_hidden=m['hidden_width'];args.n_layers=8;args.n_heads=m['heads'];args.mlp_ratio=m['ffn_ratio']
    args.dropout=m['dropout'];args.ref=m['ref'];args.unified_pos=m['unified_pos'];args.linearno_variant=m['variant'];args.linearno_rank=m['actual_M']
    args.seed=config['seed']
    identifier=run_directory_id(config);args.save_name=args.save_name or identifier
    if args.linearno_run_dir is None:
        from cdlno.experiment import timestamp
        args.linearno_run_dir=Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output'))/task/'resmlp_dual_temp_v4'/(identifier+'__'+timestamp()+'_'+uuid4().hex[:8])
    args.linearno_run_dir=args.linearno_run_dir.resolve()
    return args


def model_module(args):
    config=args._linearno_loop_config
    def construct(**kwargs):
        if config.get('architecture')==V4_ARCHITECTURE_SELECTOR:
            if kwargs != model_kwargs(args):
                raise ValueError('V4 constructor kwargs conflict with resolved model')
            from cdlno.linearno_loop.v4.construction import build_from_config
            return build_from_config(config)
        if is_v3(config) or is_v5(config):
            expected=model_kwargs(args)
            if kwargs != expected:
                raise ValueError('versioned loop legacy factory kwargs mismatch')
        elif kwargs!=config['model_spec']['constructor_kwargs']:
            raise ValueError('loop factory kwargs mismatch')
        from .versioning import construction_api
        return construction_api(config).build_from_config(config)
    return SimpleNamespace(Model=construct)


def model_kwargs(args, **grid):
    """Return the unchanged exp constructor shape for v1/v2 and V3 bridge.

    V3/V5 store their own explicit constructor kwargs; the six legacy exp files
    still call this helper with H/W, so the bridge exposes their historical
    names while validating those grid facts against the resolved V3 contract.
    """
    config=args._linearno_loop_config
    if config.get('architecture')==V4_ARCHITECTURE_SELECTOR:
        m=config['model']
        for key,value in grid.items():
            if key=='H' and value!=m['grid_height'] or key=='W' and value!=m['grid_width']:
                raise ValueError('data grid conflicts with V4 metadata')
        return dict(space_dim=m['space_dim'],n_layers=8,n_hidden=m['hidden_width'],n_head=m['heads'],dropout=m['dropout'],
          Time_Input=m['time_input'],act=m['activation'],mlp_ratio=m['ffn_ratio'],fun_dim=m['fun_dim'],out_dim=m['out_dim'],
          ref=m['ref'],unified_pos=m['unified_pos'],H=m['grid_height'],W=m['grid_width'],linearno_variant=m['variant'],linearno_rank=m['actual_M'])
    if not (is_v3(config) or is_v5(config)):
        return _legacy_model_kwargs(args, **grid)
    values=args._linearno_config['values']; m=values['model']; l=config['loop_spec']
    expected_h,expected_w=l['grid_height'],l['grid_width']
    for key,value in grid.items():
        expected={'H': expected_h, 'W': expected_w}.get(key)
        if expected is not None and value != expected:
            raise ValueError(f'data grid {key}={value} conflicts with V3 model metadata {expected}')
    return dict(space_dim=m['space_dim'], n_layers=l['unique_depth'], n_hidden=l['hidden_width'],
                n_head=l['heads'], dropout=m['dropout'], Time_Input=m['time_input'],
                act=m['activation'], mlp_ratio=m['ffn_ratio'], fun_dim=m['fun_dim'],
                out_dim=m['out_dim'], ref=m['ref'], unified_pos=m['unified_pos'],
                H=expected_h, W=expected_w, linearno_variant=l['variant'],
                linearno_rank=l['actual_M'])


def prepare_loaders(args,train_loader,test_loader):
    import torch
    from cdlno.linearno_history.fair_run import GeneratorCollate
    seeds=args._linearno_loop_config['fair_comparison']['dataloader_generators']
    for name,loader in (('train',train_loader),('test',test_loader)):
        if loader.generator is not None:raise ValueError('loop owns the explicit DataLoader generator')
        generator=torch.Generator().manual_seed(seeds[name])
        loader.generator=generator
        if isinstance(loader.sampler,torch.utils.data.RandomSampler):loader.sampler.generator=generator
        if args.linearno_task=='plasticity':
            if loader.num_workers!=0:raise ValueError('loop Plasticity resume requires native num_workers=0')
            loader.collate_fn=GeneratorCollate(loader.collate_fn,generator)


class LoopStandardRun(PureStandardRun):
    """Native observer interface; loop-only metadata and strict continuation."""
    # prepare/load/save below reuse the existing storage/RNG encodings.
    def prepare(self, optimizer, scheduler, train_loader, test_loader):
        prepare_loaders(self.args,train_loader,test_loader)
        v3 = is_v3(self.args._linearno_loop_config)
        v4 = is_v4(self.args._linearno_loop_config)
        v5 = is_v5(self.args._linearno_loop_config)
        from cdlno.linearno.checkpoint import strict_load, restore_random_state
        if v3:
            from cdlno.linearno_loop.v3.checkpoint import resume_state as v3_resume_state, measure_parameters
        elif v4:
            from cdlno.linearno_loop.v4.checkpoint import resume_state as v4_resume_state, measure_parameters
        elif v5:
            from cdlno.linearno_loop.v5.checkpoint import resume_state as v5_resume_state, measure_parameters
        else:
            from cdlno.linearno.checkpoint import resume_state
        from cdlno.training_state import _optimizer_signature
        from .versioning import checkpoint_api,provenance
        args = self.args; cfg = args._linearno_config
        loop_checkpoint=checkpoint_api(args._linearno_loop_config)
        self.optimizer, self.scheduler = optimizer, scheduler
        self.steps = len(train_loader)
        self.updates_per_batch = 20 if self.args.linearno_task == 'plasticity' else 1
        self.optimizer_steps = self.steps * self.updates_per_batch
        self.generators = {name: loader.generator for name,loader in (('train',train_loader),('test',test_loader)) if loader.generator is not None}
        self.sampler = {name: dict(type=type(loader.sampler).__name__, epoch_boundary=True,
                                  replacement=getattr(loader.sampler,'replacement',False))
                        for name,loader in (('train',train_loader),('test',test_loader))}
        self.scheduler_plan = dict(type=type(scheduler).__name__, epochs=args.epochs,
            steps_per_epoch=self.steps if type(scheduler).__name__=='OneCycleLR' else 1,
            total_steps=getattr(scheduler,'total_steps',args.epochs), state_at_construction=scheduler.state_dict())
        records = {name: normalizer_record(dict(mean=obj.mean,std=obj.std), fit_split='train only',
            data_checksum=next(iter(args._linearno_data['checksums'].values())),
            algorithm='UnitTransformer: mean(dim=(0,1)); std(dim=(0,1))+1e-8') for name,obj in args._linearno_normalizers.items()}
        normalizers = dict(policy='saved_train_fit' if records else 'none',records=records)
        data = dict(args._linearno_data, ntrain=len(train_loader.dataset), ntest=len(test_loader.dataset),
                    scheduler=self.scheduler_plan, train_batch_size=train_loader.batch_size,
                    test_batch_size=test_loader.batch_size, optimizer_signature=_optimizer_signature(self.model,optimizer))
        data = json.loads(json.dumps(data))  # canonical JSON tuples/lists before immutable comparisons
        if args.linearno_task in ('ns', 'plasticity'):
            data['temporal_progress'] = dict(optimizer_updates_per_batch=self.updates_per_batch,
                scheduler_updates_per_batch=1, forwards_per_train_batch=20 if args.linearno_task == 'plasticity' else 10,
                time_permutation='per-sample torch.randperm20' if args.linearno_task == 'plasticity' else None,
                global_step_unit='optimizer updates')
        data = dict(protocol=cfg['values']['data'],split=data['split'],sampling=data['sampling'],
            checksums=data['checksums'],scope='synthetic' if str(data['scope']).lower().startswith('synthetic') else 'real',
            runtime={k:v for k,v in data.items() if k not in ('split','sampling','checksums','scope')})
        prov = provenance(args._linearno_loop_config)
        if args.eval or args.resume:
            saved, path = loop_checkpoint.inspect_checkpoint(self.directory,args._linearno_checkpoint.stem,expected=args._linearno_loop_config)
            _, weights = loop_checkpoint.read_pair(path,expected=args._linearno_loop_config)
            for key,actual in (('data_spec',data),('normalizer_spec',normalizers)):
                if saved[key] != actual:
                    raise ValueError(f'{key} differs from saved run')
            if args.resume:
                if saved['provenance_spec']['source_sha256'] != prov['source_sha256']:
                    raise ValueError('resume source code differs from saved run')
                epoch = saved['resume_state']['epoch']
                # Reject rollback even if a newer pair committed before latest was published.
                committed = [p for p in (self.directory/'checkpoints').glob('epoch_*.json') if '.metadata.' not in p.name]
                if epoch != max(int(p.stem.split('_')[1]) for p in committed):
                    raise ValueError('resume requires the newest committed epoch in this run')
                state = saved['resume_state']
                expected_steps = epoch*self.scheduler_plan['steps_per_epoch']
                if state['global_step'] != epoch*self.optimizer_steps or unpack_state(state['scheduler'])['last_epoch'] != expected_steps:
                    raise ValueError('resume optimizer/scheduler progress mismatch')
                if unpack_state(state['sampler_state']) != self.sampler:
                    raise ValueError('sampler state mismatch')
                optim_state = unpack_state(state['optimizer'])
                scheduler_state = unpack_state(state['scheduler'])
                if scheduler_state.get('_step_count') != expected_steps+1:
                    raise ValueError('resume scheduler step counter mismatch')
                loop_checkpoint.validate_optimizer_state(optim_state,optimizer,model=self.model) if v5 else loop_checkpoint.validate_optimizer_state(optim_state,optimizer)
                for group, current in zip(optim_state['param_groups'],optimizer.param_groups):
                    if len(group['params']) != len(current['params']):
                        raise ValueError('resume optimizer group shape mismatch')
                    for index,p in zip(group['params'],current['params']):
                        for name,value in optim_state['state'].get(index,{}).items():
                            if name in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and (value.shape!=p.shape or value.dtype!=p.dtype):
                                raise ValueError('resume optimizer moment shape/dtype mismatch')
                strict_load(self.model,weights)
                optimizer.load_state_dict(optim_state)
                scheduler.load_state_dict(scheduler_state)
                self.start_epoch = self.completed_epoch = epoch
                restore_random_state(state,self.generators)  # restore LAST, after construction/loading
            self.metadata = saved
        else:
            state = (v3_resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,
                                      self.generators,self.sampler,scaler=None) if v3 else
                     v4_resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,
                                      self.generators,self.sampler) if v4 else
                     v5_resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,
                                      self.generators,self.sampler) if v5 else
                     resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,
                                  self.generators,self.sampler))
            sections = dict(data_spec=data, provenance_spec=prov, normalizer_spec=normalizers,
                            resume_state=state, ensemble_manifest=[])
            if v3 or v4 or v5:
                sections['parameter_measurement'] = measure_parameters(self.model, args._linearno_loop_config)
            self.metadata = make_metadata(args._linearno_loop_config, **sections)
            write_metadata(self.directory/'architecture.json',self.metadata)
        self.recorder.update_protocol(dict(linearno_data=data, scheduler_resolved=self.scheduler_plan))
        if hasattr(args,'_linearno_metadata'):
            del args._linearno_metadata  # do not dump optimizer/RNG base64 via the legacy print(args)


    def load(self, model):
        from cdlno.linearno.checkpoint import strict_load
        from .versioning import checkpoint_api
        _,weights=checkpoint_api(self.args._linearno_loop_config).read_pair(
            self.args._linearno_checkpoint,expected=self.args._linearno_loop_config)
        strict_load(model,weights)

    def save(self, model):
        v3 = is_v3(self.args._linearno_loop_config)
        v4 = is_v4(self.args._linearno_loop_config)
        v5 = is_v5(self.args._linearno_loop_config)
        if v3:
            from cdlno.linearno_loop.v3.checkpoint import resume_state as v3_resume_state
            from linearno_loop.v3.contracts import seal
        elif v4:
            from cdlno.linearno_loop.v4.checkpoint import resume_state as v4_resume_state
        elif v5:
            from cdlno.linearno_loop.v5.checkpoint import resume_state as v5_resume_state
        else:
            from cdlno.linearno.checkpoint import resume_state
            from linearno_loop.contracts import seal
        from .versioning import checkpoint_api
        if self.args.eval:raise ValueError('evaluation cannot save training weights')
        if self.scheduler.last_epoch != self.completed_epoch*self.scheduler_plan['steps_per_epoch']:
            raise ValueError('actual scheduler cadence differs from metadata')
        metadata=copy.deepcopy(self.metadata)
        metadata['resume_state']=(v3_resume_state(self.optimizer,self.scheduler,self.completed_epoch,
            self.optimizer_steps,self.args.epochs,self.generators,self.sampler,scaler=None) if v3 else
            v4_resume_state(self.optimizer,self.scheduler,self.completed_epoch,self.optimizer_steps,
            self.args.epochs,self.generators,self.sampler) if v4 else
            v5_resume_state(self.optimizer,self.scheduler,self.completed_epoch,self.optimizer_steps,
            self.args.epochs,self.generators,self.sampler) if v5 else resume_state(self.optimizer,self.scheduler,self.completed_epoch,
            self.optimizer_steps,self.args.epochs,self.generators,self.sampler))
        if v3:
            # The initial V3 metadata already measured the constructed model;
            # retain that immutable record while only advancing resume state.
            return checkpoint_api(self.args._linearno_loop_config).save_pair(
                self.directory,model,seal(metadata,'metadata_hash'))
        if v4:
            metadata['metadata_hash']=__import__('linearno_loop.v4.contracts',fromlist=['digest']).digest(
                {k:v for k,v in metadata.items() if k!='metadata_hash'})
            return checkpoint_api(self.args._linearno_loop_config).save_pair(self.directory,model,metadata)
        if v5:
            metadata['metadata_hash']=__import__('linearno_loop.v5.contracts',fromlist=['digest']).digest(
                {k:v for k,v in metadata.items() if k!='metadata_hash'})
            return checkpoint_api(self.args._linearno_loop_config).save_pair(self.directory,model,metadata)
        return checkpoint_api(self.args._linearno_loop_config).save_pair(
            self.directory,model,seal(metadata,'metadata_hash'))
