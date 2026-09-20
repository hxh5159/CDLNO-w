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
    model_kwargs, normalizer, verify_data, start, finish,
)
from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES
from cdlno.linearno.schema import normalizer_record, unpack_state
from linearno_loop.config import resolve_config, run_directory_id
from linearno_loop.contracts import CLI_CONTRACT, HISTORY_FLAGS, OPTIONS, PRESETS, RESIDUAL_MODES, TOPOLOGY_FIELDS
from linearno_loop.schema import read_metadata, restore_config, make_metadata, write_metadata


def _loop_parser():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for flag,spec in CLI_CONTRACT.items():
        if flag=='--linearno-rank':continue
        kw=dict(dest=spec['field'],default=argparse.SUPPRESS)
        if spec['field']=='linearno_loop':kw['choices']=('0','1')
        elif spec['field']=='topology_preset':kw['choices']=(*PRESETS,'custom')
        elif spec['field']=='residual_mode':kw['choices']=RESIDUAL_MODES
        else:kw['type']=int
        parser.add_argument(flag,**kw)
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
        options={k:v for k,v in loop_explicit.items() if k in OPTIONS}
        if 'linearno_rank' in explicit:options['linearno_rank']=explicit['linearno_rank']
        if 'linearno_rank' in options and 'rank_multiplier' in options:raise ValueError('explicit actual rank/multiplier conflict')
        if options.get('topology_preset') in PRESETS and set(options)&set(TOPOLOGY_FIELDS):
            raise ValueError('preset and custom P/C/R/S fields are mutually exclusive')
        overrides={path:explicit[name] for name,path in ARG_FIELDS.items()
                   if name in explicit and name not in ('n_layers','linearno_rank')}
        if 'model.unified_pos' in overrides:
            if overrides['model.unified_pos'] not in (0,1):raise ValueError('unified_pos must be 0 or 1')
            overrides['model.unified_pos']=bool(overrides['model.unified_pos'])
        if args.eval or args.resume:
            from .checkpoint import inspect_checkpoint, immutable_equal
            if args.linearno_run_dir is None:raise ValueError('loop eval/resume requires explicit --experiment-dir')
            initial=read_metadata(args.linearno_run_dir/'architecture.json')
            asserted=dict(options,task=task)
            if 'linearno_loop' in loop_explicit:asserted['linearno_loop']=loop_explicit['linearno_loop']
            if 'linearno_profile' in explicit:asserted['profile']=explicit['linearno_profile']
            config=restore_config(initial,explicit=asserted)['config']
            for path,value in overrides.items():
                section,field=path.split('.')
                if config['profile_spec']['values'][section][field]!=value:
                    raise ValueError(f'explicit {path} conflicts with checkpoint')
            metadata,checkpoint=inspect_checkpoint(args.linearno_run_dir,args.checkpoint or ('latest' if args.resume else 'final'),expected=config)
            immutable_equal(initial,metadata)
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
        for name,path in ARG_FIELDS.items():
            section,field=path.split('.');setattr(args,name,base['values'][section][field])
        args.n_layers=s['unique_depth'];args.linearno_rank=s['resolved_rank'];args.linearno_profile=base['profile']
        identifier=run_directory_id(config)
        args.save_name=args.save_name or identifier
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.save_name) or args.save_name in ('.','..'):
            raise ValueError('save_name must be a filename stem')
        if args.linearno_run_dir is None:
            from cdlno.experiment import timestamp
            args.linearno_run_dir=Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output'))/task/'linearno_loop'/(identifier+'__'+timestamp()+'_'+uuid4().hex[:8])
        if identifier not in args.linearno_run_dir.name:raise ValueError('loop run directory must contain '+identifier)
        args.linearno_run_dir=args.linearno_run_dir.resolve()
        if not (args.eval or args.resume) and args.linearno_run_dir.exists():raise ValueError('new loop run directory already exists')
        return args
    except (ValueError,OSError,KeyError,TypeError) as error:parser.error(str(error))


def model_module(args):
    config=args._linearno_loop_config
    def construct(**kwargs):
        if kwargs!=config['model_spec']['constructor_kwargs']:raise ValueError('loop factory kwargs mismatch')
        from .construction import build_from_config
        return build_from_config(config)
    return SimpleNamespace(Model=construct)


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
        from cdlno.linearno.checkpoint import resume_state, strict_load, restore_random_state
        from .checkpoint import read_pair, inspect_checkpoint
        from cdlno.training_state import _optimizer_signature
        self.optimizer, self.scheduler = optimizer, scheduler
        self.steps = len(train_loader)
        self.updates_per_batch = 20 if self.args.linearno_task == 'plasticity' else 1
        self.optimizer_steps = self.steps * self.updates_per_batch
        self.generators = {name: loader.generator for name,loader in (('train',train_loader),('test',test_loader)) if loader.generator is not None}
        self.sampler = {name: dict(type=type(loader.sampler).__name__, epoch_boundary=True,
                                  replacement=getattr(loader.sampler,'replacement',False))
                        for name,loader in (('train',train_loader),('test',test_loader))}
        args = self.args; cfg = args._linearno_config
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
        from .provenance import provenance
        from . import checkpoint as loop_checkpoint
        prov = provenance()
        if args.eval or args.resume:
            saved, path = inspect_checkpoint(self.directory,args._linearno_checkpoint.stem,expected=args._linearno_loop_config)
            _, weights = read_pair(path,expected=args._linearno_loop_config)
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
                loop_checkpoint.validate_optimizer_state(optim_state,optimizer)
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
            self.metadata = make_metadata(args._linearno_loop_config,data_spec=data,
                provenance_spec=prov,normalizer_spec=normalizers,
                resume_state=resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,self.generators,self.sampler),
                ensemble_manifest=[])
            write_metadata(self.directory/'architecture.json',self.metadata)
        self.recorder.update_protocol(dict(linearno_data=data, scheduler_resolved=self.scheduler_plan))
        if hasattr(args,'_linearno_metadata'):
            del args._linearno_metadata  # do not dump optimizer/RNG base64 via the legacy print(args)


    def load(self, model):
        from .checkpoint import read_pair, strict_load
        _,weights=read_pair(self.args._linearno_checkpoint,expected=self.args._linearno_loop_config)
        strict_load(model,weights)

    def save(self, model):
        from .checkpoint import resume_state,save_pair
        from linearno_loop.contracts import seal
        if self.args.eval:raise ValueError('evaluation cannot save training weights')
        if self.scheduler.last_epoch != self.completed_epoch*self.scheduler_plan['steps_per_epoch']:
            raise ValueError('actual scheduler cadence differs from metadata')
        metadata=copy.deepcopy(self.metadata)
        metadata['resume_state']=resume_state(self.optimizer,self.scheduler,self.completed_epoch,
            self.optimizer_steps,self.args.epochs,self.generators,self.sampler)
        return save_pair(self.directory,model,seal(metadata,'metadata_hash'))
