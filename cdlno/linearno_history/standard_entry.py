"""History Standard routing. Native task/data/loss/observer code stays unchanged."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from types import SimpleNamespace

from cdlno.linearno.standard_entry import (ROOT, KEYS, ARG_FIELDS, model_kwargs, start, finish,
    normalizer, verify_data, constructor_kwargs, StandardRun as PureStandardRun,
    provenance as baseline_provenance)
from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES, resolve_config, digest
from cdlno.linearno.schema import normalizer_record, unpack_state
from .config import resolve_config as resolve_history
from .factory import build_fair_model
from .cli import take_features, run_family
from . import checkpoint as history_checkpoint
from linearno_history.schema import FEATURE_FIELDS

TASKS = ('darcy','elasticity','airfoil','pipe','ns','plasticity')


def intercept(parser, task, tokens, selected_model):
    from .fair_run import take_fair, attach
    try:
        fair, tokens=take_fair(tokens)
        explicit,remaining=take_features(tokens)
        if selected_model not in KEYS:
            if explicit or fair is not None:parser.error('LinearNO history/fair-run flags require a LinearNO model')
            return None
        history_run=run_family(remaining)=='linearno_history'
        if not explicit and not history_run and fair is None:return None
        if task not in TASKS:parser.error('history extensions are not integrated for this task')
        directory_explicit=any(t.split('=')[0] in ('--experiment-dir','--linearno-run-dir') for t in remaining)
        if not history_run and not any(explicit.get(key,False) for key in FEATURE_FIELDS[:2]):
            from .config import DEFAULT_FEATURES
            from linearno_history.schema import resolve_feature_config
            resolve_feature_config({**DEFAULT_FEATURES,**explicit})
            from cdlno.linearno.standard_entry import parse_args
            args=parse_args(parser,task,remaining)
            return attach(args,fair,directory_was_explicit=directory_explicit)
        if fair is False:parser.error('research comparisons require the independent DataLoader generator protocol')
        return attach(parse_research_args(parser,task,remaining,explicit),True)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))


def model_module(args):
    config=args._linearno_history_config
    def construct(**kwargs):
        if kwargs != config['model_spec']['constructor_kwargs']:
            raise ValueError('factory kwargs conflict with resolved research configuration')
        return build_fair_model(config)
    return SimpleNamespace(Model=construct)


def select_run(args, model, baseline=PureStandardRun):
    cls=HistoryStandardRun if getattr(args,'linearno_family',None)=='linearno_history' else (FairBaselineRun if getattr(args,'_linearno_fair_run',False) else baseline)
    return cls(args,model)


def provenance():
    from .provenance import research_provenance
    return research_provenance(baseline_provenance())


def parse_research_args(parser, task, tokens, explicit_features):
    if task not in TASKS:
        parser.error('LinearNO Standard adapter supports only the six Standard tasks')
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('plain','temp','conv','conv_temp'), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-rank', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default=None, help='final, latest or epoch_XXXX within the explicit run')
    args = parser.parse_args(tokens)
    probe = copy.deepcopy(parser); probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    explicit = vars(probe.parse_args(tokens))
    try:
        expected_key = KEYS[1] if task == 'elasticity' else KEYS[0]
        if args.model != expected_key:
            raise ValueError(f'{task} requires --model {expected_key}')
        if 'slice_num' in explicit:
            raise ValueError('LinearNO rank requires --linearno-rank, not --slice_num')
        if args.eval not in (0,1) or (args.eval and args.resume):
            raise ValueError('choose exactly train, resume or eval')
        if args.checkpoint and not (args.eval or args.resume):
            raise ValueError('--checkpoint requires eval or resume')
        for name, expected in dict(downsample=1 if task == 'ns' else 5, downsamplex=1, downsampley=1).items():
            if hasattr(args,name) and getattr(args,name) != expected:
                raise ValueError(f'LinearNO task grid/sampling requires {name}={expected}')
        overrides = {path: explicit[name] for name,path in ARG_FIELDS.items() if name in explicit}
        if 'model.unified_pos' in overrides:
            if overrides['model.unified_pos'] not in (0,1):
                raise ValueError('unified_pos must be 0 or 1')
            overrides['model.unified_pos'] = bool(overrides['model.unified_pos'])
        profile = getattr(args,'linearno_profile',DEFAULT_PROFILE)
        if args.eval or args.resume:
            from .checkpoint import inspect_checkpoint
            if args.linearno_run_dir is None:
                raise ValueError('eval/resume requires an explicit --experiment-dir')
            selector = args.checkpoint or ('latest' if args.resume else 'final')
            metadata, checkpoint = inspect_checkpoint(args.linearno_run_dir, selector)
            if metadata['family'] != 'linearno_history' or metadata['profile_spec']['task'] != task:
                raise ValueError('checkpoint research family/task mismatch')
            for key,value in explicit_features.items():
                if value != metadata['resolved_config']['features'][key]:
                    raise ValueError(f'explicit {key} conflicts with checkpoint')
            initial = history_checkpoint.validate_metadata(history_checkpoint.read_json(args.linearno_run_dir/'architecture.json'))
            if any(initial[k] != metadata[k] for k in ('model_spec','profile_spec','data_spec','normalizer_spec','provenance_spec')):
                raise ValueError('checkpoint differs from immutable run metadata')
            config = metadata['profile_spec']
            if 'linearno_profile' in explicit and profile != config['profile']:
                raise ValueError('explicit profile conflicts with checkpoint')
            for path, value in overrides.items():
                section, field = path.split('.')
                if section != 'runtime' and config['values'][section][field] != value:
                    raise ValueError(f'explicit {path} conflicts with checkpoint')
                if args.resume and section == 'runtime' and config['values'][section][field] != value:
                    raise ValueError(f'resume {path} conflicts with checkpoint')
            if args.resume and metadata['resume_state']['epoch'] >= config['values']['training']['epochs']:
                raise ValueError('run has completed its resolved epochs')
            args._linearno_metadata, args._linearno_checkpoint = metadata, checkpoint
        else:
            contract = 'standard_temporal_l5' if task in ('ns', 'plasticity') else 'standard_static_l4'
            config = resolve_config(task, profile, explicit=overrides, contract=contract)
        if task == 'darcy' and config['profile'] == 'official_release' and config['values']['training']['epochs'] != 500:
            raise ValueError('Darcy official_release fixes scheduler epochs=500; non-500 training epochs are rejected')
        args.linearno_family, args.linearno_task = 'linearno_history', task
        history = (metadata['resolved_config'] if args.eval or args.resume else
                   resolve_history(config, family='linearno_history', features=explicit_features))
        if history['family'] != 'linearno_history':
            raise ValueError('research routing requires an enabled innovation')
        args._linearno_history_config = history
        for key in FEATURE_FIELDS: setattr(args,key,history['features'][key])
        args._linearno_config = config
        args._linearno_model_spec = history['model_spec']
        args._linearno_normalizers = {}
        for name,path in ARG_FIELDS.items():
            section,field = path.split('.')
            setattr(args,name,config['values'][section][field])
        args.linearno_profile = config['profile']
        if args.save_name is None:
            args.save_name = history['run_signature']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.save_name) or args.save_name in ('.','..'):
            raise ValueError('save_name must be a filename stem')
        if args.linearno_run_dir is None:
            from cdlno.experiment import timestamp
            base = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT/'output'))
            args.linearno_run_dir = base/task/'linearno_history'/f'{history["run_signature"]}__{timestamp()}_{uuid4().hex[:8]}'
        if history['run_signature'] not in args.linearno_run_dir.name:
            raise ValueError('research run directory must contain '+history['run_signature'])
        args.linearno_run_dir = args.linearno_run_dir.resolve()
        return args
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))



class FairBaselineRun(PureStandardRun):
    def __init__(self,args,model):
        super().__init__(args,model)
        from .fair_run import record
        record(args,model)

    def prepare(self,optimizer,scheduler,train_loader,test_loader):
        from .fair_run import prepare_loaders
        prepare_loaders(self.args,train_loader,test_loader)
        return super().prepare(optimizer,scheduler,train_loader,test_loader)


class HistoryStandardRun(FairBaselineRun):
    def prepare(self, optimizer, scheduler, train_loader, test_loader):
        from .fair_run import prepare_loaders
        prepare_loaders(self.args,train_loader,test_loader)
        from cdlno.linearno.checkpoint import resume_state, strict_load, restore_random_state
        from .checkpoint import _read_pair, inspect_checkpoint
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
        prov = provenance()
        if args.eval or args.resume:
            saved, path = inspect_checkpoint(self.directory,args._linearno_checkpoint.stem)
            weights = _read_pair(path,saved)
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
            self.metadata = history_checkpoint.make_metadata(args._linearno_history_config,data_spec=data,
                objective_spec=cfg['values']['objective'],evaluation_spec=cfg['values']['evaluation'],
                provenance_spec=prov,normalizer_spec=normalizers,
                resume_state=resume_state(optimizer,scheduler,0,self.optimizer_steps,args.epochs,self.generators,self.sampler),
                ensemble_manifest=[])
            with (self.directory/'architecture.json').open('x') as stream:
                json.dump(self.metadata,stream,sort_keys=True)
        self.recorder.update_protocol(dict(linearno_data=data, scheduler_resolved=self.scheduler_plan))
        if hasattr(args,'_linearno_metadata'):
            del args._linearno_metadata  # do not dump optimizer/RNG base64 via the legacy print(args)


    def load(self, model):
        from cdlno.linearno.checkpoint import strict_load
        saved,path=history_checkpoint.inspect_checkpoint(self.directory,self.args._linearno_checkpoint.stem,
            expected=self.args._linearno_history_config)
        strict_load(model,history_checkpoint._read_pair(path,saved))

    def save(self, model):
        from cdlno.linearno.checkpoint import resume_state
        if self.args.eval:raise ValueError('evaluation cannot save training weights')
        if self.scheduler.last_epoch != self.completed_epoch*self.scheduler_plan['steps_per_epoch']:
            raise ValueError('actual scheduler cadence differs from metadata')
        metadata=copy.deepcopy(self.metadata)
        metadata['resume_state']=resume_state(self.optimizer,self.scheduler,self.completed_epoch,
            self.optimizer_steps,self.args.epochs,self.generators,self.sampler)
        return history_checkpoint.save_checkpoint(self.directory,model,history_checkpoint.rehash(metadata))
