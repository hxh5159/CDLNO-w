"""Research industrial adapter. Native protocol copied at R8 with audited routing only."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4
import numpy as np
import torch
from cdlno.linearno.airfrans import AirfRANSLinearNO
from cdlno.linearno.checkpoint import (inspect_checkpoint, read_pair, restore_random_state, resume_state,
                         save_pair, strict_load)
from cdlno.linearno.profiles import PROFILES, DEFAULT_PROFILE, digest, resolve_config, require_resolved_objective
from cdlno.linearno.schema import (make_metadata, normalizer_record, numerical_state,
                     read_metadata, restore_numerical_state, write_metadata,
                     unpack_state)
from cdlno.linearno.air_entry import _has, _value, _data_root, _constructor_kwargs, _profile_explicit, _set_resolved_args, _sha256, _provenance, _hparams, _normalizer_spec, _data_spec, _json_history, _restore_coef_norm, _load_dataset, pressure_relative_l2, _pressure_rL2_dataset
from .industrial import (finish_args, check_structure, read_metadata, write_metadata, make_metadata, inspect_checkpoint, read_pair, save_pair, construct, generators, verify_resume_source)
from .factory import build_model
from .checkpoint import config_from_metadata
from .provenance import research_provenance
ROOT = Path(__file__).resolve().parents[2]

def check_structure(metadata):
    from .checkpoint import validate_metadata
    validate_metadata(metadata)
    if metadata['profile_spec']['task'] != 'airfrans':
        raise ValueError('AirfRANS checkpoint task mismatch')

def parse_args(parser, tokens, *, evaluation=False, history_features=None):
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('airfrans',), default=argparse.SUPPRESS)
    parser.add_argument('--linearno-rank', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-hidden', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-layers', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-heads', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-ffn-ratio', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-dropout', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-ref', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-unified-pos', type=int, choices=(0, 1), default=argparse.SUPPRESS)
    parser.add_argument('--subsampling', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--r', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--max-neighbors', '--max_neighbors', dest='max_neighbors', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--debug', type=int, choices=(0,), default=argparse.SUPPRESS)
    # These aliases belong only to the LinearNO branch.  The historical AirfRANS
    # parser keeps its original --nb_epochs/--batch_size/--score interface.
    parser.add_argument('--epochs', dest='nb_epochs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--save-name', '--save_name', dest='save_name', default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir',
                        type=Path, default=argparse.SUPPRESS)
    parser.add_argument('--eval', type=int, choices=(0, 1), default=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--checkpoint', default='final')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args(tokens)
    # main_evaluation.py has no --eval flag; its evaluation=True call is the
    # authoritative action selector.  The training entry defaults to train.
    args.eval = 1 if evaluation else int(getattr(args, 'eval', 0))
    # Resume is a continuation action; when the user did not explicitly pick
    # a checkpoint, consume the committed latest pointer instead of looking for
    # the final archive first.  Evaluation keeps its final-checkpoint default.
    if args.resume and not _has(tokens, '--checkpoint'):
        args.checkpoint = 'latest'
    if not hasattr(args, 'save_name'):
        args.save_name = None
    explicit = _profile_explicit(args, tokens)
    run_dir = getattr(args, 'linearno_run_dir', None) or getattr(args, 'run_dir', None)
    profile = getattr(args, 'linearno_profile', DEFAULT_PROFILE)
    if getattr(args, 'task', 'full') not in ('full', 'scarce', 'reynolds', 'aoa'):
        parser.error('LinearNO AirfRANS task must be full, scarce, reynolds or aoa')
    if args.eval and args.resume:
        parser.error('evaluation and resume are separate actions')
    if args.eval or args.resume:
        if run_dir is None:
            parser.error('LinearNO evaluation/resume requires --experiment-dir')
        metadata_path = Path(run_dir).resolve() / 'architecture.json'
        try:
            metadata = read_metadata(metadata_path, constructor=AirfRANSLinearNO)
            check_structure(metadata)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        saved = metadata['profile_spec']
        if 'linearno_profile' in vars(args) and profile != saved['profile']:
            parser.error('explicit --linearno-profile conflicts with saved metadata')
        for path, value in explicit.items():
            section, field = path.split('.')
            if saved['values'][section][field] != value:
                parser.error(f'explicit {path} conflicts with saved metadata')
        config = saved
        args._linearno_metadata = metadata
        saved_task = metadata['data_spec']['task_variant']
        if _has(tokens, '--task', '-t') and args.task != saved_task:
            parser.error('explicit task conflicts with saved data split')
        args.task = saved_task
        try:
            _, checkpoint_path = inspect_checkpoint(Path(run_dir) / 'member_000',
                                                     args.checkpoint, AirfRANSLinearNO)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        args._linearno_checkpoint = checkpoint_path
    else:
        try:
            config = resolve_config('airfrans', profile, explicit=explicit)
        except ValueError as error:
            parser.error(str(error))
    try:
        require_resolved_objective(config)
        if config['values']['training']['batch_size'] != 1:
            raise ValueError('AirfRANS LinearNO supports batch_size=1 only')
        if args.resume and config['values']['training']['nmodel'] > 1 and args.checkpoint != 'latest':
            raise ValueError('ensemble resume requires --checkpoint latest')
    except ValueError as error:
        parser.error(str(error))
    _set_resolved_args(args, config)
    args.linearno_run_dir = Path(run_dir).resolve() if run_dir is not None else None
    if args.linearno_run_dir is None:
        root = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output'))
        leaf = args.save_name or f'seed{args.seed}_{uuid4().hex[:10]}'
        args.linearno_run_dir = (root / 'airfrans' / 'linearno' / config['profile'] /
                                 f"{config['values']['model']['linearno_variant']}_M{config['values']['model']['linearno_rank']}_eval{digest(config['values']['evaluation'])[:10]}" /
                                 leaf).resolve()
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if args.gpu is not None:
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.data_path = args.my_path
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return finish_args(args, history_features or {}, tokens)

class AirRun:
    """One AirfRANS run; ``member`` state lives in its own strict pair tree."""
    def __init__(self, args, config, data_dir, coef_norm, recorder, *, manifest=None, evaluation=False):
        self.args, self.config, self.data_dir, self.coef_norm = args, config, Path(data_dir), coef_norm
        self.recorder, self.evaluation = recorder, evaluation
        self.directory = Path(args.linearno_run_dir).resolve()
        self.manifest = manifest if manifest is not None else self._read_manifest()
        self.data = copy.deepcopy(_data_spec(self.data_dir, self.manifest, config, args.task))
        checksum = digest(self.data['checksums'])
        self.data['checksums']['normalizer_fit_dataset'] = checksum
        self.normalizers = _normalizer_spec(coef_norm, checksum)
        self.provenance = research_provenance(_provenance()) if args.linearno_family == 'linearno_history' else _provenance()
        self.generators = generators(args)
        self.current_member = 0
        self._member_metadata = None
        self.completed_epoch = 0
        self.steps_per_epoch = None

        if (evaluation or getattr(args, 'resume', False)) and hasattr(args, '_linearno_metadata'):
            saved = args._linearno_metadata
            if saved['data_spec'] != self.data:
                raise ValueError('AirfRANS data manifest/split differs from saved metadata')
            if saved['normalizer_spec'] != self.normalizers:
                raise ValueError('AirfRANS normalizer differs from saved train-fit metadata')

    def _read_manifest(self):
        with (self.data_dir / 'manifest.json').open() as stream:
            return json.load(stream)

    @property
    def model_spec(self):
        return self.args._linearno_model_spec

    @property
    def member_dir(self):
        return self.directory / f'member_{self.current_member:03d}'

    def _resume_zero(self, optimizer, scheduler):
        return resume_state(optimizer, scheduler, 0,
                            self.steps_per_epoch, self.config['values']['training']['epochs'], self.generators, {})

    def _metadata(self, resume):
        return make_metadata(profile_spec=self.config, constructor=AirfRANSLinearNO,
            model_spec=self.model_spec,
            data_spec=self.data, objective_spec=self.config['values']['objective'],
            evaluation_spec=self.config['values']['evaluation'], provenance_spec=self.provenance,
            normalizer_spec=self.normalizers, resume_state=resume, ensemble_manifest=[])

    def _write_root_metadata(self, optimizer, scheduler):
        path = self.directory / 'architecture.json'
        if path.exists():
            saved = read_metadata(path, constructor=AirfRANSLinearNO)
            if saved['model_spec'] != self.model_spec or saved['profile_spec'] != self.config:
                raise ValueError('AirfRANS run metadata structure/profile mismatch')
            if saved['data_spec'] != self.data or saved['normalizer_spec'] != self.normalizers:
                raise ValueError('AirfRANS run metadata data/normalizer mismatch')
            return saved
        metadata = self._metadata(self._resume_zero(optimizer, scheduler))
        write_metadata(path, metadata, constructor=AirfRANSLinearNO)
        return metadata

    def prepare(self, model, optimizer, scheduler, train_dataset, val_dataset, criterion, reg, val_iter, val_sample):
        expected = self.config['values']['objective']
        if criterion != 'MSE_weighted' or expected['kind'] != 'MSE' or expected['space'] != 'normalized' or reg != expected['surface_weight']:
            raise ValueError('AirfRANS actual MSE criterion/weight differs from metadata')
        if len(train_dataset) < 1:
            raise ValueError('AirfRANS LinearNO requires at least one training sample')
        # The optimizer/scheduler are owned by the native train loop.  Binding
        # them here makes each checkpoint self-contained and reentrant.
        self.bind(optimizer, scheduler)
        actual_steps = (len(train_dataset) + self.config['values']['training']['batch_size'] - 1) // self.config['values']['training']['batch_size']
        if actual_steps != self.data['steps_per_epoch']:
            raise ValueError('AirfRANS loader length differs from metadata train split')
        self.steps_per_epoch = actual_steps
        self.completed_epoch = 0
        self._write_root_metadata(optimizer, scheduler)
        self.member_dir.mkdir(parents=True, exist_ok=True)
        self._member_metadata = self._metadata(self._resume_zero(optimizer, scheduler))
        if not self.args.resume:
            return 0, None
        pointer = self.member_dir / 'checkpoints' / 'latest.json'
        if not pointer.is_file() and self.current_member > 0:
            # A sequential ensemble may be interrupted before this member starts.
            # The preceding member's committed final RNG is restored by run_cli.
            inspect_checkpoint(self.directory / f'member_{self.current_member-1:03d}', 'final', AirfRANSLinearNO)
            if any(self.member_dir.iterdir()):
                raise ValueError('unfinished member artifacts without a committed checkpoint')
            return 0, None
        _, manifest_path = inspect_checkpoint(self.member_dir, self.args.checkpoint, AirfRANSLinearNO)
        saved, weights = read_pair(manifest_path, AirfRANSLinearNO)
        if saved['model_spec'] != self.model_spec or saved['profile_spec'] != self.config:
            raise ValueError('AirfRANS checkpoint structure/profile mismatch')
        if saved['data_spec'] != self.data or saved['normalizer_spec'] != self.normalizers:
            raise ValueError('AirfRANS checkpoint data/normalizer mismatch')
        verify_resume_source(saved,self.provenance)
        strict_load(model, weights)
        state = unpack_state(saved['resume_state']['optimizer'])
        optimizer.load_state_dict(state)
        scheduler.load_state_dict(unpack_state(saved['resume_state']['scheduler']))
        restore_random_state(saved['resume_state'], self.generators)
        history = unpack_state(saved['resume_state']['sampler_state']).get('history')
        if history is None:
            # Narrow compatibility with L6 archives created before history was
            # bound to the committed epoch. Never consume later-epoch curves.
            history_path = self.member_dir / 'history.json'
            history = json.loads(history_path.read_text()) if history_path.is_file() else None
            if history is None or any(len(curve) != saved['resume_state']['epoch'] for curve in history['curves'][:4]):
                raise ValueError('legacy AirfRANS checkpoint lacks matching epoch history')
        self.completed_epoch = saved['resume_state']['epoch']
        expected_step = self.completed_epoch * self.steps_per_epoch
        saved_scheduler = unpack_state(saved['resume_state']['scheduler'])
        if saved['resume_state']['global_step'] != expected_step:
            raise ValueError('AirfRANS checkpoint global_step does not match resolved loader steps')
        if saved_scheduler.get('last_epoch') != expected_step:
            raise ValueError('AirfRANS checkpoint scheduler progress mismatch')
        return self.completed_epoch, history

    def complete_epoch(self, epoch, model, history):
        if self.steps_per_epoch is None:
            raise RuntimeError('prepare must bind optimizer and scheduler before checkpointing')
        if epoch != self.completed_epoch + 1:
            raise ValueError(f'AirfRANS checkpoint epoch order differs: expected {self.completed_epoch + 1}, got {epoch}')
        self.completed_epoch = epoch
        self._member_metadata = self._metadata(resume_state(
            self.optimizer, self.scheduler, epoch, self.steps_per_epoch,
            self.config['values']['training']['epochs'], self.generators, {'history': _json_history(history)}))
        save_pair(self.member_dir, model, self._member_metadata, AirfRANSLinearNO)
        serial = _json_history(history)
        (self.member_dir / 'history.json').write_text(json.dumps(serial, allow_nan=False))

    def loader_kwargs(self, split):
        return {'generator': self.generators[split]}

    def bind(self, optimizer, scheduler):
        self.optimizer, self.scheduler = optimizer, scheduler

    def load_models(self):
        ensemble_path = self.directory / 'ensemble.json'
        if not ensemble_path.is_file():
            raise ValueError('missing LinearNO ensemble manifest')
        ensemble = json.loads(ensemble_path.read_text())
        members = ensemble.get('members') if isinstance(ensemble, dict) else None
        if (ensemble.get('family') != self.args.linearno_family or ensemble.get('checkpoint_role') != 'final'
                or not isinstance(members, list)
                or len(members) != self.config['values']['training']['nmodel']):
            raise ValueError('LinearNO ensemble manifest family/count mismatch')
        for index, row in enumerate(members):
            if (row.get('member_id') != f'member_{index:03d}' or row.get('order') != index
                    or row.get('format') != 'state_dict' or not isinstance(row.get('path'), str)):
                raise ValueError('LinearNO ensemble manifest member order mismatch')
            path = (self.directory / row['path']).resolve()
            if path.parent != self.directory / f'member_{index:03d}' / 'weights' or not path.is_file():
                raise ValueError('LinearNO ensemble manifest member path mismatch')
            if row.get('sha256') != _sha256(path):
                raise ValueError('LinearNO ensemble manifest member checksum mismatch')
        models = []
        for index in range(self.config['values']['training']['nmodel']):
            self.current_member = index
            member = self.member_dir
            _, manifest_path = inspect_checkpoint(member, self.args.checkpoint, AirfRANSLinearNO)
            pair = json.loads(manifest_path.read_text())
            if (self.args.checkpoint == 'final' and
                    (member / pair['weights']['path']).resolve() != (self.directory / members[index]['path']).resolve()):
                raise ValueError('ensemble manifest does not identify the selected final weights')
            metadata, weights = read_pair(manifest_path, AirfRANSLinearNO)
            check_structure(metadata)
            if (metadata['model_spec'] != self.model_spec or
                    metadata['profile_spec'] != self.config or
                    metadata['data_spec'] != self.data or
                    metadata['normalizer_spec'] != self.normalizers):
                raise ValueError('LinearNO checkpoint structure/profile mismatch')
            model = build_model(config_from_metadata(metadata))
            strict_load(model, weights)
            models.append(model.to(self.args.device).eval())
        if not self.evaluation:
            self._write_ensemble_manifest(models)
        return models

    def _write_ensemble_manifest(self, models):
        manifest = []
        for index in range(len(models)):
            member = self.directory / f'member_{index:03d}'
            pointer = member / 'checkpoints' / 'final.json'
            if not pointer.is_file():
                raise ValueError(f'missing final checkpoint for ensemble member {index}')
            pointer_data = json.loads(pointer.read_text())
            manifest_data = json.loads((member / 'checkpoints' / pointer_data['manifest']).read_text())
            weights = (member / manifest_data['weights']['path']).resolve()
            if weights.parent != member / 'weights':
                raise ValueError('ensemble state_dict must remain under the member weights directory')
            manifest.append(dict(member_id=f'member_{index:03d}', order=index,
                                 path=str(weights.relative_to(self.directory)),
                                 sha256=_sha256(weights), format='state_dict'))
        (self.directory / 'ensemble.json').write_text(json.dumps(
            dict(family=self.args.linearno_family, checkpoint_role='final', members=manifest), indent=2) + '\n')

    def finish_ensemble(self, models):
        states = [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in models]
        torch.save(states, self.directory / 'ensemble_state_dict.pth')
        self._write_ensemble_manifest(models)

    def eval_dir(self):
        if self.evaluation and self.recorder is not None:
            return Path(self.recorder.result_dir)
        path = self.directory / 'evaluations' / ('eval_' + uuid4().hex[:12])
        path.mkdir(parents=True, exist_ok=False)
        return path

def run_cli(args):
    """Run the original AirfRANS dataset/training/eval actions for LinearNO."""
    from cdlno.experiment import finish, start
    import train as air_train
    import utils.metrics as metrics

    config = args._linearno_config
    hparams = _hparams(config)
    recorder = start(args, 'airfrans', evaluation=bool(args.eval), hparams=hparams)
    data_dir = _data_root(args.my_path, evaluation=bool(args.eval))
    saved_coef_norm = _restore_coef_norm(getattr(args, '_linearno_metadata', None))
    if args.eval:
        manifest, coef_norm = _load_dataset(data_dir, args, train=False, coef_norm=saved_coef_norm)
    else:
        manifest, train_dataset, val_dataset, coef_norm = _load_dataset(
            data_dir, args, train=True, coef_norm=saved_coef_norm)
    if not args.eval:
        hparams['total_steps'] = ((len(train_dataset) // hparams['batch_size']) + 1) * hparams['nb_epochs']
    args._linearno_normalizers = {'coef_norm': coef_norm}
    run = AirRun(args, config, data_dir, coef_norm, recorder, manifest=manifest,
                 evaluation=bool(args.eval))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError(f'requested {device}, but CUDA is unavailable')
    if args.eval:
        models = run.load_models()
        result_dir = run.eval_dir()
        scores = metrics.Results_test(str(device), [models], [hparams], coef_norm, str(data_dir),
                                      str(result_dir), n_test=3, criterion='MSE',
                                      s=args.task + '_test' if args.task != 'scarce' else 'full_test')
        pressure = _pressure_rL2_dataset(str(device), models, hparams, coef_norm,
                                         data_dir, manifest[args.task + '_test'] if args.task != 'scarce'
                                         else manifest['full_test'])
        (result_dir / 'pressure_rL2.json').write_text(json.dumps(pressure, indent=2) + '\n')
        np.save(result_dir / 'true_coefs', scores[0]); np.save(result_dir / 'pred_coefs_mean', scores[1])
        np.save(result_dir / 'pred_coefs_std', scores[2])
        recorder.record_metrics(dict(result_dir=str(result_dir), field_metric='normalized four-channel MSE',
                                     pressure_field_metric='physical pressure-channel rL2 by volume/surface region',
                                     pressure_rL2=pressure,
                                     force_metric='relative coefficient error and Spearman', checkpoint=args.checkpoint))
        finish(args)
        return result_dir

    models = []
    for index in range(config['values']['training']['nmodel']):
        run.current_member = index
        member_pointer = run.member_dir / 'checkpoints' / 'latest.json'
        if args.resume and member_pointer.exists():
            saved, path = inspect_checkpoint(run.member_dir, args.checkpoint, AirfRANSLinearNO)
            check_structure(saved)
            if any(saved[k] != value for k, value in (
                    ('model_spec', run.model_spec), ('profile_spec', config),
                    ('data_spec', run.data), ('normalizer_spec', run.normalizers))):
                raise ValueError('AirfRANS resume member metadata mismatch')
            verify_resume_source(saved,run.provenance)
            if saved['resume_state']['checkpoint_role'] == 'final':
                model = build_model(config_from_metadata(saved)).to(device)
                _, weights = read_pair(path, AirfRANSLinearNO)
                strict_load(model, weights)
                restore_random_state(saved['resume_state'], run.generators)
                if not (run.member_dir / 'model').exists():
                    torch.save(model, run.member_dir / 'model')
                models.append(model)
                continue
        model = construct(args,index).to(device)
        recorder.attach_model(model, hparams=hparams, member=index, protocol=dict(
            family=args.linearno_family, profile=config['profile'], objective=config['values']['objective'],
            steps_per_epoch=len(train_dataset) // hparams['batch_size'] + 1,
            total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs']))
        from cdlno.linearno.air_visualization import AirFields
        if not hasattr(recorder, '_field_visualizers'):
            recorder._field_visualizers = {}
        recorder._field_visualizers[index] = AirFields(run.directory, 'airfrans', 'LinearNO',
                                                       seed=args.seed, member=index)
        # The original train function constructs Adam/OneCycle and preserves its
        # random sampling/validation cadence.  The adapter observes each epoch.
        model = air_train.main(str(device), train_dataset, val_dataset, model, hparams,
                               str(run.member_dir), criterion='MSE_weighted', reg=args.weight,
                               val_iter=10, name_mod='LinearNO', val_sample=True,
                               record=recorder, record_member=index, visualization_norm=coef_norm,
                               linearno_run=run)
        # Native final plots/pickle are observations, not the next member's RNG.
        saved, _ = inspect_checkpoint(run.member_dir, 'final', AirfRANSLinearNO)
        restore_random_state(saved['resume_state'], run.generators)
        models.append(model)
    run.finish_ensemble(models)
    finish(args)
    return run.directory
