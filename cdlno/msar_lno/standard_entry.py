"""MSAR standard task resolution, observation, and strict state_dict Run.

No data imports, resume, optimizer changes, or modifications to old selectors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import FAMILY, MSARArchitectureConfig, MSARTrainingConfig, MSARRuntimeConfig, input_layout
from .metadata import (MSARMetadata, MSARMetadataMismatch, load_metadata, save_metadata,
                       resolve_evaluation, new_run_path)
from .options import parser_for_family, explicit_arguments, resolve_training
from .standard import StaticModel
from .temporal import TemporalModel

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity')
_INPUT_OPTIONS = ('ref', 'unified_pos', 'downsample', 'downsamplex', 'downsampley', 'ntrain')


def task_record(directory):
    record = json.loads((Path(directory) / 'task.json').read_text())
    fields = {'schema_version', 'family', 'task', 'adapter', 'arguments'}
    if (type(record) is not dict or record.keys() != fields
            or type(record['schema_version']) is not int or record['schema_version'] != 1
            or record['family'] != FAMILY or record['task'] not in TASKS
            or type(record['adapter']) is not dict or type(record['arguments']) is not dict):
        raise MSARMetadataMismatch('invalid MSAR standard task sidecar')
    return record


def parse_args(parser, task, tokens):
    if task not in TASKS:
        raise ValueError('MSAR standard task not integrated: ' + task)
    parser = parser_for_family(parser, FAMILY)
    parser.add_argument('--msar-run-dir', type=Path, default=None,
                        help='new training directory, or explicit saved run for evaluation')
    args = parser.parse_args(tokens)
    explicit = explicit_arguments(parser, tokens)
    args.msar_task, args.msar_family, args.msar_evaluation = task, FAMILY, bool(args.eval)
    preset = json.loads((ROOT / 'PDE-Solving-StandardBenchmark/configs/msar_lno' / (task+'.json')).read_text())
    if preset['family'] != FAMILY or preset['task'] != task:
        raise ValueError('MSAR task preset family/task mismatch')
    objective_defaults = MSARTrainingConfig.from_dict(preset['objective']).to_dict()
    for name, value in (preset['training'] | preset['adapter']).items():
        if name not in explicit:
            setattr(args, name, value)
    if args.msar_evaluation:
        if args.msar_run_dir is None:
            parser.error('MSAR eval requires --msar-run-dir for an existing run')
        resolution = resolve_evaluation(args.msar_run_dir / 'architecture.json', explicit, task=task)
        saved = resolution.saved
        if saved.checkpoint_format != 'state_dict':
            raise MSARMetadataMismatch('standard tasks require state_dict')
        record = task_record(args.msar_run_dir)
        if record['task'] != task:
            raise MSARMetadataMismatch('task sidecar task mismatch')
        for name in _INPUT_OPTIONS:
            if hasattr(args, name):
                if name not in record['arguments']:
                    raise MSARMetadataMismatch('missing saved task input option: ' + name)
                if name in explicit and explicit[name] != record['arguments'][name]:
                    raise MSARMetadataMismatch('task input configuration mismatch: ' + name)
                setattr(args, name, record['arguments'][name])
        args.profile = saved.profile
        architecture, training = saved.architecture, saved.training
        args.msar_eval_training_overrides = resolution.requested_training.to_dict()
        args.msar_eval_training_differences = list(resolution.training_differences)
        if 'seed' not in explicit and record['arguments'].get('seed') is not None:
            args.seed = record['arguments']['seed']
    else:
        resolved = resolve_training(task, dict(profile=preset['profile']) | objective_defaults | explicit)
        architecture, training = resolved.architecture, resolved.training
        args.profile = resolved.profile
        if args.msar_run_dir is None:
            output = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output'))
            args.msar_run_dir = new_run_path(output, resolved, save_name=explicit.get('save_name'))
    if args.ref != 8 or args.unified_pos != preset['adapter']['unified_pos']:
        raise ValueError('MSAR must preserve the original reference/coordinate convention')
    # Record actual resolved fields, but never reinterpret old L/F/P/M as MSAR.
    args.msar_architecture, args.msar_training = architecture.to_dict(), training.to_dict()
    if 'save_name' in explicit:
        # Reuse the family path helper's filename validation, even for an explicit run path.
        from .profiles import ResolvedMSARConfig
        new_run_path('.', ResolvedMSARConfig(task, args.profile, architecture, training), save_name=args.save_name)
    if getattr(args, 'seed', None) is not None:
        from ..kcdno.entry import seed_process
        seed_process(args.seed, args.gpu)
    return args


def model_kwargs(args):
    if args.model != FAMILY or getattr(args, 'msar_task', None) not in TASKS:
        raise ValueError('MSAR standard model_kwargs requires an explicitly resolved task')
    return dict(config=MSARArchitectureConfig.from_dict(args.msar_architecture),
                training_config=MSARTrainingConfig.from_dict(args.msar_training),
                task_name=args.msar_task, ref=args.ref, unified_pos=args.unified_pos)


class StandardRun:
    """Original bare model.pt protocol; read-first metadata, immutable eval sidecars."""
    def __init__(self, args, model):
        from ..experiment import session, reserve_directory, timestamp
        self.args, self.evaluation = args, args.msar_evaluation
        self.directory = Path(args.msar_run_dir).resolve()
        self.recorder = session(args)
        self.architecture = MSARArchitectureConfig.from_dict(args.msar_architecture)
        self.training_config = MSARTrainingConfig.from_dict(args.msar_training)
        self._check_model(model)
        if self.evaluation:
            self.validate(model)
        else:
            parameter = next(model.parameters())
            metadata = MSARMetadata(args.msar_task, self.architecture, 'state_dict', args.profile,
                training=self.training_config, runtime=MSARRuntimeConfig(device=str(parameter.device),
                    dtype=str(parameter.dtype).removeprefix('torch.'), batch_size=args.batch_size))
            reserve_directory(args, self.directory)
            save_metadata(self.sidecar, metadata)
            record = dict(schema_version=1, family=FAMILY, task=args.msar_task,
                adapter=model.adapter_architecture(),
                arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
            with (self.directory/'task.json').open('x', encoding='utf-8') as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write('\n')
        self.result_dir = str(self.directory / ('eval_' + timestamp())) + '/'
        if self.recorder is not None:
            self.recorder.attach_model(model)
            self.result_dir = self.recorder.result_dir
        print('MSAR-LNO run:', self.directory, self.architecture.to_dict(), model.adapter_architecture())
        print('MSAR coverage:', self.training_config.to_dict(), 'effective:', self.training_config.effective_coverage_mode)
        # Point count is known from the original grid or Elasticity's fixed data contract.
        n = model.H * model.W if model.structured else 972
        print('MSAR input layout:', input_layout(self.architecture, n))

    @property
    def sidecar(self):return self.directory/'architecture.json'

    @property
    def checkpoint(self):return self.directory/'model.pt'

    def _check_model(self, model):
        expected = TemporalModel if self.args.msar_task in ('ns', 'plasticity') else StaticModel
        if (type(model) is not expected or model.config != self.architecture
                or model.core.config != self.architecture or model.training_config != self.training_config
                or model.core.training_config != self.training_config or model.task_name != self.args.msar_task):
            raise MSARMetadataMismatch('MSAR standard wrapper/core configuration mismatch')

    def validate(self, model):
        self._check_model(model)
        saved, record = load_metadata(self.sidecar), task_record(self.directory)
        if (saved.task != self.args.msar_task or saved.checkpoint_format != 'state_dict'
                or saved.architecture != model.config or saved.training != self.training_config
                or record['task'] != self.args.msar_task or record['adapter'] != model.adapter_architecture()):
            raise MSARMetadataMismatch('MSAR architecture/training/task wrapper mismatch')

    def save(self, model):
        import torch
        if self.evaluation:
            raise RuntimeError('evaluation must not save weights')
        self.validate(model)
        torch.save(model.state_dict(), self.checkpoint)

    def load(self, model):
        import torch
        self.validate(model)
        model.load_state_dict(torch.load(self.checkpoint, weights_only=True,
                                        map_location=next(model.parameters()).device), strict=True)

    def record_objective(self, epoch, accumulator, original_metrics):
        metrics = accumulator.values()
        mode = self.training_config.effective_coverage_mode
        print(f'Epoch {epoch} MSAR coverage={mode} step-mean objective: {metrics}')
        if self.recorder is not None:
            reduction = 'mean over optimizer steps; original train_loss retains task reduction'
            if self.args.msar_task == 'ns':
                reduction = 'PDE=sum over frames per update; coverage=mean over actual forwards and four levels; logs=mean over updates'
            elif self.args.msar_task == 'plasticity':
                reduction = 'one PDE+coverage objective per time-point update; coverage=mean over four levels; logs=mean over updates'
            self.recorder.record_epoch(epoch, original_metrics | metrics | dict(
                coverage_mode=mode, objective_reduction=reduction))
