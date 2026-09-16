"""New-family argument and task checkpoint integration, without data imports."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import re

from .config import KCDNOInitializationConfig, KCDNORuntimeConfig
from .matched_config import MatchedInitialization
from .metadata import KCDNOMetadataMismatch, save_metadata
from .options import explicit_arguments

from .families import (architecture_from_dict, make_metadata, load_metadata,
                       resolve_training, resolve_evaluation, new_run_path)

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / 'PDE-Solving-StandardBenchmark'
STANDARD_TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity')


def add_arguments(parser):
    parser.add_argument('--profile', choices=('kcdno_v1', 'transolver_shape_match'), default='kcdno_v1')
    parser.add_argument('--kernel-rank', '--kernel_rank', type=int, default=16)
    parser.add_argument('--history-mode', '--history_mode', choices=('all', 'off'), default='all')
    parser.add_argument('--kcdno-run-dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        help='optional RNG seed for kcdno/lrsa_matched standard tasks')


def seed_process(seed, gpu):
    """Seed before data iteration/model construction; leave backend choices intact."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    # The original entries set this too, but do so after argument resolution.
    # Select the same GPU before any torch/CUDA seeding in this optional path.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_record(path):
    payload = json.loads((Path(path) / 'task.json').read_text())
    required = {'schema_version', 'family', 'task', 'adapter', 'protocol', 'arguments'}
    if payload.keys() != required or payload['schema_version'] != 1 or payload['family'] not in ('kcdno', 'lrsa_matched'):
        raise KCDNOMetadataMismatch('invalid new-family task sidecar')
    if not all(isinstance(payload[k], dict) for k in ('adapter', 'protocol', 'arguments')):
        raise KCDNOMetadataMismatch('task sidecar objects are required')
    return payload


def resolve_args(parser, args, task, tokens, *, evaluation=None):
    """Called only for the new family; old defaults never become overrides."""
    if task not in STANDARD_TASKS:
        raise ValueError('KCDNO task entry not integrated: ' + task)
    explicit = explicit_arguments(parser, tokens)
    if 'seed' in explicit and (type(args.seed) is not int or not 0 <= args.seed < 2**32):
        parser.error('seed must be an integer in [0, 2**32)')
    args.kcdno_task, args.kcdno_family = task, args.model
    evaluation = bool(args.eval) if evaluation is None else evaluation
    args.kcdno_evaluation = evaluation
    preset = json.loads((STANDARD / 'configs/kcdno' / (task + '.json')).read_text())
    defaults = preset['training'] | preset['adapter']
    for name, value in defaults.items():
        if name not in explicit:
            setattr(args, name, value)
    if evaluation:
        if args.kcdno_run_dir is None:
            parser.error('kcdno eval requires --kcdno-run-dir for an existing run')
        resolved = resolve_evaluation(args.kcdno_run_dir / 'architecture.json', explicit, task=task)
        cfg = resolved.saved.architecture
        args.profile = resolved.saved.profile
        saved = task_record(args.kcdno_run_dir)
        if saved['task'] != task:
            raise KCDNOMetadataMismatch('task sidecar task mismatch')
        if 'seed' not in explicit and saved['arguments'].get('seed') is not None:
            args.seed = saved['arguments']['seed']
        # Restore input layout and conditioning before any dataset is read.
        for name in ('ref', 'unified_pos', 'downsample', 'ntrain'):
            if name in saved['arguments']:
                if name in explicit and explicit[name] != saved['arguments'][name]:
                    raise KCDNOMetadataMismatch('task architecture mismatch: ' + name)
                setattr(args, name, saved['arguments'][name])
    else:
        cfg = resolve_training(task, explicit)
    if args.ref != 8 or args.unified_pos not in (0, 1) or bool(args.unified_pos) != preset['adapter']['unified_pos']:
        raise ValueError('reference/coordinate semantics must preserve the task contract')
    args.kcdno_architecture = cfg.to_dict()
    for name, value in dict(n_layers=cfg.L, n_hidden=cfg.d, n_heads=cfg.h, slice_num=cfg.M,
                            kernel_rank=getattr(cfg, 'kernel_rank', None), history_mode=getattr(cfg, 'history_mode', None), dropout=0.).items():
        setattr(args, name, value)
    if 'save_name' in explicit and not re.fullmatch(r'[A-Za-z0-9_.-]+', args.save_name):
        parser.error('save_name must be a filename stem')
    if not evaluation and args.kcdno_run_dir is None:
        output = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output'))
        proposed = new_run_path(output, task, cfg)
        digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        args.kcdno_run_dir = proposed.parent / args.profile / (proposed.name + '_' + digest)
    if getattr(args, 'seed', None) is not None:
        seed_process(args.seed, args.gpu)
    return args


def model_kwargs(args):
    return dict(config=architecture_from_dict(args.kcdno_architecture),
                task_name=args.kcdno_task, ref=args.ref, unified_pos=args.unified_pos)


class StandardRun:
    """Same state_dict save protocol, with complete architecture + task checks."""

    def __init__(self, args, model):
        from ..experiment import session, reserve_directory, timestamp
        self.args = args
        self.evaluation = args.kcdno_evaluation
        self.directory = Path(args.kcdno_run_dir).resolve()
        self.recorder = session(args)
        initialization_type = (KCDNOInitializationConfig if model.config.family == 'kcdno'
                               else MatchedInitialization)
        self.metadata = make_metadata(task=args.kcdno_task, architecture=model.config,
            checkpoint_format='state_dict', profile=args.profile,
            initialization=initialization_type(seed=getattr(args, 'seed', None)),
            runtime=KCDNORuntimeConfig(device=str(next(model.parameters()).device),
                                      batch_size=args.batch_size, dtype=str(next(model.parameters()).dtype).removeprefix('torch.')))
        self.adapter = model.adapter_architecture()
        if self.evaluation:
            self.validate(model)
        else:
            reserve_directory(args, self.directory)
            save_metadata(self.sidecar, self.metadata)
            payload = dict(schema_version=1, family=model.config.family, task=args.kcdno_task,
                           adapter=self.adapter, protocol={},
                           arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
            with (self.directory / 'task.json').open('x') as f:
                json.dump(payload, f, indent=2, allow_nan=False)
        self.result_dir = str(self.directory / ('eval_' + timestamp())) + '/'
        if self.recorder is not None:
            self.recorder.attach_model(model)
            self.result_dir = self.recorder.result_dir
        print(model.config.family+' run:', self.directory, model.config.to_dict(), self.adapter)

    @property
    def sidecar(self):
        return self.directory / 'architecture.json'

    @property
    def checkpoint(self):
        return self.directory / 'model.pt'

    def validate(self, model):
        saved = load_metadata(self.sidecar)
        task = task_record(self.directory)
        if (saved.task != self.args.kcdno_task or saved.checkpoint_format != 'state_dict'
                or saved.architecture != model.config or model.core.config != model.config
                or task['family'] != saved.architecture.family or task['task'] != saved.task or task['adapter'] != model.adapter_architecture()):
            raise KCDNOMetadataMismatch('model/core/task architecture mismatch')

    def save(self, model):
        import torch
        if self.evaluation:
            raise RuntimeError('eval must not save weights')
        self.validate(model)
        torch.save(model.state_dict(), self.checkpoint)

    def load(self, model):
        import torch
        self.validate(model)
        model.load_state_dict(torch.load(self.checkpoint, weights_only=True,
                                        map_location=next(model.parameters()).device), strict=True)
