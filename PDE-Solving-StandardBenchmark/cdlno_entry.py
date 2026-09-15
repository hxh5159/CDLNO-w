"""Limited CDLNO parser, output-path and checkpoint branches; no data imports.

Legacy experiments only use parse_args; their defaults and constructor kwargs
are left intact. This module imports cdlno/torch only for the new model branch.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4


def parse_args(parser, task, argv=None):
    parser.add_argument('--profile', choices=('kcdno_v1', 'transolver_shape_match'), default='kcdno_v1')
    parser.add_argument('--kernel-rank', '--kernel_rank', type=int, default=16)
    parser.add_argument('--history-mode', '--history_mode', choices=('all', 'off'), default='all')
    parser.add_argument('--kcdno-run-dir', type=Path, default=None)
    parser.add_argument('--front-latent-mode', '--front_latent_mode',
                        choices=('full', 'no_sa', 'identity'), default=None)
    parser.add_argument('--front-blocks', type=int, default=2)
    parser.add_argument('--latent-ffn-ratio', type=float, default=2.0)
    parser.add_argument('--cdpa-mode', choices=('off', 'entry', 'every_block'), default='entry')
    parser.add_argument('--cdpa-source-chunk-size', type=int, default=0)
    parser.add_argument('--cdlno-run-dir', type=Path, default=None,
                        help='new training directory, or existing checkpoint directory for eval')
    tokens = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(tokens)
    if args.model in ('kcdno', 'lrsa_matched'):
        from cdlno.kcdno.entry import resolve_args
        return resolve_args(parser, args, task, tokens)
    if args.model != 'CDLNO':
        return args
    # Discover explicit options with argparse itself, including --flag=value
    # and accepted abbreviations. Do not guess from equality to old defaults.
    probe = copy.deepcopy(parser)
    probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    explicit = vars(probe.parse_args(tokens))
    saved_mode = None
    if args.eval:
        if args.cdlno_run_dir is None:
            parser.error('CDLNO eval requires --cdlno-run-dir pointing to an existing run')
        from cdlno.checkpoint import resolve_front_latent_mode
        saved_mode = resolve_front_latent_mode(args.cdlno_run_dir / 'architecture.json',
                                              explicit.get('front_latent_mode'))
    preset = json.loads((Path(__file__).parent / 'configs' / 'CDLNO' / f'{task}.json').read_text())
    for name, value in {**preset['model'], **preset['training']}.items():
        if name not in explicit:
            setattr(args, name, value)
    if saved_mode is not None:
        args.front_latent_mode = saved_mode
    args.cdlno_task = task
    if 'save_name' not in explicit:
        args.save_name = (f'{task}_CDLNO_L{args.n_layers}_F{args.front_blocks}'
                          f'_M{args.slice_num}_{args.cdpa_mode}')
        if args.front_latent_mode != 'full':
            args.save_name += '_' + args.front_latent_mode
    if args.eval and args.cdlno_run_dir is None:
        parser.error('CDLNO eval requires --cdlno-run-dir pointing to an existing run')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.save_name) or args.save_name in ('.', '..'):
        parser.error('CDLNO save_name must be a filename stem; use --cdlno-run-dir for paths')
    return args


def model_kwargs(args):
    """Call only on the CDLNO branch; original constructors receive no new keys."""
    return dict(task_name=args.cdlno_task, front_blocks=args.front_blocks,
                front_latent_mode=getattr(args, 'front_latent_mode', 'full'),
                latent_ffn_ratio=args.latent_ffn_ratio, cdpa_mode=args.cdpa_mode,
                cdpa_source_chunk_size=args.cdpa_source_chunk_size)


def _unique_suffix():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:12]


class StaticRun:
    """One isolated run; retains the original experiment's save frequency.

    Core architecture uses the existing sidecar schema. Adapter semantics are
    mandatory metadata.wrapper_architecture, independently compared strictly.
    Runtime settings are recorded separately and do not define weight structure.
    """

    def __init__(self, args, model):
        from cdlno.config import CDLNORuntimeConfig
        from cdlno.checkpoint import save_sidecar
        from cdlno.experiment import session, reserve_directory, default_directory
        self.recorder = session(args)
        self.architecture = model.config
        self.adapter = model.adapter_architecture()
        parameter = next(model.parameters())
        self.runtime = CDLNORuntimeConfig(
            source_chunk_size=args.cdpa_source_chunk_size, device=str(parameter.device),
            dtype=str(parameter.dtype).removeprefix('torch.'),
        ).validate()
        self.evaluation = bool(args.eval)
        if self.evaluation:
            if args.cdlno_run_dir is None:
                raise ValueError('CDLNO eval requires an explicit existing run directory')
            self.directory = Path(args.cdlno_run_dir).resolve()
            self._validate(model)  # load existing sidecar before any comparison/write
        else:
            self.directory = (Path(args.cdlno_run_dir) if args.cdlno_run_dir is not None
                              else default_directory(args.cdlno_task)).resolve()
            # Atomic directory reservation: never silently reuse another run.
            reserve_directory(args, self.directory)
            values = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()}
            try:
                commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                    stderr=subprocess.DEVNULL, text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None  # Source archives need not contain Git metadata.
            save_sidecar(self.sidecar, self.architecture, self.runtime,
                         metadata=dict(wrapper_architecture=self.adapter, resolved_arguments=values,
                                       base_repository_commit=commit))
        self.result_dir = str(self.directory / ('eval_' + _unique_suffix())) + '/'
        if self.recorder is not None:
            self.result_dir = self.recorder.result_dir
            self.recorder.attach_model(model)
        print('CDLNO run directory:', self.directory)
        print('CDLNO resolved architecture:', self.architecture.to_dict(), self.adapter)

    @property
    def sidecar(self):
        return self.directory / 'architecture.json'

    @property
    def checkpoint(self):
        return self.directory / 'model.pt'

    def _validate(self, model):
        from cdlno.checkpoint import SidecarMismatch, validate_sidecar
        payload = validate_sidecar(self.sidecar, model.config, self.runtime)
        if payload.get('metadata', {}).get('wrapper_architecture') != model.adapter_architecture():
            raise SidecarMismatch('architecture mismatch: wrapper_architecture')
        return payload

    def load(self, model):
        import torch
        self._validate(model)
        state = torch.load(self.checkpoint, map_location=next(model.parameters()).device, weights_only=True)
        model.load_state_dict(state, strict=True)

    def save(self, model):
        import torch
        if self.evaluation:
            raise RuntimeError('evaluation must not overwrite training checkpoints')
        self._validate(model)
        torch.save(model.state_dict(), self.checkpoint)
