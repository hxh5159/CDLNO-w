"""Limited AirfRANS CDLNO selection, paths and trusted checkpoint integration.

No data/metrics imports; original train.py saves each complete model and main.py
saves the model list. Only their destination paths differ for CDLNO.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4


MODEL_OPTIONS = dict(n_hidden=int, n_layers=int, n_heads=int, slice_num=int,
                     front_blocks=int, mlp_ratio=float, latent_ffn_ratio=float,
                     dropout=float, cdpa_source_chunk_size=int)


def parse_args(parser, *, evaluation=False, argv=None):
    parser.add_argument('--front_latent_mode', '--front-latent-mode',
                        choices=('full', 'no_sa', 'identity'), default=None)
    for name, kind in MODEL_OPTIONS.items():
        flags = ['--' + name]
        if '_' in name:
            flags.append('--' + name.replace('_', '-'))
        parser.add_argument(*flags, type=kind, default=None)
    parser.add_argument('--cdpa_mode', '--cdpa-mode', choices=('off', 'entry', 'every_block'), default=None)
    parser.add_argument('--run_dir', type=Path, default=None,
                        help='new CDLNO training run, or existing run for evaluation')
    for name, kind in dict(nb_epochs=int, batch_size=int, lr=float).items():
        parser.add_argument('--' + name, type=kind, default=None)
    if evaluation:
        # LinearNO is routed below before any legacy branch executes.  Keeping
        # it in this parser's choices lets main_evaluation.py accept the new
        # family while all historical choices and defaults stay unchanged.
        parser.add_argument('--model', choices=('Transolver', 'CDLNO', 'kcdno', 'lrsa_matched', 'LinearNO'), default='Transolver')
        parser.add_argument('--task', choices=('full', 'scarce', 'reynolds', 'aoa'), default='full')
        parser.add_argument('--nmodel', type=int, default=1)
        parser.add_argument('--weight', type=float, default=1.)
    parser.add_argument('--profile', choices=('kcdno_v1', 'transolver_shape_match'), default='kcdno_v1')
    parser.add_argument('--kernel-rank', '--kernel_rank', type=int, default=16)
    parser.add_argument('--history-mode', '--history_mode', choices=('all', 'off'), default='all')
    parser.add_argument('--kcdno-run-dir', type=Path, default=None)
    # Route only an explicitly selected new family; the original parser stays intact.
    import argparse
    import sys
    tokens = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--model')
    # Routing-only probe: never changes the legacy parser's actions/defaults.
    selector.add_argument('--experiment-dir', '--linearno-run-dir', '--run_dir', dest='_loop_directory', type=Path)
    selector.add_argument('--eval', dest='_loop_eval', type=int, default=0)
    selector.add_argument('--resume', dest='_loop_resume', action='store_true')
    selected, _ = selector.parse_known_args(tokens)
    loop_selected = any(t.split('=')[0].startswith('--linearno-loop') for t in tokens)
    if (evaluation or selected._loop_eval or selected._loop_resume) and selected._loop_directory is not None:
        sidecar = selected._loop_directory / 'architecture.json'
        if sidecar.is_file():
            loop_selected = loop_selected or json.loads(sidecar.read_text()).get('family') == 'linearno_loop'
    if loop_selected:
        try:
            from cdlno.linearno_loop.industrial_entry import intercept as intercept_loop
        except ModuleNotFoundError as error:
            if error.name == 'cdlno' or error.name.startswith('cdlno.linearno_loop'):
                parser.error('linearno_loop selected but its shared cdlno loop package is unavailable')
            raise
        loop_args = intercept_loop(parser, tokens, task='airfrans', evaluation=evaluation, selected_model=selected.model)
        if loop_args is not None:
            return loop_args
    if selected.model == 'LinearNO':
        from cdlno.linearno.air_entry import parse_args as parse_linearno_args
        return parse_linearno_args(parser, tokens, evaluation=evaluation)
    if selected.model == 'msar_lno':
        from cdlno.msar_lno.industrial_entry import parse_args as parse_msar_args
        return parse_msar_args(parser, tokens, task='airfrans', evaluation=evaluation)
    args = parser.parse_args(argv)
    if args.model in ('kcdno', 'lrsa_matched'):
        from cdlno.kcdno.air_entry import resolve_air
        import sys
        return resolve_air(parser, args, sys.argv[1:] if argv is None else argv, evaluation=evaluation)
    if args.model != 'CDLNO':
        return args
    if evaluation:
        if args.run_dir is None:
            parser.error('CDLNO evaluation requires an existing --run_dir')
        from cdlno.checkpoint import resolve_front_latent_mode
        args.front_latent_mode = resolve_front_latent_mode(
            args.run_dir / 'architecture.json', args.front_latent_mode)
    defaults = json.loads((Path(__file__).parent / 'configs/CDLNO/airfrans.json').read_text())['model']
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.task not in ('full', 'scarce', 'reynolds', 'aoa') or args.nmodel < 1:
        parser.error('CDLNO requires a supported AirfRANS task and nmodel>=1')
    if evaluation and args.run_dir is None:
        parser.error('CDLNO evaluation requires an existing --run_dir')
    from cdlno.airfrans import architecture
    from cdlno.config import CDLNORuntimeConfig
    values = model_kwargs(args)
    values.pop('cdpa_source_chunk_size')
    architecture(**values)
    CDLNORuntimeConfig(source_chunk_size=args.cdpa_source_chunk_size).validate()
    return args


def model_kwargs(args):
    return dict(n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_heads,
                front_latent_mode=getattr(args, 'front_latent_mode', 'full'),
                slice_num=args.slice_num, front_blocks=args.front_blocks,
                mlp_ratio=args.mlp_ratio, latent_ffn_ratio=args.latent_ffn_ratio,
                dropout=args.dropout, cdpa_mode=args.cdpa_mode,
                cdpa_source_chunk_size=args.cdpa_source_chunk_size)


def resolve_hparams(args, hparams):
    """Only explicit new-model training overrides supersede the existing YAML."""
    values = dict(hparams)
    for key in ('nb_epochs', 'batch_size', 'lr'):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if values['batch_size'] != 1:
        raise ValueError('CDLNO AirfRANS supports batch_size=1 only')
    if type(values['nb_epochs']) is not int or values['nb_epochs'] < 1:
        raise ValueError('nb_epochs must be a positive integer')
    return values


def _suffix():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:12]


class AirRun:
    def __init__(self, args, hparams, *, device, evaluation=False):
        from cdlno.airfrans import architecture, adapter_architecture
        from cdlno.config import CDLNORuntimeConfig
        from cdlno.checkpoint import save_sidecar
        from cdlno.experiment import session, reserve_directory, default_directory
        self.recorder = session(args)
        self.kwargs = model_kwargs(args)
        structural = dict(self.kwargs)
        structural.pop('cdpa_source_chunk_size')
        self.architecture = architecture(**structural)
        self.adapter = adapter_architecture()
        self.hparams = resolve_hparams(args, hparams)
        self.contract = dict(task=args.task, nmodel=args.nmodel, weight=args.weight,
                             hparams=self.hparams)
        self.device = device
        self.runtime = CDLNORuntimeConfig(source_chunk_size=args.cdpa_source_chunk_size,
                                          device=str(device), dtype='float32').validate()
        self.evaluation = evaluation
        if evaluation:
            if args.run_dir is None:
                raise ValueError('CDLNO evaluation requires an existing --run_dir')
            self.directory = Path(args.run_dir).resolve()
            self.validate()  # Sidecar is read before any whole-model load or write.
        else:
            mode_suffix = '' if self.architecture.front_latent_mode == 'full' else '_' + self.architecture.front_latent_mode
            stem = f'L{args.n_layers}_F{args.front_blocks}_M{args.slice_num}_{args.cdpa_mode}{mode_suffix}_{_suffix()}'
            self.directory = (Path(args.run_dir) if args.run_dir is not None else
                              default_directory('airfrans')).resolve()
            reserve_directory(args, self.directory)
            arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            try:
                commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            save_sidecar(self.sidecar, self.architecture, self.runtime, metadata=dict(
                wrapper_architecture=self.adapter, run_contract=self.contract,
                resolved_arguments=arguments, base_repository_commit=commit))
        self.result_dir = str(self.directory / ('eval_' + _suffix()))
        if self.recorder is not None:
            self.result_dir = self.recorder.result_dir
        print('CDLNO run directory:', self.directory)
        print('CDLNO resolved architecture:', self.architecture.to_dict(), self.adapter)
        print('CDLNO resolved training/data settings:', self.hparams)

    @property
    def sidecar(self):
        return self.directory / 'architecture.json'

    @property
    def checkpoint(self):
        return self.directory / 'CDLNO'

    def member_dir(self, index):
        if type(index) is not int or not 0 <= index < self.contract['nmodel']:
            raise ValueError('checkpoint member index is out of range')
        return str(self.directory / f'member_{index:03d}')

    def validate(self):
        from cdlno.checkpoint import SidecarMismatch, validate_sidecar
        payload = validate_sidecar(self.sidecar, self.architecture, self.runtime)
        metadata = payload.get('metadata', {})
        if metadata.get('wrapper_architecture') != self.adapter:
            raise SidecarMismatch('architecture mismatch: wrapper_architecture')
        if metadata.get('run_contract') != self.contract:
            raise SidecarMismatch('run mismatch: task/nmodel/weight/training or sampling settings')
        return payload

    def load(self, *, member=None):
        """Load this project's trusted model list, or one saved complete member."""
        import torch
        from cdlno.airfrans import AirfRANSModel
        from cdlno.checkpoint import SidecarMismatch, compare_architecture
        if not self.evaluation:
            raise RuntimeError('loading requires evaluation of an existing run')
        self.validate()
        path = self.checkpoint if member is None else Path(self.member_dir(member)) / 'model'
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if member is None:
            if type(saved) is not list or len(saved) != self.contract['nmodel']:
                raise SidecarMismatch('checkpoint must be the complete model list for this run')
            models = saved
        else:
            models = [saved]
        expected = AirfRANSModel(**self.kwargs)
        for model in models:
            if type(model) is not AirfRANSModel:
                raise SidecarMismatch('checkpoint must contain cdlno.airfrans.AirfRANSModel')
            from cdlno.checkpoint import validate_model_front_mode
            validate_model_front_mode(model, self.architecture)
            if (compare_architecture(self.architecture, model.config)
                    or compare_architecture(self.architecture, model.core.config)
                    or model.adapter_architecture() != self.adapter):
                raise SidecarMismatch('checkpoint model and sidecar architecture disagree')
            expected.load_state_dict(model.state_dict(), strict=True)
            model.core.source_chunk_size = self.runtime.source_chunk_size
            model.to(self.device)
        if self.recorder is not None:
            for index, model in enumerate(models):
                self.recorder.attach_model(model, hparams=self.hparams, member=index if member is None else member, protocol=self.contract)
        return models if member is None else models[0]
