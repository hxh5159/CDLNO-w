"""Limited ShapeNet-Car CLI, isolated paths and trusted checkpoint integration.

No dataset or training imports. The original train.py still saves the complete
model object as model_<nb_epochs>.pth at its original frequency.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4

MODEL_OPTIONS = {
    'n_hidden': int, 'n_layers': int, 'n_heads': int, 'slice_num': int,
    'front_blocks': int, 'mlp_ratio': float, 'latent_ffn_ratio': float,
    'dropout': float, 'cdpa_source_chunk_size': int,
}


def parse_args(parser, *, evaluation=False, argv=None):
    parser.add_argument('--profile', choices=('kcdno_v1', 'transolver_shape_match'), default='kcdno_v1')
    parser.add_argument('--kernel-rank', '--kernel_rank', type=int, default=16)
    parser.add_argument('--history-mode', '--history_mode', choices=('all', 'off'), default='all')
    parser.add_argument('--kcdno-run-dir', type=Path, default=None)
    parser.add_argument('--front_latent_mode', '--front-latent-mode',
                        choices=('full', 'no_sa', 'identity'), default=None)
    for name, kind in MODEL_OPTIONS.items():
        flags = ['--' + name]
        if '_' in name:
            flags.append('--' + name.replace('_', '-'))
        parser.add_argument(*flags, type=kind, default=None)
    parser.add_argument('--cdpa_mode', '--cdpa-mode', choices=('off', 'entry', 'every_block'), default=None)
    parser.add_argument('--run_dir', type=Path, default=None,
                        help='new CDLNO training directory, or existing run for evaluation')
    # Route only an explicitly selected new family; the original parser stays intact.
    import argparse
    import sys
    tokens = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--cfd_model')
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
        loop_args = intercept_loop(parser, tokens, task='car', evaluation=evaluation, selected_model=selected.cfd_model)
        if loop_args is not None:
            return loop_args
    if selected.cfd_model == 'LinearNO':
        from cdlno.linearno.car_entry import parse_args as parse_linearno_args
        return parse_linearno_args(parser, tokens, evaluation=evaluation)
    if selected.cfd_model == 'msar_lno':
        from cdlno.msar_lno.industrial_entry import parse_args as parse_msar_args
        return parse_msar_args(parser, tokens, task='car', evaluation=evaluation)
    args = parser.parse_args(argv)
    if args.cfd_model in ('kcdno', 'lrsa_matched'):
        import sys
        from cdlno.kcdno.industrial_entry import resolve_car
        return resolve_car(parser, args, sys.argv[1:] if argv is None else argv, evaluation=evaluation)
    if args.cfd_model != 'CDLNO':
        return args
    if evaluation:
        if args.run_dir is None:
            parser.error('CDLNO evaluation requires --run_dir pointing to an existing run')
        from cdlno.checkpoint import resolve_front_latent_mode
        args.front_latent_mode = resolve_front_latent_mode(
            args.run_dir / 'architecture.json', args.front_latent_mode)
    preset_path = Path(__file__).resolve().parents[1] / 'configs' / 'CDLNO' / 'shapenet_car.json'
    preset = json.loads(preset_path.read_text())
    for name, value in preset['model'].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 0 <= args.fold_id <= 8:
        parser.error('CDLNO ShapeNet-Car requires fold_id in 0..8; one run is one fold')
    if args.nb_epochs < 1:
        parser.error('nb_epochs must be positive')
    if not evaluation and args.batch_size != 1:
        parser.error('CDLNO ShapeNet-Car supports batch_size=1 only')
    if evaluation and args.run_dir is None:
        parser.error('CDLNO evaluation requires --run_dir pointing to an existing run')
    # Fail invalid architecture/runtime options before the entry reads datasets.
    from models.CDLNO import architecture
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


def _unique_suffix():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:12]


class CarRun:
    """Sidecar checks wrap the original whole-model save/load protocol.

    Eval requires matching architecture and fold before unpickling. This loader
    is only for trusted whole-model checkpoints produced by this project.
    """

    def __init__(self, args, *, device, model=None, evaluation=False):
        from models.CDLNO import architecture, adapter_architecture
        from cdlno.config import CDLNORuntimeConfig
        from cdlno.checkpoint import save_sidecar
        from cdlno.experiment import session, reserve_directory, default_directory
        self.recorder = session(args)
        self.kwargs = model_kwargs(args)
        values = dict(self.kwargs)
        values.pop('cdpa_source_chunk_size')
        self.architecture = architecture(**values)
        self.adapter = adapter_architecture()
        self.contract = dict(fold_id=args.fold_id, nb_epochs=args.nb_epochs,
                             weight=args.weight, cfd_mesh=args.cfd_mesh, r=args.r)
        self.device = device
        self.runtime = CDLNORuntimeConfig(
            source_chunk_size=args.cdpa_source_chunk_size, device=str(device),
            dtype='float32' if model is None else str(next(model.parameters()).dtype).removeprefix('torch.'),
        ).validate()
        self.evaluation = evaluation
        if evaluation:
            if args.run_dir is None:
                raise ValueError('CDLNO evaluation requires an existing --run_dir')
            self.directory = Path(args.run_dir).resolve()
            self.validate()  # Read sidecar first; never write in evaluation.
        else:
            self._check_model(model)
            mode_suffix = '' if self.architecture.front_latent_mode == 'full' else '_' + self.architecture.front_latent_mode
            stem = (f'fold{args.fold_id}_L{args.n_layers}_F{args.front_blocks}'
                    f'_M{args.slice_num}_{args.cdpa_mode}{mode_suffix}_{_unique_suffix()}')
            self.directory = (Path(args.run_dir) if args.run_dir is not None
                              else default_directory('car')).resolve()
            reserve_directory(args, self.directory)
            arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            try:
                commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                    stderr=subprocess.DEVNULL, text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            save_sidecar(self.sidecar, self.architecture, self.runtime,
                         metadata=dict(wrapper_architecture=self.adapter,
                                       run_contract=self.contract, resolved_arguments=arguments,
                                       base_repository_commit=commit))
        self.result_dir = str(self.directory / ('eval_' + _unique_suffix())) + '/'
        if self.recorder is not None:
            self.result_dir = self.recorder.result_dir
            if model is not None:
                self.recorder.attach_model(model, hparams=dict(lr=args.lr, batch_size=args.batch_size, nb_epochs=args.nb_epochs), protocol=self.contract)
        print('CDLNO run directory:', self.directory)
        print('CDLNO resolved architecture:', self.architecture.to_dict(), self.adapter)

    @property
    def sidecar(self):
        return self.directory / 'architecture.json'

    @property
    def checkpoint(self):
        return self.directory / f'model_{self.contract["nb_epochs"]}.pth'

    def validate(self):
        from cdlno.checkpoint import SidecarMismatch, validate_sidecar
        payload = validate_sidecar(self.sidecar, self.architecture, self.runtime)
        metadata = payload.get('metadata', {})
        if metadata.get('wrapper_architecture') != self.adapter:
            raise SidecarMismatch('architecture mismatch: wrapper_architecture')
        if metadata.get('run_contract') != self.contract:
            raise SidecarMismatch('run mismatch: fold_id/nb_epochs/weight/cfd_mesh/r')
        return payload

    def _check_model(self, model):
        from models.CDLNO import Model
        from cdlno.checkpoint import SidecarMismatch, compare_architecture
        if type(model) is not Model:
            raise SidecarMismatch('checkpoint must contain models.CDLNO.Model')
        from cdlno.checkpoint import validate_model_front_mode
        validate_model_front_mode(model, self.architecture)
        if (compare_architecture(self.architecture, model.config)
                or compare_architecture(self.architecture, model.core.config)
                or model.adapter_architecture() != self.adapter):
            raise SidecarMismatch('checkpoint model and requested architecture disagree')

    def load(self):
        import torch
        from models.CDLNO import Model
        if not self.evaluation:
            raise RuntimeError('CarRun.load is for evaluation of an existing trusted run')
        self.validate()  # Mandatory before the local whole-model compatibility load.
        model = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self._check_model(model)
        # Whole-object saving is retained; still reject missing/unexpected/shape
        # incompatible weights rather than accepting a partially initialized model.
        expected = Model(**self.kwargs)
        expected.load_state_dict(model.state_dict(), strict=True)
        del expected
        model.core.source_chunk_size = self.runtime.source_chunk_size
        model = model.to(self.device)
        if self.recorder is not None:
            self.recorder.attach_model(model, protocol=self.contract)
        return model
