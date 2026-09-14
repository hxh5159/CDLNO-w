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
    for name, kind in MODEL_OPTIONS.items():
        flags = ['--' + name]
        if '_' in name:
            flags.append('--' + name.replace('_', '-'))
        parser.add_argument(*flags, type=kind, default=None)
    parser.add_argument('--cdpa_mode', '--cdpa-mode', choices=('off', 'entry', 'every_block'), default=None)
    parser.add_argument('--run_dir', type=Path, default=None,
                        help='new CDLNO training directory, or existing run for evaluation')
    args = parser.parse_args(argv)
    if args.cfd_model != 'CDLNO':
        return args
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
            stem = (f'fold{args.fold_id}_L{args.n_layers}_F{args.front_blocks}'
                    f'_M{args.slice_num}_{args.cdpa_mode}_{_unique_suffix()}')
            self.directory = (Path(args.run_dir) if args.run_dir is not None
                              else Path('runs') / 'CDLNO' / 'shapenet-car' / stem).resolve()
            self.directory.mkdir(parents=True, exist_ok=False)
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
        return model.to(self.device)
