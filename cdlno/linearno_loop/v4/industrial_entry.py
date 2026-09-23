"""V4-only industrial parsing; saved sidecars are validated before allocation."""
import os
from pathlib import Path
import random
from uuid import uuid4

from cdlno.linearno.profiles import DEFAULT_PROFILE
from linearno_loop.v4.config import resolve_config, run_directory_id
from linearno_loop.v4.contracts import ARCHITECTURE
from linearno_loop.v4.schema import read_metadata, restore_config

ROOT = Path(__file__).resolve().parents[3]
TRAIN_FIELDS = dict(lr='lr', nb_epochs='epochs', batch_size='batch_size')
FIXED_TRAIN_FIELDS = dict(nmodel='nmodel', subsampling='subsampling',
                         max_neighbors='max_neighbors', debug='debug', preprocessed='preprocessed')


def parse_v4(parser, args, supplied, loop_explicit, *, task):
    unsupported = set(loop_explicit) - {'linearno_loop', 'architecture', 'temperature_mode'}
    if unsupported:
        raise ValueError('V4 does not accept legacy loop fields: ' + str(sorted(unsupported)))
    structural = {'linearno_variant', 'linearno_rank', 'linearno_hidden', 'linearno_heads',
                  'linearno_ffn_ratio', 'linearno_dropout', 'linearno_ref', 'linearno_unified_pos'} & set(supplied)
    if structural:
        raise ValueError('V4 model fields come from the pure LinearNO profile: ' + str(sorted(structural)))
    restoring = bool(args.eval or args.resume)
    if restoring:
        if args.linearno_run_dir is None:
            raise ValueError('V4 eval/resume requires --experiment-dir')
        root = read_metadata(args.linearno_run_dir / 'architecture.json')
        assertions = dict(task=task)
        assertions.update({k: v for k, v in loop_explicit.items() if k != 'linearno_loop'})
        if 'linearno_loop' in loop_explicit:
            assertions['linearno_loop'] = loop_explicit['linearno_loop'] == '1'
        if 'linearno_profile' in supplied:
            assertions['profile'] = args.linearno_profile
        if 'seed' in supplied:
            assertions['seed'] = args.seed
        config = restore_config(root, explicit=assertions)['config']
        for name, field in TRAIN_FIELDS.items():
            if name in supplied and supplied[name] != config['profile_spec']['values']['training'][field]:
                raise ValueError('explicit training.' + field + ' conflicts with V4 metadata')
        if 'save_name' in supplied and supplied['save_name'] != config['profile_spec']['values']['runtime']['save_name']:
            raise ValueError('save_name conflicts with V4 metadata')
        data = root['data_spec']['runtime']
        for field in (('task',) if task == 'airfrans' else ('fold_id', 'cfd_mesh', 'r', 'val_iter')):
            stored = data['task_variant'] if field == 'task' else data[field]
            if field in supplied and supplied[field] != stored:
                raise ValueError('explicit ' + field + ' conflicts with V4 data protocol')
            setattr(args, field, stored)
        from . import checkpoint
        if task == 'car':
            saved, path = checkpoint.inspect_checkpoint(args.linearno_run_dir, args.checkpoint, expected=config)
            args._linearno_metadata, args._linearno_checkpoint = saved, path
        else:
            if args.resume and args.checkpoint != 'latest':
                raise ValueError('Air ensemble resume requires latest')
            from cdlno.linearno_loop.industrial_state import inspect_members
            pairs = inspect_members(args.linearno_run_dir, root, evaluation=bool(args.eval),
                                    selector=args.checkpoint, checkpoint_module=checkpoint)
            args._linearno_metadata = root
            args._linearno_checkpoint = next(pair[1] for pair in reversed(pairs) if pair is not None)
    else:
        overrides = {'training.' + field: supplied[name] for name, field in TRAIN_FIELDS.items()
                     if name in supplied}
        if 'save_name' in supplied:
            overrides['runtime.save_name'] = supplied['save_name']
        config = resolve_config(task, profile=getattr(args, 'linearno_profile', DEFAULT_PROFILE),
            options=dict(architecture=ARCHITECTURE, temperature_mode=loop_explicit.get('temperature_mode', 'latent_k_point_q'),
                         seed=args.seed), profile_overrides=overrides)
    base = config['profile_spec']; model = config['model']; training = base['values']['training']
    if training['batch_size'] != 1:
        raise ValueError('industrial V4 requires batch_size=1, one graph per forward')
    fixed = dict(FIXED_TRAIN_FIELDS)
    if task == 'airfrans':
        fixed['r'] = 'r'
    for name, field in fixed.items():
        if name in supplied and supplied[name] != training[field]:
            raise ValueError('V4 preserves profile training.' + field)
    if 'weight' in supplied and supplied['weight'] != base['values']['objective']['surface_weight']:
        raise ValueError('V4 preserves profile objective.surface_weight')
    args.linearno_family = 'linearno_loop'; args.linearno_task = task; args.linearno_profile = base['profile']
    args._linearno_loop_config = config; args._linearno_config = base; args._linearno_model_spec = config['model_spec']
    args.linearno_rank = model['actual_M']; args.linearno_variant = model['variant']
    args.n_hidden = model['hidden_width']; args.n_layers = 8; args.n_heads = model['heads']
    args.mlp_ratio = model['ffn_ratio']; args.dropout = model['dropout']; args.ref = model['ref']
    args.unified_pos = int(model['unified_pos']); args.seed = config['seed']
    args.nb_epochs = training['epochs']; args.lr = training['lr']; args.batch_size = training['batch_size']
    args.weight = base['values']['objective']['surface_weight']; args.save_name = base['values']['runtime']['save_name']
    if task == 'airfrans':
        for name in ('nmodel', 'subsampling', 'r', 'max_neighbors', 'debug'):
            setattr(args, name, training[name])
        if args.task not in ('full', 'scarce', 'reynolds', 'aoa'):
            raise ValueError('invalid Air task')
        args.data_path = args.my_path
    else:
        if not 0 <= args.fold_id <= 8 or args.val_iter < 1:
            raise ValueError('Car requires fold0..8 and positive val_iter')
        args.preprocessed = training['preprocessed']
    if args.linearno_run_dir is None:
        from cdlno.experiment import timestamp
        args.linearno_run_dir = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output')) / task / ARCHITECTURE / (
            run_directory_id(config) + '__' + timestamp() + '_' + uuid4().hex[:8])
    args.linearno_run_dir = Path(args.linearno_run_dir).resolve()
    if not restoring and args.linearno_run_dir.exists():
        raise ValueError('new V4 run directory already exists')
    if getattr(args, 'gpu', None) is not None and args.gpu < 0:
        raise ValueError('gpu must be nonnegative')
    import numpy as np
    import torch
    args.device = args.device or (f'cuda:{args.gpu or 0}' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('unsupported/unavailable device')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    return args
