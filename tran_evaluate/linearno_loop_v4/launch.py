"""Eight-task V4 launcher through the real, data-free native parser."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tran_evaluate.linearno_loop.launch import TASKS, plan, execute


def plan_v4(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('action', choices=('train', 'resume', 'eval', 'train_eval'))
    parser.add_argument('--temperature-mode', choices=('base', 'latent_k_point_q', 'point_k_point_q'))
    parser.add_argument('--linearno-profile', dest='profile')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT/'output')))
    parser.add_argument('--experiment-dir', type=Path)
    parser.add_argument('--dry-run', '--preview', action='store_true')
    parser.add_argument('--print-config', '--plan-json', action='store_true')
    parser.add_argument('--print-run-dir', action='store_true')
    parser.add_argument('--then-eval', action='store_true')
    args, rest = parser.parse_known_args(argv)
    explicit_data_root = args.data_root is not None
    args.data_root = args.data_root or Path(os.environ.get('CDLNO_DATA_ROOT', ROOT.parent/'data'))
    if args.then_eval and args.action == 'eval':
        parser.error('--then-eval is only for train/resume')
    action = 'train' if args.action == 'train_eval' else args.action
    if action != 'train' and args.experiment_dir is None:
        parser.error('resume/eval requires --experiment-dir')
    forbidden = {'--linearno-loop-architecture', '--linearno-loop', '--linearno-loop-temperature-mode'}
    if any(token.split('=')[0] in forbidden for token in rest):
        parser.error('V4 architecture is fixed; use --temperature-mode')
    tokens = ['--linearno-loop', '1', '--linearno-loop-architecture', 'resmlp_dual_temp_v4', '--gpu', str(args.gpu)]
    if action == 'train' or args.temperature_mode is not None:
        tokens += ['--linearno-loop-temperature-mode', args.temperature_mode or 'latent_k_point_q']
    if args.seed is not None:
        tokens += ['--seed', str(args.seed)]
    if args.profile is not None:
        tokens += ['--linearno-profile', args.profile]
    if args.experiment_dir is not None:
        tokens += ['--experiment-dir', str(args.experiment_dir.resolve())]
    flags = {token.split('=')[0] for token in rest}
    def data_default(environment_name, fallback):
        return str(fallback if explicit_data_root else os.environ.get(environment_name, fallback))
    if args.task not in ('airfrans', 'car') and not {'--data_path', '--data-path'} & flags:
        if args.task == 'plasticity':
            tokens += ['--data_path', data_default('CDLNO_PLASTICITY_FILE', args.data_root/'fno'/'plas_N987_T20.mat')]
        else:
            tokens += ['--data_path', data_default('CDLNO_FNO_ROOT', args.data_root/'fno')]
    elif args.task == 'airfrans' and '--my_path' not in flags:
        tokens += ['--my_path', data_default('CDLNO_AIRFRANS_DATASET', args.data_root/'AirfRANS'/'Dataset')]
    elif args.task == 'car':
        if '--data_dir' not in flags:
            tokens += ['--data_dir', data_default('CDLNO_CAR_RAW_ROOT', args.data_root/'mlcfd_data'/'training_data')]
        if '--save_dir' not in flags:
            tokens += ['--save_dir', data_default('CDLNO_CAR_CACHE_ROOT', args.data_root/'mlcfd_data'/'preprocessed_data')]
    # Both planner and parser use this root. Restore the caller's environment.
    prior = os.environ.get('CDLNO_RUNS_ROOT')
    os.environ['CDLNO_RUNS_ROOT'] = str(args.output_root.resolve())
    try:
        value = plan(args.task, action, [*tokens, *rest])
    finally:
        if prior is None:
            os.environ.pop('CDLNO_RUNS_ROOT', None)
        else:
            os.environ['CDLNO_RUNS_ROOT'] = prior
    if value['config']['architecture'] != 'resmlp_dual_temp_v4':
        parser.error('V4 launcher requires V4 saved metadata')
    return args, value


def main(argv=None):
    args, value = plan_v4(argv)
    if args.print_run_dir:
        print(value['run']); return 0
    if args.print_config or args.dry_run:
        print(json.dumps({k:v for k,v in value.items() if k != 'environment'}, indent=2, sort_keys=True))
        print('DRY RUN: actual parser and saved metadata only; no data/tensor reads, no run created.')
        return 0
    code = execute(value)
    if code or not (args.then_eval or args.action == 'train_eval'):
        return code
    # Keep train/eval on the GPU explicitly selected by the caller.  The
    # saved metadata still supplies all model settings; only the runtime
    # device is carried into the second command.
    follow = ['--experiment-dir', value['run'], '--gpu', str(args.gpu)]
    paths = {'--data_path', '--my_path', '--data_dir', '--save_dir', '--device'}
    index = 0
    while index < len(value['argv']):
        token = value['argv'][index]
        if token.split('=')[0] in paths:
            follow.append(token)
            if '=' not in token:
                index += 1; follow.append(value['argv'][index])
        index += 1
    evaluation = plan(args.task, 'eval', follow, value['environment'])
    return execute(evaluation)


if __name__ == '__main__':
    raise SystemExit(main())
