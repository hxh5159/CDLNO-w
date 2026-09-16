"""Sequential shell orchestration and exact result reporting; no task imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from uuid import uuid4


TASKS = ('darcy', 'airfoil', 'plasticity', 'elasticity', 'ns', 'pipe')
LAUNCHERS = Path(__file__).resolve().parent
ROOT = LAUNCHERS.parents[1]


def write_record(path, record):
    """Publish an incremental suite record without partially written JSON."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def task_summary(directory, task, seed):
    """Read the original metrics; never recompute losses or parse log numbers."""
    training = json.loads((directory / 'train_results.json').read_text())
    evaluations = json.loads((directory / 'eval_results.json').read_text())['evaluations']
    # The suite owns a NEW run and performs exactly one independent evaluation.
    if len(evaluations) != 1:
        raise ValueError(f'{task}: expected exactly one evaluation in this new run')
    evaluation = evaluations[0]
    for label, result in (('training', training), ('evaluation', evaluation)):
        if result.get('status') != 'completed' or result.get('task') != task or result.get('seed') != seed:
            raise ValueError(f'{task}: {label} result is incomplete or has a different task/seed')
    train_metrics = training.get('final_recorded_metrics_by_member', {})
    if not train_metrics or not evaluation.get('metrics'):
        raise ValueError(f'{task}: no recorded training/evaluation metrics; refusing an empty success summary')
    return dict(task=task, seed=seed, status='completed', run_directory=str(directory),
                training_metrics=train_metrics, evaluation_metrics=evaluation['metrics'],
                train_result_file=str(directory / 'train_results.json'),
                eval_result_file=str(directory / evaluation['result_file']))


def run_suite(seed, gpu='0', *, dry_run=False):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    tag = f'{stamp}_seed{seed}_{uuid4().hex[:8]}'
    output = Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT / 'output')).resolve()
    suite = output / '_seed_suites' / 'kcdno' / tag
    record = dict(seed=seed, model='kcdno', profile='kcdno_v1', task_order=list(TASKS),
                  gpu=str(gpu), status='running', completed_tasks=[], active_task=None,
                  active_phase=None, started_at_utc=stamp)
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    print(f'KCDNO seed={seed}; order: darcy -> airfoil -> plas -> elas -> ns -> pipe', flush=True)
    print(f'Suite results: {suite / "summary.json"}', flush=True)
    if not dry_run:
        suite.mkdir(parents=True, exist_ok=False)
        write_record(suite / 'summary.json', record)
    try:
        for task in TASKS:
            directory = output / task / 'kcdno' / tag
            for phase in ('train', 'eval'):
                record.update(active_task=task, active_phase=phase)
                print(f'\n[seed={seed}] {task}: {phase}', flush=True)
                if not dry_run:
                    write_record(suite / 'summary.json', record)
                command = ['bash', str(LAUNCHERS / f'{task}.sh'), phase,
                           '--gpu', str(gpu), '--seed', str(seed), '--kcdno-run-dir', str(directory)]
                if dry_run:
                    command.append('--dry-run')
                print('Launch: ' + shlex.join(command), flush=True)
                subprocess.run(command, env=env, check=True)
            if dry_run:
                print(f'[seed={seed}] {task}: DRY RUN; results will be printed here after evaluation.', flush=True)
                continue
            record['active_phase'] = 'report'
            result = task_summary(directory, task, seed)
            write_record(directory / 'seed_summary.json', result)
            record['completed_tasks'].append(result)
            write_record(suite / 'summary.json', record)
            print(f'\n[seed={seed}] {task}: training + evaluation completed', flush=True)
            print('Training: ' + json.dumps(result['training_metrics'], ensure_ascii=False), flush=True)
            print('Evaluation: ' + json.dumps(result['evaluation_metrics'], ensure_ascii=False), flush=True)
            print(f'Results: {directory / "seed_summary.json"}', flush=True)
        record.update(status='completed', active_task=None, active_phase=None)
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        if not dry_run:
            write_record(suite / 'summary.json', record)
        print(f'[seed={seed}] stopped at {record["active_task"]}/{record["active_phase"]}: {error}',
              file=sys.stderr, flush=True)
        if isinstance(error, KeyboardInterrupt):
            return 130
        return error.returncode if isinstance(error, subprocess.CalledProcessError) and error.returncode > 0 else 1
    if dry_run:
        print(f'\n[seed={seed}] all 12 commands previewed; no dataset entry or result directory created.', flush=True)
    else:
        write_record(suite / 'summary.json', record)
        print(f'\n[seed={seed}] all six tasks completed; summary: {suite / "summary.json"}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--gpu', default='0', help='original standard-task GPU selector')
    parser.add_argument('--dry-run', action='store_true', help='preview commands without dataset execution')
    args = parser.parse_args(argv)
    return run_suite(args.seed, args.gpu, dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
