"""Seed propagation and sequential reporting; no real dataset entry imports."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_kcdno_tasks import ROOT, SMALL, arguments, construct, StandardRun
from cdlno.experiment import start, finish
from cdlno.kcdno.entry import seed_process

spec = importlib.util.spec_from_file_location('seed_suite', ROOT / 'tran_evaluate/kcdlno/_seed_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class SeedSuiteChecks(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        self.env.stop()

    def test_same_seed_replays_six_task_initializations(self):
        # Real constructors, small widths; original parsers extracted via AST.
        for task in suite.TASKS:
            states = []
            for seed in (0, 1, 2):
                def capture():
                    args = arguments(task, [*SMALL, '--gpu', '0', '--seed', str(seed)])
                    model = construct(task, args).eval()
                    return copy.deepcopy(model.state_dict()), (random.random(), np.random.rand(), torch.rand(4))
                first, draws = capture()
                repeat, repeat_draws = capture()
                self.assertEqual(first.keys(), repeat.keys())
                self.assertTrue(all(torch.equal(first[k], repeat[k]) for k in first), task)
                self.assertEqual(draws[:2], repeat_draws[:2])
                self.assertTrue(torch.equal(draws[2], repeat_draws[2]))
                states.append(first)
            for a, b in ((0, 1), (1, 2), (0, 2)):
                self.assertTrue(any(not torch.equal(states[a][k], states[b][k]) for k in states[a]), task)

    def test_seed_validation_and_omitted_seed_preserves_rng(self):
        seed_process(12, '0')
        expected = (random.random(), np.random.rand(), torch.rand(4))
        seed_process(12, '0')
        args = arguments('darcy')
        self.assertFalse(hasattr(args, 'seed'))
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        for invalid in ('-1', str(2**32), '0.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments('darcy', ['--seed', invalid])

    def test_recording_strict_checkpoint_and_eval_seed_recovery(self):
        for family in ('kcdno', 'lrsa_matched'):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                run_dir = Path(tmp) / 'run'
                small = ['--n-hidden', '8', '--n-heads', '2', '--n-layers', '2', '--slice_num', '3']
                args = arguments('darcy', ['--model', family, *small, '--seed', '2', '--kcdno-run-dir', str(run_dir)])
                recorder = start(args, 'darcy')
                try:
                    model = construct('darcy', args).eval()
                    run = StandardRun(args, model)
                    points = model.H * model.W
                    x, fx = torch.randn(2, points, 2), torch.randn(2, points, 1)
                    expected = model(x, fx).detach()
                    run.save(model)
                    # Explicit observation-only fixture, not a training loss.
                    recorder.record_epoch(1, {'synthetic_record_only': 0.25})
                finally:
                    finish(args)
                paths = [run_dir / name for name in ('architecture.json', 'config.json', 'task.json')]
                before = [p.read_bytes() for p in paths]
                self.assertEqual(json.loads(before[0])['initialization']['seed'], 2)
                self.assertEqual(json.loads(before[1])['resolved_arguments']['seed'], 2)
                ev = arguments('darcy', ['--model', family, '--eval', '1', '--kcdno-run-dir', str(run_dir)])
                self.assertEqual(ev.seed, 2)
                recorder = start(ev, 'darcy', evaluation=True)
                try:
                    loaded = construct('darcy', ev).eval()
                    erun = StandardRun(ev, loaded)
                    erun.load(loaded)
                    self.assertTrue(torch.equal(expected, loaded(x, fx)))
                    recorder.record_metrics({'synthetic_record_only': 0.5})
                finally:
                    finish(ev)
                self.assertEqual(before, [p.read_bytes() for p in paths])
                summary = suite.task_summary(run_dir, 'darcy', 2)
                self.assertEqual(summary['seed'], 2)
                self.assertEqual(summary['evaluation_metrics'], {'synthetic_record_only': 0.5})
                with self.assertRaisesRegex(ValueError, 'different task/seed'):
                    suite.task_summary(run_dir, 'darcy', 1)

    @staticmethod
    def recorded_command(command, env, check):
        """Only result-record fixtures; never creates fake physical dataset files."""
        task = Path(command[1]).stem
        phase = command[2]
        seed = int(command[command.index('--seed') + 1])
        directory = Path(command[command.index('--kcdno-run-dir') + 1])
        assert env['PYTHONHASHSEED'] == str(seed)
        result = dict(status='completed', task=task, seed=seed)
        if phase == 'train':
            directory.mkdir(parents=True, exist_ok=False)
            result['final_recorded_metrics_by_member'] = {'0': {'record_fixture': seed + .1}}
            (directory / 'train_results.json').write_text(json.dumps(result))
        else:
            result.update(metrics={'record_fixture': seed + .2}, result_file='evaluations/check/results.json')
            (directory / 'eval_results.json').write_text(json.dumps({'evaluations': [result]}))
        return subprocess.CompletedProcess(command, 0)

    def test_exact_order_immediate_reports_and_seed_isolation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CDLNO_RUNS_ROOT=tmp):
            tags = set()
            for seed in (0, 1, 2, 0):
                calls = []
                output = io.StringIO()
                def run(command, **kwargs):
                    task, phase = Path(command[1]).stem, command[2]
                    if phase == 'train' and calls:
                        previous = calls[-1][0]
                        self.assertIn(f'[seed={seed}] {previous}: training + evaluation completed', output.getvalue())
                    calls.append((task, phase))
                    return self.recorded_command(command, **kwargs)
                with patch.object(suite.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(output):
                    self.assertEqual(suite.run_suite(seed), 0)
                self.assertEqual(calls, [(task, phase) for task in suite.TASKS for phase in ('train', 'eval')])
                reports = sorted(Path(tmp).glob('_seed_suites/kcdno/*/summary.json'))
                report = json.loads(reports[-1].read_text())
                self.assertEqual((report['seed'], report['status']), (seed, 'completed'))
                self.assertEqual([r['task'] for r in report['completed_tasks']], list(suite.TASKS))
                for result in report['completed_tasks']:
                    self.assertEqual(result['seed'], seed)
                    saved = json.loads((Path(result['run_directory']) / 'seed_summary.json').read_text())
                    self.assertEqual(result, saved)
                tags.add(reports[-1].parent.name)
            self.assertEqual(len(tags), 4)

    def test_failure_stops_at_current_phase_and_keeps_previous_results(self):
        for failure in (('airfoil', 'train'), ('airfoil', 'eval')):
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CDLNO_RUNS_ROOT=tmp):
                calls = []
                def run(command, **kwargs):
                    call = (Path(command[1]).stem, command[2])
                    calls.append(call)
                    if call == failure:
                        raise subprocess.CalledProcessError(7, command)
                    return self.recorded_command(command, **kwargs)
                with patch.object(suite.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(suite.run_suite(1), 7)
                self.assertEqual(calls[-1], failure)
                self.assertFalse(any(task == 'plasticity' for task, phase in calls))
                report = json.loads(next(Path(tmp).glob('_seed_suites/kcdno/*/summary.json')).read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual([r['task'] for r in report['completed_tasks']], ['darcy'])

    def test_guarded_real_shell_previews_and_help(self):
        with tempfile.TemporaryDirectory(prefix='seed suite ') as tmp:
            root = Path(tmp)
            guard = root / 'forbidden python'
            marker = root / 'unexpected-python'
            guard.write_text('#!/usr/bin/env bash\nprintf called > "$SEED_TEST_MARKER"\nexit 97\n')
            guard.chmod(0o755)
            env = dict(os.environ, CDLNO_PYTHON=str(guard), SEED_TEST_MARKER=str(marker),
                       CDLNO_RUNS_ROOT=str(root / 'output'))
            for seed in (0, 1, 2):
                command = [sys.executable, str(ROOT / 'tran_evaluate/kcdlno/_seed_suite.py'),
                           '--seed', str(seed), '--gpu', '0', '--dry-run']
                result = subprocess.run(command, env=env, cwd=root, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.count('Command:'), 12)
                self.assertFalse(marker.exists())
                self.assertFalse((root / 'output').exists())
            launcher = ROOT / 'tran_evaluate/kcdlno/run_seed.sh'
            for args in (['--help'], []):
                result = subprocess.run(['bash', str(launcher), *args], env=env, cwd=root,
                                        stdin=subprocess.DEVNULL, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0 if args else 2)
                self.assertFalse(marker.exists())
            env['CDLNO_PYTHON'] = sys.executable
            for args, code in ((['0', '--dry-run'], 0), (['--seed', '2', '--dry-run'], 0),
                               (['--seed', '3', '--dry-run'], 2), (['--seed', '-1', '--dry-run'], 2)):
                result = subprocess.run(['bash', str(launcher), *args], env=env, cwd=root,
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
