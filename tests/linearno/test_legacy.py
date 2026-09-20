import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class LegacyRegression(unittest.TestCase):
    def test_real_parser_AST_defaults_unchanged_and_new_flags_isolated(self):
        baseline = json.loads((ROOT/'docs/linearno_audit/l1/baseline.json').read_text())
        paths = [f'PDE-Solving-StandardBenchmark/exp_{t}.py' for t in ('darcy','elas','airfoil','pipe','ns','plas')]
        paths += [f'{project}/{entry}.py' for project in ('Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS')
                  for entry in ('main','main_evaluation')]
        def parser_only(path):
            tree = ast.parse(path.read_text())
            nodes = [n for n in tree.body if
                     isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser' or
                     isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'parser.add_argument']
            scope = {'argparse':argparse, 'Path':Path}
            exec(compile(ast.Module(body=nodes,type_ignores=[]), str(path)+':parser AST only','exec'),scope)
            return scope['parser']
        from cdlno.linearno.profiles import parse_overrides, resolve_config
        for path in paths:
            with self.subTest(path=path):
                old = parser_only(Path(baseline['snapshot'])/'source'/path)
                current = parser_only(ROOT/path)
                args = vars(current.parse_args([]))
                self.assertEqual(args, vars(old.parse_args([])))
                profile, explicit = parse_overrides(['--linearno-hidden','64'])
                resolve_config('darcy', profile, explicit=explicit, legacy_defaults=args)
                self.assertEqual(args, vars(current.parse_args([])))
                _, unknown = current.parse_known_args(['--linearno-rank','32'])
                self.assertEqual(unknown, ['--linearno-rank','32'])

    def test_eight_real_models_same_weights_and_checkpoints_fresh_workdirs(self):
        baseline = json.loads((HERE/'fixtures/transolver_cpu.json').read_text())
        for task, expected in baseline['tasks'].items():
            project = ('Car-Design-ShapeNetCar' if task == 'car' else 'Airfoil-Design-AirfRANS' if task == 'airfrans'
                       else 'PDE-Solving-StandardBenchmark')
            with self.subTest(task=task):
                proc = subprocess.run([sys.executable, '-B', str(HERE/'legacy_worker.py'), task], cwd=ROOT/project,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT)),
                                      text=True, capture_output=True, timeout=90)
                self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
                actual = json.loads(proc.stdout)
                for key in ('config','selection','input_hashes','parameters','keys','output_shape','roundtrip_max_error','whole_object_roundtrip'):
                    self.assertEqual(actual[key], expected[key], f'{task}/{key}; fixture versions={expected["torch"]}, {expected["python"]}')
                torch.testing.assert_close(torch.tensor(actual['output']), torch.tensor(expected['output']), atol=1e-6, rtol=1e-5)

    def test_all_preexisting_files_and_absent_frozen_directories(self):
        baseline = json.loads((ROOT/'docs/linearno_audit/l1/baseline.json').read_text())
        allowed = {'docs/LINEARNO_IMPLEMENTATION_STATUS.md'}
        # L6 authorizes the AirfRANS-only LinearNO adapter and its three thin
        # entry/train guards.  Their old behavior is covered by the dedicated
        # AirfRANS legacy fixture; this inventory must not mistake the new
        # branch for an accidental edit to the frozen Transolver path.
        air_integrated = {
            'Airfoil-Design-AirfRANS/cdlno_entry.py',
            'Airfoil-Design-AirfRANS/main.py',
            'Airfoil-Design-AirfRANS/main_evaluation.py',
            'Airfoil-Design-AirfRANS/train.py',
        }
        # L4 authorizes guarded entry/recording branches. Recover and compare the
        # ENTIRE old AST; retain byte hashes for every other baseline file.
        from linearno_entry_projection import strip_linearno
        integrated = {'PDE-Solving-StandardBenchmark/' + name for name in
                      ('exp_darcy.py','exp_elas.py','exp_airfoil.py','exp_pipe.py','exp_ns.py','exp_plas.py','model_dict.py','cdlno_entry.py')}
        integrated.add('cdlno/experiment.py')
        integrated.update('Car-Design-ShapeNetCar/'+name for name in
                          ('main.py','main_evaluation.py','train.py','models/cdlno_run.py'))
        for row in baseline['files']:
            if '__pycache__' in Path(row['path']).parts or row['path'].endswith(('.pyc', '.pyo')):
                continue
            if row['path'] in allowed:
                continue
            if row['path'] in air_integrated:
                continue
            with self.subTest(path=row['path']):
                if row['path']=='docs/LINEARNO_REPRODUCTION_MATRIX.md':
                    from frozen_revisions import expected_hash
                    self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),
                                     expected_hash(row['path'],row['sha256']))
                    continue
                if row['path'] in integrated:
                    before=(Path(baseline['snapshot'])/'source'/row['path']).read_text()
                    self.assertEqual(ast.dump(strip_linearno(ast.parse((ROOT/row['path']).read_text()))),
                                     ast.dump(ast.parse(before)))
                    continue
                if row['path']=='tests/msar_entry_projection.py':
                    before=(Path(baseline['snapshot'])/'source'/row['path']).read_text()
                    expected=before.replace('def strip_msar(tree):\n', 'def strip_msar(tree):\n    from linearno_entry_projection import strip_linearno\n    tree = strip_linearno(tree)\n')
                    self.assertEqual((ROOT/row['path']).read_text(),expected)
                    continue
                if row['path']=='tests/test_front_task_modes.py':
                    before=(Path(baseline['snapshot'])/'source'/row['path']).read_text()
                    expected=before.replace('next(n for n in main.body if isinstance(n, ast.If)',
                                            'next(n for n in ast.walk(main) if isinstance(n, ast.If)')
                    self.assertEqual((ROOT/row['path']).read_text(),expected)
                    continue
                if row['path'] in ('tests/test_airfrans.py','tests/test_shapenet_car.py'):
                    # L6's exact Air projection is reused by the old cross-project
                    # Car audit. Check the complete tests against pre-L7 content,
                    # allowing only that precise projection call adjustment.
                    l7=json.loads((ROOT/'docs/linearno_audit/l7/integration/baseline.json').read_text())
                    expected=(Path(l7['snapshot'])/'source'/row['path']).read_text()
                    if row['path'].endswith('test_shapenet_car.py'):
                        expected=expected.replace(
                            '                self.assertEqual(ast.dump(strip_recording(ast.parse((ROOT/name).read_text()))), ast.dump(ast.parse(before)), name)',
                            '                from test_airfrans import AirFrozenChecks\n'
                            '                tree = AirFrozenChecks._strip_linearno_air(ast.parse((ROOT/name).read_text()))\n'
                            '                self.assertEqual(ast.dump(strip_recording(tree)), ast.dump(ast.parse(before)), name)')
                    self.assertEqual((ROOT/row['path']).read_text(),expected)
                    continue
                from frozen_revisions import expected_hash
                from ll9r_test_source_projection import project_test_source
                raw=project_test_source(row['path'],(ROOT/row['path']).read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash(row['path'],row['sha256']))
        for path in ('LINEARNO','train_and_evaluate','CODEX/train_and_evaluate','evaluate'):
            self.assertFalse((ROOT/path).exists(), path)

    def test_no_task_parser_factory_imports_from_new_namespace(self):
        baseline = json.loads((ROOT/'docs/linearno_audit/l1/baseline.json').read_text())
        from linearno_entry_projection import strip_linearno
        for row in baseline['files']:
            if row['kind'] == 'tracked' and row['path'].endswith('.py'):
                if row['path'].startswith('Airfoil-Design-AirfRANS/'):
                    continue
                self.assertNotIn('cdlno.linearno', ast.unparse(strip_linearno(ast.parse((ROOT/row['path']).read_text()))), row['path'])
        # There is no monitor in this checkout; absence isn't called a passing monitor test.
        self.assertFalse((ROOT/'LINEARNO').exists())
