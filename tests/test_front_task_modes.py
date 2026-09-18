"""A2: actual parsers/constructors/checkpoints; synthetic tensors, never exp imports."""
import ast
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import test_static_standard as static
import test_temporal_standard as temporal
import test_shapenet_car as car
import test_airfrans as air
if car.HAS_PYG:
    from torch_geometric.data import Data, Batch
from cdlno import CDLNOArchitectureConfig
from cdlno.checkpoint import SidecarMismatch, save_sidecar, load_sidecar

ROOT = Path(__file__).resolve().parents[1]
MODES = ('full', 'no_sa', 'identity')
STANDARD = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity')
TASKS = (*STANDARD, 'car', 'airfrans')
SMALL = ['--n-hidden', '8', '--n-heads', '2', '--slice_num', '4']
LEGACY = Path(os.environ.get('CDLNO_A1_BASELINE', '/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0'))


def parse(task, options=(), evaluation=False):
    if task in STANDARD:
        p = static.parser_for(task) if task in static.TASKS else temporal.parser(task)
        return static.parse_args(p, task, ['--model', 'CDLNO', '--eval', str(int(evaluation)), *options])
    return car.args(options, evaluation) if task == 'car' else air.args(options, evaluation)


def build(task, args):
    if task in static.TASKS:
        return static.construct(task, args)
    if task in ('ns', 'plasticity'):
        # Evaluate the actual existing model constructor AST without .cuda/data.
        stem = 'ns' if task == 'ns' else 'plas'
        tree = ast.parse((static.PROJECT / f'exp_{stem}.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "args.model == 'CDLNO'")
        call = branch.body[0].value.func.value
        return eval(compile(ast.Expression(call), '<task-constructor-only>', 'eval'),
                    dict(args=args, get_model=static.get_model, cdlno_model_kwargs=static.model_kwargs,
                     h=64, s1=101, s2=31, T_in=10, Deformation=4))
    return car.Model(**car.model_kwargs(args)) if task == 'car' else air.AirfRANSModel(**air.entry.model_kwargs(args))


def make_run(task, args, model=None, evaluation=False):
    if task in STANDARD:
        return static.StaticRun(args, model)
    if task == 'car':
        return car.new_run(args, model, evaluation)
    return air.run(args, evaluation)


def run_option(task, path):
    return ['--cdlno-run-dir' if task in STANDARD else '--run_dir', str(path)]


def inputs(task, model, n=11):
    if task in STANDARD:
        count = model.H * model.W if model.config.structured else n
        x = torch.randn(2, count, 2)
        fx = torch.randn(2, count, model.fun_dim) if model.fun_dim else None
        return (x, fx, torch.tensor([[.15], [.8]]) if task == 'plasticity' else None)
    if not car.HAS_PYG:
        raise unittest.SkipTest('real PyG unavailable: industrial tensor integration not run')
    graph = Batch.from_data_list([Data(x=torch.randn(n, 7), pos=torch.randn(n, 2),
                                      y=torch.randn(n, 4), surf=torch.arange(n) % 2 == 0)])
    return ((graph, Data()),) if task == 'car' else (graph,)


def save(task, run, model):
    if task in STANDARD:
        run.save(model)
    elif task == 'car':
        torch.save(model, run.checkpoint)
    else:
        torch.save([model], run.checkpoint)
        Path(run.member_dir(0)).mkdir()
        torch.save(model, Path(run.member_dir(0)) / 'model')


def load(task, run, model):
    if task in STANDARD:
        run.load(model)
        return model
    saved = run.load()
    return saved if task == 'car' else saved[0]


class TaskFrontModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(222)
        context = sdpa_kernel(SDPBackend.MATH)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_eight_task_three_mode_real_constructor_backward_and_roundtrip(self):
        for task in TASKS:
            for mode in MODES:
                with self.subTest(task=task, mode=mode), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                    options = [*SMALL, *run_option(task, Path(tmp) / 'run')]
                    a = parse(task, [*options, '--front-latent-mode', mode])
                    m = build(task, a).eval()
                    self.assertEqual(m.config.front_latent_mode, mode)
                    self.assertTrue(all(b.front_latent_mode == mode for b in m.core.front_blocks))
                    args = inputs(task, m)
                    output = m(*args)
                    self.assertEqual(output.shape[-1], 4 if task in ('plasticity', 'car', 'airfrans') else 1)
                    if task in STANDARD:
                        loss = static.TestLoss(size_average=False)(output.flatten(1), torch.randn_like(output).flatten(1))
                    else:
                        graph = args[0][0] if task == 'car' else args[0]
                        # Execute the original loss assignments only. Air's default
                        # MSE backward uses total_loss; masks also supply metrics.
                        path = (car.CAR if task == 'car' else air.AIR) / 'train.py'
                        original = ast.parse(path.read_text())
                        train = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
                        names = {'loss_press', 'loss_velo_var', 'loss_velo', 'total_loss'} if task == 'car' else {
                            'loss_per_var', 'total_loss', 'loss_surf_var', 'loss_vol_var', 'loss_surf', 'loss_vol'}
                        assignments = [n for n in ast.walk(train) if isinstance(n, ast.Assign)
                                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
                        assignments.sort(key=lambda n: n.lineno)
                        scope = dict(out=output, targets=graph.y, cfd_data=graph, data_clone=graph,
                                     criterion_func=torch.nn.MSELoss(reduction='none'),
                                     loss_criterion=torch.nn.MSELoss(reduction='none'), reg=.5)
                        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<original-mask-loss>', 'exec'), scope)
                        loss = scope['total_loss']
                    loss.backward()
                    for name, p in m.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    run = make_run(task, a, m)
                    save(task, run, m)
                    before = run.sidecar.read_bytes()
                    self.assertEqual(json.loads(before)['architecture']['front_latent_mode'], mode)
                    self.assertEqual(json.loads(before)['metadata']['resolved_arguments']['front_latent_mode'], mode)
                    if task in STANDARD and mode == 'full':
                        legacy = json.loads(before)
                        legacy['architecture'].pop('front_latent_mode')
                        legacy['architecture']['history_rule'] = 'front-t-after-ffn2-before-up-v1'
                        run.sidecar.write_text(json.dumps(legacy))
                        before = run.sidecar.read_bytes()
                    # OMIT mode: eval must infer it before a constructor sees defaults.
                    ev = parse(task, [*options, '--cdpa-source-chunk-size', '1'], True)
                    self.assertEqual(ev.front_latent_mode, mode)
                    restored = build(task, ev)
                    evaluation = make_run(task, ev, restored, True)
                    restored = load(task, evaluation, restored).eval()
                    torch.testing.assert_close(restored(*args), output, atol=1e-5, rtol=3e-4)
                    if task == 'airfrans':
                        torch.testing.assert_close(evaluation.load(member=0).eval()(*args), output, atol=1e-5, rtol=3e-4)
                    for conflict in set(MODES) - {mode}:
                        with patch('torch.load') as unpickle, self.assertRaisesRegex(SidecarMismatch, 'front_latent_mode'):
                            parse(task, [*options, '--front-latent-mode', conflict], True)
                        unpickle.assert_not_called()
                    self.assertEqual(run.sidecar.read_bytes(), before)
                    with self.assertRaises(FileExistsError):
                        make_run(task, a, m)

    def test_defaults_cli_precedence_frozen_presets_and_legacy_branches(self):
        presets = json.loads((ROOT / 'docs/front_ablation_audit/a2/before-presets.json').read_text())
        for task in TASKS:
            a = parse(task)
            self.assertEqual(a.front_latent_mode, 'full')
            self.assertEqual(a.slice_num, 32 if task == 'pipe' else 64)
            self.assertEqual(parse(task, ['--front-latent-mode=no_sa', '--front_latent_mode', 'identity']).front_latent_mode, 'identity')
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                parse(task, ['--front-latent-mode', 'invalid'])
        for name, before in presets.items():
            now = json.loads((ROOT / name).read_text())
            self.assertEqual(now['model'].pop('front_latent_mode'), 'full')
            self.assertEqual(now, before)
        # Both legacy train and eval parsers return before sidecar resolution.
        with patch('cdlno.checkpoint.resolve_front_latent_mode', side_effect=AssertionError('legacy read')):
            for task in STANDARD:
                p = static.parser_for(task) if task in static.TASKS else temporal.parser(task)
                result = static.parse_args(p, task, ['--eval', '1'])
                self.assertNotEqual(result.model, 'CDLNO')
            for ev in (False, True):
                car.parse_args(car.entry_parser(ev), evaluation=ev, argv=['--cfd_model', 'Transolver'])
                air.entry.parse_args(air.parser(ev), evaluation=ev, argv=[] if ev else ['--model', 'Transolver'])

    def test_sidecar_missing_fields_and_unknown_legacy_rejected(self):
        cfg = CDLNOArchitectureConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'architecture.json'
            save_sidecar(path, cfg)
            original = json.loads(path.read_text())
            legacy = copy.deepcopy(original)
            legacy['architecture'].pop('front_latent_mode')
            legacy['architecture']['history_rule'] = 'front-t-after-ffn2-before-up-v1'
            path.write_text(json.dumps(legacy)); before = path.read_bytes()
            self.assertEqual(load_sidecar(path)['architecture']['front_latent_mode'], 'full')
            self.assertEqual(path.read_bytes(), before)
            for field in cfg.to_dict():
                bad = copy.deepcopy(legacy)
                if field != 'front_latent_mode':
                    bad['architecture'].pop(field)
                else:
                    bad['architecture']['history_rule'] = cfg.history_rule
                path.write_text(json.dumps(bad)); before = path.read_bytes()
                with self.assertRaises(SidecarMismatch):
                    load_sidecar(path)
                self.assertEqual(path.read_bytes(), before)

    def test_industrial_object_modes_configs_and_weights_must_agree(self):
        for task in ('car', 'airfrans'):
            for mode in MODES:
                with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                    flags = [*SMALL, *run_option(task, Path(tmp) / 'run'), '--front-latent-mode', mode]
                    a = parse(task, flags); m = build(task, a)
                    run = make_run(task, a, m); save(task, run, m)
                    ev = make_run(task, parse(task, flags, True), None, True)
                    before = run.sidecar.read_bytes()
                    for issue in ('block_mode', 'core_config', 'partial_config', 'missing_weight', 'missing_mode'):
                        broken = copy.deepcopy(m)
                        if issue == 'block_mode': broken.core.front_blocks[0].front_latent_mode = 'identity' if mode != 'identity' else 'full'
                        if issue == 'core_config': broken.core.config = replace(m.config, front_latent_mode='identity' if mode != 'identity' else 'full')
                        if issue == 'partial_config': broken.config = {'front_latent_mode': mode}
                        if issue == 'missing_weight': del broken.placeholder
                        if issue == 'missing_mode':
                            if mode == 'full': continue  # Legitimate legacy case checked with genuine fixtures below.
                            del broken.core.front_blocks[0].front_latent_mode
                        torch.save(broken if task == 'car' else [broken], run.checkpoint)
                        with self.assertRaises((SidecarMismatch, RuntimeError)):
                            ev.load()
                        self.assertEqual(run.sidecar.read_bytes(), before)

    @unittest.skipUnless(car.HAS_PYG and LEGACY.is_dir(), 'requires real PyG and genuine pre-A1 CDLNO_A1_BASELINE artifacts')
    def test_legacy_industrial_objects_through_actual_run_loaders(self):
        for task, group in (('car', 'car'), ('airfrans', 'airfrans')):
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                flags = [*SMALL, *run_option(task, Path(tmp) / 'run')]
                a = parse(task, flags); m = build(task, a)
                run = make_run(task, a, m)
                fixture_dir = LEGACY / group / '11'
                self.assertTrue(fixture_dir.exists(), 'real pre-A1 artifacts required; do not manufacture legacy objects')
                payload = json.loads(run.sidecar.read_text())
                payload['architecture'] = json.loads((fixture_dir / 'architecture.json').read_text())['architecture']
                run.sidecar.write_text(json.dumps(payload))
                if task == 'airfrans':
                    payload['metadata']['run_contract']['nmodel'] = 2
                    run.sidecar.write_text(json.dumps(payload)); flags += ['--nmodel', '2']
                shutil.copyfile(fixture_dir / ('legacy-model.pt' if task == 'car' else 'legacy-list.pt'), run.checkpoint)
                before = run.sidecar.read_bytes()
                ev = make_run(task, parse(task, flags, True), None, True)
                saved = ev.load(); models = [saved] if task == 'car' else saved
                fixture = torch.load(fixture_dir / 'fixture.pt', weights_only=True)
                for old in models:
                    self.assertTrue(all(not hasattr(b, 'front_latent_mode') for b in old.core.front_blocks))
                    x = fixture['input']; graph = Batch.from_data_list([Data(x=x, pos=x[:, :2], y=torch.zeros(x.shape[0], 4))])
                    y = old.eval()((graph, Data()) if task == 'car' else graph)
                    torch.testing.assert_close(y, fixture['output'][0], atol=0, rtol=0)
                self.assertEqual(run.sidecar.read_bytes(), before)

    def test_root_scripts_three_modes_paths_overrides_and_parser(self):
        for task in TASKS:
            for mode in MODES:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'run'
                    path.mkdir()
                    save_sidecar(path / 'architecture.json', CDLNOArchitectureConfig(front_latent_mode=mode))
                    for evaluation in (False, True):
                        tokens = subprocess.check_output(['bash', str(ROOT / 'tran_evaluate' / f'{task}.sh'),
                            'eval' if evaluation else 'train', '--front-latent-mode', 'identity',
                            '--front_latent_mode=' + mode, '--dry-run'], cwd=ROOT, text=True,
                            env=dict(os.environ, CDLNO_RUN_TAG='test_timestamp'))
                        command = shlex.split(next(s.removeprefix('Command:') for s in tokens.splitlines() if s.startswith('Command:')))
                        argv = command[3:]
                        option = '--cdlno-run-dir' if task in STANDARD else '--run_dir'
                        implicit = argv[argv.index(option) + 1]
                        self.assertEqual(Path(implicit).name, 'test_timestamp')
                        self.assertEqual(Path(implicit).parent.name, task)
                        # Actual parser with a synthetic config fixture, no task/data execution.
                        argv += run_option(task, path)
                        if task in STANDARD:
                            p = static.parser_for(task) if task in static.TASKS else temporal.parser(task)
                            a = static.parse_args(p, task, argv)
                        elif task == 'car': a = car.parse_args(car.entry_parser(evaluation), evaluation=evaluation, argv=argv)
                        else: a = air.entry.parse_args(air.parser(evaluation), evaluation=evaluation, argv=argv)
                        self.assertEqual(a.front_latent_mode, mode)
                        self.assertEqual(a.slice_num, 32 if task == 'pipe' else 64)

    def test_automatic_run_names_explicit_save_name_and_missing_eval(self):
        for task in TASKS:
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                previous = Path.cwd()
                try:
                    os.chdir(tmp)
                    directories = []
                    for mode in MODES:
                        a = parse(task, [*SMALL, '--front-latent-mode', mode])
                        model = build(task, a)
                        with patch.dict(os.environ, CDLNO_RUNS_ROOT=str(Path(tmp) / 'output')):
                            run = make_run(task, a, model)
                        directories.append(run.directory)
                        self.assertEqual(run.directory.parent, Path(tmp) / 'output' / task)
                        self.assertRegex(run.directory.name, r'^\d{8}T\d{12}Z$')
                        if task in STANDARD:
                            named = parse(task, [*SMALL, '--front-latent-mode', mode, '--save_name', 'chosen'])
                            self.assertEqual(named.save_name, 'chosen')
                    self.assertEqual(len(set(directories)), 3)
                    with patch('torch.load') as unpickle, self.assertRaises(FileNotFoundError):
                        parse(task, [*SMALL, *run_option(task, Path(tmp) / 'absent')], True)
                    unpickle.assert_not_called()
                    self.assertFalse((Path(tmp) / 'absent').exists())
                finally:
                    os.chdir(previous)

    def test_three_original_workdirs_new_process_import_and_all_mode_loads(self):
        for project, tasks in (('PDE-Solving-StandardBenchmark', STANDARD),
                               ('Car-Design-ShapeNetCar', ('car',)), ('Airfoil-Design-AirfRANS', ('airfrans',))):
            with self.subTest(project=project), tempfile.TemporaryDirectory() as tmp:
                if project != 'PDE-Solving-StandardBenchmark' and not car.HAS_PYG:
                    self.skipTest('real PyG unavailable: industrial fresh-process integration not run')
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', tmp, *tasks],
                    cwd=ROOT / project, env=dict(os.environ, PYTHONPATH=str(ROOT)), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def worker(directory, tasks):
    torch.set_num_threads(1)
    for task in tasks:
        for mode in MODES:
            flags = [*SMALL, *run_option(task, Path(directory) / (task + '_' + mode))]
            a = parse(task, [*flags, '--front-latent-mode', mode]); model = build(task, a).eval()
            run = make_run(task, a, model); save(task, run, model)
            ev = parse(task, flags, True); target = build(task, ev)
            restored = load(task, make_run(task, ev, target, True), target).eval()
            x = inputs(task, model)
            torch.testing.assert_close(model(*x), restored(*x), atol=0, rtol=0)
            assert restored.config.front_latent_mode == mode
    assert not any(name.startswith('exp_') for name in sys.modules)
    print('fresh process modes/load PASS', tasks)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': worker(sys.argv[2], sys.argv[3:])
    else: unittest.main()
