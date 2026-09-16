"""Four actual static interfaces, without importing/running experiment modules."""

import argparse
import ast
from output_recording_projection import strip_recording
from collections import Counter
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'PDE-Solving-StandardBenchmark'
sys.path.insert(0, str(PROJECT))
from cdlno_entry import parse_args, model_kwargs, StaticRun
from model_dict import get_model
from cdlno.config import CDLNORuntimeConfig
from cdlno.checkpoint import SidecarMismatch, save_sidecar, load_sidecar, validate_sidecar
from cdlno.modules import ConvFFN
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss

TASKS = dict(darcy=('darcy', 85, 85, 8, 64, 4),
             elasticity=('elas', None, None, 8, 64, 1),
             airfoil=('airfoil', 221, 51, 4, 64, 4),
             pipe=('pipe', 129, 129, 4, 32, 8))
UPSTREAM = '75e0f67643806a81cd1d3f6adc88dd8c02416fe7'


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def tree(task):
    from msar_entry_projection import strip_msar
    return strip_msar(ast.parse((PROJECT / f'exp_{TASKS[task][0]}.py').read_text()))


def parser_for(task):
    # Execute only ArgumentParser/add_argument AST nodes, never an exp import.
    nodes = [node for node in tree(task).body if (
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'parser') or (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'parser' and node.value.func.attr == 'add_argument')]
    scope = dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<parser-only>', 'exec'), scope)
    return scope['parser']


def arguments(task, extras=()):
    return parse_args(parser_for(task), task, ['--model', 'CDLNO', *extras])


def construct(task, args=None, *, real_n=False):
    args = args or arguments(task)
    _, height, width, *_ = TASKS[task]
    height, width = (height, width) if real_n else (5, 7)
    main = next(node for node in tree(task).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    branch = next(node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args.model == 'CDLNO'")
    cuda_call = branch.body[0].value
    model_call = cuda_call.func.value  # exact constructor, omitting only device transfer
    return eval(compile(ast.Expression(body=model_call), '<constructor-only>', 'eval'),
                dict(args=args, get_model=get_model, cdlno_model_kwargs=model_kwargs,
                     s=height, s1=height, s2=width))


def sample(model, n=None, batch=2):
    n = n or (model.H * model.W if model.config.structured else 37)
    # Curved, deliberately unsorted physical coordinates; never replace with a
    # regular coordinate set to test Airfoil/Pipe input lifting.
    row = torch.linspace(-0.7, 1.4, n)
    x = torch.stack((row.sin() + row.square(), row.cos() - row), -1).repeat(batch, 1, 1).requires_grad_()
    fx = torch.randn(batch, n, 1, requires_grad=True) if model.fun_dim else None
    return x, fx


def assert_gradients(test, model, inputs):
    for name, p in [*model.named_parameters(), *inputs]:
        test.assertIsNotNone(p.grad, name)
        test.assertTrue(torch.isfinite(p.grad).all(), name)


class StaticInterfaceChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(505)

    def test_defaults_and_explicit_overrides_leave_legacy_parser_unchanged(self):
        for task, (_, _, _, heads, latents, batch) in TASKS.items():
            with self.subTest(task=task):
                old = parser_for(task).parse_args([])
                legacy = parse_args(parser_for(task), task, [])
                for name, value in vars(old).items():
                    self.assertEqual(getattr(legacy, name), value)
                self.assertEqual((legacy.n_layers, legacy.n_hidden, legacy.slice_num), (3, 64, 32))
                args = arguments(task)
                self.assertEqual((args.n_layers, args.front_blocks, args.n_hidden, args.n_heads, args.slice_num), (8, 2, 128, heads, latents))
                self.assertEqual((args.epochs, args.batch_size, args.lr, args.max_grad_norm), (500, batch, .001, .1))
                self.assertEqual((args.mlp_ratio, args.latent_ffn_ratio, args.cdpa_mode), (2, 2, 'entry'))
                # Values equal to old defaults still count as explicit overrides.
                changed = arguments(task, ['--n-layers=3', '--n-hidden', '64', '--slice_num', '32', '--front-blocks', '0', '--epochs', '7'])
                self.assertEqual((changed.n_layers, changed.n_hidden, changed.slice_num, changed.front_blocks, changed.epochs), (3, 64, 32, 0, 7))
                with redirect_stdout(io.StringIO()):
                    model = construct(task)
                self.assertEqual((model.config.L, model.config.P, model.config.M, model.config.num_heads), (8, 6, latents, heads))

    def test_four_interfaces_three_modes_backward_placeholder_and_no_time(self):
        for task in TASKS:
            for mode in ('off', 'entry', 'every_block'):
                with self.subTest(task=task, mode=mode):
                    model = construct(task, arguments(task, ['--cdpa-mode', mode]))
                    x, fx = sample(model)
                    with patch('numpy.load', side_effect=AssertionError('data load')):
                        y = model(x, fx, T=None)
                    self.assertEqual(y.shape, (*x.shape[:2], 1))
                    self.assertTrue(torch.isfinite(y).all())
                    TestLoss(size_average=False)(y.squeeze(-1), torch.randn_like(y[..., 0]) + 1).backward()
                    active_inputs = [('fx', fx)] if fx is not None else [('x', x)]
                    assert_gradients(self, model, active_inputs)
                    self.assertEqual(hasattr(model, 'placeholder'), task != 'darcy')
                    self.assertFalse(hasattr(model, 'time_fc'))
                    self.assertEqual(sum(isinstance(m, ConvFFN) for m in model.modules()), 0 if task == 'elasticity' else 3)
                    with self.assertRaises(ValueError):
                        model(x, fx, T=torch.ones(2, 1))
                    with self.assertRaises(ValueError):
                        model(x, None if fx is not None else torch.randn(2, x.shape[1], 1))

    def test_original_lifting_reference_order_coords_and_initialization(self):
        from model import Transolver_Structured_Mesh_2D as baseline
        for task in TASKS:
            model = construct(task, arguments(task, ['--n-hidden', '8', '--n-heads', '2', '--slice_num', '4']))
            x, fx = sample(model)
            seen = []
            handle = model.core.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
            model(x, fx)
            handle.remove()
            stem = baseline.MLP(model.preprocess[0].in_features, 16, 8, n_layers=0, res=False)
            with torch.no_grad():
                stem.linear_pre[0].load_state_dict(model.preprocess[0].state_dict())
                stem.linear_post.load_state_dict(model.preprocess[2].state_dict())
            if task == 'darcy':
                # Use the baseline's actual get_grid method, removing only its
                # unconditional CUDA transfer so this formula check runs on CPU.
                source = ast.parse((PROJECT / 'model/Transolver_Structured_Mesh_2D.py').read_text())
                cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Model')
                function = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_grid'))
                class CPUGrid(ast.NodeTransformer):
                    def visit_Call(self, node):
                        self.generic_visit(node)
                        return node.func.value if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda' else node
                function = CPUGrid().visit(function)
                scope = dict(torch=torch, np=np)
                exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<baseline-grid>', 'exec'), scope)
                position = scope['get_grid'](SimpleNamespace(H=model.H, W=model.W, ref=8)).reshape(1, model.H * model.W, 64)
                torch.testing.assert_close(model.pos, position, atol=0, rtol=0)
                self.assertEqual(model.preprocess[0].in_features, 65)
                expected = stem(torch.cat((position.expand(2, -1, -1), fx), -1))
                torch.testing.assert_close(model(x, fx), model(x.flip(1), fx), atol=0, rtol=0)
            else:
                self.assertEqual(model.preprocess[0].in_features, 2)
                expected = stem(x) + model.placeholder
            torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
        with patch('torch.nn.init.trunc_normal_', wraps=nn.init.trunc_normal_) as calls:
            model = construct('darcy')
        self.assertEqual(Counter(call.args[0].data_ptr() for call in calls.call_args_list),
                         Counter(m.weight.data_ptr() for m in model.modules() if isinstance(m, nn.Linear)))
        self.assertTrue(torch.equal(model.core.cdpa_at['0'].w, torch.zeros(128)))

    def test_real_n_with_smaller_width_latents_preserves_layout(self):
        for task in TASKS:
            model = construct(task, arguments(task, ['--n-hidden', '8', '--n-heads', '2', '--slice_num', '4']), real_n=True)
            x, fx = sample(model, n=972 if task == 'elasticity' else None, batch=1)
            y = model(x, fx)
            expected_n = dict(darcy=7225, elasticity=972, airfoil=11271, pipe=16641)[task]
            self.assertEqual(y.shape, (1, expected_n, 1))
            y.square().mean().backward()
            assert_gradients(self, model, [('fx', fx)] if fx is not None else [('x', x)])
            print('static real-N:', task, tuple(y.shape), 'backward passed')

    def test_darcy_exact_training_decode_and_gradient_loss_ast(self):
        module = tree('darcy')
        central = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'central_diff')
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        batch_loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and any(isinstance(q, ast.Expr) and ast.unparse(q.value) == 'optimizer.zero_grad()' for q in n.body))
        start = next(i for i, n in enumerate(batch_loop.body) if isinstance(n, ast.Expr) and ast.unparse(n.value) == 'optimizer.zero_grad()') + 1
        end = next(i for i, n in enumerate(batch_loop.body) if isinstance(n, ast.Expr) and ast.unparse(n.value) == 'loss.backward()')
        selected = [central, *batch_loop.body[start:end]]
        # Real loss statements and central_diff, no loops/readers/optimizers.
        code = compile(ast.Module(body=selected, type_ignores=[]), '<darcy-loss-only>', 'exec')
        model = construct('darcy', arguments('darcy', ['--n-hidden', '8', '--n-heads', '2', '--slice_num', '4']), real_n=True)
        n = 7225
        xn = UnitTransformer(torch.randn(4, n))
        yn = UnitTransformer(torch.randn(4, n) * 2 + 1)
        x, _ = sample(model, batch=2)
        coeff = torch.randn(2, n, requires_grad=True)
        target = torch.randn(2, n) * 2 + 1
        scope = dict(model=model, x=x, fx=xn.encode(coeff), y=yn.encode(target), y_normalizer=yn,
                     myloss=TestLoss(size_average=False), de_x=TestLoss(size_average=False),
                     de_y=TestLoss(size_average=False), torch=torch, rearrange=rearrange, F=F, s=85, dx=1/85)
        exec(code, scope)
        torch.testing.assert_close(scope['y'], target)
        for key in ('l2loss', 'deriv_loss'):
            gradient = torch.autograd.grad(scope[key], coeff, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 0)
        torch.testing.assert_close(scope['loss'], scope['l2loss'] + .1 * scope['deriv_loss'])
        scope['loss'].backward()
        assert_gradients(self, model, [('coeff', coeff)])

    def test_pipe_and_elasticity_original_normalizer_loss_connections(self):
        for task in ('pipe', 'elasticity'):
            model = construct(task, arguments(task, ['--n-hidden', '8', '--n-heads', '2', '--slice_num', '4']))
            x, _ = sample(model)
            if task == 'pipe':
                x = UnitTransformer(torch.randn(4, 35, 2)).encode(x)
            normalizer = UnitTransformer(torch.randn(4, x.shape[1]))
            y = normalizer.decode(model(x, None).squeeze(-1))
            TestLoss(size_average=False)(y, torch.randn_like(y) + 1).backward()
            assert_gradients(self, model, [])


class StaticCheckpointChecks(unittest.TestCase):
    def test_strict_roundtrip_sidecar_no_overwrite_and_separate_outputs(self):
        for task in TASKS:
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                path = Path(folder) / 'run'
                extras = ['--n-hidden','8','--n-heads','2','--slice_num','4','--cdlno-run-dir',str(path)]
                args = arguments(task, extras)
                model = construct(task, args)
                run = StaticRun(args, model)
                run.save(model)
                sidecar, state = run.sidecar.read_bytes(), run.checkpoint.read_bytes()
                with self.assertRaises(FileExistsError):
                    StaticRun(args, model)
                x, fx = sample(model)
                expected = model(x, fx)
                eval_args = arguments(task, [*extras, '--eval','1','--cdpa-source-chunk-size','1'])
                restored = construct(task, eval_args)
                evaluation = StaticRun(eval_args, restored)
                evaluation.load(restored)
                torch.testing.assert_close(restored(x, fx), expected, atol=1e-5, rtol=3e-4)
                self.assertNotEqual(run.result_dir, evaluation.result_dir)
                again = StaticRun(eval_args, restored)
                self.assertNotEqual(evaluation.result_dir, again.result_dir)
                with self.assertRaises(RuntimeError):
                    evaluation.save(restored)
                for flags in (['--front-blocks','0'], ['--n-layers','9'], ['--slice_num','5'], ['--cdpa-mode','off'], ['--ref','7']):
                    wrong = construct(task, arguments(task, [*extras, '--eval','1', *flags]))
                    with self.assertRaises(SidecarMismatch):
                        StaticRun(eval_args, wrong)
                if task != 'elasticity':
                    wrong = get_model(eval_args).Model(task_name=task, H=7, W=5, n_hidden=8, n_head=2, slice_num=4)
                    with self.assertRaises(SidecarMismatch):
                        StaticRun(eval_args, wrong)
                self.assertEqual(run.sidecar.read_bytes(), sidecar)
                self.assertEqual(run.checkpoint.read_bytes(), state)
                broken = torch.load(run.checkpoint, weights_only=True)
                broken.pop(next(iter(broken)))
                torch.save(broken, run.checkpoint)
                with self.assertRaises(RuntimeError):
                    evaluation.load(restored)
                run.checkpoint.write_bytes(state)
                # Adapter mismatch even when the core and tensor shapes agree.
                payload = json.loads(sidecar)
                payload['metadata']['wrapper_architecture']['version'] = 'different'
                run.sidecar.write_text(json.dumps(payload))
                corrupted = run.sidecar.read_bytes()
                with self.assertRaises(SidecarMismatch):
                    StaticRun(eval_args, restored)
                self.assertEqual(run.sidecar.read_bytes(), corrupted)

    def test_automatic_run_names_and_eval_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            original_cwd = Path.cwd()
            try:
                os.chdir(folder)
                args = arguments('elasticity', ['--n-hidden','8','--n-heads','2','--slice_num','4'])
                model = construct('elasticity', args)
                with patch.dict(os.environ, CDLNO_RUNS_ROOT=str(Path(folder) / 'output')):
                    a, b = StaticRun(args, model), StaticRun(args, model)
                self.assertNotEqual(a.directory, b.directory)
                self.assertEqual(a.directory.parent, Path(folder) / 'output' / 'elasticity')
                self.assertRegex(a.directory.name, r'^\d{8}T\d{12}Z$')
                args.eval = 1
                args.cdlno_run_dir = Path(folder) / 'missing'
                with self.assertRaises(FileNotFoundError):
                    StaticRun(args, model)
                self.assertFalse(args.cdlno_run_dir.exists())
            finally:
                os.chdir(original_cwd)

    def test_invalid_runtime_chunk_all_sidecar_boundaries(self):
        model = construct('elasticity', arguments('elasticity', ['--n-hidden','8','--n-heads','2','--slice_num','4']))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'architecture.json'
            save_sidecar(path, model.config)
            original = path.read_bytes()
            for bad in (True, 1.5, float('nan'), float('inf'), -1, '2'):
                runtime = CDLNORuntimeConfig(source_chunk_size=bad)
                with self.assertRaises(ValueError):
                    runtime.validate()
                with self.assertRaises(ValueError):
                    CDLNORuntimeConfig.from_dict({'source_chunk_size': bad})
                absent = Path(folder) / 'absent.json'
                with self.assertRaises(ValueError):
                    save_sidecar(absent, model.config, runtime)
                self.assertFalse(absent.exists())
                with self.assertRaises(ValueError):
                    validate_sidecar(path, model.config, runtime)
                self.assertEqual(path.read_bytes(), original)
                payload = json.loads(original)
                payload['runtime']['source_chunk_size'] = bad
                malformed = Path(folder) / 'bad.json'
                malformed.write_text(json.dumps(payload))
                before = malformed.read_bytes()
                with self.assertRaises(ValueError):
                    load_sidecar(malformed)
                self.assertEqual(malformed.read_bytes(), before)

    def test_fresh_process_wrapper_checkpoint_restore(self):
        for task in TASKS:
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                flags = ['--n-hidden','8','--n-heads','2','--slice_num','4', '--cdlno-run-dir',str(Path(folder) / 'run')]
                args = arguments(task, flags)
                model = construct(task, args)
                run = StaticRun(args, model)
                run.save(model)
                x, fx = sample(model)
                with torch.no_grad():
                    torch.save(dict(x=x.detach(), fx=None if fx is None else fx.detach(), y=model(x, fx)), Path(folder) / 'sample.pt')
                script = '''
import json, sys
from pathlib import Path
from argparse import Namespace
import torch
from cdlno_entry import StaticRun, model_kwargs
from model_dict import get_model
torch.set_num_threads(1)
folder = Path(sys.argv[1])
sidecar = folder / 'run/architecture.json'
original = sidecar.read_bytes()
payload = json.loads(original)
args = Namespace(**payload['metadata']['resolved_arguments'])
args.eval = 1
args.cdpa_source_chunk_size = 1
args.cdlno_run_dir = folder / 'run'
cfg = payload['architecture']
adapter = payload['metadata']['wrapper_architecture']
shape = cfg['grid_shape']
model = get_model(args).Model(n_layers=cfg['L'], n_hidden=cfg['d_model'], n_head=cfg['num_heads'],
    slice_num=cfg['M'], mlp_ratio=cfg['ffn_ratio'], ref=adapter['ref'], unified_pos=adapter['unified_pos'],
    H=None if shape is None else shape[0], W=None if shape is None else shape[1], **model_kwargs(args))
run = StaticRun(args, model)
run.load(model)
sample = torch.load(folder / 'sample.pt', weights_only=True)
with torch.no_grad():
    torch.testing.assert_close(model(sample['x'], sample['fx']), sample['y'], atol=1e-5, rtol=3e-4)
assert sidecar.read_bytes() == original
assert not any(name.startswith('exp_') for name in sys.modules)
'''
                result = subprocess.run([sys.executable, '-B', '-c', script, folder], cwd=PROJECT,
                                        env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class StaticGPUChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'GPU unavailable: wrapper GPU checks not run')
    def test_static_wrapper_device_and_amp_backward(self):
        print('static GPU:', torch.cuda.get_device_name())
        for task in TASKS:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(task=task, dtype=dtype):
                    model = construct(task, arguments(task, ['--n-hidden','16','--n-heads','2','--slice_num','4'])).cuda()
                    x, fx = sample(model)
                    x = x.detach().cuda().requires_grad_()
                    fx = fx.detach().cuda().requires_grad_() if fx is not None else None
                    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                        y = model(x, fx)
                    self.assertTrue(torch.isfinite(y).all())
                    y.float().square().mean().backward()
                    assert_gradients(self, model, [('fx', fx)] if fx is not None else [('x', x)])


class StaticFrozenChecks(unittest.TestCase):
    def test_legacy_ast_recovered_exactly_from_limited_branches(self):
        class LegacyProjection(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == 'cdlno_entry' else node
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('cdlno_run', 'results_dir'):
                    return None
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'parse_cdlno_args':
                    node.value = ast.parse('parser.parse_args()', mode='eval').body
                return self.generic_visit(node)
            def visit_If(self, node):
                if ast.unparse(node.test) in ("args.model == 'CDLNO'", 'cdlno_run is not None'):
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)
            def visit_Name(self, node):
                if node.id == 'results_dir':
                    return ast.parse("'./results/' + save_name + '/'", mode='eval').body
                return node
        for task, (name, *_) in TASKS.items():
            relative = f'PDE-Solving-StandardBenchmark/exp_{name}.py'
            original = subprocess.check_output(['git','show',f'{UPSTREAM}:{relative}'], cwd=ROOT, text=True)
            projected = LegacyProjection().visit(strip_recording(tree(task)))
            self.assertEqual(ast.dump(projected), ast.dump(ast.parse(original)), task)
        current = strip_recording(ast.parse((PROJECT / 'model_dict.py').read_text()))
        # M4 adds exactly one independent lifted-core selection branch. Assert
        # its entire AST before removing it from the legacy factory comparison.
        factory = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'get_model')
        msar = ast.parse("if args.model == 'msar_lno':\n    if getattr(args, 'msar_task', None) in ('ns', 'plasticity'):\n        from model import MSAR_Temporal\n        return MSAR_Temporal\n    if getattr(args, 'msar_task', None) in ('darcy', 'elasticity', 'airfoil', 'pipe'):\n        from model import MSAR_Standard\n        return MSAR_Standard\n    from model import MSAR_LNO\n    return MSAR_LNO").body[0]
        self.assertEqual(ast.dump(factory.body[0]), ast.dump(msar))
        factory.body.pop(0)
        original = subprocess.check_output(['git','show',f'{UPSTREAM}:PDE-Solving-StandardBenchmark/model_dict.py'], cwd=ROOT, text=True)
        self.assertEqual(ast.dump(LegacyProjection().visit(current)), ast.dump(ast.parse(original)))

    # This test captures arguments only, with a placeholder /existing-run.
    # Real eval sidecar resolution is covered by checkpoint and A2 task tests.
    @patch('cdlno.checkpoint.resolve_front_latent_mode', return_value='full')
    def test_scripts_syntax_arguments_and_fresh_imports(self, _mode):
        with tempfile.TemporaryDirectory() as folder:
            # Capture argv with a temporary executable, never run exp scripts or
            # manufacture datasets. This exercises shell paths and "$@" quoting.
            executable = Path(folder) / 'python'
            executable.write_text('#!/bin/sh\nexec "' + sys.executable + '" -B -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@"\n')
            executable.chmod(0o755)
            env = dict(os.environ, PATH=folder + os.pathsep + os.environ['PATH'])
            for script in sorted((PROJECT / 'scripts').glob('CDLNO_*.sh')):
                subprocess.run(['bash','-n',str(script)], check=True)
                command = ['bash',str(script),'--front-blocks','0','--data_path','/not a dataset/path']
                result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, check=True)
                argv = json.loads(result.stdout)
                task = next((t for t, (name, *_) in TASKS.items() if argv[0] == f'exp_{name}.py'), None)
                if task is None:
                    # Phase-6 temporal scripts are syntax/argument checked in
                    # test_temporal_standard; this test only validates the
                    # four static preset defaults.
                    continue
                if '_Eval' in script.stem:
                    argv += ['--cdlno-run-dir', '/existing-run']
                args = parse_args(parser_for(task), task, argv[1:])
                self.assertEqual(args.front_blocks, 0)
                self.assertEqual(args.data_path, '/not a dataset/path')
                self.assertEqual((args.model, args.n_layers, args.n_hidden), ('CDLNO', 8, 128))
                self.assertEqual(args.n_heads, TASKS[task][3])
                self.assertEqual(args.slice_num, TASKS[task][4])
                self.assertEqual(args.batch_size, TASKS[task][5])
        code = "from argparse import Namespace; from model_dict import get_model; m=get_model(Namespace(model='CDLNO',cdlno_task='elasticity')).Model(task_name='elasticity', n_hidden=8,n_head=2,slice_num=4); assert not hasattr(m,'time_fc'); import sys; assert not any(n.startswith('exp_') for n in sys.modules)"
        subprocess.run([sys.executable,'-B','-c',code], cwd=PROJECT, env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
