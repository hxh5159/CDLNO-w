"""A3: synthetic optimizer/eval steps through existing task code, never exp imports.

Only loss/batch AST fragments are compiled from standard experiments. Industrial
train.py defines safe functions, called with real PyG loaders. Optional JSON is
evidence of completed cases, not a replacement test runner or a data fixture.
"""
import ast
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import test_front_task_modes as modes
from test_front_task_modes import static, car, air
from cdlno.modules import ConvFFN, PlainFFN

if car.HAS_PYG:
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader

RECORDS = []
FRAGMENTS = {}
STEMS = dict(darcy='darcy', elasticity='elas', airfoil='airfoil', pipe='pipe',
             ns='ns', plasticity='plas')


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)
    if os.environ.get('CDLNO_A3_EVIDENCE'):
        out = Path(os.environ['CDLNO_A3_EVIDENCE'])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'training-cases.json').write_text(json.dumps(RECORDS, indent=2) + '\n')
        (out / 'executed-standard-fragments.json').write_text(json.dumps(FRAGMENTS, indent=2) + '\n')


def execute(nodes, scope, task, label):
    """Record and execute original selected nodes without rewriting their math."""
    fragment = ast.Module(body=nodes, type_ignores=[])
    FRAGMENTS[f'{task}/{label}'] = ast.unparse(fragment)
    exec(compile(fragment, f'<original-{task}-{label}>', 'exec'), scope)


def source(task):
    from msar_entry_projection import strip_msar
    return strip_msar(ast.parse((static.PROJECT / f'exp_{STEMS[task]}.py').read_text()))


def training_nodes(task):
    tree = source(task)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == 'train_loader')
    # Synthetic tensors have already been placed on the test device. This is
    # the ONLY statement removed from the actual complete train batch body.
    nodes = [n for n in loop.body if not (isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Tuple) and '.cuda()' in ast.unparse(n.value))]
    assert len(nodes) == len(loop.body) - 1
    if task == 'elasticity':
        # Its original scheduler is outside the batch loop: execute once for
        # the synthetic one-batch epoch, without changing its location in prod.
        epoch = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and loop in n.body)
        nodes += [n for n in epoch.body if ast.unparse(n) == 'scheduler.step()']
        assert ast.unparse(nodes[-1]) == 'scheduler.step()'
    return nodes


def evaluation_nodes(task):
    loop = next(n for n in ast.walk(source(task)) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == 'test_loader')
    if task in static.TASKS:
        # Original forward, decoder and relative-L2 expression; no plot/file IO.
        return [n for n in loop.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('out', 'tl')]
    return [n for n in loop.body if (isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('bsz', 'loss'))
            or isinstance(n, ast.For)
            or (isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                and n.target.id in ('test_l2_step', 'test_l2_full'))]


def build(task, args, real_n=True):
    return static.construct(task, args, real_n=real_n) if task in static.TASKS else modes.build(task, args)


class FrontTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(915)
        self.backend = sdpa_kernel(SDPBackend.MATH)
        self.backend.__enter__()
        self.addCleanup(self.backend.__exit__, None, None, None)

    def finite_gradients(self, model):
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        # Zero initial depth scorer can yield zero norm gradients legitimately.

    def roundtrip(self, task, flags, model, inputs, *, real_n):
        model.eval()
        with torch.no_grad():
            expected = model(*inputs).detach().cpu()
        args = modes.parse(task, flags)
        run = modes.make_run(task, args, model)
        modes.save(task, run, model)
        sidecar = run.sidecar.read_bytes()
        # Omitted mode must come from sidecar; change chunk, not architecture.
        omitted = flags[:-2]
        self.assertEqual(flags[-2], '--front-latent-mode')
        ev = modes.parse(task, [*omitted, '--cdpa-source-chunk-size', '1'], True)
        self.assertEqual(ev.front_latent_mode, model.config.front_latent_mode)
        target = build(task, ev, real_n)
        evaluation = modes.make_run(task, ev, target, True)
        restored = modes.load(task, evaluation, target).to(next(model.parameters()).device).eval()
        with torch.no_grad():
            actual = restored(*inputs).detach().cpu()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=3e-4)
        if task == 'airfrans':
            member = evaluation.load(member=0).eval()
            with torch.no_grad():
                torch.testing.assert_close(member(*inputs), expected, atol=1e-5, rtol=3e-4)
        self.assertEqual(run.sidecar.read_bytes(), sidecar)
        for other in set(modes.MODES) - {model.config.front_latent_mode}:
            with patch('torch.load') as loading, self.assertRaisesRegex(modes.SidecarMismatch, 'front_latent_mode'):
                modes.parse(task, [*omitted, '--front-latent-mode', other], True)
            loading.assert_not_called()
            self.assertEqual(run.sidecar.read_bytes(), sidecar)
        return float((actual - expected).abs().max())

    def standard_case(self, task, mode, device='cpu', real_n=True):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            flags = [*modes.SMALL, *modes.run_option(task, Path(tmp) / 'run'), '--front-latent-mode', mode]
            args = modes.parse(task, flags)
            model = build(task, args, real_n).to(device)
            n = model.H * model.W if model.config.structured else 972
            batch = 2 if task in ('ns', 'plasticity') or not real_n else 1
            if task in static.TASKS:
                x, fx = static.sample(model, n=n, batch=batch)
                x = x.detach().to(device)
                fx = fx.detach().squeeze(-1).to(device) if fx is not None else None
            else:
                x = torch.randn(batch, n, 2, device=device)
                fx = torch.randn(batch, n, model.fun_dim, device=device)
            if task in ('pipe', 'plasticity'):
                x = static.UnitTransformer(torch.randn(3, n, 2, device=device)).encode(x)
            if task == 'darcy':
                fx = static.UnitTransformer(torch.randn(3, n, device=device)).encode(fx)
            scope = dict(torch=torch, F=static.F, rearrange=static.rearrange,
                         args=args, epochs=args.epochs, model=model, train_loader=[None],
                         x=x, fx=fx, myloss=static.TestLoss(size_average=False),
                         de_x=static.TestLoss(size_average=False), de_y=static.TestLoss(size_average=False),
                         s=model.H, dx=1 / model.H if model.H else None, T=20 if task == 'plasticity' else 10,
                         step=1, train_loss=0., reg=0., train_l2_step=0., train_l2_full=0.)
            if task == 'darcy':
                central = next(n for n in source(task).body if isinstance(n, ast.FunctionDef) and n.name == 'central_diff')
                execute([central], scope, task, 'central_diff')
            if task in static.TASKS:
                raw_y = torch.randn(batch, n, device=device) + 2
                if task != 'airfoil':
                    norm = static.UnitTransformer(torch.randn(3, n, device=device) + 2)
                    scope['y_normalizer'] = norm
                    scope['y'] = norm.encode(raw_y)
                    torch.testing.assert_close(norm.decode(scope['y']), raw_y)
                else:
                    scope['y'] = raw_y
            else:
                scope['yy'] = torch.randn(batch, n, 10, device=device) if task == 'ns' else torch.randn(batch, n, 4, 20, device=device)
                scope['tim'] = torch.stack((torch.linspace(0, 1, 20), torch.linspace(.1, .9, 20))).to(device)
            tree = source(task)
            setup = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('optimizer', 'scheduler')]
            execute(sorted(setup, key=lambda n: n.lineno), scope, task, 'optimizer_scheduler')
            optimizer, scheduler = scope['optimizer'], scope['scheduler']
            before = model.preprocess[0].weight.detach().clone()
            x_before, fx_before = x.clone(), fx.clone() if fx is not None else None
            observations, bridges, histories, history_grads, windows, predictions = [], [], [], [], [], []
            handles = []
            def front_hook(_m, _a, result):
                t = result[1]
                observations.append(t)
                if t.requires_grad:
                    t.register_hook(lambda grad: history_grads.append(grad.detach().clone()))
            for block in model.core.front_blocks:
                handles.append(block.register_forward_hook(front_hook))
                self.assertIsInstance(block.point_ffn, ConvFFN if model.config.structured else PlainFFN)
            handles.append(model.core.bridge.register_forward_hook(lambda _m, _a, result: bridges.append(result)))
            for module in model.core.cdpa_at.values():
                handles.append(module.register_forward_pre_hook(lambda _m, a: histories.append(tuple(a[1]))))
            if task == 'ns':
                handles.append(model.register_forward_pre_hook(
                    lambda _m, _a, kw: windows.append(kw['fx'].detach().clone()), with_kwargs=True))
                handles.append(model.register_forward_hook(lambda _m, _a, out: predictions.append(out.detach().clone())))
            backward_calls = []
            original_backward = torch.Tensor.backward
            def backward(tensor, *a, **kw):
                backward_calls.append(1)
                return original_backward(tensor, *a, **kw)
            model.train()
            with patch.object(optimizer, 'step', wraps=optimizer.step) as steps, \
                 patch.object(scheduler, 'step', wraps=scheduler.step) as schedules, \
                 patch.object(torch.Tensor, 'backward', backward):
                execute(training_nodes(task), scope, task, 'train_batch')
            updates = 20 if task == 'plasticity' else 1
            calls = 20 if task == 'plasticity' else 10 if task == 'ns' else 1
            self.assertEqual((steps.call_count, schedules.call_count, len(backward_calls)), (updates, 1, updates))
            self.assertFalse(torch.equal(before, model.preprocess[0].weight))
            self.assertTrue(optimizer.state)
            self.finite_gradients(model)
            self.assertEqual(len(bridges), calls)
            self.assertEqual(len(observations), calls * 2)
            self.assertEqual(len(history_grads), calls * 2)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().max() > 0 for g in history_grads))
            self.assertEqual(len(histories), calls)
            for i, history in enumerate(histories):
                self.assertEqual([id(t) for t in history], [id(t) for t in observations[2*i:2*i+2]])
            if task == 'ns':
                for t, value in enumerate(windows):
                    torch.testing.assert_close(value, torch.cat((fx_before[..., t:], scope['yy'][..., :t]), -1), atol=0, rtol=0)
                torch.testing.assert_close(scope['fx'], scope['yy'], atol=0, rtol=0)
            train_tensors = list(observations)
            observations.clear(); bridges.clear(); histories.clear(); windows.clear(); predictions.clear()
            model.eval()
            scope.update(fx=fx_before, test_l2_step=0., test_l2_full=0., loss=0.)
            if task in static.TASKS:
                scope['y'] = raw_y  # Original test labels are not encoded.
            with torch.no_grad():
                execute(evaluation_nodes(task), scope, task, 'eval_batch_without_plots')
            self.assertEqual(len(bridges), calls)
            self.assertFalse(set(map(id, train_tensors)) & set(map(id, observations)))
            if task == 'ns':
                for t, value in enumerate(windows):
                    torch.testing.assert_close(value, torch.cat((fx_before[..., t:], *predictions[:t]), -1), atol=0, rtol=0)
                self.assertEqual(scope['pred'].shape, (batch, n, 10))
                torch.testing.assert_close(scope['fx'], scope['pred'], atol=0, rtol=0)
            elif task == 'plasticity':
                self.assertEqual(scope['pred'].shape, (batch, n, 4, 20))
            else:
                self.assertTrue(np.isfinite(scope['tl']))
            for handle in handles:
                handle.remove()
            torch.testing.assert_close(x, x_before, atol=0, rtol=0)
            if fx is not None:
                torch.testing.assert_close(fx, fx_before, atol=0, rtol=0)
            if task == 'plasticity':
                model.zero_grad(set_to_none=True)
                t = torch.tensor([[.15], [.7]], device=device, requires_grad=True)
                out = model(x, fx, t)
                self.assertGreater((out - model(x, fx, t + .2)).abs().max().item(), 0)
                scope['myloss'](out.reshape(batch, -1), scope['yy'][..., 0].reshape(batch, -1)).backward()
                self.assertTrue(torch.isfinite(t.grad).all())
                self.assertTrue((t.grad.abs() > 0).all())
                for p in model.time_fc.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().max().item(), 0)
            inputs = (x, fx.unsqueeze(-1) if task == 'darcy' else fx,
                      scope['tim'][:, :1] if task == 'plasticity' else None)
            error = self.roundtrip(task, flags, model, inputs, real_n=real_n)
            RECORDS.append(dict(task=task, mode=mode, device=device, dtype='float32', backend='math',
                                real_N=real_n or task in ('ns', 'plasticity'), N=n, B=batch, d=8, M=4, L=8, F=2, cdpa='entry',
                                optimizer_steps=updates, backward_calls=updates, scheduler_steps=1,
                                train_forwards=calls, eval_forwards=calls, original_loss=True,
                                history_gradients=len(history_grads), checkpoint_max_abs_error=error,
                                status='passed', real_data=False))

    def industrial_case(self, task, mode, device='cpu'):
        if not car.HAS_PYG:
            self.skipTest('real torch_geometric unavailable: industrial integration NOT run')
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            flags = [*modes.SMALL, *modes.run_option(task, Path(tmp) / 'run'), '--front-latent-mode', mode]
            args = modes.parse(task, flags)
            model = modes.build(task, args).to(device)
            factory = car.RealPyGChecks.data if task == 'car' else air.AirRealPyG.data
            graph = factory(19 if task == 'car' else 23).to(device)
            def invoke(net, data):
                return net((data, None)) if task == 'car' else net(data)
            before_fields = {k: (v.clone(), v.data_ptr()) for k, v in graph if torch.is_tensor(v)}
            reference = copy.deepcopy(model)
            out = invoke(reference, graph)
            per_var = (out - graph.y).square()
            expected_loss = per_var[:, :3].mean() + .5 * per_var[graph.surf, 3].mean() if task == 'car' else per_var.mean()
            expected_loss.backward()
            training = air.module((car.CAR if task == 'car' else air.AIR) / 'train.py', f'a3_{task}_safe_train')
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001,
                total_steps=2 * (200 if task == 'car' else 398),
                **({'final_div_factor': 1000.} if task == 'car' else {}))
            loader = DataLoader([(graph, torch.randn(6, 3, device=device))] if task == 'car' else [graph], batch_size=1)
            before = model.preprocess[0].weight.detach().clone()
            with patch.object(optimizer, 'step', wraps=optimizer.step) as steps, \
                 patch.object(scheduler, 'step', wraps=scheduler.step) as schedules:
                result = training.train(device, model, loader, optimizer, scheduler,
                                        **({'reg': .5} if task == 'car' else {'criterion': 'MSE'}))
            self.assertEqual((steps.call_count, schedules.call_count), (1, 1))
            self.assertFalse(torch.equal(before, model.preprocess[0].weight))
            self.finite_gradients(model)
            for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)
            if task == 'car':
                self.assertAlmostEqual(result[0], per_var[graph.surf, 3].mean().item(), places=6)
                self.assertAlmostEqual(result[1], per_var[:, :3].mean().item(), places=6)
            else:
                self.assertAlmostEqual(float(result[0]), expected_loss.item(), places=6)
            metrics = training.test(device, model, loader)
            self.assertTrue(all(np.isfinite(v).all() for v in metrics))
            with torch.no_grad():
                output = invoke(model, graph)
                changed = graph.clone(); changed.y = torch.randn_like(graph.y) * 1000
                torch.testing.assert_close(invoke(model, changed), output, atol=0, rtol=0)
                del changed.y
                torch.testing.assert_close(invoke(model, changed), output, atol=0, rtol=0)
                for n in (7, 29):
                    data = factory(n).to(device)
                    self.assertEqual(invoke(model, Batch.from_data_list([data])).shape, (n, 4))
                multi = Batch.from_data_list([factory(3), factory(5)]).to(device)
                with self.assertRaisesRegex(ValueError, 'one physical graph'):
                    invoke(model, multi)
                if task == 'airfrans':
                    sampled = Batch.from_data_list([factory(29)]).to(device)
                    idx = torch.tensor([27, 0, 5, 9, 3, 12, 20], device=device)
                    for name in ('pos', 'x', 'y', 'surf', 'batch'):
                        setattr(sampled, name, getattr(sampled, name)[idx])
                    plain = air.Data(x=sampled.x, pos=sampled.pos)
                    torch.testing.assert_close(model(sampled), model(plain), atol=0, rtol=0)
                    self.assertEqual(sampled.ptr.tolist(), [0, 29])
                # Full point-count layout only; not a claim of full-width training.
                large_n = 32186 if task == 'car' else 32000
                large = invoke(model, factory(large_n).to(device))
                self.assertEqual(large.shape, (large_n, 4))
                self.assertTrue(torch.isfinite(large).all())
            for key, (value, pointer) in before_fields.items():
                torch.testing.assert_close(graph[key], value, atol=0, rtol=0)
                self.assertEqual(graph[key].data_ptr(), pointer)
            inputs = ((graph, None),) if task == 'car' else (graph,)
            error = self.roundtrip(task, flags, model, inputs, real_n=False)
            RECORDS.append(dict(task=task, mode=mode, device=device, dtype='float32', backend='math',
                                N=graph.num_nodes, layout_only_N=large_n, B=1, d=8, M=4, L=8, F=2,
                                cdpa='entry', optimizer_steps=1, scheduler_steps=1, original_loss=True,
                                real_PyG=True, variable_N=True, checkpoint_max_abs_error=error,
                                status='passed', real_data=False))

    def test_no_experiment_module_was_imported(self):
        for module in sys.modules.values():
            path = getattr(module, '__file__', None)
            if path and Path(path).parent in (static.PROJECT, car.CAR, air.AIR):
                self.assertFalse(Path(path).stem.startswith('exp_') or Path(path).name in ('main.py', 'main_evaluation.py'), path)


def case(task, mode, device='cpu', real_n=True):
    def check(self):
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable: GPU check NOT run')
        matmul, conv = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            if device == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            if task in modes.STANDARD:
                self.standard_case(task, mode, device, real_n)
            else:
                self.industrial_case(task, mode, device)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = conv
    return check


# Named unittest cases expose each coverage-table cell and independent failures.
for task in modes.TASKS:
    for mode in modes.MODES:
        setattr(FrontTraining, f'test_cpu_{task}_{mode}', case(task, mode))
for task in ('airfoil', 'pipe'):
    for mode in modes.MODES:
        setattr(FrontTraining, f'test_cpu_5x7_{task}_{mode}', case(task, mode, real_n=False))
# Finite representatives only: spatial conv, two time protocols and real PyG.
for task in ('darcy', 'ns', 'plasticity', 'car'):
    for mode in modes.MODES:
        setattr(FrontTraining, f'test_cuda_{task}_{mode}', case(task, mode, 'cuda', real_n=False))


if __name__ == '__main__':
    unittest.main()
