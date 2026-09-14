"""Data-free ShapeNet-Car integration and frozen-source checks.

Real PyG tests skip only if torch_geometric itself is absent. Import errors in
our wrapper or installed dependencies remain failures; no fake PyG is provided.
"""
from argparse import ArgumentParser
import ast
import copy
from contextlib import redirect_stdout
import importlib.util
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

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CAR = ROOT / 'Car-Design-ShapeNetCar'
sys.path.insert(0, str(CAR))
from models.CDLNO import Model
from models.cdlno_run import CarRun, parse_args, model_kwargs
from cdlno.checkpoint import SidecarMismatch

HAS_PYG = importlib.util.find_spec('torch_geometric') is not None
if HAS_PYG:
    from torch_geometric.data import Batch, Data
    from torch_geometric.loader import DataLoader


def original(filename):
    return subprocess.check_output(['git', 'show', '75e0f67643806a81cd1d3f6adc88dd8c02416fe7:'
                                    + 'Car-Design-ShapeNetCar/' + filename], cwd=ROOT, text=True)


def entry_parser(evaluation=False, baseline=False):
    filename = 'main_evaluation.py' if evaluation else 'main.py'
    source = original(filename) if baseline else (CAR / filename).read_text()
    nodes = []
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'parser'):
            nodes.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and ast.unparse(node.value.func.value) == 'parser'):
            nodes.append(node)
    namespace = {'argparse': __import__('argparse')}
    # Extract only argument definitions; never execute/import an entry or loader.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<car-parser-only>', 'exec'), namespace)
    return namespace['parser']


def args(extra=(), evaluation=False):
    return parse_args(entry_parser(evaluation), evaluation=evaluation,
                      argv=['--cfd_model', 'CDLNO', *extra])


SMALL = ['--n_hidden', '16', '--n_heads', '4', '--slice_num', '4']


def new_run(a, model=None, evaluation=False):
    with redirect_stdout(io.StringIO()):
        return CarRun(a, device=torch.device('cpu'), model=model, evaluation=evaluation)


class WrapperAndProtocol(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(417)

    def test_defaults_active_parameters_and_three_modes(self):
        a = args()
        self.assertEqual((a.n_hidden, a.n_heads, a.slice_num, a.n_layers, a.front_blocks),
                         (256, 8, 64, 8, 2))
        self.assertEqual((a.nb_epochs, a.batch_size, a.lr, a.weight), (200, 1, .001, .5))
        for mode in ('off', 'entry', 'every_block'):
            m = Model(**model_kwargs(args(['--cdpa_mode', mode])))
            x = torch.randn(9, 7, requires_grad=True)
            out = m((SimpleNamespace(x=x), None))
            self.assertEqual(out.shape, (9, 4))
            out.square().sum().backward()
            self.assertFalse(hasattr(m, 'time_fc'))
            self.assertFalse(any(isinstance(n, nn.Conv2d) for n in m.modules()))
            self.assertEqual(m.preprocess[0].in_features, 7)
            self.assertEqual(m.placeholder.shape, (256,))
            for name, p in m.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(x.grad.abs().max().item(), 0)
            self.assertGreater(m.placeholder.grad.abs().max().item(), 0)
            for cdpa in m.core.cdpa_at.values():
                self.assertTrue(torch.equal(cdpa.w, torch.zeros_like(cdpa.w)))

    def test_baseline_lifting_placeholder_and_query_initialization(self):
        from models.Transolver import Model as Baseline
        m = Model(**model_kwargs(args(SMALL)))
        old = Baseline(space_dim=7, fun_dim=0, out_dim=4, n_hidden=16,
                       n_head=4, slice_num=4, n_layers=1, mlp_ratio=2, unified_pos=0)
        with torch.no_grad():
            old.preprocess.linear_pre[0].load_state_dict(m.preprocess[0].state_dict())
            old.preprocess.linear_post.load_state_dict(m.preprocess[2].state_dict())
            old.placeholder.copy_(m.placeholder)
        x = torch.randn(1, 13, 7)
        captured = []
        hook = m.core.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        m((SimpleNamespace(x=x[0]), None))
        hook.remove()
        torch.testing.assert_close(captured[0], old.preprocess(x) + old.placeholder[None, None], rtol=0, atol=0)
        self.assertTrue(((m.placeholder >= 0) & (m.placeholder < 1 / 16)).all())
        for front in m.core.front_blocks:
            q = front.down.latent_queries.reshape(4, 16)
            torch.testing.assert_close(q @ q.T, torch.eye(4), atol=5e-7, rtol=1e-6)

    def test_parser_overrides_legacy_and_invalid_contracts(self):
        a = args(['--n_layers', '12', '--front_blocks', '7', '--slice_num', '32',
                  '--mlp_ratio', '3', '--latent_ffn_ratio', '4', '--fold_id', '8'])
        m = Model(**model_kwargs(a))
        self.assertEqual((m.config.L, m.config.F, m.config.P, m.config.M), (12, 7, 5, 32))
        self.assertEqual((m.config.ffn_ratio, m.config.latent_ffn_ratio), (3, 4))
        old = parse_args(entry_parser(), argv=['--cfd_model', 'Transolver'])
        base = entry_parser(baseline=True).parse_args(['--cfd_model', 'Transolver'])
        self.assertEqual({k: getattr(old, k) for k in vars(base)}, vars(base))
        for flags in (['--batch_size', '2'], ['--fold_id', '-1'], ['--fold_id', '9']):
            with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    args(flags)
        for flags in (['--front_blocks', '8'], ['--cdpa_source_chunk_size', '-1']):
            with self.assertRaises(ValueError):
                args(flags)
        with self.assertRaises(TypeError):
            Model(rear_depth=6)

    def test_run_isolation_and_sidecar_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            a = args([*SMALL, '--run_dir', str(Path(temp) / 'run'), '--fold_id', '2'])
            m = Model(**model_kwargs(a))
            run = new_run(a, m)
            torch.save(m, run.checkpoint)
            before = run.sidecar.read_bytes()
            checkpoint_before = run.checkpoint.read_bytes()
            self.assertEqual(json.loads(before)['runtime']['source_chunk_size'], 0)
            with self.assertRaises(FileExistsError):
                new_run(a, m)
            flags = [*SMALL, '--run_dir', str(run.directory), '--fold_id', '2']
            evaluation = new_run(args([*flags, '--cdpa_source_chunk_size', '1'], True), evaluation=True)
            another = new_run(args(flags, True), evaluation=True)
            self.assertNotEqual(evaluation.result_dir, another.result_dir)
            self.assertEqual(Path(evaluation.result_dir).parent, run.directory)
            with patch('torch.load', wraps=torch.load) as loading:
                loaded = evaluation.load()
                self.assertEqual(loading.call_count, 1)
                self.assertIs(loading.call_args.kwargs['weights_only'], False)
                self.assertEqual(loading.call_args.kwargs['map_location'], torch.device('cpu'))
            self.assertEqual(loaded.core.source_chunk_size, 1)
            x = SimpleNamespace(x=torch.randn(10, 7))
            torch.testing.assert_close(m((x, None)), loaded((x, None)), atol=1e-5, rtol=3e-4)
            for change in (['--fold_id', '3'], ['--slice_num', '5'], ['--cdpa_mode', 'off'], ['--nb_epochs', '201']):
                with patch('torch.load') as forbidden:
                    with self.assertRaises(SidecarMismatch):
                        new_run(args([*flags, *change], True), evaluation=True)
                    forbidden.assert_not_called()
            self.assertEqual(run.sidecar.read_bytes(), before)
            self.assertEqual(run.checkpoint.read_bytes(), checkpoint_before)
            broken = json.loads(before)
            broken['metadata']['wrapper_architecture']['placeholder'] = False
            run.sidecar.write_text(json.dumps(broken))
            with patch('torch.load') as forbidden:
                with self.assertRaises(SidecarMismatch):
                    evaluation.load()
                forbidden.assert_not_called()
            run.sidecar.unlink()
            with patch('torch.load') as forbidden:
                with self.assertRaises(FileNotFoundError):
                    evaluation.load()
                forbidden.assert_not_called()
            self.assertFalse(run.sidecar.exists())
            # Test automatic run names without creating anything in the repo.
            auto = args(SMALL)
            saved_cwd = Path.cwd()
            try:
                os.chdir(temp)
                r1, r2 = new_run(auto, m), new_run(auto, m)
                self.assertNotEqual(r1.directory, r2.directory)
            finally:
                os.chdir(saved_cwd)

    def test_checkpoint_rejects_wrong_object_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            a = args([*SMALL, '--run_dir', str(Path(temp) / 'run')])
            m = Model(**model_kwargs(a))
            run = new_run(a, m)
            e = new_run(args([*SMALL, '--run_dir', str(run.directory)], True), evaluation=True)
            torch.save(nn.Linear(7, 4), run.checkpoint)
            with self.assertRaises(SidecarMismatch):
                e.load()
            bad = copy.deepcopy(m)
            del bad.placeholder
            torch.save(bad, run.checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                e.load()

    def test_whole_model_save_and_load_in_separate_car_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            # Capture the exact original torch.save expression (not a new protocol).
            original_tree = ast.parse((CAR / 'train.py').read_text())
            save_call = next(n for n in ast.walk(original_tree) if isinstance(n, ast.Call)
                             and ast.unparse(n.func) == 'torch.save')
            save_line = ast.unparse(save_call)
            producer = '''
import os, sys, torch
from argparse import Namespace
from models.CDLNO import Model
from models.cdlno_run import CarRun
# The working directory supplies the stable local models package; only the
# shared package root is on PYTHONPATH, as an editable installation would be.
torch.set_num_threads(1); torch.manual_seed(15)
a=Namespace(cfd_model='CDLNO', n_hidden=16,n_layers=8,n_heads=4,slice_num=4,
 front_blocks=2,mlp_ratio=2,latent_ffn_ratio=2,dropout=0.,cdpa_mode='entry',
 cdpa_source_chunk_size=0,fold_id=3,nb_epochs=200,weight=.5,cfd_mesh=False,r=.2,
 run_dir=sys.argv[1])
model=Model(n_hidden=16,n_head=4,slice_num=4)
run=CarRun(a,device=torch.device('cpu'),model=model)
path=str(run.directory); hparams={'nb_epochs':200}
'''
            producer += save_line + '\n'
            producer += '''
from types import SimpleNamespace
x=torch.randn(17,7); model.eval()
torch.save({'x':x,'expected':model((SimpleNamespace(x=x),None)).detach()},run.directory/'expected.pt')
'''
            consumer = '''
import sys, torch
from argparse import Namespace
from types import SimpleNamespace
from models.cdlno_run import CarRun
from pathlib import Path
torch.set_num_threads(1)
a=Namespace(cfd_model='CDLNO', n_hidden=16,n_layers=8,n_heads=4,slice_num=4,
 front_blocks=2,mlp_ratio=2,latent_ffn_ratio=2,dropout=0.,cdpa_mode='entry',
 cdpa_source_chunk_size=1,fold_id=3,nb_epochs=200,weight=.5,cfd_mesh=False,r=.2,
 run_dir=sys.argv[1])
sidecar=Path(a.run_dir)/'architecture.json'; before=sidecar.read_bytes()
run=CarRun(a,device=torch.device('cpu'),evaluation=True); model=run.load().eval()
payload=torch.load(Path(a.run_dir)/'expected.pt',weights_only=True)
assert type(model).__module__=='models.CDLNO'
assert model.core.source_chunk_size==1
torch.testing.assert_close(model((SimpleNamespace(x=payload['x']),None)), payload['expected'],atol=1e-5,rtol=3e-4)
assert before==sidecar.read_bytes()
print('fresh-process whole-model output and sidecar passed')
'''
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
            for code in (producer, consumer):
                result = subprocess.run([sys.executable, '-B', '-c', code, str(directory)], cwd=CAR,
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(HAS_PYG, 'real torch_geometric unavailable: PyG integration not completed')
class RealPyGChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(79)

    @staticmethod
    def data(n):
        return Data(x=torch.randn(n, 7), pos=torch.randn(n, 3), y=torch.randn(n, 4),
                    surf=torch.arange(n) >= n // 2,
                    edge_index=torch.tensor([[0, 1], [1, 0]]))

    def test_variable_n_single_batch_no_mutation_no_label_geometry_read(self):
        m = Model(**model_kwargs(args(SMALL))).eval()
        for n in (7, 29):
            raw = self.data(n)
            for data in (raw, Batch.from_data_list([raw]), Data(x=raw.x, batch=torch.zeros(n, dtype=torch.long))):
                before = {k: (v.clone(), v.data_ptr()) for k, v in data if isinstance(v, torch.Tensor)}
                out = m((data, object()))
                self.assertEqual(out.shape, (n, 4))
                for key, (value, pointer) in before.items():
                    torch.testing.assert_close(data[key], value, atol=0, rtol=0)
                    self.assertEqual(data[key].data_ptr(), pointer)
            expected = m((raw, None))
            modified = raw.clone()
            modified.y = torch.randn_like(raw.y) * 1000
            modified.surf = ~raw.surf
            modified.pos = torch.randn_like(raw.pos)
            torch.testing.assert_close(m((modified, object())), expected, rtol=0, atol=0)
            del modified.y
            torch.testing.assert_close(m((modified, None)), expected, rtol=0, atol=0)
            perm = torch.randperm(n)
            torch.testing.assert_close(m((Data(x=raw.x[perm]), None)), expected[perm], atol=2e-6, rtol=2e-5)

    def test_multigraph_batch_ptr_and_malformed_inputs_rejected(self):
        m = Model(**model_kwargs(args(SMALL)))
        multi = Batch.from_data_list([self.data(3), self.data(5)])
        with self.assertRaisesRegex(ValueError, 'one physical graph'):
            m((multi, None))
        for kwargs in (dict(ptr=torch.tensor([0, 3, 8])),
                       dict(batch=torch.zeros(8, dtype=torch.long), ptr=torch.tensor([0, 8, 8])),
                       dict(batch=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]))):
            with self.assertRaisesRegex(ValueError, 'one physical graph'):
                m((Data(x=torch.randn(8, 7), **kwargs), None))
        for kwargs in (dict(ptr=torch.tensor([0, 9])), dict(batch=torch.zeros(7)),
                       dict(batch=torch.zeros(8)), dict(ptr=torch.tensor([0., 8.]))):
            with self.assertRaises(ValueError):
                m((Data(x=torch.randn(8, 7), **kwargs), None))
        for x in (torch.randn(8, 6), torch.randn(0, 7), torch.ones(8, 7, dtype=torch.long)):
            with self.assertRaises(ValueError):
                m((Data(x=x), None))

    def test_original_train_mask_loss_backward_optimizer_scheduler(self):
        spec = importlib.util.spec_from_file_location('car_train_for_test', CAR / 'train.py')
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)  # train.py defines functions only; no dataset IO.
        m = Model(**model_kwargs(args(SMALL)))
        reference = copy.deepcopy(m)
        data = self.data(19)
        criterion = nn.MSELoss(reduction='none')
        y = reference((data, None))
        pressure = criterion(y[data.surf, -1], data.y[data.surf, -1]).mean()
        velocity = criterion(y[:, :-1], data.y[:, :-1]).mean(dim=0).mean()
        (velocity + .5 * pressure).backward()
        optimizer = torch.optim.Adam(m.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, total_steps=3, final_div_factor=1000.)
        loader = DataLoader([(data, torch.randn(6, 3))], batch_size=1)
        start_step = scheduler.last_epoch
        before = m.preprocess[0].weight.detach().clone()
        lp, lv = training.train(torch.device('cpu'), m, loader, optimizer, scheduler, reg=.5)
        self.assertAlmostEqual(lp, pressure.item(), places=6)
        self.assertAlmostEqual(lv, velocity.item(), places=6)
        self.assertEqual(scheduler.last_epoch, start_step + 1)
        self.assertFalse(torch.equal(before, m.preprocess[0].weight))
        for (name, p), (_, q) in zip(m.named_parameters(), reference.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        press, vel = training.test(torch.device('cpu'), m, loader)
        self.assertTrue(torch.isfinite(torch.tensor([press, vel])).all())

    def test_large_synthetic_point_count_and_input_gradients(self):
        m = Model(n_hidden=8, n_head=2, slice_num=4)
        data = self.data(32186)
        data.x.requires_grad_()
        output = m((data, None))
        self.assertEqual(output.shape, (32186, 4))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(data.x.grad).all())
        self.assertGreater(data.x.grad.abs().max().item(), 0)


class FrozenSourceAndScripts(unittest.TestCase):
    def test_original_entries_project_to_baseline_ast(self):
        class Project(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == 'models.cdlno_run' else node

            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ('cdlno_run', 'results_dir'):
                        return None
                    if name == 'args':
                        node.value = ast.parse('parser.parse_args()', mode='eval').body
                    if name == 'path' and isinstance(node.value, ast.IfExp):
                        node.value = node.value.orelse
                return self.generic_visit(node)

            def visit_If(self, node):
                if ast.unparse(node.test) == "args.cfd_model == 'CDLNO'":
                    return [self.visit(n) for n in node.orelse if not
                            (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'results_dir')]
                return self.generic_visit(node)

            def visit_Call(self, node):
                if ast.unparse(node.func) == 'torch.load':
                    node.keywords = [k for k in node.keywords if k.arg not in ('map_location', 'weights_only')]
                if ast.unparse(node.func) == 'parser.add_argument' and ast.literal_eval(node.args[0]) == '--nb_epochs':
                    if evaluation:
                        for k in node.keywords:
                            if k.arg == 'type':
                                k.value = ast.Name(id='float', ctx=ast.Load())
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == 'results_dir':
                    return ast.parse("'./results/' + args.cfd_model + '/'", mode='eval').body
                return node

        for evaluation, filename in ((False, 'main.py'), (True, 'main_evaluation.py')):
            restored = Project().visit(ast.parse((CAR / filename).read_text()))
            self.assertEqual(ast.dump(restored), ast.dump(ast.parse(original(filename))), filename)

    def test_frozen_car_and_airfrans_original_files(self):
        names = subprocess.check_output(['git', 'ls-files', 'Car-Design-ShapeNetCar',
                                         'Airfoil-Design-AirfRANS'], cwd=ROOT, text=True).splitlines()
        count = 0
        for name in names:
            # Phase8 now separately audits the authorized AirfRANS entry/YAML
            # changes; its train/dataset/metrics/model files remain frozen here.
            if name in ('Car-Design-ShapeNetCar/main.py', 'Car-Design-ShapeNetCar/main_evaluation.py',
                        'Airfoil-Design-AirfRANS/main.py', 'Airfoil-Design-AirfRANS/main_evaluation.py',
                        'Airfoil-Design-AirfRANS/params.yaml'):
                continue
            before = subprocess.check_output(['git', 'show', '75e0f67643806a81cd1d3f6adc88dd8c02416fe7:' + name], cwd=ROOT)
            self.assertEqual((ROOT / name).read_bytes(), before, name)
            count += 1
        print('Frozen Car/AirfRANS original files byte-identical:', count)

    def test_epoch_path_fix_and_launch_arguments(self):
        old = entry_parser(True, True).parse_args(['--nb_epochs', '200'])
        self.assertEqual(f'model_{old.nb_epochs}.pth', 'model_200.0.pth')
        fixed = parse_args(entry_parser(True), evaluation=True, argv=['--cfd_model', 'Transolver', '--nb_epochs', '200'])
        self.assertEqual(f'model_{fixed.nb_epochs}.pth', 'model_200.pth')
        preset = json.loads((CAR / 'configs/CDLNO/shapenet_car.json').read_text())
        for evaluation, filename in ((False, 'CDLNO.sh'), (True, 'CDLNO_Evaluation.sh')):
            script = CAR / 'scripts' / filename
            subprocess.run(['bash', '-n', str(script)], check=True)
            extra = ['--fold_id', '4', '--n_layers', '12', '--front_blocks', '7', '--run_dir', '/tmp/user-chosen-run']
            # Capture the shell arguments only. No Python entry/data is executed.
            code = 'python() { printf "%s\\0" "$@"; }; export -f python; bash "$@"'
            result = subprocess.run(['bash', '-c', code, 'capture', str(script), *extra],
                                    cwd=CAR, capture_output=True, check=True)
            tokens = result.stdout.decode().rstrip('\0').split('\0')
            resolved = parse_args(entry_parser(evaluation), evaluation=evaluation, argv=tokens[1:])
            self.assertEqual((resolved.n_layers, resolved.front_blocks, resolved.fold_id), (12, 7, 4))
            default = args(['--run_dir', '/tmp/example'], evaluation)
            for key, value in preset['model'].items():
                self.assertEqual(getattr(default, key), value)
            if not evaluation:
                for key, value in preset['training'].items():
                    self.assertEqual(getattr(default, key), value)
        for filename in ('main.py', 'main_evaluation.py', 'models/CDLNO.py', 'models/cdlno_run.py'):
            ast.parse((CAR / filename).read_text(), feature_version=(3, 10))


if __name__ == '__main__':
    unittest.main()
