"""Isolated old-Transolver CPU fixtures; never import exp/main entry modules.

The ONLY device adaptation is a test-local Tensor.cuda -> Tensor.cpu shim for
original hard-coded position builders. Model source/weights/math are untouched.
This is not native GPU evidence. Whole-object loads concern files just created
by this worker, never external or user-supplied pickle.
"""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = dict(standard='PDE-Solving-StandardBenchmark', car='Car-Design-ShapeNetCar', airfrans='Airfoil-Design-AirfRANS')
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')


@contextmanager
def cpu_positions():
    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self.cpu()):
        yield


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def input_hashes(value):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value))
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [input_hashes(item) for item in value]
    return {k: input_hashes(v) for k, v in value.to_dict().items()}


def build(task):
    project = task if task in ('car', 'airfrans') else 'standard'
    cwd = ROOT / PROJECTS[project]
    sys.path.insert(0, str(cwd))
    kw = dict(n_layers=2, n_hidden=8, n_head=2, mlp_ratio=2, dropout=0., slice_num=4, ref=3,
              space_dim=2, fun_dim=0, out_dim=1, unified_pos=task in ('darcy', 'ns', 'airfrans'))
    if project == 'standard':
        kw.update(Time_Input=task == 'plasticity')
        if task != 'elasticity':
            kw.update(H=5, W=7)
        kw['fun_dim'] = 10 if task == 'ns' else 1 if task in ('darcy', 'plasticity') else 0
        kw['out_dim'] = 4 if task == 'plasticity' else 1
        key = 'Transolver_Irregular_Mesh' if task == 'elasticity' else 'Transolver_Structured_Mesh_2D'
        module = importlib.import_module('model_dict').get_model(SimpleNamespace(model=key))
        cls = module.Model
        assert cls.__module__ == 'model.' + key
        selection = dict(kind='real model_dict.get_model', key=key, class_path=cls.__module__+'.Model')
    else:
        # Audit real selection AST without importing the side-effecting main.
        tree = ast.parse((cwd/'main.py').read_text())
        field = 'cfd_model' if task == 'car' else 'model'
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == f"args.{field} == 'Transolver'")
        class_name = 'Model' if task == 'car' else 'Transolver'
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == class_name
                   for statement in branch.body for n in ast.walk(statement))
        cls = getattr(importlib.import_module('models.Transolver'), class_name)
        kw.update(space_dim=7, fun_dim=0, out_dim=4)
        selection = dict(kind='real main selection AST + safe original class import', key='Transolver',
                         class_path=cls.__module__+'.'+cls.__name__)
    torch.manual_seed(1701 + TASKS.index(task))
    with cpu_positions():
        model = cls(**kw).eval()
    return model, kw, selection


def inputs(task):
    generator = torch.Generator().manual_seed(2701 + TASKS.index(task))
    if task in ('car', 'airfrans'):
        from torch_geometric.data import Data
        x = torch.randn(19, 7, generator=generator)
        data = Data(x=x, pos=torch.randn(19, 3 if task == 'car' else 2, generator=generator),
                    y=torch.randn(19, 4, generator=generator), surf=torch.arange(19) % 3 == 0)
        return ((data, torch.randn(9, 3, generator=generator)),) if task == 'car' else (data,)
    N = 19 if task == 'elasticity' else 35
    x = torch.randn(2, N, 2, generator=generator)
    channels = 10 if task == 'ns' else 1 if task in ('darcy', 'plasticity') else 0
    fx = torch.randn(2, N, channels, generator=generator) if channels else None
    return (x, fx, torch.tensor([[0.2], [0.7]])) if task == 'plasticity' else (x, fx)


def fixture(task):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config, selection = build(task)
    args = inputs(task)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with cpu_positions(), torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = model(*args)
        assert torch.isfinite(expected).all()
        with tempfile.TemporaryDirectory(prefix='linearno-l1-legacy-') as directory:
            path = Path(directory)/'state.pt'
            torch.save(state, path)
            # Corrupt current weights first: loading must actually restore SAME weights.
            for param in model.parameters():
                param.add_(1.)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            actual = model(*args)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if task in ('car', 'airfrans'):
                path = Path(directory)/'local-object.pt'
                torch.save([model] if task == 'airfrans' else model, path)
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                if task == 'airfrans':
                    loaded = loaded[0]
                torch.testing.assert_close(loaded(*args), expected, atol=0, rtol=0)
    flat = expected.flatten().double()
    return dict(task=task, seed=1701+TASKS.index(task), input_seed=2701+TASKS.index(task),
                input_hashes=input_hashes(args),
                config=config, selection=selection, parameters=sum(p.numel() for p in model.parameters()),
                keys={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hash(v)) for k,v in state.items()},
                output_shape=list(expected.shape), output=expected.tolist(),
                output_summary=dict(sum=flat.sum().item(), norm=flat.norm().item(), sha256=tensor_hash(expected)),
                roundtrip_max_error=float((actual-expected).abs().max()),
                whole_object_roundtrip=task in ('car','airfrans'),
                device='cpu', dtype='float32', threads=1, atol=1e-6, rtol=1e-5,
                device_adaptation='test-only Tensor.cuda->cpu for legacy position builders',
                python=sys.version.split()[0], torch=str(torch.__version__))


if __name__ == '__main__':
    print(json.dumps(fixture(sys.argv[1]), sort_keys=True))
