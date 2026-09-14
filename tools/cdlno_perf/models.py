"""Task-shaped synthetic cases and the thin matched LRSA composition.

No experiment/loader import. Original Transolver source is executed with only
model-local import aliases and a device-only replacement of hardcoded .cuda().
All attention, MLP, normalization, parameters and forward math stay original.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from cdlno.airfrans import AirfRANSModel
from cdlno.modules import LRSAFrontBlock
from cdlno.standard import StaticStandardModel, TemporalStandardModel

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / 'PDE-Solving-StandardBenchmark'
TASKS = {
    # grid, N, d, new heads, new M, original M, batch, fx, output
    'darcy': ((85, 85), 7225, 128, 8, 64, 64, 4, 1, 1),
    'elasticity': (None, 972, 128, 8, 64, 64, 1, 0, 1),
    'airfoil': ((221, 51), 11271, 128, 4, 64, 64, 4, 0, 1),
    'pipe': ((129, 129), 16641, 128, 4, 32, 64, 8, 0, 1),
    'ns': ((64, 64), 4096, 256, 8, 64, 32, 2, 10, 1),
    'plasticity': ((101, 31), 3131, 128, 8, 64, 64, 8, 1, 4),
    'shapenet-car': (None, 32186, 256, 8, 64, 32, 1, 0, 4),
    'airfrans': (None, 32000, 256, 8, 64, 32, 1, 0, 4),
}
MODELS = ('transolver', 'lrsa_matched', 'cdlno_off', 'cdlno_entry', 'cdlno_every_block')


@dataclass(frozen=True)
class Case:
    task: str = 'elasticity'
    comparison: str = 'matched'
    B: int = 1
    N: int = 972
    d: int = 128
    h: int = 8
    M: int = 64
    L: int = 8
    F: int = 2
    ratio: int = 2
    grid: tuple[int, int] | None = None

    @classmethod
    def preset(cls, task, comparison='matched', **overrides):
        grid, n, d, h, m, _, b, _, _ = TASKS[task]
        case = cls(task, comparison, b, n, d, h, m, grid=grid)
        if comparison == 'task' and any(k in overrides for k in ('d', 'h', 'M', 'L', 'F', 'ratio')):
            raise ValueError('task comparison keeps original/new task architecture presets; use matched for overrides')
        if 'grid' in overrides:
            if grid is None:
                raise ValueError('irregular task cannot acquire a grid')
            overrides['grid'] = tuple(overrides['grid'])
            overrides['N'] = overrides['grid'][0] * overrides['grid'][1]
        case = replace(case, **overrides)
        case.validate()
        return case

    def validate(self):
        if self.task not in TASKS or self.comparison not in ('matched', 'task'):
            raise ValueError('unknown task/comparison')
        if any(type(v) is not int or v < 1 for v in (self.B, self.N, self.d, self.h, self.M, self.L, self.ratio)):
            raise ValueError('B/N/d/h/M/L/ratio must be positive integers')
        if type(self.F) is not int or not 0 <= self.F < self.L or self.d % self.h:
            raise ValueError('require 0<=F<L and d divisible by h')
        grid = TASKS[self.task][0]
        if (self.grid is None) != (grid is None):
            raise ValueError('task grid policy cannot change')
        if self.grid is not None and (len(self.grid) != 2 or any(type(i) is not int or i < 1 for i in self.grid)
                                      or self.N != self.grid[0] * self.grid[1]):
            raise ValueError('N must match explicit H*W')
        if self.task in ('ns', 'plasticity') and self.grid != grid:
            raise ValueError('temporal wrappers preserve their fixed spatial grid')
        if self.task in ('shapenet-car', 'airfrans') and self.B != 1:
            raise ValueError('industrial tasks accept single graphs only')
        return self

    def for_model(self, name):
        if name not in MODELS:
            raise ValueError(name)
        if name == 'transolver' and self.comparison == 'task':
            return replace(self, h=8, M=TASKS[self.task][5])
        return self

    def description(self):
        result = asdict(self)
        result['P'] = self.L - self.F
        result['task_geometry_and_batch_preserved'] = (self.N == TASKS[self.task][1] and self.B == TASKS[self.task][6])
        result['preset_source'] = ('original Transolver launchers + industrial main constructors; '
                                   'CDLNO configs/CDLNO/*.json (v1.2), not old parser defaults')
        return result


class Structured(StaticStandardModel):
    structured_adapter = True


class Irregular(StaticStandardModel):
    structured_adapter = False


class LRSAMatched(nn.Module):
    """L full blocks; last block already upsamples. Only LN/head follows.

    The surrounding task wrapper, stem and output head are the same as CDLNO.
    This class is not registered in any training factory or checkpoint protocol.
    """
    def __init__(self, config, output_norm, output):
        super().__init__()
        config.validate()  # never pretend F=L is a valid CDLNO configuration
        self.config = config
        self.blocks = nn.ModuleList([
            LRSAFrontBlock(config.d_model, config.num_heads, config.M,
                           structured=config.structured, grid_shape=config.grid_shape,
                           ffn_ratio=config.ffn_ratio) for _ in range(config.L)
        ])
        self.output_norm = copy.deepcopy(output_norm)
        self.output = copy.deepcopy(output)

    def forward(self, h):
        for block in self.blocks:
            h, t = block(h)
            del t  # no CDPA history list; autograd of the complete block stays live
        return self.output(self.output_norm(h))


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_class(case, device):
    """Isolated source namespace; no sys.modules replacements or fake libraries."""
    if case.task in ('shapenet-car', 'airfrans'):
        project = 'Car-Design-ShapeNetCar' if case.task == 'shapenet-car' else 'Airfoil-Design-AirfRANS'
        path = ROOT / project / 'models/Transolver.py'
    else:
        path = STANDARD / 'model' / ('Transolver_Structured_Mesh_2D.py' if case.grid else 'Transolver_Irregular_Mesh.py')
    tree = ast.parse(path.read_text())
    # Dependencies remain real modules. Only project-local import resolution is
    # isolated; timm's initialization utility is imported without substitution.
    attention = load_file(STANDARD / 'model/Physics_Attention.py', '_perf_physics')
    embedding = load_file(STANDARD / 'model/Embedding.py', '_perf_embedding')
    namespace = {'_perf_device': device, 'timestep_embedding': embedding.timestep_embedding,
                 **{k: v for k, v in vars(attention).items() if k.startswith('Physics_Attention_')}}
    changes = []
    class DeviceOnly(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module in ('model.Embedding', 'model.Physics_Attention'):
                changes.append('resolve ' + node.module + ' in isolated namespace')
                return None
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                if node.args or node.keywords:
                    raise ValueError('unexpected baseline cuda signature')
                changes.append(f'.cuda() -> .to(device) at line {node.lineno}')
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=node.func.value, attr='to', ctx=ast.Load()),
                    args=[ast.Name(id='_perf_device', ctx=ast.Load())], keywords=[]), node)
            return node
    exec(compile(ast.fix_missing_locations(DeviceOnly().visit(tree)), str(path), 'exec'), namespace)
    cls = namespace['Transolver' if case.task == 'airfrans' else 'Model']
    return cls, dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     adaptations=changes)


def build(case, name, device='cpu', chunk=0, seed=20260914):
    case = case.for_model(name).validate()
    torch.manual_seed(seed)
    kwargs = dict(n_hidden=case.d, n_layers=case.L, n_head=case.h,
                  slice_num=case.M, mlp_ratio=case.ratio, dropout=0.)
    industrial = case.task in ('shapenet-car', 'airfrans')
    if not industrial:
        kwargs.update(space_dim=2, fun_dim=TASKS[case.task][7], out_dim=TASKS[case.task][8],
                      Time_Input=case.task == 'plasticity', unified_pos=case.task in ('darcy', 'ns'), ref=8)
        if case.grid:
            kwargs.update(H=case.grid[0], W=case.grid[1])
    provenance = {}
    if name == 'transolver':
        cls, provenance = baseline_class(case, device)
        if industrial:
            kwargs.update(space_dim=7, fun_dim=0, out_dim=4, unified_pos=case.task == 'airfrans')
        model = cls(**kwargs)
    else:
        kwargs.update(front_blocks=case.F, cdpa_mode=name.removeprefix('cdlno_') if name != 'lrsa_matched' else 'off',
                      cdpa_source_chunk_size=chunk, latent_ffn_ratio=case.ratio)
        if case.task == 'airfrans':
            cls = AirfRANSModel
        elif case.task == 'shapenet-car':
            cls = load_file(ROOT / 'Car-Design-ShapeNetCar/models/CDLNO.py', '_perf_car').Model
        else:
            cls = TemporalStandardModel if case.task in ('ns', 'plasticity') else Structured if case.grid else Irregular
            kwargs['task_name'] = case.task
        model = cls(**kwargs)
        if name == 'lrsa_matched':
            model.core = LRSAMatched(model.config, model.core.readout.output_norm, model.core.readout.output)
    return model.to(device), case, provenance


def synthetic_inputs(case, device='cpu', seed=101):
    generator = torch.Generator(device=device).manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, device=device)
    if case.task in ('shapenet-car', 'airfrans'):
        from torch_geometric.data import Data
        data = Data(x=rand(case.N, 7), pos=rand(case.N, 2 if case.task == 'airfrans' else 3))
        # Ordinary real PyG Data, without Batch metadata; no graph construction.
        args = (data,) if case.task == 'airfrans' else ((data, None),)
        shape = (case.N, 4)
    else:
        fx = TASKS[case.task][7]
        args = (rand(case.B, case.N, 2), rand(case.B, case.N, fx) if fx else None)
        if case.task == 'plasticity':
            args += (rand(case.B, 1),)
        shape = (case.B, case.N, TASKS[case.task][8])
    return args, rand(*shape)
