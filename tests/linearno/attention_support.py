"""Load unchanged attention class ASTs from the fixed, external official source.

No vendoring/import of exp/main/Embedding/data or full-model construction. Whole
class AST nodes are compiled without rewriting. Source hashes fail closed.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import types

import torch
from einops import rearrange
from timm.layers import trunc_normal_

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('LINEARNO_REFERENCE_ROOT', '/home/hwz/LinearNO'))
MANIFEST = json.loads((Path(__file__).parent/'fixtures/official_attention_sources.json').read_text())
CLASSES = dict(plain='LinearNO', temp='LinearNO_temp', conv='LinearNO_Conv', conv_temp='LinearNO_Conv_temp',
               airfrans='LinearNO', shapenet='LinearNO')
LOCAL = dict(standard='PDE-Solving-StandardBenchmark/model/LinearNO_Attention.py',
             airfrans='Airfoil-Design-AirfRANS/models/LinearNO_Attention.py',
             shapenet='Car-Design-ShapeNetCar/models/LinearNO_Attention.py')


def kind(variant):
    return variant if variant in ('airfrans','shapenet') else 'standard'


def load_classes(variant):
    task = kind(variant)
    origin = MANIFEST['files'][task]
    path = SOURCE/origin['path']
    if not path.is_file():
        raise RuntimeError(f'fixed source missing: set LINEARNO_REFERENCE_ROOT to the audited checkout: {path}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != origin['sha256']:
        raise RuntimeError(f'official source hash mismatch: {path}')
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and
                n.name in (CLASSES[variant], 'LinearAttentionNeuralOperator')]
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, rearrange=rearrange, trunc_normal_=trunc_normal_)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path), 'exec'), scope)
    spec = importlib.util.spec_from_file_location('linearno_l2_'+task, ROOT/LOCAL[task])
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module, CLASSES[variant]), scope[CLASSES[variant]], scope['LinearAttentionNeuralOperator']._init_weights


def kwargs(variant, *, dim=12, heads=3, dim_head=4, rank=8, dropout=0., H=3, W=5):
    args = dict(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
    if variant=='shapenet':
        assert rank % dim_head == 0
        args['key_ratio'] = rank//dim_head
    elif variant=='airfrans':
        args['slice_num'] = rank
    else:
        args['key_ratio'] = rank
    if variant in ('conv','conv_temp'):
        args.update(H=H,W=W)
    return args


def errors(actual, expected):
    a,b = actual.detach().double(),expected.detach().double()
    delta=(a-b).abs()
    relative=delta/b.abs().clamp_min(1e-12)
    return dict(max_abs=delta.max().item(),mean_abs=delta.mean().item(),max_relative=relative.max().item(),
                mean_relative=relative.mean().item(),relative_floor=1e-12,
                relative_l2=(delta.norm()/b.norm().clamp_min(1e-12)).item())
