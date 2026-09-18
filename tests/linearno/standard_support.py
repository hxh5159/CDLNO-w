"""L3 fixed-source loader. Execute unchanged model/class/function AST nodes only.

No task imports, data reads, device monkeypatches, or source math edits. The
official unified-position constructor genuinely requires a CUDA device.
"""
import ast
import hashlib
import importlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from timm.layers import trunc_normal_

from linearno.attention_support import ROOT, SOURCE, MANIFEST

EMBEDDING_SHA256 = 'e3a4f71c4378b7cce09a43b8c306d6d8a17edb4cd69db5eb069e55a57d2affe2'
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity')
VARIANTS = ('plain', 'temp', 'conv', 'conv_temp')


def official_class():
    source = MANIFEST['files']['standard']
    path = SOURCE / source['path']
    embedding = path.with_name('Embedding.py')
    for file, digest in ((path, source['sha256']), (embedding, EMBEDDING_SHA256)):
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'fixed official source mismatch: {file}')
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
                 math=math, rearrange=rearrange, trunc_normal_=trunc_normal_)
    nodes = [n for n in ast.parse(embedding.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'timestep_embedding']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(embedding), 'exec'), scope)
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['LinearAttentionNeuralOperator']


def local_class():
    project = str(ROOT / 'PDE-Solving-StandardBenchmark')
    if project not in sys.path:
        sys.path.insert(0, project)
    return importlib.import_module('model.LinearNO').Model


def release_configs():
    inventory = json.loads((ROOT/'docs/linearno_audit/l0/profile-inventory.json').read_text())
    return {row['task']: row['official_constructor_kwargs'] for row in inventory['rows']
            if row['profile'] == 'official_release' and row['task'] in TASKS}


def official_kwargs(config):
    result = dict(config)
    result['args'] = SimpleNamespace(**config['args'])
    return result


def local_kwargs(config):
    result = dict(config)
    result['linearno_variant'] = result.pop('args')['model']
    if result['linearno_variant'] == 'no_temp':
        result['linearno_variant'] = 'plain'
    result['linearno_rank'] = result.pop('key_ratio')
    return result


def small_config(variant='plain', *, B=2, fun_dim=1, time=True, unified=False, dropout=0., layers=3):
    return dict(space_dim=2, n_layers=layers, n_hidden=12, dropout=dropout,
                n_head=3, Time_Input=time, act='gelu', mlp_ratio=2,
                fun_dim=fun_dim, out_dim=2, key_ratio=4, ref=3,
                unified_pos=unified, H=3, W=5, args={'model': variant})


def inputs(config, *, B=2, device='cpu', dtype=torch.float32, T=True, N=None):
    count = config['H'] * config['W'] if N is None else N
    x = torch.randn(B, count, config['space_dim'], device=device, dtype=dtype, requires_grad=True)
    fx = (torch.randn(B, count, config['fun_dim'], device=device, dtype=dtype, requires_grad=True)
          if config['fun_dim'] else None)
    time = torch.rand(B, 1, device=device, dtype=dtype, requires_grad=True) if T else None
    return x, fx, time
