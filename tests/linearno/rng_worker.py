"""Test-only epoch continuation with a real Transolver, not task-native resume.

Uses current RNG capture/restore, current PeriodicFields, actual Plasticity collate
and the actual Air random.sample expression extracted from source. Loss is labelled
synthetic MSE; this is not the twenty-step Plasticity or Air graph training loop.
"""
import ast
import copy
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch.nn.attention import sdpa_kernel, SDPBackend

from legacy_worker import ROOT, build
sys.path.insert(0, str(ROOT))
from cdlno.training_state import capture_rng, restore_rng, isolated_evaluation, _same
from cdlno.periodic_visualization import PeriodicFields


def actual_sampling():
    tree = ast.parse((ROOT/'PDE-Solving-StandardBenchmark/exp_plas.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'random_collate_fn')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_plasticity_collate', 'exec'), scope)
    tree = ast.parse((ROOT/'Airfoil-Design-AirfRANS/train.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    sample = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'random.sample')
    return scope['random_collate_fn'], compile(ast.Expression(sample), 'actual_air_sampling_expression', 'eval')


def run(action, directory, schedule):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model, _, _ = build('elasticity')
    random.seed(481); np.random.seed(482); torch.manual_seed(483)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.001, total_steps=8) if schedule == 'onecycle'
             else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4))
    generator = torch.Generator().manual_seed(484)
    loader = torch.utils.data.DataLoader(torch.arange(4), batch_size=2, shuffle=True, generator=generator)
    collate, sample = actual_sampling()
    # fixed synthetic inputs; labels/time indices make permutation comparison observable.
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 1, 3), torch.linspace(0, 1, 5), indexing='ij'), -1).reshape(15, 2)
    times = torch.arange(20).float()
    items = [(xy, times, torch.ones(15), times.expand(15, 4, 20).clone()) for _ in range(2)]
    graph = Data(x=torch.arange(40).float().reshape(20, 2), pos=torch.zeros(20, 2))
    viz = PeriodicFields(directory, 'elasticity', 'original Transolver synthetic', seed=483)
    viz_data = [(xy, torch.zeros(15), torch.ones(15))]
    records, start = [], 0
    if action == 'resume':
        payload = torch.load(directory/'test-only-epoch.pt', weights_only=True, map_location='cpu')
        model.load_state_dict(payload['model'], strict=True)
        opt.load_state_dict(payload['optimizer']); sched.load_state_dict(payload['scheduler'])
        start, records = payload['epoch'], payload['records']
        random.random(); np.random.rand(4); torch.rand(4); torch.rand(4, generator=generator)
        restore_rng(payload['rng'], {'loader': generator})  # Restore last, after construction/loading.
    for ep in range(start, 2 if action == 'split' else 4):
        model.train()
        for ids in loader:
            shuffled = collate(items)
            tperm = shuffled[1].tolist()
            assert torch.equal(shuffled[3][:, 0, 0, :], shuffled[1])
            picks = eval(sample, {'random': random, 'range': range},
                         {'data_sampled': graph, 'hparams': {'subsampling': 7}})
            np_perm = np.random.permutation(20).tolist()  # Additional NumPy stream, NOT Plasticity's implementation.
            x = xy[None].repeat(2, 1, 1) + ids[:, None, None] / 100
            x = x + torch.randn_like(x) / 100 + sum(picks)/1000
            target = torch.ones(2, 15, 1) + shuffled[1][:, :1, None]/100 + sum(np_perm[:3])/100
            opt.zero_grad()
            with sdpa_kernel(SDPBackend.MATH):
                loss = (model(x, None)-target).square().mean()
                loss.backward()
            opt.step()
            if schedule == 'onecycle': sched.step()
            records.append(dict(batch=ids.tolist(), plasticity_torch_permutation=tperm, air_python_sample=picks,
                                extra_numpy_permutation=np_perm, loss=loss.item(), lr=opt.param_groups[0]['lr']))
        if schedule == 'cosine': sched.step()
        if action == 'instrumented':
            before = capture_rng({'loader': generator})
            flags = [m.training for m in model.modules()]
            # Real renderer, real model and extra RNG draws, all within the existing isolation primitive.
            with isolated_evaluation(model, generators={'loader': generator}):
                random.random(); np.random.rand(3); torch.rand(3); torch.rand(3, generator=generator)
                event = viz.after_epoch(model, (ep+1)*50, 200, dataset=viz_data)
                assert event['status'] == 'completed', event
                torch.save(model.state_dict(), directory/f'weights-{ep}.pt')
            assert _same(before, capture_rng({'loader': generator}))
            assert flags == [m.training for m in model.modules()]
    result = dict(model=copy.deepcopy(model.state_dict()), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                  records=records, rng=capture_rng({'loader': generator}), epoch=2 if action=='split' else 4)
    if action == 'split':
        torch.save(result, directory/'test-only-epoch.pt')
    torch.save(result, directory/f'{action}.pt')
    print(action, schedule, len(records), 'synthetic updates passed')


if __name__ == '__main__':
    run(*sys.argv[1:])
