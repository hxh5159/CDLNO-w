"""Run real temporal main ASTs with in-memory, full-spatial synthetic inputs.

Only the raw-data prefix, CUDA transfers and standalone showcase are adapted.
Original collate, normalizer, time loops, losses, recording and archives execute.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'PDE-Solving-StandardBenchmark'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))
from cdlno_entry import parse_args
from model_dict import get_model
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from cdlno.linearno.standard_entry import StandardRun, model_kwargs, normalizer, start, finish
from cdlno.linearno.schema import pack_state, unpack_state
from cdlno.experiment import session

SHORT = dict(ns='ns', plasticity='plas')


def parser_for(task):
    tree = ast.parse((PROJECT / f'exp_{SHORT[task]}.py').read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser' or
             isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
             ast.unparse(n.value.func) == 'parser.add_argument']
    scope = dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<real temporal parser>', 'exec'), scope)
    return scope['parser']


def args_for(task, *tokens):
    return parse_args(parser_for(task), task, ['--model', 'LinearNO_Structured_Mesh_2D', *map(str, tokens)])


def synthetic_values(task):
    H, W = (64, 64) if task == 'ns' else (101, 31)
    # Intentionally preserve meshgrid's xy order even for Plasticity's H,W reshape.
    xx, yy = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W))
    pos = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)[None]
    sample = torch.arange(6).float().reshape(6, 1, 1)
    t = torch.linspace(0, 1, 20)
    if task == 'ns':
        data = 1 + sample * .1 + pos[..., :1] + pos[..., 1:] * .2 + t.reshape(1, 1, 20) ** 2
        values = dict(h=64, ntrain=4, ntest=2, T_in=10, T=10, step=1,
                      train_a=data[:4, :, :10], train_u=data[:4, :, 10:],
                      test_a=data[4:, :, :10], test_u=data[4:, :, 10:],
                      pos_train=pos.repeat(4, 1, 1), pos_test=pos.repeat(2, 1, 1))
    else:
        fx = (1 + sample + torch.linspace(0, 1, H).repeat_interleave(W).reshape(1, -1, 1))
        field = torch.cat((pos[..., :1]+1, pos[..., 1:]+1, pos[..., :1]+2, pos[..., 1:]+2), -1)
        data = (field + sample*.1).unsqueeze(-1) + t.reshape(1, 1, 1, 20)*.1
        values = dict(ntrain=4, ntest=2, s1=H, s2=W, T=20, Deformation=4,
                      x_train=fx[:4], x_test=fx[4:], y_train=data[:4], y_test=data[4:])
    sha = hashlib.sha256()
    for v in values.values():
        if isinstance(v, torch.Tensor): sha.update(v.numpy().tobytes())
    return values, sha.hexdigest()


def native_main(task, values, scope):
    tree = ast.parse((PROJECT / f'exp_{SHORT[task]}.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    index = next(i for i, n in enumerate(main.body) if
                 ('linearno_normalizer(' in ast.unparse(n) if task == 'plasticity' else
                  isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'train_loader'))
    main.body = ast.parse('\n'.join(f'{key} = _synthetic_values[{key!r}]' for key in values)).body + main.body[index:]
    class CPU(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            return node.func.value if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda' else node
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == 'showcase': node.value = ast.Constant(0)
            return self.generic_visit(node)
    functions = [CPU().visit(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('random_collate_fn', 'count_parameters', 'main')]
    scope['_synthetic_values'] = values
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(PROJECT / f'exp_{SHORT[task]}.py') + ':synthetic-prefix/CPU', 'exec'), scope)
    return scope['main']


def run(task, action, directory, report, signature):
    torch.set_num_threads(1)
    tokens = ['--experiment-dir', directory]
    if action in ('train', 'interrupt'):
        tokens += ['--epochs', 3, '--n-hidden', 8, '--n-heads', 2, '--n-layers', 4,
                   '--linearno-rank', 4, '--batch-size', 2, '--seed', 17, '--linearno-fair-run', 1,
                   '--linearno_latent_attnres', signature[1], '--linearno_history_k_conditioning', signature[3]]
    elif action == 'resume': tokens += ['--resume']
    elif action == 'eval': tokens += ['--eval', 1]
    else: raise ValueError(action)
    args = args_for(task, *tokens)
    global StandardRun
    from cdlno.linearno_history.standard_entry import HistoryStandardRun, FairBaselineRun
    StandardRun = HistoryStandardRun if args.linearno_family == 'linearno_history' else FairBaselineRun
    values, checksum = synthetic_values(task)
    data = args._linearno_config['values']['data']
    args._linearno_data = dict(split=data['split'], sampling=data['sampling'],
        checksums={'SYNTHETIC_full_spatial_time_shape': checksum}, scope='SYNTHETIC 4train/2test, no real loader')
    def immutable():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (directory/'architecture.json', directory/'config.json') if p.exists()}
    before = immutable()
    start(args, task)
    runs, batches, times, masks, history_lengths = [], [], [], [], []
    counters = dict(train_forward=0, backward=0, optimizer=0, scheduler=0, eval_forward=0, window_checks=0)
    state = dict(batch=None, step=0, losses=[])
    init, prepare, save = StandardRun.__init__, StandardRun.prepare, StandardRun.save
    iterate, backward = torch.utils.data.DataLoader.__iter__, torch.Tensor.backward
    scheduler_step = torch.optim.lr_scheduler.OneCycleLR.step

    def capture_init(self, *a, **kw):
        init(self, *a, **kw); runs.append(self)
        def pre(model, inputs, kwargs):
            batch = state['batch']
            if batch is None: return  # PeriodicFields outside a loader; measured separately.
            counters['train_forward' if model.training else 'eval_forward'] += 1
            k = state['step']
            fx = kwargs.get('fx', inputs[1] if len(inputs) > 1 else None)
            if task == 'ns':
                assert fx.shape[-1] == 10
                torch.testing.assert_close(fx, state['expected_fx'], atol=0, rtol=0)
                counters['window_checks'] += 1
            else:
                T = kwargs['T']; assert T.shape == (batch[0].shape[0], 1)
                torch.testing.assert_close(T, batch[1][:, k:k+1], atol=0, rtol=0)
                if model.training: times.append(T.detach().tolist())
        def post(model, inputs, kwargs, output):
            batch = state['batch']
            if batch is None: return
            k = state['step']; B = output.shape[0]
            truth = batch[-1][..., k:k+1]
            if model.training:
                a = output.detach().reshape(B, -1); b = truth.reshape(B, -1)
                state['losses'].append((torch.linalg.vector_norm(a-b, dim=1)/torch.linalg.vector_norm(b, dim=1)).sum())
            if task == 'ns':
                state['expected_fx'] = torch.cat((state['expected_fx'][..., 1:], truth if model.training else output.detach()), -1)
            state['step'] += 1
        self.model.register_forward_pre_hook(pre, with_kwargs=True)
        self.model.register_forward_hook(post, with_kwargs=True)

    def capture_prepare(self, optimizer, scheduler, train_loader, test_loader):
        saved_rng = unpack_state(args._linearno_metadata['resume_state']['rng']) if action == 'resume' else None
        if action == 'resume': np.random.seed(999)  # prove restore, not accidental unchanged NumPy stream
        prepare(self, optimizer, scheduler, train_loader, test_loader)
        if saved_rng is not None: assert pack_state(np.random.get_state()) == pack_state(saved_rng['numpy'])
        optimizer.register_step_post_hook(lambda *a, **kw: counters.__setitem__('optimizer', counters['optimizer']+1))
        state['prepared'] = True

    def capture_scheduler(self, *a, **kw):
        if state.get('prepared'): counters['scheduler'] += 1
        return scheduler_step(self, *a, **kw)

    def capture_iter(loader):
        for batch in iterate(loader):
            state.update(batch=batch, step=0, losses=[])
            if task == 'ns': state['expected_fx'] = batch[1].clone()
            training = runs[0].model.training
            if training:
                batches.append(hashlib.sha256(b''.join(v.numpy().tobytes() for v in batch)).hexdigest())
            yield batch
            assert state['step'] == (10 if task == 'ns' else 20)
            state['batch'] = None

    def capture_backward(value, *a, **kw):
        if runs and runs[0].model.training:
            reference = sum(state['losses']) if task == 'ns' else state['losses'][-1]
            torch.testing.assert_close(value.detach(), reference, atol=1e-6, rtol=1e-5)
            counters['backward'] += 1
        return backward(value, *a, **kw)

    from cdlno.linearno_history.context import RawHistoryContext
    from cdlno.linearno_history.attnres import LatentSummaryAttnRes
    before_block = RawHistoryContext.before_block
    original_a = LatentSummaryAttnRes.forward
    def context_before(context,index):
        assert len(context.raw) == index
        history_lengths.append(index)
        return before_block(context,index)
    def a_forward(operator,index,current,history,**kwargs):
        def observe(trace):
            if operator.training:
                masks.append(dict(index=index,mask=trace.drop_mask.tolist()))
        return original_a(operator,index,current,history,observe=observe)
    class Interruption(Exception): pass
    def capture_save(self, model):
        result = save(self, model)
        if action == 'interrupt': raise Interruption()
        return result
    scope = dict(args=args, eval=bool(args.eval), save_name=args.save_name, torch=torch, np=np,
                 plt=plt, os=os, get_model=get_model, TestLoss=TestLoss, UnitTransformer=UnitTransformer,
                 linearno_model_kwargs=model_kwargs, LinearNORun=StandardRun, linearno_normalizer=normalizer)
    try:
        with patch.object(StandardRun, '__init__', capture_init), patch.object(StandardRun, 'prepare', capture_prepare), \
             patch.object(StandardRun, 'save', capture_save), patch.object(torch.utils.data.DataLoader, '__iter__', capture_iter), \
             patch.object(torch.Tensor, 'backward', capture_backward), \
             patch.object(torch.optim.lr_scheduler.OneCycleLR, 'step', capture_scheduler), \
             patch.object(RawHistoryContext, 'before_block', context_before), \
             patch.object(LatentSummaryAttnRes, 'forward', a_forward):
            if action in ('resume', 'eval'):
                with patch.object(UnitTransformer, '__init__', side_effect=AssertionError('no normalizer refit')):
                    native_main(task, values, scope)()
            else: native_main(task, values, scope)()
    except Interruption:
        session(args).finish(status='interrupted', error='synthetic epoch1 checkpoint boundary')
    else: finish(args)
    if action in ('resume', 'eval'): assert before == immutable()
    model = runs[0].model.eval()
    with torch.no_grad():
        if task == 'ns':
            prediction = model(values['pos_test'][:1], values['test_a'][:1])
        else:
            xx, yy = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 31))
            pos = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)[None]
            fx = args._linearno_normalizers['input'].encode(values['x_test'][:1])
            prediction = model(pos, fx, T=torch.tensor([[.5]]))
    expected_epochs = dict(train=3, interrupt=1, resume=2, eval=0)[action]
    outer = expected_epochs * 2
    assert counters['train_forward'] == outer * (10 if task == 'ns' else 20)
    assert counters['backward'] == counters['optimizer'] == outer * (1 if task == 'ns' else 20)
    assert counters['scheduler'] == outer
    report.write_text(json.dumps(dict(task=task, action=action, synthetic=True, counters=counters,
        batches=batches, time_queries=times, masks=masks, history_lengths=history_lengths, immutable_metadata=True, normalizer_no_refit=True,
        prediction_hash=hashlib.sha256(prediction.numpy().tobytes()).hexdigest(), prediction_shape=list(prediction.shape),
        numpy_next=np.random.random(5).tolist()), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('task', choices=SHORT); p.add_argument('action')
    p.add_argument('directory', type=Path); p.add_argument('report', type=Path); p.add_argument('signature')
    options=p.parse_args(); run(options.task, options.action, options.directory, options.report, options.signature)
