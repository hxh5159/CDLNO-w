"""R2 real backbone parity, independent factors and context lifetime tests."""
import ast
import copy
import gc
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT / 'PDE-Solving-StandardBenchmark'
if str(STANDARD) not in sys.path:
    sys.path.insert(0, str(STANDARD))
from model.LinearNO import Model as PureStandard
from model.LinearNO_History import Model as HistoryStandard
from cdlno.linearno.airfrans import AirfRANSLinearNO
from cdlno.linearno.shapenet import ShapeNetLinearNO
from cdlno.linearno_history.models import AirfRANSHistoryModel, ShapeNetHistoryModel
from cdlno.linearno_history.context import RawHistoryContext
from cdlno.linearno_history.core import LinearNOHistoryCore, attention_factors
from linearno.attention_reference import reference
from linearno.attention_support import errors
from linearno.test_attention_structure import Trace


VARIANTS = ('plain', 'temp', 'conv', 'conv_temp', 'airfrans', 'shapenet')


def pair(variant, *, layers=4, dropout=0., **overrides):
    kw = dict(n_layers=layers, n_hidden=12, n_head=3, linearno_rank=8,
              mlp_ratio=2, dropout=dropout, ref=3)
    if variant in ('airfrans', 'shapenet'):
        classes = ((AirfRANSLinearNO, AirfRANSHistoryModel) if variant == 'airfrans'
                   else (ShapeNetLinearNO, ShapeNetHistoryModel))
    else:
        classes = PureStandard, HistoryStandard
        kw.update(space_dim=2, fun_dim=1, out_dim=2, H=3, W=5,
                  Time_Input=True, unified_pos=False, linearno_variant=variant)
    kw.update(overrides)
    torch.manual_seed(21002)
    baseline = classes[0](**kw)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(21002)
    model = classes[1](**kw)
    assert torch.equal(rng, torch.get_rng_state()), 'research construction advanced RNG'
    torch.testing.assert_close(model.state_dict(), baseline.state_dict(), atol=0, rtol=0)
    model.load_state_dict(baseline.state_dict(), strict=True)
    return baseline, model, kw


def inputs(variant, kw, *, B=2, N=15, dtype=torch.float32, time=True):
    if variant in ('airfrans', 'shapenet'):
        values = {'x': torch.randn(N, 7, dtype=dtype).requires_grad_()}
        if variant == 'airfrans':
            values['pos'] = (torch.rand(N, 2, dtype=dtype) + .13).requires_grad_()
    else:
        values = {'x': torch.randn(B, N, 2, dtype=dtype).requires_grad_()}
        if kw['fun_dim']:
            values['fx'] = torch.randn(B, N, kw['fun_dim'], dtype=dtype).requires_grad_()
        if kw['Time_Input'] and time:
            values['T'] = torch.rand(B, 1, dtype=dtype).requires_grad_()
    return values


def call(model, variant, values, **kwargs):
    if variant == 'airfrans':
        return model(Data(**values), **kwargs)
    if variant == 'shapenet':
        return model((Data(**values), None), **kwargs)
    return model(values['x'], values.get('fx'), values.get('T'), **kwargs)


class HistoryCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if path := os.environ.get('LINEARNO_R2_REPORT'):
            Path(path).write_text(json.dumps(dict(scope='R2 no-op CPU synthetic, not A/K',
                torch=str(torch.__version__), rows=cls.rows), indent=2))

    def checked(self, a, b, label, metrics):
        if a is None or b is None:
            self.assertIs(a, b, label)
            return
        metrics[label] = errors(a, b)
        self.assertTrue(torch.isfinite(a).all(), label)
        # Identical operation ordering on this backend; no relaxed R2 tolerance.
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=label)

    def parity(self, variant, dtype, train, layers=4, **overrides):
        baseline, model, kw = pair(variant, layers=layers, dropout=.2 if train else 0., **overrides)
        baseline.to(dtype=dtype).train(train)
        model.to(dtype=dtype).train(train)
        self.assertEqual(list(baseline.state_dict()), list(model.state_dict()))
        for a, b in zip(baseline.parameters(), model.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        va = inputs(variant, kw, dtype=dtype)
        vb = {k: v.detach().clone().requires_grad_() for k, v in va.items()}
        block_outputs, traces = [], []
        handles = [b.register_forward_hook(lambda m, i, o: block_outputs.append(o)) for b in baseline.blocks]
        rng = torch.get_rng_state().clone()
        ya = call(baseline, variant, va)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        yb = call(model, variant, vb, observe=traces.append)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for h in handles:
            h.remove()
        metrics = {}
        self.assertEqual(len(traces), layers)
        for i, trace in enumerate(traces):
            self.checked(trace.output, block_outputs[i], f'block/{i}', metrics)
            self.assertEqual(len(trace.history), i)
            for j, raw in enumerate(trace.history):
                self.assertIs(raw, traces[j].factors.C_raw)
                self.assertIsNotNone(raw.grad_fn)
        self.checked(yb, ya, 'forward', metrics)
        target = torch.randn_like(ya)
        la, lb = (ya-target).square().mean(), (yb-target).square().mean()
        self.checked(lb, la, 'loss', metrics)
        la.backward(); lb.backward()
        for k in va:
            self.checked(vb[k].grad, va[k].grad, 'input/'+k, metrics)
        pa = dict(baseline.named_parameters())
        inactive = []
        for name, p in model.named_parameters():
            self.checked(p.grad, pa[name].grad, 'grad/'+name, metrics)
            if p.grad is None:
                inactive.append(name)
        oa = torch.optim.AdamW(baseline.parameters(), lr=.001)
        ob = torch.optim.AdamW(model.parameters(), lr=.001)
        oa.step(); ob.step()
        for name, value in model.state_dict().items():
            self.checked(value, baseline.state_dict()[name], 'step/'+name, metrics)
        # Honest internal no-op strict state roundtrip, not a research metadata loader.
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = type(model)(**kw).to(dtype=dtype).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            self.checked(call(restored, variant, vb), call(model, variant, vb), 'roundtrip', metrics)
        self.rows.append(dict(kind='baseline_parity', variant=variant, dtype=str(dtype),
            train=train, L=layers, config=kw, parameters=sum(p.numel() for p in model.parameters()),
            keys_shapes_equal=True, init_rng_equal=True, forward_rng_equal=True,
            inactive=inactive, atol=0, rtol=0, metrics=metrics))

    def test_six_variants_forward_gradients_step_keys_and_RNG(self):
        for variant in VARIANTS:
            for dtype in (torch.float64, torch.float32):
                for train in (False, True):
                    with self.subTest(variant=variant, dtype=dtype, train=train):
                        self.parity(variant, dtype, train)

    def test_position_placeholder_time_and_eight_complete_blocks(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                kw = {}
                if variant in VARIANTS[:4]:
                    kw = dict(fun_dim=0, unified_pos=True)
                elif variant == 'airfrans':
                    kw = dict(unified_pos=False)
                else:
                    kw = dict(space_dim=7, fun_dim=0, unified_pos=True, H=3, W=5, Time_Input=True)
                self.parity(variant, torch.float32, False, layers=8, **kw)
        for layers in (5, 6, 7):
            self.parity('plain', torch.float32, False, layers=layers, Time_Input=False, fun_dim=0)

    def test_exposed_factors_independent_oracle_and_actual_shapes(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                _, model, kw = pair(variant)
                model.double().eval()
                block = model.blocks[0]
                x = torch.randn(2, 15, 12, dtype=torch.float64)
                f = attention_factors(block.Attn, x)
                ref_out, oracle = reference(x, block.Attn.state_dict(), heads=3, variant=variant, H=3, W=5)
                oracle_metrics = {}
                for k, r in [('Q','Q'), ('K','K'), ('V','V'), ('C_raw','C')]:
                    torch.testing.assert_close(getattr(f,k), oracle[r], atol=1e-12, rtol=1e-10)
                    oracle_metrics[k] = errors(getattr(f,k), oracle[r])
                torch.testing.assert_close(f.base_q_logits, f.Z @ block.Attn.to_q.weight.T, atol=0, rtol=0)
                torch.testing.assert_close(f.base_k_logits, f.Z @ block.Attn.to_k.weight.T, atol=0, rtol=0)
                torch.testing.assert_close(f.QC_raw, oracle['Q'] @ oracle['C'], atol=1e-12, rtol=1e-10)
                out = block.Attn.to_out(f.QC_raw.transpose(1,2).reshape(2,15,12))
                torch.testing.assert_close(out, ref_out, atol=1e-12, rtol=1e-10)
                oracle_metrics['QC_raw'] = errors(f.QC_raw, oracle['Q'] @ oracle['C'])
                oracle_metrics['output'] = errors(out, ref_out)
                for k,axis in [('Q',-1),('K',-2)]:
                    total = getattr(f,k).sum(axis)
                    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-12, rtol=1e-12)
                trace = Trace()
                with trace:
                    call(model, variant, inputs(variant, kw, dtype=torch.float64))
                self.assertEqual(len(trace.bmms), 2*len(model.blocks))
                self.assertEqual(len(trace.softmax), 2*len(model.blocks))
                for shape, axis, _ in trace.softmax:
                    self.assertEqual(shape[-2:], (15,8))
                    self.assertIn(axis, (-1,-2))
                self.assertFalse(any(out[-2:] in ((15,15),(8,8)) for _,_,out in trace.bmms))
                self.rows.append(dict(kind='oracle_and_execution', variant=variant,
                    bmm_shapes=trace.bmms, softmax_axes=[axis for _,axis,_ in trace.softmax],
                    N=15, M=8, d_h=4, no_NN_or_MM_attention=True,
                    metrics=oracle_metrics, atol=1e-12, rtol=1e-10))

    def test_temperature_boundaries_and_inert_air_temperature(self):
        for variant in ('temp', 'conv_temp', 'shapenet', 'airfrans'):
            _, model, _ = pair(variant)
            attention = model.double().blocks[0].Attn
            x = torch.randn(2, 15, 12, dtype=torch.float64, requires_grad=True)
            for temperature in (-3., 0.5, 8.):
                with torch.no_grad():
                    for name, p in attention.named_parameters():
                        if 'temperature' in name or 'tempreature' in name:
                            p.fill_(temperature)
                f = attention_factors(attention, x)
                raw_q, raw_k = f.base_q_logits.clone(), f.base_k_logits.clone()
                if variant == 'airfrans':
                    expected_q, expected_k = raw_q.softmax(-1), raw_k.softmax(-2)
                else:
                    lo, hi = ((.1, 2.) if variant == 'shapenet' else (.01, 1.))
                    tau = min(hi, max(lo, temperature))
                    expected_q, expected_k = (raw_q / tau).softmax(-1), (raw_k / tau).softmax(-2)
                torch.testing.assert_close(f.Q, expected_q, atol=0, rtol=0)
                torch.testing.assert_close(f.K, expected_k, atol=0, rtol=0)
                expected = attention(x)
                actual = attention.to_out(f.QC_raw.transpose(1, 2).reshape(2, 15, 12))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                ga = torch.autograd.grad(actual.square().sum(), tuple(attention.parameters()), allow_unused=True)
                gb = torch.autograd.grad(expected.square().sum(), tuple(attention.parameters()), allow_unused=True)
                for a, b in zip(ga, gb):
                    if a is None or b is None:
                        self.assertIs(a, b)
                    else:
                        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_real_PyG_multigraph_rejection_and_fresh_process_strict_roundtrip(self):
        from torch_geometric.data import Batch
        worker = '''
import importlib, json, sys, torch
from torch_geometric.data import Data
torch.set_num_threads(1)
p = torch.load(sys.argv[1], weights_only=True)
module, name = p['class_path'].rsplit('.', 1)
cls = getattr(importlib.import_module(module), name)
model = cls(**p['kwargs']).eval()
model.load_state_dict(p['state'], strict=True)
v = p['values']
def invoke(**extra):
    if p['variant'] == 'airfrans': return model(Data(**v), **extra)
    if p['variant'] == 'shapenet': return model((Data(**v), None), **extra)
    return model(v['x'], v.get('fx'), v.get('T'), **extra)
sizes = []
with torch.no_grad():
    first = invoke(observe=lambda t: sizes.append(len(t.history)))
    second = invoke()
torch.testing.assert_close(first, p['output'], atol=0, rtol=0)
torch.testing.assert_close(second, first, atol=0, rtol=0)
assert sizes == [0, 1, 2, 3]
assert type(model.blocks).__module__ == 'cdlno.linearno_history.core'
print(json.dumps({'class_path': p['class_path'], 'history_sizes': sizes, 'max_abs': 0.0}))
'''
        for variant, cwd in [('conv_temp', STANDARD), ('airfrans', ROOT/'Airfoil-Design-AirfRANS'),
                             ('shapenet', ROOT/'Car-Design-ShapeNetCar')]:
            _, model, kw = pair(variant)
            model.eval()
            values = {k: v.detach() for k, v in inputs(variant, kw).items()}
            with torch.no_grad():
                expected = call(model, variant, values)
            if variant in ('airfrans', 'shapenet'):
                data = Data(**values)
                multiple = Batch.from_data_list([data, data.clone()])
                with self.assertRaises(ValueError):
                    model(multiple if variant == 'airfrans' else (multiple, None))
            with tempfile.TemporaryDirectory(prefix='linearno-history-r2-') as temp:
                path = Path(temp)/'internal-state.pt'
                class_path = type(model).__module__ + '.' + type(model).__name__
                torch.save(dict(class_path=class_path, kwargs=kw, state=model.state_dict(),
                                values=values, output=expected, variant=variant), path)
                result = subprocess.run([sys.executable, '-B', '-c', worker, str(path)], cwd=cwd,
                    env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                             CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.rows.append(dict(kind='fresh_process_strict_state', cwd=str(cwd),
                    **json.loads(result.stdout)))

    def test_context_is_raw_graph_connected_immutable_and_not_serializable(self):
        _, model, kw = pair('plain')
        traces = []
        y = call(model, 'plain', inputs('plain', kw), observe=traces.append)
        raw = traces[0].factors.C_raw
        self.assertIs(traces[1].history[0], raw)
        grad = torch.autograd.grad(traces[-1].history[0].square().sum(),
                                   model.blocks[0].Attn.to_v.weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertIsNotNone(torch.autograd.grad(y.sum(), raw)[0])
        empty = RawHistoryContext(); full = empty.after_block(0, raw)
        self.assertEqual(empty.raw, ())
        self.assertIs(full.raw[0], raw)
        with self.assertRaises(ValueError): full.before_block(0)
        with self.assertRaises(TypeError): pickle.dumps(full)
        with self.assertRaises(TypeError): RawHistoryContext([raw])
        with self.assertRaises(ValueError): full.after_block(1, raw[:1])
        with self.assertRaises(Exception): full.raw = ()

    def test_append_after_FFN_final_head_and_no_state_retained(self):
        _, model, kw = pair('conv_temp')
        events, weak_contexts = [], []
        old_append = RawHistoryContext.after_block
        def append(ctx, index, raw):
            events.append(('append', index))
            weak_contexts.append(weakref.ref(ctx))
            result = old_append(ctx, index, raw)
            weak_contexts.append(weakref.ref(result))
            return result
        handles = [b.mlp.register_forward_hook(lambda m, a, o, i=i: events.append(('ffn',i))) for i,b in enumerate(model.blocks)]
        handles.append(model.blocks[-1].mlp2.register_forward_hook(lambda m,a,o: events.append(('head',3))))
        before = {n: set(vars(m)) for n,m in model.named_modules()}
        with patch.object(RawHistoryContext, 'after_block', append):
            y = call(model, 'conv_temp', inputs('conv_temp', kw))
        for h in handles: h.remove()
        self.assertEqual(events, [('ffn',0),('append',0),('ffn',1),('append',1),('ffn',2),('append',2),('ffn',3),('head',3),('append',3)])
        gc.collect()
        self.assertTrue(all(w() is None for w in weak_contexts))
        self.assertEqual(before, {n: set(vars(m)) for n,m in model.named_modules()})
        self.assertFalse(any(isinstance(v, RawHistoryContext) for m in model.modules() for v in vars(m).values()))
        y.sum().backward()

    def test_repeated_backward_batch_changes_exceptions_and_reentry(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                base, model, kw = pair(variant)
                base.eval(); model.eval()
                for B,N in ((2,15),(1,15),(2,15 if variant.startswith('conv') else 11)):
                    vals = inputs(variant, kw, B=B, N=N)
                    sizes = []
                    y = call(model, variant, vals, observe=lambda t: sizes.append(len(t.history)))
                    self.assertEqual(sizes, [0,1,2,3])
                    torch.testing.assert_close(y, call(base, variant, vals), atol=0, rtol=0)
                    y.square().sum().backward()
                    model.zero_grad(set_to_none=True)
                vals = inputs(variant, kw)
                def fail(t):
                    if t.index == 2: raise RuntimeError('injected observer failure')
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    call(model, variant, vals, observe=fail)
                sizes = []
                expected = call(base, variant, vals)
                def nested(t):
                    sizes.append(len(t.history))
                    if t.index == 1:
                        inner_sizes = []
                        inner = call(model, variant, vals, observe=lambda q: inner_sizes.append(len(q.history)))
                        torch.testing.assert_close(inner, expected, atol=0, rtol=0)
                        self.assertEqual(inner_sizes,[0,1,2,3])
                torch.testing.assert_close(call(model, variant, vals, observe=nested), expected, atol=0, rtol=0)
                self.assertEqual(sizes,[0,1,2,3])

    def test_invalid_topology_unimplemented_flags_and_original_factory(self):
        from model_dict import get_model
        from types import SimpleNamespace
        for key in ('LinearNO_Structured_Mesh_2D','LinearNO_Irregular_Mesh'):
            self.assertIs(get_model(SimpleNamespace(model=key)).Model, PureStandard)
        with self.assertRaises(TypeError): HistoryStandard(linearno_latent_attnres=True)
        _, model, _ = pair('plain')
        with self.assertRaises(ValueError): LinearNOHistoryCore([model.blocks[0]]*3+[model.blocks[-1]])
        broken = copy.deepcopy(list(model.blocks))
        broken[1].Attn.heads = 1
        with self.assertRaises(ValueError): LinearNOHistoryCore(broken)
        with self.assertRaises(ValueError): LinearNOHistoryCore(list(model.blocks)[:3])
        _, conv, _ = pair('conv')
        with self.assertRaises(ValueError): conv(torch.randn(2,14,2), torch.randn(2,14,1))
        self.assertEqual(set(model.state_dict()), set(pair('plain')[0].state_dict()))

    def test_wrapper_prefixes_match_unchanged_baseline_AST(self):
        pairs = [(STANDARD/'model/LinearNO.py','Model',STANDARD/'model/LinearNO_History.py','Model'),
                 (ROOT/'cdlno/linearno/airfrans.py','AirfRANSLinearNO',ROOT/'cdlno/linearno_history/models.py','AirfRANSHistoryModel'),
                 (ROOT/'cdlno/linearno/shapenet.py','ShapeNetLinearNO',ROOT/'cdlno/linearno_history/models.py','ShapeNetHistoryModel')]
        def body(path, name, research):
            cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
            f = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            nodes = []
            for n in f.body:
                if isinstance(n, ast.For): continue
                if research and isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.blocks': continue
                for v in ast.walk(n):
                    if isinstance(v,ast.Name) and v.id in ('air_single_graph','car_single_graph'):v.id='single_graph'
                nodes.append(n)
            return ast.dump(ast.Module(body=nodes,type_ignores=[]))
        for pure,pc,research,rc in pairs:
            self.assertEqual(body(pure,pc,False), body(research,rc,True))

    def test_temporal_context_fixtures_NS_and_Plasticity(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/history_r2_temporal.json').read_text())
        _, ns, kw = pair('plain', fun_dim=10, out_dim=1, Time_Input=False, unified_pos=True)
        x = torch.rand(2,15,2)
        initial = torch.randn(2,15,10)
        target = torch.randn(2,15,10)
        ns_optimizer = torch.optim.AdamW(ns.parameters(), lr=.001)
        for feedback in ('truth','prediction'):
            window = initial.clone(); loss = 0; calls = []
            for i in range(fixture['ns']['calls']):
                sizes = []
                pred = ns(x, window, observe=lambda t:sizes.append(len(t.history)))
                self.assertEqual(sizes,[0,1,2,3]); self.assertEqual(pred.shape,(2,15,1))
                frame = target[...,i:i+1] if feedback=='truth' else pred
                loss = loss + (pred-target[...,i:i+1]).square().mean()
                window = torch.cat((window[...,1:],frame),dim=-1)
                self.assertEqual(window.shape,initial.shape)
                calls.append(sizes)
            loss.backward()
            # Both feedback paths can backpropagate through this synthetic fixture;
            # only teacher forcing performs the single training update.
            if feedback == 'truth':
                ns_optimizer.step()
            ns_optimizer.zero_grad(set_to_none=True)
            self.rows.append(dict(kind='synthetic_NS_context', feedback=feedback, calls=len(calls), history_sizes=calls))
        def rollout(labels):
            window = initial.clone(); predictions = []; eval_loss = 0
            with torch.no_grad():
                for i in range(10):
                    pred = ns(x, window)
                    predictions.append(pred)
                    eval_loss += (pred - labels[...,i:i+1]).square().mean()
                    window = torch.cat((window[...,1:], pred), dim=-1)
            return torch.cat(predictions, dim=-1), eval_loss
        y1, loss1 = rollout(target)
        y2, loss2 = rollout(target + 100)
        torch.testing.assert_close(y1, y2, atol=0, rtol=0)
        self.assertNotEqual(loss1.item(), loss2.item())
        _, plas, _ = pair('conv', fun_dim=1, out_dim=4)
        optimizer = torch.optim.AdamW(plas.parameters(),lr=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
        times = torch.stack([torch.randperm(20),torch.randperm(20)]).float()/19
        fx = torch.randn(2,15,1); counts=[]
        for i in range(fixture['plasticity']['queries']):
            optimizer.zero_grad(set_to_none=True); sizes=[]
            pred = plas(x,fx,times[:,i:i+1],observe=lambda t:sizes.append(len(t.history)))
            self.assertEqual(sizes,[0,1,2,3]);self.assertEqual(pred.shape,(2,15,4))
            pred.square().mean().backward()
            self.assertGreater(plas.time_fc[0].weight.grad.abs().sum().item(),0)
            optimizer.step(); counts.append(sizes)
        scheduler.step()
        self.assertEqual(scheduler.last_epoch,1)
        self.rows.append(dict(kind='synthetic_Plasticity_context', calls=len(counts), history_sizes=counts, optimizer_steps=20,scheduler_steps=1))


if __name__ == '__main__':
    unittest.main()
