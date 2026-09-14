"""Phase-4 core contracts. All hooks, raw-state traces and counters are test-only."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import io
import math
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from torch.nn import functional as functional

from cdlno import CDLNO
from cdlno.cdpa import CDPA
from cdlno.config import CDLNOArchitectureConfig
from cdlno.modules import ConvFFN
from cdpa_reference import cdpa_reference

MODES = ('off', 'entry', 'every_block')


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


def config(*, L=8, F=2, mode='entry', structured=False):
    return CDLNOArchitectureConfig(
        L=L, F=F, M=4, d_model=8, num_heads=2, output_dim=3,
        cdpa_mode=mode, structured=structured,
        grid_shape=(5, 7) if structured else None,
    )


def inputs(cfg, *, device='cpu', dtype=torch.float32, n=11):
    n = math.prod(cfg.grid_shape) if cfg.structured else n
    return torch.randn(2, n, cfg.d_model, device=device, dtype=dtype, requires_grad=True)


@contextmanager
def trace(model):
    records = dict(front=[], bridge=[], rear=[], cdpa=[], readout=[])
    with ExitStack() as stack:
        def register(module, callback):
            stack.callback(module.register_forward_hook(callback).remove)
        for block in model.front_blocks:
            register(block, lambda _m, args, out: records['front'].append((args[0], *out)))
        register(model.bridge, lambda _m, args, out: records['bridge'].append((args[0], out)))
        for block in model.latent_blocks:
            register(block, lambda _m, args, out: records['rear'].append((args[0], out)))
        for key, module in model.cdpa_at.items():
            register(module, lambda _m, args, out, j=int(key): records['cdpa'].append((j, *args, out)))
        register(model.readout, lambda _m, args, out: records['readout'].append((*args, out)))
        yield records


def explicit_core(model, h_0):
    """Independent schedule via complete raw-state prefixes; CDPA uses math oracle.

    Reuses verified phase-2 blocks; never calls CDLNO.forward. Histories are
    derived by prefix slicing, independently of the production append loop.
    """
    cfg = model.config
    h_f, front = h_0, []
    for block in model.front_blocks:
        h_f, t = block(h_f)
        front.append(t)
    states = [model.bridge(h_f)]
    for j, block in enumerate(model.latent_blocks):
        z = states[-1]
        sources = front + states[:j] if cfg.cdpa_mode == 'every_block' else front
        active = cfg.cdpa_mode == 'every_block' or (cfg.cdpa_mode == 'entry' and j == 0)
        if active and sources:
            fusion = model.cdpa_at[str(j)]
            z = cdpa_reference(z, sources, dict(fusion.named_parameters()), cfg.num_heads)[0]
        states.append(block(z))
    return model.readout(h_f, states[-1])


class CoreChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(404)

    def assertFinite(self, tensor, name='tensor'):
        self.assertIsNotNone(tensor, name)
        self.assertTrue(torch.isfinite(tensor).all().item(), name)

    def assertAllGradients(self, model, h):
        for name, parameter in [*model.named_parameters(), ('H_0', h)]:
            self.assertFinite(parameter.grad, name)

    def test_default_and_42_depth_mode_point_grid_forward_backward_cases(self):
        default = CDLNO()
        self.assertEqual((default.config.L, default.config.F, default.config.P), (8, 2, 6))
        self.assertEqual(default.config.cdpa_mode, 'entry')
        self.assertEqual(default(torch.randn(1, 7, 64)).shape, (1, 7, 1))
        for structured in (False, True):
            for mode in MODES:
                for front in range(7):
                    with self.subTest(structured=structured, mode=mode, F=front):
                        cfg = config(F=front, mode=mode, structured=structured)
                        model, h = CDLNO(cfg), inputs(cfg)
                        y = model(h)
                        self.assertIsInstance(y, torch.Tensor)
                        self.assertEqual(y.shape, (2, h.shape[1], 3))
                        self.assertFinite(y)
                        (y * torch.randn_like(y)).sum().backward()
                        self.assertAllGradients(model, h)

    def test_extended_boundary_and_front_above_six_all_modes(self):
        for total, front in ((12, 2), (16, 6), (1, 0), (8, 7), (10, 8)):
            for mode in MODES:
                with self.subTest(L=total, F=front, mode=mode):
                    cfg = config(L=total, F=front, mode=mode)
                    model, h = CDLNO(cfg), inputs(cfg)
                    y = model(h)
                    self.assertEqual(y.shape, (2, 11, 3))
                    self.assertEqual(len(model.front_blocks), front)
                    self.assertEqual(len(model.latent_blocks), total - front)
                    y.square().mean().backward()
                    self.assertAllGradients(model, h)

    def test_f0_entry_off_same_weights_exact_outputs_gradients_and_calls(self):
        for structured in (False, True):
            for total in (1, 8):
                cfg = config(L=total, F=0, mode='off', structured=structured)
                off, entry = CDLNO(cfg), CDLNO(replace(cfg, cdpa_mode='entry'))
                entry.load_state_dict(off.state_dict(), strict=True)
                self.assertEqual(set(off.state_dict()), set(entry.state_dict()))
                self.assertFalse(any(isinstance(m, CDPA) for m in entry.modules()))
                h = inputs(cfg)
                he = h.detach().clone().requires_grad_()
                with patch.object(CDPA, 'forward', side_effect=AssertionError('inactive CDPA')):
                    with patch.object(functional, 'scaled_dot_product_attention', wraps=functional.scaled_dot_product_attention) as spy:
                        a = off(h)
                        off_shapes = [tuple(tuple(t.shape) for t in c.args[:3]) for c in spy.call_args_list]
                        spy.reset_mock()
                        b = entry(he)
                        self.assertEqual(off_shapes, [tuple(tuple(t.shape) for t in c.args[:3]) for c in spy.call_args_list])
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                cotangent = torch.randn_like(a)
                ga = torch.autograd.grad(a, (h, *off.parameters()), cotangent)
                gb = torch.autograd.grad(b, (he, *entry.parameters()), cotangent)
                for x, y in zip(ga, gb):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_registration_storage_independence_and_single_initialization(self):
        for front in range(7):
            for mode in MODES:
                model = CDLNO(config(F=front, mode=mode))
                expected = [] if mode == 'off' else ([0] if front else [])
                if mode == 'every_block':
                    expected = list(range(0 if front else 1, model.config.P))
                self.assertEqual(list(model.cdpa_at), [str(j) for j in expected])
                # Do not let parameters() silently deduplicate accidental sharing.
                params = list(model.named_parameters(remove_duplicate=False))
                self.assertEqual(len(params), len({id(p) for _, p in params}))
                self.assertEqual(len(params), len({p.untyped_storage().data_ptr() for _, p in params}))
                for family in (model.front_blocks, model.latent_blocks, model.cdpa_at.values()):
                    objects = list(family)
                    self.assertEqual(len(objects), len({id(x) for x in objects}))
                for module in (*model.latent_blocks, *model.cdpa_at.values()):
                    self.assertFalse(any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in module.modules()))
        with patch('torch.nn.init.trunc_normal_', wraps=nn.init.trunc_normal_) as trunc:
            model = CDLNO(config(mode='every_block', structured=True))
        expected = Counter(m.weight.data_ptr() for m in model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(Counter(c.args[0].data_ptr() for c in trunc.call_args_list), expected)
        for fusion in model.cdpa_at.values():
            torch.testing.assert_close(fusion.w, torch.zeros_like(fusion.w), atol=0, rtol=0)
            torch.testing.assert_close(fusion.depth_norm.weight, torch.ones_like(fusion.w), atol=0, rtol=0)

    def test_operation_counts_and_constant_m_all_front_depths(self):
        for mode in MODES:
            for front in range(7):
                cfg = config(F=front, mode=mode, structured=True)
                model, counts = CDLNO(cfg), Counter()
                with ExitStack() as stack:
                    groups = {
                        'down_bridge': [*(b.down for b in model.front_blocks), model.bridge],
                        'up_readout': [*(b.up for b in model.front_blocks), model.readout.up],
                        'latent_sa': [*(b.latent_sa for b in model.front_blocks), *(b.self_attention for b in model.latent_blocks)],
                        'conv': [m for m in model.modules() if isinstance(m, ConvFFN)],
                    }
                    for name, group in groups.items():
                        for module in group:
                            def count(_m, _a, out, label=name):
                                counts[label] += 1
                                if label in ('down_bridge', 'latent_sa'):
                                    self.assertEqual(out.shape, (2, cfg.M, cfg.d_model))
                            stack.callback(module.register_forward_hook(count).remove)
                    model(inputs(cfg))
                self.assertEqual(counts, {'down_bridge': front + 1, 'up_readout': front + 1,
                                          'latent_sa': cfg.L, 'conv': front + 1})

    def test_actual_history_sdpa_counts_chunks_and_each_location_projections(self):
        for front in (0, 2, 6):
            for mode in MODES:
                for chunk in (0, 1, 2, 99):
                    cfg = config(F=front, mode=mode)
                    model, measured = CDLNO(cfg, source_chunk_size=chunk), {}
                    with ExitStack() as stack:
                        for key, fusion in model.cdpa_at.items():
                            original = fusion.forward
                            def measured_fusion(z, history, *, _key=key, _fn=original, _module=fusion, **kwargs):
                                with ExitStack() as nested:
                                    projection = {name: nested.enter_context(patch.object(getattr(_module, name), 'forward', wraps=getattr(_module, name).forward))
                                                  for name in ('ln_q', 'to_q', 'ln_kv', 'to_k', 'to_v', 'to_out')}
                                    sdpa = nested.enter_context(patch.object(functional, 'scaled_dot_product_attention', wraps=functional.scaled_dot_product_attention))
                                    result = _fn(z, history, **kwargs)
                                measured[_key] = (len(history), sdpa.call_count)
                                groups = 1 if chunk == 0 else math.ceil(len(history) / chunk)
                                self.assertEqual(sdpa.call_count, groups)
                                for name, spy in projection.items():
                                    self.assertEqual(spy.call_count, 1 if name in ('ln_q', 'to_q') else groups, name)
                                for call in sdpa.call_args_list:
                                    for qkv in call.args[:3]:
                                        self.assertEqual(qkv.shape[1:], (cfg.num_heads, cfg.M, cfg.d_model // cfg.num_heads))
                                return result
                            stack.enter_context(patch.object(fusion, 'forward', side_effect=measured_fusion))
                        with patch.object(functional, 'scaled_dot_product_attention', wraps=functional.scaled_dot_product_attention) as all_sdpa:
                            model(inputs(cfg))
                    sizes = [] if mode == 'off' else ([front] if front else [])
                    if mode == 'every_block':
                        sizes = [s for s in range(front, front + cfg.P) if s > 0]
                    sources = 0 if mode == 'off' else (front if mode == 'entry' else cfg.P * front + cfg.P * (cfg.P - 1) // 2)
                    self.assertEqual([s for s, _ in measured.values()], sizes)
                    self.assertEqual(sum(s for s, _ in measured.values()), sources)
                    calls = sum(c for _, c in measured.values())
                    self.assertEqual(calls, sum(1 if chunk == 0 else math.ceil(s / chunk) for s in sizes))
                    self.assertEqual(all_sdpa.call_count, 2 * (front + 1) + cfg.L + calls)
                    if front == 2 and chunk == 0:
                        print(f'phase-4 default {mode}: logical histories={sources}, actual history SDPA={calls}, total SDPA={all_sdpa.call_count}')

    def assertSchedule(self, model, record, h_0):
        cfg = model.config
        h_f = record['front'][-1][1] if cfg.F else h_0
        self.assertIs(record['bridge'][0][0], h_f)
        front = [item[2] for item in record['front']]
        raw = [record['bridge'][0][1], *(item[1] for item in record['rear'])]
        self.assertEqual(len(record['rear']), cfg.P)
        self.assertIs(record['readout'][0][0], h_f)
        self.assertIs(record['readout'][0][1], raw[-1])
        fusions = {j: (z, hs, fused) for j, z, hs, fused in record['cdpa']}
        for j in range(cfg.P):
            sources = front + raw[:j] if cfg.cdpa_mode == 'every_block' else front
            active = cfg.cdpa_mode == 'every_block' or (cfg.cdpa_mode == 'entry' and j == 0)
            if active and sources:
                current, actual, fused = fusions[j]
                self.assertIsInstance(actual, tuple)
                self.assertIs(current, raw[j])
                self.assertEqual([id(t) for t in actual], [id(t) for t in sources])
                self.assertNotIn(id(current), [id(t) for t in actual])
                self.assertIs(record['rear'][j][0], fused)
                self.assertTrue(all(t.requires_grad and t.grad_fn is not None for t in actual))
            else:
                self.assertNotIn(j, fusions)
                self.assertIs(record['rear'][j][0], raw[j])

    def test_history_objects_timing_readout_and_two_forward_graph_isolation(self):
        for front in (0, 2, 6):
            for mode in MODES:
                model = CDLNO(config(F=front, mode=mode))
                before = {n: set(vars(m)) for n, m in model.named_modules()}
                h1, h2 = inputs(model.config), inputs(model.config, n=9)
                saved1, saved2 = h1.detach().clone(), h2.detach().clone()
                with trace(model) as first:
                    y1 = model(h1)
                with trace(model) as second:
                    y2 = model(h2)
                self.assertSchedule(model, first, h1)
                self.assertSchedule(model, second, h2)
                old = [t for _, z, hs, _ in first['cdpa'] for t in (z, *hs)]
                new = [t for _, z, hs, _ in second['cdpa'] for t in (z, *hs)]
                self.assertTrue({id(t) for t in old}.isdisjoint(id(t) for t in new))
                self.assertIsNone(torch.autograd.grad(y2.sum(), h1, allow_unused=True, retain_graph=True)[0])
                self.assertIsNone(torch.autograd.grad(y1.sum(), h2, allow_unused=True, retain_graph=True)[0])
                (y1.square().mean() + y2.square().mean()).backward()
                self.assertFinite(h1.grad)
                self.assertFinite(h2.grad)
                torch.testing.assert_close(h1, saved1, atol=0, rtol=0)
                torch.testing.assert_close(h2, saved2, atol=0, rtol=0)
                self.assertEqual(before, {n: set(vars(m)) for n, m in model.named_modules()})
                for module in model.modules():
                    self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(module).values()))

    def test_off_discards_t_and_entry_releases_history_after_only_use(self):
        # no_grad isolates Python references from autograd's legitimate saved tensors.
        for mode in ('off', 'entry'):
            model, refs = CDLNO(config(mode=mode)), []
            with ExitStack() as stack:
                for block in model.front_blocks:
                    def remember(_m, _args, out):
                        refs.append(weakref.ref(out[1]))
                    stack.callback(block.register_forward_hook(remember).remove)
                def check_bridge(_m, _args):
                    self.assertEqual(len(refs), model.config.F)
                    self.assertTrue(all((r() is None) == (mode == 'off') for r in refs))
                stack.callback(model.bridge.register_forward_pre_hook(check_bridge).remove)
                def check_later(_m, _args):
                    self.assertTrue(all(r() is None for r in refs))
                stack.callback(model.latent_blocks[1].register_forward_pre_hook(check_later).remove)
                with torch.no_grad():
                    model(inputs(model.config))

    def test_explicit_raw_state_schedule_output_and_all_gradient_parity(self):
        for front in (0, 2):
            for mode in MODES:
                model = CDLNO(config(F=front, mode=mode))
                with torch.no_grad():
                    for fusion in model.cdpa_at.values():
                        fusion.w.normal_(std=0.1)
                h = inputs(model.config)
                expected = explicit_core(model, h)
                actual = model(h, source_chunk_size=2)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=3e-4)
                cotangent = torch.randn_like(actual)
                ga = torch.autograd.grad(actual, (h, *model.parameters()), cotangent)
                gb = torch.autograd.grad(expected, (h, *model.parameters()), cotangent)
                for a, b in zip(ga, gb):
                    self.assertFinite(a)
                    self.assertFinite(b)
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=3e-4)

    def test_direct_history_readout_gradients_zero_w_then_updated_w(self):
        for mode in ('entry', 'every_block'):
            model = CDLNO(config(mode=mode, structured=True)).double()
            h = inputs(model.config, dtype=torch.float64)
            with trace(model) as records:
                y = model(h)
            intermediates = [item[2] for item in records['front']]
            intermediates += [records['bridge'][0][1], *(item[1] for item in records['rear'])]
            intermediates += [records['readout'][0][0]]
            grads = torch.autograd.grad(y, intermediates, torch.randn_like(y), retain_graph=True)
            for gradient in grads:
                self.assertFinite(gradient)
                self.assertGreater(gradient.abs().max().item(), 0)
            for _, z, sources, fused in records['cdpa']:
                direct = torch.autograd.grad(fused, (z, *sources), torch.randn_like(fused), retain_graph=True)
                self.assertTrue(all(g.abs().max().item() > 0 for g in direct))
            y.square().mean().backward()
            self.assertAllGradients(model, h)
            for fusion in model.cdpa_at.values():
                torch.testing.assert_close(fusion.depth_norm.weight.grad, torch.zeros_like(fusion.w), atol=0, rtol=0)
                self.assertGreater(fusion.w.grad.abs().max().item(), 0)
                with torch.no_grad():
                    fusion.w.add_(fusion.w.grad, alpha=-0.1)
            model.zero_grad(set_to_none=True)
            model(h).square().mean().backward()
            for fusion in model.cdpa_at.values():
                self.assertFinite(fusion.depth_norm.weight.grad)
                self.assertGreater(fusion.depth_norm.weight.grad.abs().max().item(), 0)

    def test_state_dict_roundtrip_actual_chunk_changes_and_all_gradients(self):
        for structured in (False, True):
            for mode in MODES:
                cfg = config(mode=mode, structured=structured)
                model = CDLNO(cfg).eval()
                with torch.no_grad():
                    for fusion in model.cdpa_at.values():
                        fusion.w.normal_(std=0.1)
                state = io.BytesIO()
                torch.save(model.state_dict(), state)
                h = inputs(cfg)
                expected = model(h)
                cotangent = torch.randn_like(expected)
                reference_grads = torch.autograd.grad(expected, (h, *model.parameters()), cotangent)
                for chunk in (0, 1, 2, 99):
                    restored = CDLNO(cfg, source_chunk_size=chunk).eval()
                    state.seek(0)
                    restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
                    actual = restored(h)  # Do not accidentally override every chunk back to 0.
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=3e-4)
                    grads = torch.autograd.grad(actual, (h, *restored.parameters()), cotangent)
                    for a, b in zip(grads, reference_grads):
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=3e-4)

    def test_point_order_batch_independence_and_variable_n(self):
        model = CDLNO(config(mode='every_block'))
        h = inputs(model.config)
        order = torch.randperm(h.shape[1])
        y = model(h)
        torch.testing.assert_close(model(h[:, order]), y[:, order], atol=2e-6, rtol=2e-5)
        single = torch.cat([model(h[i:i+1]) for i in range(2)])
        torch.testing.assert_close(single, y, atol=2e-6, rtol=2e-5)
        self.assertEqual(model(inputs(model.config, n=17)).shape, (2, 17, 3))


class CoreCUDAChecks(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('GPU unavailable: core CUDA checks not run')
        torch.manual_seed(404)

    def test_small_core_fp32_and_amp_modes_point_and_grid(self):
        print('phase-4 GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__)
        for structured in (False, True):
            for mode in MODES:
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                        continue
                    with self.subTest(structured=structured, mode=mode, dtype=dtype):
                        cfg = config(mode=mode, structured=structured)
                        model, h = CDLNO(cfg).cuda(), inputs(cfg, device='cuda')
                        with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                            y = model(h)
                        self.assertTrue(torch.isfinite(y).all().item())
                        self.assertEqual(y.shape, (2, h.shape[1], 3))
                        y.float().square().mean().backward()
                        for name, parameter in [*model.named_parameters(), ('H_0', h)]:
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
