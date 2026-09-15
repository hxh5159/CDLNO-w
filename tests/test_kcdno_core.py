"""K3 assembly acceptance; no dataset, wrapper or task entry imports."""

import copy
import io
import unittest
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.core import KCDNO, KCDNOBlock
from cdlno.kcdno.history import KernelHistoryReader, KernelHistoryWriter
from cdlno.modules import ConvFFN, PlainFFN, RMSNorm, _SelfAttention
from kernel_history_reference import history_reference


def small(**changes):
    return KCDNOArchitectureConfig(**dict(dict(L=2, d=8, h=2, M=3, kernel_rank=5), **changes))


# Test-only whitelist, never a production cross-architecture weight loader.
COMMON = {"point_norm", "down", "latent_norm_1", "latent_ffn_1", "latent_norm_2",
          "latent_ffn_2", "up_latent_norm", "up", "point_ffn_norm", "point_ffn"}


def copy_common(source, target):
    state = source.state_dict()
    target_state = target.state_dict()
    for name in target_state:
        if name.split('.')[2] not in COMMON:
            raise AssertionError(f"not a whitelisted common parameter: {name}")
    target.load_state_dict({name: state[name].clone() for name in target_state}, strict=True)


@contextmanager
def count_execution(model):
    """Observe real primitive calls, independent of declared config counts."""
    counts = Counter()
    handles = []

    def hook(label):
        def count(*_):
            counts[label] += 1
        return count

    def reader_hook(module, args):
        counts['reader_calls'] += 1
        counts['logical_reads'] += len(args[1])

    for block in model.blocks:
        for name in ('down', 'up', 'latent_ffn_1', 'latent_ffn_2', 'point_ffn', 'up_latent_norm'):
            handles.append(getattr(block, name).register_forward_hook(hook(name)))
        if hasattr(block, 'writer'):
            handles.append(block.writer.register_forward_hook(hook('writer')))
        if hasattr(block, 'reader'):
            handles.append(block.reader.register_forward_pre_hook(reader_hook))
            # K2 uses F.linear(weight), so observe query_norm AND actual linear.
            handles.append(block.reader.query_norm.register_forward_hook(hook('query_norm')))
    actual_sdpa, actual_einsum, actual_linear = F.scaled_dot_product_attention, torch.einsum, F.linear
    actual_softmax = torch.Tensor.softmax
    query_weights = {b.reader.to_q.weight.data_ptr() for b in model.blocks if hasattr(b, 'reader')}

    def sdpa(*args, **kwargs):
        counts['sdpa'] += 1
        if kwargs['scale'] != (model.config.d // model.config.h) ** -.5:
            raise AssertionError('Down/Up scale changed')
        return actual_sdpa(*args, **kwargs)

    def einsum(equation, *args, **kwargs):
        counts['einsum:' + equation] += 1
        return actual_einsum(equation, *args, **kwargs)

    def linear(x, weight, *args, **kwargs):
        if weight.data_ptr() in query_weights:
            counts['query_projection'] += 1
        return actual_linear(x, weight, *args, **kwargs)

    def softmax(x, dim, *args, **kwargs):
        if x.ndim != 3 or x.shape[1] != model.config.M or dim != 2:
            raise AssertionError('kernel source softmax axes changed')
        counts['source_softmax'] += 1
        return actual_softmax(x, dim, *args, **kwargs)

    try:
        with patch.object(F, 'scaled_dot_product_attention', side_effect=sdpa), \
             patch.object(torch, 'einsum', side_effect=einsum), patch.object(F, 'linear', side_effect=linear), \
             patch.object(torch.Tensor, 'softmax', new=softmax):
            yield counts
    finally:
        for handle in handles:
            handle.remove()


def manual_core(model, x, grid_shape=None):
    """Hand composition of primitives + double, per-source K2 oracle.

    Does not call Block/core/Writer/Reader.forward, and keeps raw histories.
    model and input must be double; history oracle does not force FP32.
    """
    raw_history, writer_parameters = [], []
    for block in model.blocks:
        h = block.point_norm(x)
        s = block.down(h)
        u = s + block.latent_ffn_1(block.latent_norm_1(s))
        if raw_history:
            uhat = history_reference(u, raw_history, writer_parameters,
                                    dict(block.reader.named_parameters()), explicit_tokens=True).output
        else:
            uhat = u
        t = uhat + block.latent_ffn_2(block.latent_norm_2(uhat))
        v = x + block.up(h, block.up_latent_norm(t))
        normed = block.point_ffn_norm(v)
        x = v + (block.point_ffn(normed, grid_shape) if grid_shape else block.point_ffn(normed))
        if hasattr(block, 'writer'):
            raw_history.append(t)
            writer_parameters.append(dict(block.writer.named_parameters()))
    return x


class KCDNOCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(3103)

    def test_depths_shapes_and_all_active_gradients(self):
        cases = [(1, 4, 1, 1, 3, 'point_ffn'), (2, 8, 2, 3, 5, 'conv_ffn'),
                 (4, 12, 3, 5, 2, 'point_ffn'), (8, 8, 2, 3, 5, 'conv_ffn'),
                 (12, 8, 4, 4, 7, 'point_ffn')]
        for L, d, h, M, r, point in cases:
            for mode in ('all', 'off'):
                with self.subTest(L=L, mode=mode):
                    model = KCDNO(small(L=L, d=d, h=h, M=M, kernel_rank=r,
                                        history_mode=mode, point_module=point))
                    x = torch.randn(2, 35, d, requires_grad=True)
                    original = x.detach().clone()
                    y = model(x, grid_shape=(5, 7) if point == 'conv_ffn' else None)
                    self.assertIsInstance(y, torch.Tensor)
                    self.assertEqual(y.shape, x.shape)
                    self.assertTrue(torch.isfinite(y).all())
                    (y.square().mean() + y.mean()).backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for name, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    # w=0 legitimately gives zero score-only depth scale grad.
                    for b in model.blocks:
                        if hasattr(b, 'reader'):
                            torch.testing.assert_close(b.reader.depth_norm.weight.grad,
                                                       torch.zeros(d), atol=0, rtol=0)
                    torch.testing.assert_close(x.detach(), original, atol=0, rtol=0)

    def test_default_L8_counts_and_history_boundaries(self):
        for mode in ('all', 'off'):
            model = KCDNO(small(L=8, history_mode=mode, point_module='conv_ffn'))
            with count_execution(model) as c:
                model(torch.randn(2, 35, 8), grid_shape=(5, 7))
            for key in ('down', 'up', 'latent_ffn_1', 'latent_ffn_2', 'point_ffn', 'up_latent_norm'):
                self.assertEqual(c[key], 8)
            self.assertEqual(c['sdpa'], 16)
            for key in ('writer', 'query_norm', 'query_projection', 'reader_calls', 'source_softmax',
                        'einsum:bmr,bsrd->bsmd', 'einsum:bmr,bsr->bsm'):
                self.assertEqual(c[key], 7 if mode == 'all' else 0, key)
            self.assertEqual(c['logical_reads'], 28 if mode == 'all' else 0)
            self.assertFalse(any(isinstance(m, _SelfAttention) for m in model.modules()))
            for i, block in enumerate(model.blocks):
                expected = COMMON | ({'reader'} if mode == 'all' and i > 0 else set())
                expected |= {'writer'} if mode == 'all' and i < 7 else set()
                self.assertEqual(set(dict(block.named_children())), expected)
            self.assertEqual(set(dict(model.named_children())), {'blocks'})
            print('L8 observed', mode, dict(c))
        one = KCDNO(small(L=1))
        off = KCDNO(small(L=1, history_mode='off'))
        self.assertEqual(list(one.state_dict()), list(off.state_dict()))
        self.assertFalse(any(isinstance(m, (KernelHistoryReader, KernelHistoryWriter)) for m in one.modules()))

    def test_initialization_projection_contract_and_parameter_independence(self):
        for point in ('point_ffn', 'conv_ffn'):
            with patch.object(nn.init, 'trunc_normal_', wraps=nn.init.trunc_normal_) as trunc, \
                 patch.object(nn.init, 'xavier_uniform_', wraps=nn.init.xavier_uniform_) as xavier, \
                 patch.object(nn.init, 'orthogonal_', wraps=nn.init.orthogonal_) as orthogonal:
                model = KCDNO(small(L=4, point_module=point))
            self.assertEqual(trunc.call_count, 4 * 13)  # Down3 + FFNs4 + Up4 + point2
            self.assertEqual(xavier.call_count, 6)
            self.assertEqual(orthogonal.call_count, 4)
            all_params = list(model.named_parameters(remove_duplicate=False))
            self.assertEqual(len({id(p) for _, p in all_params}), len(all_params))
            self.assertEqual(len({p.untyped_storage().data_ptr() for _, p in all_params}), len(all_params))
            trunc_ids = [id(c.args[0]) for c in trunc.call_args_list]
            self.assertEqual(len(trunc_ids), len(set(trunc_ids)))
            kernel_ids = [id(c.args[0]) for c in xavier.call_args_list]
            self.assertFalse(set(trunc_ids) & set(kernel_ids))
            for call in xavier.call_args_list:
                self.assertEqual(call.kwargs, {'gain': 1.0})
            for block in model.blocks:
                self.assertFalse(hasattr(block.down, 'to_q'))
                q = block.down.latent_queries.reshape(3, 8)
                torch.testing.assert_close(q @ q.T, torch.eye(3), atol=3e-7, rtol=3e-7)
                for attention in (block.down, block.up):
                    self.assertIsNone(attention.to_k.bias)
                    self.assertIsNone(attention.to_v.bias)
                    self.assertIsNotNone(attention.to_out.bias)
                    self.assertIsInstance(attention.q_norm, RMSNorm)
                    self.assertIsInstance(attention.k_norm, RMSNorm)
                self.assertIsNone(block.up.to_q.bias)
                for name in ('latent_ffn_1', 'latent_ffn_2', 'point_ffn'):
                    ffn = getattr(block, name)
                    self.assertEqual(tuple(ffn.fc1.weight.shape), (16, 8))
                    self.assertEqual(tuple(ffn.fc2.weight.shape), (8, 16))
                    self.assertIsNotNone(ffn.fc2.bias)
                    if name != 'point_ffn' or point == 'point_ffn':
                        self.assertIsInstance(ffn, PlainFFN)
                        self.assertIsNotNone(ffn.fc1.bias)
                if point == 'conv_ffn':
                    self.assertIsNone(block.point_ffn.fc1.bias)
                    self.assertEqual(block.point_ffn.conv.groups, 1)
                    self.assertEqual(block.point_ffn.conv.kernel_size, (3, 3))
                    self.assertEqual(block.point_ffn.conv.stride, (1, 1))
                    self.assertEqual(block.point_ffn.conv.padding, (1, 1))
                    self.assertIsInstance(block.point_ffn.norm, nn.LayerNorm)
                for m in block.modules():
                    if isinstance(m, (RMSNorm, nn.LayerNorm)):
                        torch.testing.assert_close(m.weight, torch.ones_like(m.weight), atol=0, rtol=0)
                if hasattr(block, 'reader'):
                    self.assertAlmostEqual(block.reader.gamma.item(), .1, places=7)
                    self.assertEqual(block.reader.gamma.ndim, 0)
                    self.assertEqual(torch.count_nonzero(block.reader.w).item(), 0)

    def test_manual_one_two_layer_equations_and_gradients(self):
        max_output, max_grad = 0., 0.
        for L in (1, 2):
            for point in ('point_ffn', 'conv_ffn'):
                model = KCDNO(small(L=L, point_module=point)).eval()
                # Nontrivial trained norms/scorer expose wrong T, double norm,
                # or bypassing a reader which zero initialization could hide.
                with torch.no_grad():
                    for b in model.blocks:
                        b.up_latent_norm.weight.copy_(torch.linspace(.3, 1.8, 8))
                        b.latent_ffn_2.fc2.bias.copy_(torch.linspace(-.2, .3, 8))
                        if hasattr(b, 'reader'):
                            b.reader.w.copy_(torch.linspace(-.4, .5, 8))
                            b.reader.gamma.fill_(.7)
                reference = copy.deepcopy(model).double()
                x = torch.randn(2, 35, 8, requires_grad=True)
                xr = x.detach().double().requires_grad_()
                grid = (5, 7) if point == 'conv_ffn' else None
                actual = model(x, grid_shape=grid)
                with patch.object(KCDNO, 'forward', side_effect=AssertionError('core in oracle')), \
                     patch.object(KCDNOBlock, 'forward', side_effect=AssertionError('block in oracle')), \
                     patch.object(KernelHistoryReader, 'forward', side_effect=AssertionError('reader in oracle')), \
                     patch.object(KernelHistoryWriter, 'forward', side_effect=AssertionError('writer in oracle')):
                    expected = manual_core(reference, xr, grid)
                torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-5)
                max_output = max(max_output, (actual.double() - expected).abs().max().item())
                probe = torch.randn_like(actual)
                actual_grad = torch.autograd.grad((actual * probe).sum(), (x, *model.parameters()))
                reference_grad = torch.autograd.grad((expected * probe.double()).sum(), (xr, *reference.parameters()))
                for a, e in zip(actual_grad, reference_grad, strict=True):
                    torch.testing.assert_close(a.double(), e, atol=2e-5, rtol=3e-4)
                    max_grad = max(max_grad, (a.double() - e).abs().max().item())
        print(f'Hand composition FP32/double: output maxabs={max_output:.9g}; gradient maxabs={max_grad:.9g}')

    def test_raw_T_up_norm_and_causal_tuple_schedule(self):
        model = KCDNO(small(L=4))
        records = [dict() for _ in model.blocks]
        handles = []
        events = []
        def save_input(row, key):
            def hook(module, args):
                row[key] = args
            return hook
        def save_output(row, key):
            def hook(module, args, out):
                row[key] = out
            return hook
        for i, b in enumerate(model.blocks):
            row = records[i]
            for name in ('point_norm', 'down', 'latent_ffn_1', 'latent_ffn_2', 'up_latent_norm', 'point_ffn'):
                handles.append(getattr(b, name).register_forward_hook(save_output(row, name)))
            for name in ('up', 'up_latent_norm', 'latent_norm_2'):
                handles.append(getattr(b, name).register_forward_pre_hook(save_input(row, name + '_input')))
            handles.append(b.register_forward_pre_hook(save_input(row, 'block_input')))
            handles.append(b.register_forward_hook(lambda m, a, o, i=i: events.append(('done', i))))
            if hasattr(b, 'writer'):
                handles.append(b.writer.register_forward_pre_hook(save_input(row, 'writer_input')))
                handles.append(b.writer.register_forward_hook(save_output(row, 'cache')))
                handles.append(b.writer.register_forward_hook(lambda m, a, o, i=i: events.append(('write', i))))
            if hasattr(b, 'reader'):
                handles.append(b.reader.register_forward_pre_hook(save_input(row, 'reader_input')))
                handles.append(b.reader.register_forward_hook(save_output(row, 'reader_output')))
                handles.append(b.reader.register_forward_hook(lambda m, a, o, i=i: events.append(('read', i))))
        x = torch.randn(2, 11, 8, requires_grad=True)
        out = model(x)
        for i, row in enumerate(records):
            incoming = row['block_input'][1]
            self.assertIsInstance(incoming, tuple)
            self.assertEqual(len(incoming), i)
            for s, cache in enumerate(incoming):
                self.assertIs(cache, records[s]['cache'])
                self.assertLess(events.index(('write', s)), events.index(('read', i)))
            self.assertIs(row['up_input'][0], row['point_norm'])
            self.assertIs(row['up_input'][1], row['up_latent_norm'])
            t = row['up_latent_norm_input'][0]
            torch.testing.assert_close(t, row['latent_norm_2_input'][0] + row['latent_ffn_2'], atol=0, rtol=0)
            if i < 3:
                self.assertIs(row['writer_input'][0], t)
            if i:
                torch.testing.assert_close(row['reader_input'][0], row['down'] + row['latent_ffn_1'], atol=0, rtol=0)
        # Isolated last reader output -> first writer weights cannot use the
        # point residual bypass; proves the assembled core's history graph.
        early_t = records[0]['writer_input'][0]
        key = model.blocks[0].writer.to_k.weight
        grads = torch.autograd.grad(records[-1]['reader_output'].square().sum(), (early_t, key), retain_graph=True)
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(g.abs().max().item(), 0.)
        final_key_grad = torch.autograd.grad(out.square().sum(), key)[0]
        self.assertTrue(torch.isfinite(final_key_grad).all())
        self.assertGreater(final_key_grad.abs().max().item(), 0.)
        for h in handles:
            h.remove()

    def test_gamma_zero_all_equals_off_same_common_weights_and_cost_differs(self):
        for point in ('point_ffn', 'conv_ffn'):
            cfg = small(L=4, point_module=point)
            model = KCDNO(cfg).eval()
            off = KCDNO(replace(cfg, history_mode='off')).eval()
            copy_common(model, off)
            with torch.no_grad():
                for b in model.blocks[1:]:
                    b.reader.gamma.zero_()
            x = torch.randn(2, 35, 8)
            grid = (5, 7) if point == 'conv_ffn' else None
            with count_execution(model) as active:
                a = model(x, grid_shape=grid)
            with count_execution(off) as inactive:
                b = off(x, grid_shape=grid)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(active['logical_reads'], 6)
            self.assertEqual(active['writer'], 3)
            self.assertEqual(inactive['logical_reads'], 0)
            self.assertEqual(inactive['writer'], 0)

    def test_forward_isolation_variable_N_and_no_persistent_state(self):
        model = KCDNO(small(L=4)).eval()
        a, b = torch.randn(2, 11, 8), torch.randn(2, 19, 8)
        caches = []
        handles = [block.writer.register_forward_hook(lambda m, a, o: caches.append(o))
                   for block in model.blocks if hasattr(block, 'writer')]
        state = {n: t.clone() for n, t in model.state_dict().items()}
        first = model(a)
        snapshots = [(c.memory.clone(), c.mass.clone()) for c in caches]
        other = model(b)
        last = model(a)
        self.assertEqual(other.shape, b.shape)
        torch.testing.assert_close(first, last, atol=0, rtol=0)
        self.assertEqual(len({id(c) for c in caches}), 9)
        for c, (memory, mass) in zip(caches[:3], snapshots, strict=True):
            torch.testing.assert_close(c.memory, memory, atol=0, rtol=0)
            torch.testing.assert_close(c.mass, mass, atol=0, rtol=0)
        torch.testing.assert_close(state, model.state_dict(), atol=0, rtol=0)
        self.assertEqual(list(model.named_buffers()), [])
        for module in model.modules():
            for value in vars(module).values():
                self.assertFalse(isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter))
                self.assertFalse(isinstance(value, list))
        # Distinct calls' graphs remain independent after backward frees first.
        first.sum().backward()
        model.zero_grad(set_to_none=True)
        last.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for h in handles:
            h.remove()

    def test_strict_config_weight_roundtrip(self):
        for point in ('point_ffn', 'conv_ffn'):
            for mode in ('all', 'off'):
                model = KCDNO(small(history_mode=mode, point_module=point)).eval()
                x = torch.randn(2, 35, 8)
                grid = (5, 7) if point == 'conv_ffn' else None
                # Include learned gate values; loading must not reinitialize.
                with torch.no_grad():
                    if mode == 'all':
                        model.blocks[1].reader.gamma.fill_(-.3)
                f = io.BytesIO()
                torch.save({'architecture': model.config.to_dict(), 'state_dict': model.state_dict()}, f)
                f.seek(0)
                saved = torch.load(f, weights_only=True)
                clone = KCDNO(KCDNOArchitectureConfig.from_dict(saved['architecture'])).eval()
                clone.load_state_dict(saved['state_dict'], strict=True)
                torch.testing.assert_close(model(x, grid_shape=grid), clone(x, grid_shape=grid), atol=0, rtol=0)
                if mode == 'all':
                    wrong = KCDNO(replace(model.config, history_mode='off'))
                    with self.assertRaises(RuntimeError):
                        wrong.load_state_dict(saved['state_dict'], strict=True)
                    wrong = KCDNO(replace(model.config, kernel_rank=7))
                    with self.assertRaises(RuntimeError):
                        wrong.load_state_dict(saved['state_dict'], strict=True)

    def test_validation_and_grid_contract(self):
        for kwargs in ({'L': 0}, {'L': True}, {'d': 7}, {'h': 0}, {'M': 2.5},
                       {'kernel_rank': False}, {'history_mode': 'entry'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KCDNO(small(**kwargs))
        for name in ('ffn1_hidden', 'ffn2_hidden', 'point_hidden'):
            with self.assertRaisesRegex(ValueError, name):
                KCDNO(small(**{name: 24}))
        with self.assertRaises(TypeError):
            KCDNO({})
        for i in (-1, 2, 1.5, True):
            with self.assertRaises(ValueError):
                KCDNOBlock(small(), i)
        conv = KCDNO(small(point_module='conv_ffn'))
        x = torch.randn(2, 35, 8)
        self.assertEqual(conv(x, grid_shape=(5, 7)).shape, x.shape)
        for grid in (None, [5, 7], (7,), (5, 8), (True, 35), (5., 7)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                conv(x, grid_shape=grid)
        point = KCDNO(small())
        with self.assertRaises(ValueError):
            point(x, grid_shape=(5, 7))
        for bad in (None, torch.ones(2, 35, 8, dtype=torch.int64), torch.randn(2, 8),
                    torch.randn(2, 35, 7), torch.empty(0, 35, 8), torch.empty(2, 0, 8)):
            with self.assertRaises(ValueError):
                point(bad)
        with self.assertRaises(TypeError):
            point.blocks[0](x, [])
        with self.assertRaises(ValueError):
            point.blocks[1](x, ())
        with self.assertRaises(ValueError):
            point.blocks[0](x, (None,))

    @unittest.skipUnless(torch.cuda.is_available(), 'no actual CUDA GPU')
    def test_limited_gpu_fp32_and_amp_backward(self):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            for point in ('point_ffn', 'conv_ffn'):
                cpu = KCDNO(small(point_module=point)).eval()
                gpu = copy.deepcopy(cpu).cuda()
                x = torch.randn(2, 35, 8)
                grid = (5, 7) if point == 'conv_ffn' else None
                with sdpa_kernel(SDPBackend.MATH):
                    expected = cpu(x, grid_shape=grid)
                    actual = gpu(x.cuda(), grid_shape=grid)
                torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=3e-5)
                actual.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters()))
                for dtype in (torch.float16, torch.bfloat16):
                    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                        continue
                    gpu.zero_grad(set_to_none=True)
                    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=dtype):
                        amp = gpu(x.cuda(), grid_shape=grid)
                        loss = amp.square().mean()
                    self.assertTrue(torch.isfinite(amp).all())
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters()))
                    torch.testing.assert_close(amp.cpu(), expected, atol=3e-3, rtol=1e-2)
                    print('GPU core finite forward/backward', point, str(dtype))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn


if __name__ == '__main__':
    unittest.main(verbosity=2)
