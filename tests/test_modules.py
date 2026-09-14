"""Data-free phase-2 contract checks; no experiment modules are imported.

Run: python -B -m unittest discover -s tests -p test_modules.py -v
Only small individual modules are tested, with no CDLNO model assembly.
"""

from collections import Counter
from contextlib import ExitStack
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from cdlno.modules import (
    RMSNorm, make_norm, PlainFFN, GEGLUFFN, ConvFFN,
    LRSAFrontBlock, IPOTBridge, PersistentLatentBlock, LRSAFeatureReadout,
)


def setUpModule():
    torch.set_num_threads(1)
    print(f"phase-2 tests: torch={torch.__version__}, runtime={torch.version.cuda}")


def rms(x, weight, eps=1e-6):
    return x / (x.square().mean(-1, keepdim=True) + eps).sqrt() * weight


def layer_norm(x, module):
    centered = x - x.mean(-1, keepdim=True)
    return centered / (centered.square().mean(-1, keepdim=True) + module.eps).sqrt() * module.weight + module.bias


def linear(x, layer):
    y = x @ layer.weight.T
    return y if layer.bias is None else y + layer.bias


def heads(x, h):
    return x.reshape(x.shape[0], x.shape[1], h, -1).transpose(1, 2)


def explicit_attention(module, q_input, context):
    """Independent QK^T/softmax/AV formula; never calls SDPA."""
    if hasattr(module, "latent_queries"):
        q = module.latent_queries.transpose(0, 1).unsqueeze(0).expand(context.shape[0], -1, -1, -1)
    else:
        q = heads(linear(q_input, module.to_q), module.heads)
    k = heads(linear(context, module.to_k), module.heads)
    v = heads(linear(context, module.to_v), module.heads)
    if isinstance(module.q_norm, RMSNorm):
        q = rms(q, module.q_norm.weight)
        k = rms(k, module.k_norm.weight)
    scores = q @ k.transpose(-1, -2) / math.sqrt(module.head_dim)
    out = scores.softmax(-1) @ v
    out = out.transpose(1, 2).reshape(context.shape[0], q.shape[-2], module.dim)
    return linear(out, module.to_out)


def explicit_geglu(x, ffn):
    value, gate = linear(x, ffn.fc_in).chunk(2, dim=-1)
    # GELU exact erf formula, independent of the production activation call.
    activated = 0.5 * gate * (1 + torch.erf(gate / math.sqrt(2)))
    return linear(value * activated, ffn.fc_out)


class ModuleChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(714)

    def assertParity(self, actual, reference, leaves):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-8)
        cotangent = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, leaves, cotangent, retain_graph=True)
        reference_grads = torch.autograd.grad(reference, leaves, cotangent)
        for a, b in zip(actual_grads, reference_grads):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-7)

    def assertLiveGradients(self, module, inputs):
        for name, tensor in list(module.named_parameters()) + [(f"input{i}", x) for i, x in enumerate(inputs)]:
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(torch.isfinite(tensor.grad).all().item(), name)
            self.assertGreater(tensor.grad.abs().max().item(), 0, name)

    def test_norm_formula_outputs_and_gradients(self):
        for kind in ("rmsnorm", "layernorm"):
            with self.subTest(kind=kind):
                norm = make_norm(8, kind).double()
                x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
                with torch.no_grad():
                    norm.weight.uniform_(0.7, 1.3)
                ref = rms(x, norm.weight) if kind == "rmsnorm" else layer_norm(x, norm)
                self.assertParity(norm(x), ref, (x, *norm.parameters()))

    def test_rms_small_large_zero_and_low_precision(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            for scale in (0.0, 1e-4, 1e4):
                with self.subTest(dtype=dtype, scale=scale):
                    # Mixed parameter/input dtypes also occur under autocast.
                    norm = RMSNorm(8)
                    x = (torch.randn(2, 3, 8) * scale).to(dtype).requires_grad_()
                    y = norm(x)
                    self.assertEqual(y.dtype, dtype)
                    self.assertTrue(torch.isfinite(y).all().item())
                    y.float().square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_plain_and_geglu_formula(self):
        for cls in (PlainFFN, GEGLUFFN):
            with self.subTest(cls=cls.__name__):
                m = cls(8).double()
                x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
                if cls is PlainFFN:
                    a = linear(x, m.fc1)
                    ref = linear(0.5 * a * (1 + torch.erf(a / math.sqrt(2))), m.fc2)
                    self.assertEqual(m.fc1.out_features, 16)
                else:
                    ref = explicit_geglu(x, m)
                    self.assertEqual(m.fc_in.out_features, 32)
                    self.assertEqual(m.fc_out.in_features, 16)
                self.assertParity(m(x), ref, (x, *m.parameters()))

    def test_sdpa_matches_explicit_attention_and_gradients(self):
        front = LRSAFrontBlock(16, 4, 5).double()
        modules = [front.down, front.latent_sa.attn, front.up,
                   IPOTBridge(16, 4, 5).double().cross.attn]
        for m in modules:
            with self.subTest(module=type(m).__name__, qk_norm=type(m.q_norm).__name__):
                context = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
                if hasattr(m, "latent_queries"):
                    q = None
                    actual = m(context)
                    inputs = (context,)
                else:
                    q = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                    actual = m(q, context)
                    inputs = (q, context)
                reference = explicit_attention(m, q, context)
                self.assertParity(actual, reference, (*inputs, *m.parameters()))

    def test_front_order_history_object_and_live_gradients(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                block = LRSAFrontBlock(16, 4, 5, structured=structured, grid_shape=(5, 7))
                names = ["point_norm", "down", "latent_norm_1", "latent_ffn_1",
                         "latent_norm_sa", "latent_sa", "latent_norm_2", "latent_ffn_2",
                         "up_latent_norm", "up", "point_ffn_norm", "point_ffn"]
                calls, history_seen = [], []
                with ExitStack() as stack:
                    for name in names:
                        handle = getattr(block, name).register_forward_hook(
                            lambda _m, _a, _o, name=name: calls.append(name))
                        stack.callback(handle.remove)
                    handle = block.up_latent_norm.register_forward_pre_hook(
                        lambda _m, args: history_seen.append(args[0]))
                    stack.callback(handle.remove)
                    x = torch.randn(2, 35, 16, requires_grad=True)
                    saved = x.detach().clone()
                    y, t = block(x)
                self.assertEqual(calls, names)
                self.assertEqual(tuple(y.shape), (2, 35, 16))
                self.assertEqual(tuple(t.shape), (2, 5, 16))
                self.assertIs(t, history_seen[0])
                t.retain_grad()
                (y * torch.randn_like(y)).mean().backward()
                self.assertGreater(t.grad.abs().max().item(), 0)
                self.assertLiveGradients(block, (x,))
                torch.testing.assert_close(x.detach(), saved, atol=0, rtol=0)
                with torch.no_grad():
                    block.point_ffn.fc2.bias.add_(1)
                y2, t2 = block(x.detach())
                torch.testing.assert_close(t, t2, atol=0, rtol=0)
                torch.testing.assert_close(y2, y.detach() + 1)

    def test_front_down_zero_and_bridge_query_residual(self):
        x = torch.randn(2, 35, 16)
        front, bridge = LRSAFrontBlock(16, 4, 5), IPOTBridge(16, 4, 5)
        self.assertFalse(hasattr(front.down, "to_q"))
        for attn in (front.down, bridge.cross.attn):
            with torch.no_grad():
                attn.to_out.weight.zero_()
                attn.to_out.bias.zero_()
        torch.testing.assert_close(front.down(x), torch.zeros(2, 5, 16), atol=0, rtol=0)
        torch.testing.assert_close(bridge(x), bridge.latent_queries.expand(2, -1, -1), atol=0, rtol=0)
        self.assertFalse(any(isinstance(m, (PlainFFN, GEGLUFFN, ConvFFN)) for m in bridge.modules()))
        self.assertFalse(any("encoder_ff" in name for name, _ in bridge.named_parameters()))

    def test_bridge_formula_and_gradients(self):
        bridge = IPOTBridge(16, 4, 5).double()
        x = torch.randn(2, 35, 16, dtype=torch.float64, requires_grad=True)
        q = bridge.latent_queries.expand(2, -1, -1)
        reference = q + explicit_attention(bridge.cross.attn,
                    layer_norm(q, bridge.query_norm), layer_norm(x, bridge.context_norm))
        self.assertParity(bridge(x), reference, (x, *bridge.parameters()))
        (bridge(x) * torch.randn_like(q)).mean().backward()
        self.assertLiveGradients(bridge, (x,))

    def test_persistent_same_normalized_qkv_and_formula(self):
        m = PersistentLatentBlock(16, 4).double()
        z = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        seen, norm_calls = [], []
        with ExitStack() as stack:
            for layer in (m.self_attention.attn.to_q, m.self_attention.attn.to_k, m.self_attention.attn.to_v):
                handle = layer.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
                stack.callback(handle.remove)
            handle = m.norm_1.register_forward_hook(lambda _m, _a, out: norm_calls.append(out))
            stack.callback(handle.remove)
            actual = m(z)
        self.assertEqual(len(norm_calls), 1)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(x is norm_calls[0] for x in seen))
        n = layer_norm(z, m.norm_1)
        a = z + explicit_attention(m.self_attention.attn, n, n)
        ref = a + explicit_geglu(layer_norm(a, m.norm_2), m.ffn)
        self.assertParity(actual, ref, (z, *m.parameters()))
        (m(z) * torch.randn_like(z)).mean().backward()
        self.assertLiveGradients(m, (z,))

    def test_parameter_identity_and_storage_are_independent(self):
        front1, front2 = LRSAFrontBlock(16, 4, 5), LRSAFrontBlock(16, 4, 5)
        rear = [PersistentLatentBlock(16, 4) for _ in range(6)]
        modules = [front1, front2, IPOTBridge(16, 4, 5), *rear, LRSAFeatureReadout(16, 4, 3)]
        params = [p for m in modules for p in m.parameters()]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual(len(params), len({p.data_ptr() for p in params}))
        self.assertIsNot(front1.down.to_k, front1.up.to_k)
        for m in rear:
            self.assertEqual(sum(isinstance(layer, GEGLUFFN) for layer in m.modules()), 1)
            self.assertFalse(any(isinstance(layer, nn.Conv2d) for layer in m.modules()))
        snapshot = copy.deepcopy(rear[1].state_dict())
        with torch.no_grad():
            next(rear[0].parameters()).add_(1)
        for name, value in rear[1].state_dict().items():
            torch.testing.assert_close(value, snapshot[name], atol=0, rtol=0)

    def test_readout_query_context_residual_and_output_norm(self):
        m = LRSAFeatureReadout(16, 4, 3)
        x, z = torch.randn(2, 35, 16), torch.randn(2, 5, 16)
        seen = []
        handle = m.up.register_forward_pre_hook(lambda _m, args: seen.append(args))
        y = m(x, z)
        handle.remove()
        self.assertEqual(tuple(y.shape), (2, 35, 3))
        torch.testing.assert_close(seen[0][0], rms(x, m.query_norm.weight))
        torch.testing.assert_close(seen[0][1], rms(z, m.latent_norm.weight))
        self.assertIsInstance(m.output_norm, nn.LayerNorm)
        with torch.no_grad():
            m.up.to_out.weight.zero_()
            m.up.to_out.bias.zero_()
            m.point_ffn.fc2.weight.zero_()
            m.point_ffn.fc2.bias.zero_()
        features = m.forward_features(x, z)
        torch.testing.assert_close(features, x, atol=0, rtol=0)
        torch.testing.assert_close(m(x, z), linear(layer_norm(x, m.output_norm), m.output))

    def test_readout_formula_and_gradients(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                m = LRSAFeatureReadout(16, 4, 3, structured=structured, grid_shape=(5, 7)).double()
                x = torch.randn(2, 35, 16, dtype=torch.float64, requires_grad=True)
                z = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                # The explicit attention branch covers query/KV/output projections.
                delta = explicit_attention(m.up, rms(x, m.query_norm.weight), rms(z, m.latent_norm.weight))
                hd = x + delta
                normalized = rms(hd, m.point_ffn_norm.weight)
                update = m.point_ffn(normalized, (5, 7)) if structured else m.point_ffn(normalized)
                ref = linear(layer_norm(hd + update, m.output_norm), m.output)
                self.assertParity(m(x, z), ref, (x, z, *m.parameters()))
                (m(x, z) * torch.randn(2, 35, 3, dtype=torch.float64)).mean().backward()
                self.assertLiveGradients(m, (x, z))

    def test_conv_5_by_7_row_major_channel_mixing(self):
        # Distinct row, column and channel values catch swaps and flattened 1D convolution.
        m = ConvFFN(3).double()
        x = (torch.arange(2 * 35 * 3, dtype=torch.float64).reshape(2, 35, 3) / 31).requires_grad_()
        with torch.no_grad():
            m.conv.weight.zero_()
            m.conv.bias.copy_(torch.tensor([0.1, -0.3, 0.7]))
            m.conv.weight[:, :, 0, 1].copy_(torch.tensor([[1., 2., 3.], [-2., 4., 1.], [3., 1., -1.]]))
            m.conv.weight[:, :, 1, 2].copy_(torch.tensor([[.1, .2, -.5], [.3, -.4, .1], [-.2, .1, .8]]))
        expected_rows = []
        for row in range(5):
            for col in range(7):
                value = m.conv.bias.expand(2, -1)
                if row > 0:
                    value = value + x[:, (row - 1) * 7 + col] @ m.conv.weight[:, :, 0, 1].T
                if col < 6:
                    value = value + x[:, row * 7 + col + 1] @ m.conv.weight[:, :, 1, 2].T
                expected_rows.append(value)
        expected_conv = torch.stack(expected_rows, 1)
        seen = []
        hook = m.norm.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
        actual = m(x, (5, 7))
        hook.remove()
        torch.testing.assert_close(seen[0], expected_conv, atol=1e-12, rtol=1e-12)
        ref = linear(F.gelu(linear(layer_norm(expected_conv, m.norm), m.fc1)), m.fc2)
        # Conv weights outside the test stencil are excluded from this reference graph.
        leaves = (x, m.conv.bias, *m.norm.parameters(), *m.fc1.parameters(), *m.fc2.parameters())
        self.assertParity(actual, ref, leaves)
        self.assertEqual(m.conv.groups, 1)
        # Non-contiguous input must preserve the same point order too.
        noncontiguous = x.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(noncontiguous.is_contiguous())
        torch.testing.assert_close(m(noncontiguous, (5, 7)), actual)

    def test_point_permutation_latent_permutation_and_batch_isolation(self):
        front, readout = LRSAFrontBlock(16, 4, 5), LRSAFeatureReadout(16, 4, 3)
        x, z = torch.randn(2, 35, 16), torch.randn(2, 5, 16)
        perm, lp = torch.randperm(35), torch.randperm(5)
        y, t = front(x)
        yp, tp = front(x[:, perm])
        torch.testing.assert_close(yp, y[:, perm], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(tp, t, atol=2e-6, rtol=1e-5)
        out = readout(x, z)
        torch.testing.assert_close(readout(x[:, perm], z[:, lp]), out[:, perm], atol=2e-6, rtol=1e-5)
        for index in range(2):
            single, single_t = front(x[index:index+1])
            torch.testing.assert_close(single, y[index:index+1], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(single_t, t[index:index+1], atol=2e-6, rtol=1e-5)

    def test_norm_bias_and_ffn_default_policies(self):
        front = LRSAFrontBlock(16, 4, 5, structured=True)
        bridge, rear = IPOTBridge(16, 4, 5), PersistentLatentBlock(16, 4)
        readout = LRSAFeatureReadout(16, 4, 3, structured=True)
        for attn in (front.down, front.latent_sa.attn, front.up, bridge.cross.attn, rear.self_attention.attn, readout.up):
            for name in ("to_q", "to_k", "to_v"):
                if hasattr(attn, name):
                    self.assertIsNone(getattr(attn, name).bias)
            self.assertIsNotNone(attn.to_out.bias)
        for attn in (front.down, front.latent_sa.attn, front.up, readout.up):
            for norm in (attn.q_norm, attn.k_norm):
                self.assertIsInstance(norm, RMSNorm)
                self.assertEqual(tuple(norm.weight.shape), (4,))
            self.assertIsNot(attn.q_norm.weight, attn.k_norm.weight)
        for attn in (bridge.cross.attn, rear.self_attention.attn):
            self.assertIsInstance(attn.q_norm, nn.Identity)
            self.assertIsInstance(attn.k_norm, nn.Identity)
        for norm in (bridge.query_norm, bridge.context_norm, rear.norm_1, rear.norm_2, readout.output_norm):
            self.assertIsInstance(norm, nn.LayerNorm)
        for name in ("point_norm", "latent_norm_1", "latent_norm_sa", "latent_norm_2", "up_latent_norm", "point_ffn_norm"):
            self.assertIsInstance(getattr(front, name), RMSNorm)
        for name in ("query_norm", "latent_norm", "point_ffn_norm"):
            self.assertIsInstance(getattr(readout, name), RMSNorm)
        for m in (front, bridge, rear, readout):
            for layer in m.modules():
                if isinstance(layer, (RMSNorm, nn.LayerNorm)):
                    self.assertEqual(layer.eps, 1e-6)
                    torch.testing.assert_close(layer.weight, torch.ones_like(layer.weight), atol=0, rtol=0)
                if isinstance(layer, nn.LayerNorm):
                    torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), atol=0, rtol=0)
                if isinstance(layer, nn.Linear) and layer.bias is not None:
                    torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), atol=0, rtol=0)
        for ffn in (front.point_ffn, readout.point_ffn):
            self.assertIsNone(ffn.fc1.bias)
            self.assertIsNotNone(ffn.fc2.bias)
            self.assertIsInstance(ffn.norm, nn.LayerNorm)
            self.assertEqual(ffn.conv.kernel_size, (3, 3))
            self.assertEqual(ffn.conv.groups, 1)

    def test_initialization_once_queries_and_default_conv(self):
        with patch("torch.nn.init.trunc_normal_", wraps=nn.init.trunc_normal_) as trunc:
            with patch("torch.nn.init.orthogonal_", wraps=nn.init.orthogonal_) as ortho:
                with patch("torch.nn.init.normal_", wraps=nn.init.normal_) as normal:
                    modules = [LRSAFrontBlock(16, 4, 5), LRSAFrontBlock(16, 4, 5, structured=True),
                               IPOTBridge(16, 4, 5), PersistentLatentBlock(16, 4), LRSAFeatureReadout(16, 4, 3)]
        expected = {id(layer.weight) for m in modules for layer in m.modules() if isinstance(layer, nn.Linear)}
        actual = Counter(id(call.args[0]) for call in trunc.call_args_list)
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(count == 1 for count in actual.values()))
        self.assertTrue(all(call.kwargs["std"] == 0.02 for call in trunc.call_args_list))
        self.assertEqual(ortho.call_count, 2)
        self.assertEqual(normal.call_count, 1)
        self.assertEqual(normal.call_args.kwargs["std"], 0.02)
        for count in (5, 20):
            q = LRSAFrontBlock(16, 4, count).down.latent_queries.reshape(count, 16)
            gram = q @ q.T if count <= 16 else q.T @ q
            torch.testing.assert_close(gram, torch.eye(min(count, 16)), atol=1e-6, rtol=1e-6)
        torch.manual_seed(831)
        reference_conv = nn.Conv2d(8, 8, 3, padding=1)
        torch.manual_seed(831)
        conv = ConvFFN(8).conv
        torch.testing.assert_close(conv.weight, reference_conv.weight, atol=0, rtol=0)
        torch.testing.assert_close(conv.bias, reference_conv.bias, atol=0, rtol=0)

    def test_invalid_dimensions_ratios_and_grid_fail_early(self):
        for args in ((16, 0, 5), (15, 4, 5), (16, 4, 0), (-16, 4, 5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                LRSAFrontBlock(*args)
        for ratio in (0, -1, float("nan"), float("inf"), 1.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                GEGLUFFN(16, ratio)
        block = LRSAFrontBlock(16, 4, 5, structured=True)
        with patch.object(block.down, "forward", wraps=block.down.forward) as down:
            for grid in (None, (5, 8), (0, 7), (5, -7), (5, 7, 1)):
                with self.subTest(grid=grid), self.assertRaises(ValueError):
                    block(torch.randn(2, 35, 16), grid_shape=grid)
            down.assert_not_called()
        readout = LRSAFeatureReadout(16, 4, 3)
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            readout(torch.randn(2, 35, 16), torch.randn(1, 5, 16))

    def test_cpu_autocast_bfloat16_individual_modules(self):
        # The learned down-Q is FP32 even when projected K/V are BF16.
        for module, shapes in (
            (LRSAFrontBlock(16, 4, 5, structured=True, grid_shape=(5, 7)), [(2, 35, 16)]),
            (IPOTBridge(16, 4, 5), [(2, 35, 16)]),
            (PersistentLatentBlock(16, 4), [(2, 5, 16)]),
            (LRSAFeatureReadout(16, 4, 3, structured=True, grid_shape=(5, 7)), [(2, 35, 16), (2, 5, 16)]),
        ):
            with self.subTest(module=type(module).__name__):
                inputs = [torch.randn(shape, requires_grad=True) for shape in shapes]
                # Portable CPU math check: oneDNN BF16 convolution backward is
                # unavailable on some CPUs. No production backend is changed.
                with torch.backends.mkldnn.flags(enabled=False):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        out = module(*inputs)
                    outputs = out if isinstance(out, tuple) else (out,)
                    self.assertTrue(all(torch.isfinite(x).all().item() for x in outputs))
                    sum(x.float().square().mean() for x in outputs).backward()
                self.assertLiveGradients(module, inputs)


class CUDAModuleChecks(unittest.TestCase):
    def setUp(self):
        # CPU/CUDA FP32 parity needs matching precision, including convolution.
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_conv = torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", old_conv)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; GPU module checks not run")
    def test_small_cuda_fp32_and_amp(self):
        torch.manual_seed(19)
        print("phase-2 GPU:", torch.cuda.get_device_name())
        modes = [None, torch.float16]
        if torch.cuda.is_bf16_supported():
            modes.append(torch.bfloat16)
        for dtype in modes:
            for structured in (False, True):
                with self.subTest(dtype=dtype, structured=structured):
                    cases = [
                        (LRSAFrontBlock(16, 4, 5, structured=structured, grid_shape=(5, 7)), [(2, 35, 16)]),
                        (IPOTBridge(16, 4, 5), [(2, 35, 16)]),
                        (PersistentLatentBlock(16, 4), [(2, 5, 16)]),
                        (LRSAFeatureReadout(16, 4, 3, structured=structured, grid_shape=(5, 7)), [(2, 35, 16), (2, 5, 16)]),
                    ]
                    for module, shapes in cases:
                        module = module.cuda()
                        inputs = [torch.randn(shape, device="cuda", requires_grad=True) for shape in shapes]
                        reference = copy.deepcopy(module).cpu()(*(x.detach().cpu() for x in inputs))
                        with torch.autocast("cuda", enabled=dtype is not None, dtype=dtype or torch.float16):
                            out = module(*inputs)
                        outputs = out if isinstance(out, tuple) else (out,)
                        references = reference if isinstance(reference, tuple) else (reference,)
                        for actual, expected in zip(outputs, references):
                            self.assertTrue(torch.isfinite(actual).all().item())
                            tolerance = 2e-6 if dtype is None else 3e-3
                            torch.testing.assert_close(actual.float().cpu(), expected, atol=tolerance, rtol=0.02 if dtype else 1e-5)
                        sum(x.float().square().mean() for x in outputs).backward()
                        for p in (*module.parameters(), *inputs):
                            self.assertIsNotNone(p.grad)
                            self.assertTrue(torch.isfinite(p.grad).all().item())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
