"""Data-free, single-fusion CDPA acceptance tests (v1.2 section 8.3).

Run: python -B -m unittest discover -s tests -p test_cdpa.py -v
The explicit oracle was written separately, before the production module.
"""

from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.cdpa import CDPA
from cdlno.checkpoint import SidecarMismatch, save_sidecar, validate_sidecar
from cdlno.config import CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdpa_reference import cdpa_reference


def setUpModule():
    torch.set_num_threads(1)
    print(f"phase-3 execution: torch={torch.__version__}, CUDA runtime={torch.version.cuda}")


def inputs(sources=5, *, b=3, m=5, d=8, dtype=torch.float32, device="cpu", scale=1.0):
    z = (torch.randn(b, m, d, dtype=dtype, device=device) * scale).requires_grad_()
    history = [(torch.randn_like(z) * scale + (s + 1) * scale / 7).requires_grad_() for s in range(sources)]
    return z, history


def leaves(module, z, history):
    return (z, *history, *module.parameters())


def excite(module):
    # Exercise trained, nonuniform source scoring and nonzero output/LN biases.
    with torch.no_grad():
        module.w.normal_(std=0.35)
        module.depth_norm.weight.uniform_(0.5, 1.5)
        module.to_out.bias.normal_(std=0.1)
        module.ln_q.bias.normal_(std=0.1)
        module.ln_kv.bias.normal_(std=0.1)


class DepthOperations(TorchDispatchMode):
    """Record actual math dispatch dtypes/autocast state within depth fusion."""

    def __init__(self, device_type):
        super().__init__()
        self.device_type = device_type
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        math_ops = ("aten.pow.", "aten.mean.", "aten.rsqrt.", "aten.mul.",
                    "aten.mv.", "aten.mm.", "aten._softmax.", "aten.sum.")
        if str(func).startswith(math_ops):
            tensors = [a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
            outputs = [result] if isinstance(result, torch.Tensor) else []
            self.records.append((str(func), [x.dtype for x in tensors + outputs],
                                 torch.is_autocast_enabled(self.device_type)))
        return result


class CDPAChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)

    def assertFinite(self, tensors):
        for index, tensor in enumerate(tensors):
            self.assertIsNotNone(tensor, index)
            self.assertTrue(torch.isfinite(tensor).all().item(), index)

    def assertParity(self, actual, expected, variables, *, atol=3e-6, rtol=3e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        cotangent = torch.randn_like(actual)
        ga = torch.autograd.grad(actual, variables, cotangent, retain_graph=True)
        gb = torch.autograd.grad(expected, variables, cotangent, retain_graph=True)
        self.assertFinite((*ga, *gb))
        for index, (a, b) in enumerate(zip(ga, gb)):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=lambda msg: f"gradient {index}: {msg}")
        return max((a - b).abs().max().item() for a, b in zip(ga, gb))

    def test_reference_does_not_call_production_or_sdpa(self):
        module = CDPA(8, 2)
        z, history = inputs(2)
        with patch.object(CDPA, "forward", side_effect=AssertionError("oracle called CDPA")):
            with patch.object(F, "scaled_dot_product_attention", side_effect=AssertionError("oracle called SDPA")):
                out, weights, token, _ = cdpa_reference(z, history, dict(module.named_parameters()), 2)
        self.assertEqual(out.shape, z.shape)
        self.assertEqual(weights.shape, (3, 5, 3))
        self.assertEqual(len(token), 2)

    def test_initialization_and_only_shared_parameters(self):
        with patch("torch.nn.init.trunc_normal_", wraps=nn.init.trunc_normal_) as trunc:
            module = CDPA(8, 2)
        self.assertEqual(Counter(id(c.args[0]) for c in trunc.call_args_list),
                         Counter(id(layer.weight) for layer in (module.to_q, module.to_k, module.to_v, module.to_out)))
        self.assertTrue(all(c.kwargs["std"] == 0.02 for c in trunc.call_args_list))
        torch.testing.assert_close(module.w, torch.zeros(8), atol=0, rtol=0)
        torch.testing.assert_close(module.depth_norm.weight, torch.ones(8), atol=0, rtol=0)
        self.assertFalse(hasattr(module.depth_norm, "bias"))
        self.assertEqual(module.depth_norm.eps, 1e-6)
        for norm in (module.ln_q, module.ln_kv):
            self.assertIsInstance(norm, nn.LayerNorm)
            self.assertEqual(norm.eps, 1e-6)
            torch.testing.assert_close(norm.weight, torch.ones(8), atol=0, rtol=0)
            torch.testing.assert_close(norm.bias, torch.zeros(8), atol=0, rtol=0)
        self.assertTrue(all(layer.bias is None for layer in (module.to_q, module.to_k, module.to_v)))
        torch.testing.assert_close(module.to_out.bias, torch.zeros(8), atol=0, rtol=0)
        self.assertEqual(set(dict(module.named_parameters())), {
            'w', 'ln_q.weight', 'ln_q.bias', 'ln_kv.weight', 'ln_kv.bias',
            'to_q.weight', 'to_k.weight', 'to_v.weight', 'to_out.weight', 'to_out.bias', 'depth_norm.weight',
        })
        # No per-source parameter families and no per-head QK norm/extra gate.
        ids = [id(p) for p in module.parameters()]
        for s in (1, 2, 5):
            module(*inputs(s))
            self.assertEqual(ids, [id(p) for p in module.parameters()])
        another = CDPA(8, 2)
        self.assertFalse({p.data_ptr() for p in module.parameters()} & {p.data_ptr() for p in another.parameters()})

    def test_empty_history_is_exact_identity_without_computation(self):
        module = CDPA(8, 2)
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            for chunk in (0, 1, 2, 8):
                with self.subTest(dtype=dtype, chunk=chunk):
                    z, _ = inputs(0, dtype=dtype)
                    with ExitStack() as stack:
                        for layer in (module.ln_q, module.to_q, module.ln_kv, module.depth_norm):
                            stack.enter_context(patch.object(layer, "forward", side_effect=AssertionError("empty history computation")))
                        stack.enter_context(patch.object(F, "scaled_dot_product_attention", side_effect=AssertionError("empty SDPA")))
                        out, weights = module(z, (), source_chunk_size=chunk, return_weights=True)
                    self.assertIs(out, z)
                    torch.testing.assert_close(weights, torch.ones(3, 5, 1), atol=0, rtol=0)
                    torch.testing.assert_close(torch.autograd.grad(out.sum(), z)[0], torch.ones_like(z), atol=0, rtol=0)

    def test_uniform_initialization_raw_values_and_no_outer_residual(self):
        for s in (1, 2, 5):
            module = CDPA(8, 2)
            with torch.no_grad():
                module.to_out.weight.zero_()
                module.to_out.bias.copy_(torch.arange(8).float() + 2)
                # Key normalization may change scores but must not touch values.
                module.depth_norm.weight.fill_(7)
            z, history = inputs(s)
            expected = (z + s * module.to_out.bias) / (s + 1)
            for chunk in (0, 1, 2, 9):
                out, weights = module(z, history, source_chunk_size=chunk, return_weights=True)
                torch.testing.assert_close(out, expected)
                torch.testing.assert_close(weights, torch.full((3, 5, s + 1), 1 / (s + 1)))
                torch.testing.assert_close(torch.autograd.grad(out.sum(), z, retain_graph=True)[0], torch.full_like(z, 1 / (s + 1)))
                self.assertFalse(torch.allclose(out, z))

    def test_two_softmax_axes_and_tokenwise_weights(self):
        module = CDPA(8, 2)
        excite(module)
        # Larger Q/K projections make a merged-history softmax distinguishable.
        with torch.no_grad():
            module.to_q.weight.copy_(torch.eye(8))
            module.to_k.weight.copy_(torch.eye(8))
        z, history = inputs(2)
        expected, alpha, token, raw = cdpa_reference(z, history, dict(module.named_parameters()), 2)
        actual, weights = module(z, history, return_weights=True)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(weights, alpha, atol=2e-6, rtol=2e-5)
        self.assertEqual(weights.shape, (3, 5, 3))
        torch.testing.assert_close(weights.sum(2), torch.ones(3, 5))
        for a in token:
            self.assertEqual(a.shape, (3, 2, 5, 5))
            torch.testing.assert_close(a.sum(-1), torch.ones(3, 2, 5))
        self.assertGreater((weights[:, 1:] - weights[:, :1]).abs().max().item(), 0.01)
        # Normalizing values or adding Z again yields a different answer.
        wrong_values = raw / torch.sqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
        self.assertFalse(torch.allclose(actual, (weights.unsqueeze(-1) * wrong_values).sum(2)))
        self.assertFalse(torch.allclose(actual, actual + z))
        # Deliberately concatenate sources on token axis to demonstrate a bad alternative.
        query = module.to_q(module.ln_q(z)).reshape(3, 5, 2, 4).transpose(1, 2)
        context = module.ln_kv(torch.cat(history, dim=1))
        k = module.to_k(context).reshape(3, 10, 2, 4).transpose(1, 2)
        v = module.to_v(context).reshape(3, 10, 2, 4).transpose(1, 2)
        merged = ((query @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(3, 5, 8)
        merged = module.to_out(merged)
        self.assertFalse(torch.allclose(merged, raw[:, :, 1:].mean(2), atol=1e-5, rtol=1e-4))

    def test_sdpa_reference_output_and_all_gradients_matrix(self):
        errors = []
        for dtype in (torch.float32, torch.float64):
            for s in (1, 2, 5):
                for nonzero in (False, True):
                    module = CDPA(8, 2).to(dtype)
                    if nonzero:
                        excite(module)
                    z, history = inputs(s, dtype=dtype)
                    expected, alpha, _, _ = cdpa_reference(z, history, dict(module.named_parameters()), 2)
                    for chunk in (0, 1, 2, s + 3):
                        with self.subTest(dtype=dtype, S=s, nonzero_w=nonzero, chunk=chunk):
                            with sdpa_kernel(SDPBackend.MATH):
                                actual, weights = module(z, history, source_chunk_size=chunk, return_weights=True)
                            self.assertEqual(actual.dtype, dtype)
                            torch.testing.assert_close(weights, alpha, atol=3e-6, rtol=3e-5)
                            errors.append(self.assertParity(actual, expected, leaves(module, z, history)))
        print(f"CDPA reference matrix: 48 cases, all parameter/input gradients; max abs grad error={max(errors):.3g}")

    def test_chunk_layout_counts_q_once_and_single_global_fusion(self):
        for s in (1, 2, 5):
            for chunk in (0, 1, 2, s + 3):
                with self.subTest(S=s, chunk=chunk):
                    module = CDPA(8, 2, source_chunk_size=chunk)
                    z, history = inputs(s)
                    with ExitStack() as stack:
                        spies = {name: stack.enter_context(patch.object(getattr(module, name), "forward", wraps=getattr(module, name).forward))
                                 for name in ('ln_q', 'to_q', 'ln_kv', 'to_k', 'to_v', 'to_out')}
                        depth = stack.enter_context(patch.object(module, '_depth_fusion', wraps=module._depth_fusion))
                        sdpa = stack.enter_context(patch.object(F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention))
                        module(z, history)
                    step = s if chunk == 0 else chunk
                    sizes = [min(step, s - i) for i in range(0, s, step)]
                    self.assertEqual(sdpa.call_count, len(sizes))
                    for name in ('ln_q', 'to_q'):
                        self.assertEqual(spies[name].call_count, 1)
                    for name in ('ln_kv', 'to_k', 'to_v', 'to_out'):
                        self.assertEqual(spies[name].call_count, len(sizes))
                    self.assertEqual(depth.call_count, 1)
                    self.assertIs(depth.call_args.args[0], z)
                    self.assertEqual(len(depth.call_args.args[1]), s)
                    for call, size in zip(sdpa.call_args_list, sizes):
                        self.assertTrue(all(qkv.shape == (3 * size, 2, 5, 4) for qkv in call.args))
                        self.assertEqual(call.kwargs, dict(dropout_p=0.0, is_causal=False, scale=0.5))

    def test_source_reorder_and_historical_token_permutations(self):
        module = CDPA(8, 2)
        excite(module)
        z, history = inputs(5)
        base, base_weights = module(z, history, return_weights=True)
        order = [4, 0, 3, 1, 2]
        reordered = [history[i] for i in order]
        for chunk in (0, 1, 2, 8):
            actual, weights = module(z, reordered, source_chunk_size=chunk, return_weights=True)
            self.assertParity(actual, base, leaves(module, z, history))
            torch.testing.assert_close(weights, base_weights[:, :, [0, *(i + 1 for i in order)]], atol=2e-6, rtol=2e-5)
            permuted = [t[:, torch.randperm(5)] for t in history]
            self.assertParity(module(z, permuted, source_chunk_size=chunk), base, leaves(module, z, history))

    def test_current_token_permutation_equivariance(self):
        module = CDPA(8, 2)
        excite(module)
        z, history = inputs(5)
        perm = torch.tensor([4, 1, 3, 0, 2])
        base, weights = module(z, history, return_weights=True)
        for chunk in (0, 1, 2, 8):
            actual, reordered_weights = module(z[:, perm], history, source_chunk_size=chunk, return_weights=True)
            self.assertParity(actual, base[:, perm], leaves(module, z, history))
            torch.testing.assert_close(reordered_weights, weights[:, perm], atol=2e-6, rtol=2e-5)

    def test_batch_isolation_forward_and_gradient(self):
        module = CDPA(8, 2)
        excite(module)
        z, history = inputs(5)
        for chunk in (0, 1, 2, 8):
            actual = module(z, history, source_chunk_size=chunk)
            single = torch.cat([module(z[i:i+1], [t[i:i+1] for t in history], source_chunk_size=chunk) for i in range(3)])
            self.assertParity(actual, single, leaves(module, z, history))
        grads = torch.autograd.grad(actual[0].sum(), (z, *history))
        for grad in grads:
            torch.testing.assert_close(grad[1:], torch.zeros_like(grad[1:]), atol=0, rtol=0)

    def test_zero_w_gradient_then_updated_w(self):
        module = CDPA(8, 2)
        z, history = inputs(5)
        cotangent = torch.randn_like(z)
        (module(z, history) * cotangent).sum().backward()
        self.assertFinite([x.grad for x in leaves(module, z, history)])
        torch.testing.assert_close(module.depth_norm.weight.grad, torch.zeros(8), atol=0, rtol=0)
        self.assertGreater(module.w.grad.abs().max().item(), 0)
        for tensor in (z, *history, module.to_q.weight, module.to_k.weight, module.to_v.weight, module.to_out.weight):
            self.assertGreater(tensor.grad.abs().max().item(), 0)
        # A synthetic scorer step only; no optimizer, data or training loop.
        with torch.no_grad():
            module.w.add_(module.w.grad, alpha=-0.01)
        module.zero_grad(set_to_none=True)
        (module(z, history) * cotangent).sum().backward()
        self.assertGreater(module.depth_norm.weight.grad.abs().max().item(), 0)
        self.assertTrue(torch.isfinite(module.depth_norm.weight.grad).all().item())

    def check_depth_precision(self, device, amp_dtype):
        module = CDPA(8, 2).to(device)
        excite(module)
        z, history = inputs(2, device=device)
        audit = DepthOperations(torch.device(device).type)
        original_depth = module._depth_fusion
        depth_io = []

        def audited_depth(current, aligned):
            with audit:
                result = original_depth(current, aligned)
            depth_io.append((current, aligned, result))
            return result

        with patch.object(module, '_depth_fusion', side_effect=audited_depth):
            with torch.autocast(torch.device(device).type, dtype=amp_dtype):
                actual, weights = module(z, history, return_weights=True)
                self.assertTrue(torch.is_autocast_enabled(torch.device(device).type))
        self.assertEqual(actual.dtype, z.dtype)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertTrue(audit.records)
        for name, dtypes, autocast_enabled in audit.records:
            self.assertTrue(dtypes and all(dtype == torch.float32 for dtype in dtypes), (name, dtypes))
            self.assertFalse(autocast_enabled, name)
        for op in ('aten.pow.', 'aten.mean.', 'aten.rsqrt.', 'aten.mul.', 'aten._softmax.', 'aten.sum.'):
            self.assertTrue(any(name.startswith(op) for name, _, _ in audit.records), op)
        self.assertTrue(any(name.startswith(('aten.mv.', 'aten.mm.')) for name, _, _ in audit.records))
        aligned = depth_io[0][1]
        self.assertTrue(all(t.dtype == amp_dtype for t in aligned))
        actual.square().mean().backward()
        self.assertFinite([t.grad for t in leaves(module, z, history)])

    def test_depth_fp32_dispatch_under_cpu_autocast(self):
        self.check_depth_precision('cpu', torch.bfloat16)

    def test_extreme_magnitudes_and_all_gradients_finite(self):
        # LayerNorm and depth mean-square are not arbitrary-range real arithmetic.
        # Explicitly test representative zero/tiny/large finite FP32 inputs.
        for scale in (0.0, 1e-12, 1e-6, 1.0, 1e4, 1e8):
            for chunk in (0, 1, 2, 8):
                with self.subTest(scale=scale, chunk=chunk):
                    module = CDPA(8, 2)
                    excite(module)
                    z, history = inputs(5, scale=scale)
                    out, weights = module(z, history, source_chunk_size=chunk, return_weights=True)
                    self.assertFinite((out, weights))
                    grads = torch.autograd.grad((out / max(scale, 1.0)).sum(), leaves(module, z, history))
                    self.assertFinite(grads)
        module = CDPA(8, 2)
        z, history = inputs(5)
        with torch.no_grad():
            module.w.fill_(1e4)
        out = module(z, history)
        self.assertFinite((out, *torch.autograd.grad(out.sum(), leaves(module, z, history))))

    def test_noncontiguous_inputs_and_minimal_sizes(self):
        module = CDPA(8, 2)
        excite(module)
        z, history = inputs(5)
        def strided(x):
            return x.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(strided(z).is_contiguous())
        self.assertParity(module(strided(z), [strided(t) for t in history], source_chunk_size=2), module(z, history), leaves(module, z, history))
        for s in (1, 2, 5):
            z1, h1 = inputs(s, b=1, m=1)
            expected = cdpa_reference(z1, h1, dict(module.named_parameters()), 2)[0]
            self.assertParity(module(z1, h1), expected, leaves(module, z1, h1))

    def test_mixed_history_dtypes_preserve_cast_gradients(self):
        module = CDPA(8, 2)
        excite(module)
        z, history = inputs(5)
        history = [t.detach().to(torch.bfloat16 if i % 2 else torch.float32).requires_grad_()
                   for i, t in enumerate(history)]
        expected = cdpa_reference(z, history, dict(module.named_parameters()), 2)[0]
        for chunk in (0, 1, 2, 8):
            self.assertParity(module(z, history, source_chunk_size=chunk), expected,
                              leaves(module, z, history), atol=1e-4, rtol=0.01)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = module(z, history, source_chunk_size=chunk)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertFinite(torch.autograd.grad(actual.sum(), leaves(module, z, history)))

    def test_stateless_calls_input_immutability_and_chunk_weight_loading(self):
        module = CDPA(8, 2, source_chunk_size=1)
        excite(module)
        z, history = inputs(5)
        tensors = [z, *history]
        snapshots = [t.detach().clone() for t in tensors]
        state = copy.deepcopy(module.state_dict())
        expected = module(z, history)
        module(*inputs(2))
        actual = module(z, history, source_chunk_size=2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for t, saved in zip(tensors, snapshots):
            torch.testing.assert_close(t, saved, atol=0, rtol=0)
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        self.assertEqual(set(module.__dict__) & {'history', 'aligned', 'weights', 'query', 'key', 'value'}, set())
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        for chunk in (0, 1, 2, 8):
            restored = CDPA(8, 2, source_chunk_size=chunk)
            buffer.seek(0)
            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            torch.testing.assert_close(restored(z, history), expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(module.source_chunk_size, 1)

    def test_sidecar_runtime_changes_and_architecture_rejection_without_overwrite(self):
        # Check the existing protocol without assembling or registering a model.
        architecture = CDLNOArchitectureConfig(M=5, d_model=8, num_heads=2)
        runtime = CDLNORuntimeConfig(source_chunk_size=1, device='cpu')
        with tempfile.TemporaryDirectory(prefix='cdlno-phase3-') as directory:
            path = Path(directory) / 'architecture.json'
            save_sidecar(path, architecture, runtime)
            saved = path.read_bytes()
            for chunk in (0, 1, 2, 8):
                requested_runtime = replace(runtime, source_chunk_size=chunk,
                                            device='cuda', dtype='bfloat16', amp=True)
                payload = validate_sidecar(path, architecture, requested_runtime)
                self.assertEqual(payload['runtime'], runtime.to_dict())
                self.assertEqual(path.read_bytes(), saved)
            for changes in ({'cdpa_mode': 'off'}, {'cdpa_mode': 'every_block'},
                            {'F': 0}, {'F': 3}, {'L': 12}, {'M': 6}):
                with self.subTest(changes=changes):
                    with self.assertRaisesRegex(SidecarMismatch, next(iter(changes))):
                        validate_sidecar(path, replace(architecture, **changes), runtime)
                    self.assertEqual(path.read_bytes(), saved)
            with self.assertRaises(FileExistsError):
                save_sidecar(path, replace(architecture, F=3), runtime)
            self.assertEqual(path.read_bytes(), saved)

    def test_invalid_inputs_rejected_before_projection(self):
        for dim, heads in ((0, 2), (8, 0), (7, 2)):
            with self.assertRaises(ValueError):
                CDPA(dim, heads)
        for chunk in (-1, 0.5, True, '2'):
            with self.assertRaises(ValueError):
                CDPA(8, 2, source_chunk_size=chunk)
        module = CDPA(8, 2)
        z, history = inputs(2)
        with patch.object(module.to_q, 'forward', side_effect=AssertionError('invalid input projected')):
            for bad in ([history[0][:, :4]], [history[0][:1]], [history[0].long()], ['not tensor']):
                with self.assertRaises(ValueError):
                    module(z, bad)
            with self.assertRaises(TypeError):
                module(z, torch.stack(history))
            with self.assertRaises(ValueError):
                module(z, history, source_chunk_size=-1)
            with self.assertRaises(ValueError):
                module(z.long(), [])


class CDPACUDAChecks(unittest.TestCase):
    assertFinite = CDPAChecks.assertFinite
    assertParity = CDPAChecks.assertParity
    check_depth_precision = CDPAChecks.check_depth_precision

    def setUp(self):
        torch.manual_seed(913)
        if not torch.cuda.is_available():
            self.skipTest('GPU unavailable; CUDA precision checks not run')
        old_matmul, old_conv = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', old_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, 'allow_tf32', old_conv)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def test_gpu_fp32_reference_and_fused_sdpa(self):
        print('phase-3 GPU:', torch.cuda.get_device_name())
        module = CDPA(8, 2).cuda()
        excite(module)
        for s in (1, 2, 5):
            z, history = inputs(s, device='cuda')
            expected = cdpa_reference(z, history, dict(module.named_parameters()), 2)[0]
            for chunk in (0, 1, 2, s + 3):
                with self.subTest(S=s, chunk=chunk):
                    with sdpa_kernel(SDPBackend.MATH):
                        math_result = module(z, history, source_chunk_size=chunk)
                    self.assertParity(math_result, expected, leaves(module, z, history), atol=5e-6, rtol=1e-4)
                    self.assertParity(module(z, history, source_chunk_size=chunk), math_result, leaves(module, z, history), atol=5e-6, rtol=1e-4)

    def test_gpu_depth_autocast_precision_and_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype):
                self.check_depth_precision('cuda', dtype)

    def test_gpu_half_bfloat16_outputs_chunks_and_extremes(self):
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for direct_half in (False, True):
                module = CDPA(8, 2).cuda()
                excite(module)
                if direct_half:
                    module = module.to(dtype)
                for scale in (0.0, 1e-4, 1.0, 1e4):
                    z, history = inputs(5, device='cuda', dtype=dtype if direct_half else torch.float32, scale=scale)
                    with torch.autocast('cuda', dtype=dtype, enabled=not direct_half):
                        base = module(z, history, source_chunk_size=1)
                    for chunk in (0, 2, 8):
                        with self.subTest(dtype=dtype, direct_half=direct_half, scale=scale, chunk=chunk):
                            with torch.autocast('cuda', dtype=dtype, enabled=not direct_half):
                                actual, alpha = module(z, history, source_chunk_size=chunk, return_weights=True)
                            self.assertEqual(actual.dtype, z.dtype)
                            self.assertEqual(alpha.dtype, torch.float32)
                            self.assertFinite((actual, alpha))
                            # Normalize scale before comparing low-precision gradients.
                            self.assertParity(actual / max(1.0, scale), base / max(1.0, scale), leaves(module, z, history), atol=0.03, rtol=0.04)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
