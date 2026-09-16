"""M2 primitive acceptance; synthetic tensors only, no trainer/entry imports."""
from collections import Counter
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno.modules import _DownAttention
from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.modules import (
    LearnedQueryDown, LatentFFNSAFFNBlock, QueryAlignedUpCross,
    PairwiseAttnResFusion, CoverageFloorLoss, coverage_diagnostics,
)
import msar_reference as oracle


def setUpModule():
    torch.set_num_threads(1)
    print(f'M2 environment: torch={torch.__version__}, CUDA={torch.version.cuda}, '
          f'GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')


def state(module):
    return dict(module.named_parameters())


class MSARPrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9162)

    def parity(self, actual, expected, leaves, *, double=False):
        atol, rtol = (2e-10, 2e-8) if double else (3e-6, 1e-4)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        probe = torch.randn_like(actual)
        ga = torch.autograd.grad(actual, leaves, probe, retain_graph=True)
        ge = torch.autograd.grad(expected, leaves, probe, retain_graph=True)
        for a, e in zip(ga, ge):
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, e, atol=atol * 5, rtol=rtol * 5)

    def test_down_explicit_oracle_outputs_masks_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for b, n, m, d, h in ((1, 7, 3, 8, 2), (2, 35, 5, 12, 3), (2, 3, 7, 8, 4)):
                with self.subTest(dtype=dtype, shape=(b, n, m, d, h)):
                    down = LearnedQueryDown(d, h, m).to(dtype)
                    x = torch.randn(b, n, d, dtype=dtype, requires_grad=True)
                    valid = torch.ones(b, n, dtype=torch.bool)
                    valid[:, -1] = False
                    if b == 2:
                        valid[1, 0] = False
                    y, a = down(x, valid_mask=valid, return_aux=True, coverage=MSARTrainingConfig())
                    ref, ar = oracle.down(x, state(down), h, valid)
                    self.parity(y, ref, (x, *down.parameters()), double=dtype == torch.float64)
                    torch.testing.assert_close(a, ar, atol=1e-7 if dtype == torch.float32 else 1e-12, rtol=1e-6)
                    torch.testing.assert_close(a.sum(-1), torch.ones(b, h, m, dtype=dtype))
                    self.assertTrue((a.masked_select(~valid[:, None, None].expand_as(a)) == 0).all())
                    grad = torch.autograd.grad(y.sum(), x)[0]
                    self.assertEqual(grad[~valid].abs().max().item(), 0)

    def test_aux_gate_and_sdpa_path_equivalence(self):
        down = LearnedQueryDown(8, 2, 5)
        x = torch.randn(2, 7, 8)
        valid = torch.tensor([[True] * 6 + [False], [True] * 7])
        active = MSARTrainingConfig()
        expected, a = down(x, valid_mask=valid, return_aux=True, coverage=active)
        self.assertIsNotNone(a)
        for mode, weight, train, request in (
            ('off', .01, True, True), ('floor', 0., True, True),
            ('floor', .01, False, True), ('floor', .01, True, False),
        ):
            with self.subTest(mode=mode, weight=weight, train=train, request=request):
                down.train(train)
                cfg = MSARTrainingConfig(coverage_mode=mode, coverage_weight=weight, diagnostics=True)
                # Any explicit Tensor.softmax here would mean an unwanted aux A.
                with patch.object(torch.Tensor, 'softmax', side_effect=AssertionError('unexpected explicit A')):
                    with patch('cdlno.msar_lno.modules.F.scaled_dot_product_attention',
                               wraps=F.scaled_dot_product_attention) as sdpa:
                        result = down(x, valid_mask=valid, return_aux=request, coverage=cfg)
                        self.assertEqual(sdpa.call_count, 1)
                if request:
                    y, absent = result
                    self.assertIsNone(absent)
                else:
                    y = result
                    self.assertIsInstance(y, torch.Tensor)
                torch.testing.assert_close(y, expected, atol=2e-7, rtol=2e-5)
        down.train()
        self.assertIsNone(down(x, return_aux=True)[1])
        self.assertFalse(any(isinstance(v, torch.Tensor) for v in down.__dict__.values()))

    def test_aux_and_sdpa_gradient_equivalence(self):
        down = LearnedQueryDown(12, 3, 5).double()
        x = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        off = down(x, valid_mask=mask)
        active, _ = down(x, valid_mask=mask, return_aux=True, coverage=MSARTrainingConfig())
        self.parity(off, active, (x, *down.parameters()), double=True)

    def test_down_matches_accepted_old_primitive_same_weights(self):
        old = _DownAttention(8, 2, 5).double()
        new = LearnedQueryDown(8, 2, 5).double()
        weights = old.state_dict()
        weights['latent_queries'] = weights['latent_queries'].reshape(5, 8)
        new.load_state_dict(weights, strict=True)
        x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(old(x), new(x), atol=0, rtol=0)
        self.assertFalse(hasattr(new, 'to_q'))
        with torch.no_grad():
            new.to_out.weight.zero_(); new.to_out.bias.zero_()
        torch.testing.assert_close(new(x), torch.zeros(2, 5, 8, dtype=torch.float64), atol=0, rtol=0)

    def test_latent_three_residual_formula_and_calls(self):
        for b, n, d, h, dtype in ((1, 7, 8, 2, torch.float64), (2, 35, 12, 3, torch.float32)):
            block = LatentFFNSAFFNBlock(d, h).to(dtype)
            x = torch.randn(b, n, d, dtype=dtype, requires_grad=True)
            counts = Counter()
            def hook(name):
                return lambda *args: counts.update([name])
            handles = [getattr(block, name).register_forward_hook(hook(name))
                       for name in ('norm1', 'ffn1', 'norm_sa', 'sa', 'norm2', 'ffn2')]
            y = block(x)
            for handle in handles:
                handle.remove()
            self.assertEqual(counts, Counter(dict.fromkeys(('norm1', 'ffn1', 'norm_sa', 'sa', 'norm2', 'ffn2'), 1)))
            self.parity(y, oracle.latent(x, state(block), h), (x, *block.parameters()), double=dtype == torch.float64)
            for ffn in (block.ffn1, block.ffn2):
                self.assertEqual(ffn.fc1.weight.shape, (2 * d, d))
                self.assertIsNotNone(ffn.fc1.bias)
            with torch.no_grad():
                for layer in (block.ffn1.fc2, block.sa.attn.to_out, block.ffn2.fc2):
                    layer.weight.zero_(); layer.bias.zero_()
            torch.testing.assert_close(block(x), x, atol=0, rtol=0)

    def test_up_formula_permutation_branch_and_batch_isolation(self):
        for dtype in (torch.float32, torch.float64):
            up = QueryAlignedUpCross(12, 3).to(dtype)
            e = torch.randn(2, 7, 12, dtype=dtype, requires_grad=True)
            d = torch.randn(2, 5, 12, dtype=dtype, requires_grad=True)
            y = up(e, d)
            self.parity(y, oracle.up(e, d, state(up), 3), (e, d, *up.parameters()), double=dtype == torch.float64)
            perm = torch.randperm(7)
            torch.testing.assert_close(up(e[:, perm], d), y[:, perm])
            torch.testing.assert_close(up(e, d), torch.cat([up(e[i:i+1], d[i:i+1]) for i in range(2)]))
            with torch.no_grad():
                up.to_out.weight.zero_(); up.to_out.bias.zero_()
            torch.testing.assert_close(up(e, d), torch.zeros_like(e), atol=0, rtol=0)

    def test_fusion_zero_initialization_jacobian_and_learnable_w(self):
        fusion = PairwiseAttnResFusion(8).double()
        e = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
        u = torch.randn_like(e, requires_grad=True)
        self.assertEqual(list(state(fusion)), ['w'])
        self.assertEqual(torch.count_nonzero(fusion.w).item(), 0)
        y = fusion(e, u)
        torch.testing.assert_close(y, e + u, atol=0, rtol=0)
        de, du, dw = torch.autograd.grad(y, (e, u, fusion.w), torch.ones_like(y))
        torch.testing.assert_close(de, torch.ones_like(e), atol=0, rtol=0)
        torch.testing.assert_close(du, torch.ones_like(u), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(dw).all())
        self.assertGreater(dw.abs().max().item(), 0)

    def test_fusion_raw_values_slot_alignment_and_oracle(self):
        for dtype in (torch.float32, torch.float64):
            f = PairwiseAttnResFusion(8).to(dtype)
            with torch.no_grad():
                f.w.normal_(0, .3)
            e = torch.randn(2, 7, 8, dtype=dtype, requires_grad=True) * 3
            u = torch.randn(2, 7, 8, dtype=dtype, requires_grad=True) * .2
            self.parity(f(e, u), oracle.fusion(e, u, f.w), (e, u, f.w), double=dtype == torch.float64)
            perm = torch.arange(6, -1, -1)
            torch.testing.assert_close(f(e[:, perm], u[:, perm]), f(e, u)[:, perm])
            self.assertFalse(torch.allclose(f(e, u[:, perm]), f(e, u)))
            torch.testing.assert_close(f(e, u), torch.cat([f(e[i:i+1], u[i:i+1]) for i in range(2)]))

    def test_down_permutations_and_batch_isolation(self):
        down = LearnedQueryDown(8, 2, 5).double()
        x = torch.randn(2, 7, 8, dtype=torch.float64)
        mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)
        y, a = down(x, valid_mask=mask, return_aux=True, coverage=MSARTrainingConfig())
        perm = torch.randperm(7)
        yp, ap = down(x[:, perm], valid_mask=mask[:, perm], return_aux=True, coverage=MSARTrainingConfig())
        torch.testing.assert_close(y, yp)
        torch.testing.assert_close(a[:, :, :, perm], ap)
        singles = [down(x[i:i+1], valid_mask=mask[i:i+1]) for i in range(2)]
        torch.testing.assert_close(y, torch.cat(singles))

    def test_coverage_hand_values_and_layer_reduction(self):
        loss = CoverageFloorLoss()
        a = torch.tensor([1., 0., 0., 0.]).reshape(1, 1, 1, 4).requires_grad_()
        expected = 3 * (.2 / 4) ** 2 / (.25 + 1e-6)
        torch.testing.assert_close(loss(a), torch.tensor(expected))
        self.assertGreater(loss(a).item(), 0)
        uniform = torch.full((2, 3, 5, 7), 1 / 7)
        self.assertEqual(loss(uniform).item(), 0)
        # Mean over layers AFTER each layer's source-sum and batch-mean.
        torch.testing.assert_close(loss.mean_layers([a, uniform]), loss(a) / 2)
        for cfg in (MSARTrainingConfig(coverage_mode='off'), MSARTrainingConfig(coverage_weight=0)):
            disabled = CoverageFloorLoss(cfg)
            self.assertEqual(disabled(None).item(), 0)
            self.assertFalse(disabled(a).requires_grad)
        self.assertEqual(CoverageFloorLoss(MSARTrainingConfig(coverage_kappa=0))(a).item(), 0)
        self.assertEqual(list(loss.parameters()), [])

    def test_coverage_reference_gradient_mask_and_measure(self):
        for batch in (1, 2):
            logits = (torch.randn(batch, 2, 3, 7) * 3).requires_grad_()
            mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1]] * batch, dtype=torch.bool)
            if batch == 2:
                mask[1, 1] = False  # mu normalizes independently for unequal valid counts.
            a = logits.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
            for measure in (None, torch.rand(batch, 7) + .1):
                loss = CoverageFloorLoss(MSARTrainingConfig(coverage_kappa=.8))
                actual = loss(a, valid_mask=mask, source_measure=measure)
                reference = oracle.coverage(a, .8, valid_mask=mask, source_measure=measure)
                self.parity(actual, reference, (logits,))
                # Independent double oracle also agrees within FP32 loss tolerance.
                double_ref = oracle.coverage(a.double(), .8, valid_mask=mask,
                    source_measure=None if measure is None else measure.double())
                torch.testing.assert_close(actual.double(), double_ref, atol=2e-7, rtol=2e-5)
            self.assertEqual(torch.autograd.grad(actual, logits)[0][~mask[:, None, None].expand_as(logits)].abs().max().item(), 0)

    def test_coverage_masks_ignore_padding_and_measure_scale(self):
        a = torch.tensor([.9, .05, .05, float('nan')]).reshape(1, 1, 1, 4).requires_grad_()
        mask = torch.tensor([[True, True, True, False]])
        measure = torch.tensor([[1e38, 2e38, 3e38, float('nan')]])
        loss = CoverageFloorLoss(MSARTrainingConfig(coverage_kappa=.8))
        actual = loss(a, valid_mask=mask, source_measure=measure)
        cropped = a[..., :3]
        expected = oracle.coverage(cropped, .8, source_measure=torch.tensor([[1., 2., 3.]]))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertEqual(a.grad[..., -1].item(), 0)

    def test_single_sample_and_joint_batch_agree_for_latent_and_fusion(self):
        # Each primitive is exercised with both B1 and B2, without reshaping
        # points across samples or treating a PyG graph batch as one field.
        x, context = torch.randn(2, 7, 8), torch.randn(2, 3, 8)
        block, up, fuse = LatentFFNSAFFNBlock(8, 2), QueryAlignedUpCross(8, 2), PairwiseAttnResFusion(8)
        for fn in (lambda a, b: block(a), lambda a, b: up(a, b), lambda a, b: fuse(a, a * .3 + 1)):
            torch.testing.assert_close(fn(x, context),
                torch.cat([fn(x[i:i+1], context[i:i+1]) for i in range(2)]))

    def test_isolated_coverage_gradient_to_learned_q_and_key_projection(self):
        down = LearnedQueryDown(8, 1, 2)
        with torch.no_grad():
            down.to_k.weight.copy_(torch.eye(8))
            down.latent_queries.uniform_(.7, 1.3)
        x = -torch.ones(2, 7, 8) + .2 * torch.randn(2, 7, 8)
        x[:, 0] = -x[:, 0]
        x.requires_grad_()
        _, a = down(x, return_aux=True, coverage=MSARTrainingConfig())
        loss = CoverageFloorLoss()(a)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for name in ('latent_queries', 'to_k.weight', 'q_norm.weight', 'k_norm.weight'):
            g = state(down)[name].grad
            self.assertIsNotNone(g, name)
            self.assertTrue(torch.isfinite(g).all(), name)
            self.assertGreater(g.abs().max().item(), 0, name)
        self.assertIsNone(down.to_v.weight.grad)
        self.assertIsNone(down.to_out.weight.grad)
        self.assertGreater(x.grad.abs().max().item(), 0)

    def test_float64_independent_oracle_gradcheck(self):
        down = LearnedQueryDown(4, 2, 3).double()
        x = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
        names, tensors = zip(*state(down).items())
        def formula(x, *values):
            return oracle.down(x, dict(zip(names, values)), 2)[0]
        self.assertTrue(torch.autograd.gradcheck(formula, (x, *tensors), fast_mode=True))
        e = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        u = torch.randn_like(e, requires_grad=True)
        w = torch.randn(4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(oracle.fusion, (e, u, w), fast_mode=True))
        # Safely away from ReLU floor boundary; double reference never casts float.
        a = torch.tensor([.97, .01, .01, .01], dtype=torch.float64).reshape(1, 1, 1, 4).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(oracle.coverage, (a,), fast_mode=True))

    def test_parameters_initialization_independence_and_checkpoint(self):
        factories = [lambda: LearnedQueryDown(8, 2, 5), lambda: LatentFFNSAFFNBlock(8, 2),
                     lambda: QueryAlignedUpCross(8, 2), lambda: PairwiseAttnResFusion(8)]
        all_parameters = []
        for make in factories:
            first, second = make(), make()
            all_parameters.extend(list(first.parameters()) + list(second.parameters()))
            self.assertFalse(any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in first.modules()))
            for name, p in first.named_parameters():
                if name.endswith('bias'):
                    self.assertEqual(torch.count_nonzero(p).item(), 0)
            buffer = io.BytesIO(); torch.save(first.state_dict(), buffer); buffer.seek(0)
            second.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            for k, v in first.state_dict().items():
                torch.testing.assert_close(v, second.state_dict()[k], atol=0, rtol=0)
        self.assertEqual(len({p.untyped_storage().data_ptr() for p in all_parameters}), len(all_parameters))
        for m in (3, 13):
            p = LearnedQueryDown(8, 2, m).latent_queries.detach()
            gram = p @ p.T if m < 8 else p.T @ p
            torch.testing.assert_close(gram, torch.eye(min(m, 8)), atol=6e-7, rtol=1e-6)
        up = QueryAlignedUpCross(8, 2)
        self.assertTrue(all(getattr(up, name).bias is None for name in ('to_q', 'to_k', 'to_v')))
        self.assertIsNotNone(up.to_out.bias)

    def test_inputs_not_mutated_and_synthetic_optimizer_step(self):
        down, latent = LearnedQueryDown(8, 2, 5), LatentFFNSAFFNBlock(8, 2)
        up, fusion = QueryAlignedUpCross(8, 2), PairwiseAttnResFusion(8)
        modules = (down, latent, up, fusion)
        optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=1e-3)
        x = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor([[1] * 6 + [0], [1] * 7], dtype=torch.bool)
        saved = x.detach().clone(); mask_before = mask.clone()
        s, a = down(x, valid_mask=mask, return_aux=True, coverage=MSARTrainingConfig())
        a_before, s_before = a.detach().clone(), s.detach().clone()
        t = latent(s)
        y = fusion(s, up(s, t))
        # Explicitly synthetic target; not an original task loss or a full core.
        objective = (y - .3).square().mean() + .01 * CoverageFloorLoss()(a, valid_mask=mask)
        objective.backward(); optimizer.step()
        for module in modules:
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.isfinite(parameter).all())
        for value, before in ((x, saved), (a, a_before), (s, s_before), (mask, mask_before)):
            torch.testing.assert_close(value, before, atol=0, rtol=0)

    def test_diagnostics_no_grad_and_not_automatic(self):
        same = torch.full((2, 2, 4, 4), .25, requires_grad=True)
        diag = coverage_diagnostics(same)
        torch.testing.assert_close(diag['coverage_ratio'], torch.ones(2))
        torch.testing.assert_close(diag['diversity_ratio'], torch.full((2,), .25), atol=2e-6, rtol=2e-5)
        identity = torch.eye(4)[None, None]
        d = coverage_diagnostics(identity)
        torch.testing.assert_close(d['diversity_ratio'], torch.ones(1))
        self.assertFalse(any(x.requires_grad for x in diag.values()))
        with patch('cdlno.msar_lno.modules.coverage_diagnostics', side_effect=AssertionError('automatic diagnostics')):
            LearnedQueryDown(8, 2, 5)(torch.randn(1, 7, 8))

    def test_invalid_shapes_masks_and_configs(self):
        for values in ((0, 2, 3), (8, 0, 3), (8, 3, 5), (8, 2, 0), (True, 1, 2), (8, 2., 3)):
            with self.assertRaises(ValueError):
                LearnedQueryDown(*values)
        down = LearnedQueryDown(8, 2, 5)
        x = torch.randn(2, 7, 8)
        for invalid in (torch.ones(2, 7), torch.ones(1, 7, dtype=torch.bool), torch.zeros(2, 7, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                down(x, valid_mask=invalid)
        for bad in (None, torch.ones(2, 7, 8, dtype=torch.int64), torch.randn(2, 0, 8), torch.randn(2, 7, 9)):
            with self.assertRaises(ValueError):
                down(bad)
        up = QueryAlignedUpCross(8, 2)
        with self.assertRaisesRegex(ValueError, 'batch'):
            up(x, x[:1])
        with self.assertRaises(ValueError):
            up(x, x.double())
        with self.assertRaises(ValueError):
            PairwiseAttnResFusion(8)(x, x[:, :5])
        loss = CoverageFloorLoss()
        a = torch.full((2, 2, 3, 7), 1 / 7)
        for measure in (torch.zeros(2, 7), -torch.ones(2, 7), torch.full((2, 7), torch.nan), torch.ones(2, 6)):
            with self.assertRaises(ValueError):
                loss(a, source_measure=measure)
        with self.assertRaises(ValueError):
            loss.mean_layers([])
        with self.assertRaises(ValueError):
            loss.mean_layers([a], valid_masks=[None, None])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no AMP acceptance')
    def test_cuda_fp32_and_amp_primitives_finite_and_critical_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), sdpa_kernel(SDPBackend.MATH):
                down = LearnedQueryDown(8, 2, 5).cuda()
                latent = LatentFFNSAFFNBlock(8, 2).cuda()
                up, fusion = QueryAlignedUpCross(8, 2).cuda(), PairwiseAttnResFusion(8).cuda()
                x = torch.randn(2, 7, 8, device='cuda', requires_grad=True)
                observed = []
                original_relu = F.relu
                def checked_relu(tensor, *args, **kwargs):
                    observed.append((tensor.dtype, torch.is_autocast_enabled('cuda')))
                    return original_relu(tensor, *args, **kwargs)
                with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                    y, a = down(x, return_aux=True, coverage=MSARTrainingConfig())
                    z = fusion(y, up(y, latent(y)))
                    with patch('cdlno.msar_lno.modules.F.relu', side_effect=checked_relu):
                        auxiliary = CoverageFloorLoss()(a)
                    objective = z.float().square().mean() + auxiliary
                self.assertEqual(a.dtype, torch.float32)
                self.assertEqual(auxiliary.dtype, torch.float32)
                self.assertEqual(observed, [(torch.float32, False)])
                self.assertEqual(z.dtype, dtype)
                objective.backward()
                for module in (down, latent, up, fusion):
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
                self.assertTrue(torch.isfinite(x.grad).all())
                print(f'M2 CUDA {dtype}: finite forward/backward, FP32 A/coverage, autocast disabled for loss')


if __name__ == '__main__':
    unittest.main()
