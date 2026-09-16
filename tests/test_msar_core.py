"""Four-scale M3 validation, without task entry imports or real dataset access."""
from collections import Counter
from dataclasses import replace
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno.modules import PlainFFN, _SelfAttention
from cdlno.msar_lno.config import MSARArchitectureConfig, MSARTrainingConfig
from cdlno.msar_lno.profiles import resolve_profile
from cdlno.msar_lno.core import MSARLNO, MSARAuxOutput
from cdlno.msar_lno.modules import (
    LearnedQueryDown, LatentFFNSAFFNBlock, QueryAlignedUpCross,
    PairwiseAttnResFusion, CoverageFloorLoss, coverage_diagnostics,
)
from cdlno.msar_lno.diagnostics import observe_down, observe_fusion
import msar_core_reference as oracle
import msar_reference as primitives

ROOT = Path(__file__).resolve().parents[1]


def small():
    # Only test dimensions shrink. Confirmed depths/topology stay [3,1,1,1].
    return MSARArchitectureConfig(d=8, num_latents=(7, 5, 3, 2), heads=(2, 2, 4, 4))


def setUpModule():
    torch.set_num_threads(1)


class MSARCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9163)

    def test_module_counts_ownership_and_initialization(self):
        model = MSARLNO(small(), output_dim=4)
        counts = Counter(type(m).__name__ for m in model.modules())
        for cls, expected in ((LearnedQueryDown, 4), (QueryAlignedUpCross, 4),
                              (LatentFFNSAFFNBlock, 12), (PairwiseAttnResFusion, 3),
                              (_SelfAttention, 12), (PlainFFN, 24)):
            self.assertEqual(counts[cls.__name__], expected)
        self.assertEqual([len(s) for s in model.encoders], [3, 1, 1, 1])
        self.assertEqual([len(s) for s in model.decoders], [3, 1, 1, 1])
        self.assertEqual([m.heads for m in model.ups], [2, 2, 4])
        self.assertEqual(model.final_up.heads, 2)
        self.assertFalse(any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules()))
        forbidden = ('history', 'reader', 'writer', 'bridge', 'cdpa', 'persistent')
        self.assertFalse(any(word in k.lower() for k in model.state_dict() for word in forbidden))
        named = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(named), len({id(p) for _, p in named}))
        self.assertEqual(len(named), len({p.untyped_storage().data_ptr() for _, p in named}))
        self.assertEqual(sum(p.numel() for m in model.fusions for p in m.parameters()), 3 * small().d)
        for fusion in model.fusions:
            self.assertEqual(list(dict(fusion.named_parameters())), ['w'])
            self.assertEqual(fusion.w.count_nonzero().item(), 0)
        for down in model.downs:
            p = down.latent_queries.detach()
            torch.testing.assert_close(p @ p.T, torch.eye(p.shape[0]), atol=6e-7, rtol=1e-6)

    def test_execution_order_trace_and_no_point_sa(self):
        model = MSARLNO(small(), output_dim=2)
        trace, saved, handles = [], {}, []
        def record(name):
            def hook(module, inputs, output):
                trace.append(name)
                saved[name] = (inputs, output)
            return hook
        for group in ('downs', 'encoders', 'decoders', 'ups', 'fusions'):
            for i, module in enumerate(getattr(model, group)):
                handles.append(module.register_forward_hook(record(f'{group}.{i}')))
        for name in ('final_up', 'output_norm', 'output'):
            handles.append(getattr(model, name).register_forward_hook(record(name)))
        sa_sizes = []
        for module in model.modules():
            if isinstance(module, _SelfAttention):
                handles.append(module.register_forward_pre_hook(lambda m, a: sa_sizes.append(a[0].shape[1])))
        x = torch.randn(2, 35, 8)
        y = model(x)
        for h in handles:
            h.remove()
        self.assertEqual(trace, [f'{g}.{i}' for i in range(4) for g in ('downs', 'encoders')] +
                         ['decoders.3'] + [f'{g}.{i}' for i in (2, 1, 0) for g in ('ups', 'fusions', 'decoders')] +
                         ['final_up', 'output_norm', 'output'])
        self.assertEqual(Counter(sa_sizes), Counter({7: 6, 5: 2, 3: 2, 2: 2}))
        self.assertIs(saved['decoders.3'][0][0], saved['encoders.3'][1])
        for i in (2, 1, 0):
            self.assertIs(saved[f'ups.{i}'][0][0], saved[f'encoders.{i}'][1])
            self.assertIs(saved[f'ups.{i}'][0][1], saved[f'decoders.{i+1}'][1])
            self.assertIs(saved[f'decoders.{i}'][0][0], saved[f'fusions.{i}'][1])
            torch.testing.assert_close(saved[f'fusions.{i}'][1],
                                       saved[f'encoders.{i}'][1] + saved[f'ups.{i}'][1], atol=0, rtol=0)
        self.assertIs(saved['final_up'][0][0], x)
        self.assertEqual(y.shape, (2, 35, 2))

    def test_full_topology_independent_forward_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            model = MSARLNO(small(), output_dim=3).to(dtype)
            x = torch.randn(2, 11, 8, dtype=dtype, requires_grad=True)
            mask = torch.ones(2, 11, dtype=torch.bool); mask[0, -2:] = False; mask[1, -1] = False
            cfg = MSARTrainingConfig(coverage_kappa=.9)
            actual = model(x, return_aux=True, training_config=cfg, valid_mask=mask)
            expected, raw = oracle.core(x, dict(model.named_parameters()), small().heads,
                small().encoder_depths, small().decoder_depths, valid_mask=mask, kappa=.9)
            atol, rtol = (3e-10, 3e-8) if dtype == torch.float64 else (5e-6, 2e-4)
            torch.testing.assert_close(actual.prediction, expected, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual.coverage_per_level.double(), raw.double(), atol=2e-7, rtol=2e-4)
            probe = torch.randn_like(expected)
            leaves = (x, *model.parameters())
            ga = torch.autograd.grad(actual.prediction, leaves, probe, retain_graph=True)
            gr = torch.autograd.grad(expected, leaves, probe)
            for a, r in zip(ga, gr):
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a, r, atol=atol * 20, rtol=rtol * 10)

    def test_off_floor_and_zero_weight_prediction_and_gradient_parity(self):
        model = MSARLNO(small()).double()
        x = torch.randn(2, 9, 8, dtype=torch.float64, requires_grad=True)
        floor = model(x, return_aux=True)
        for cfg in (MSARTrainingConfig(coverage_mode='off'), MSARTrainingConfig(coverage_weight=0)):
            with patch.object(CoverageFloorLoss, 'forward', side_effect=AssertionError('off loss called')):
                outputs = []
                handles = [d.register_forward_hook(lambda m, a, out: outputs.append(out)) for d in model.downs]
                off = model(x, return_aux=True, training_config=cfg)
                for h in handles:
                    h.remove()
            self.assertTrue(all(isinstance(t, torch.Tensor) for t in outputs))
            self.assertFalse(off.coverage_active)
            self.assertFalse(off.coverage_mean.requires_grad)
            self.assertEqual(off.coverage_per_level.tolist(), [0] * 4)
            torch.testing.assert_close(off.prediction, floor.prediction, atol=3e-10, rtol=3e-8)
            go = torch.autograd.grad(off.prediction.sum(), (x, *model.parameters()), retain_graph=True)
            gf = torch.autograd.grad(floor.prediction.sum(), (x, *model.parameters()), retain_graph=True)
            for a, b in zip(go, gf):
                torch.testing.assert_close(a, b, atol=6e-9, rtol=3e-7)
        self.assertEqual(model.training_config, MSARTrainingConfig())

    def test_default_eval_aux_and_diagnostics_gate(self):
        model = MSARLNO(small(), training_config=MSARTrainingConfig(diagnostics=True))
        x = torch.randn(1, 11, 8)
        for training in (True, False):
            model.train(training)
            with (patch('cdlno.msar_lno.core.observe_down', side_effect=AssertionError('default diagnostic')),
                  patch('cdlno.msar_lno.core.observe_fusion', side_effect=AssertionError('default diagnostic')),
                  patch.object(CoverageFloorLoss, 'forward', side_effect=AssertionError('default coverage'))):
                self.assertIsInstance(model(x), torch.Tensor)
        model.eval()
        aux = model(x, return_aux=True)
        self.assertIsInstance(aux, MSARAuxOutput)
        self.assertFalse(aux.coverage_active)
        self.assertEqual(aux.coverage_mean.item(), 0)
        self.assertIsNotNone(aux.diagnostics)

    def test_diagnostics_no_grad_reuse_and_prediction_rng_invariance(self):
        model = MSARLNO(small())
        x = torch.randn(2, 9, 8, requires_grad=True)
        for training, coverage in ((True, 'floor'), (True, 'off'), (False, 'floor')):
            model.train(training)
            cfg = MSARTrainingConfig(coverage_mode=coverage)
            before_rng = torch.get_rng_state().clone()
            plain = model(x, return_aux=True, training_config=cfg)
            captured = []
            def down_observer(module, source, **kw):
                captured.append(kw['attention'] is not None)
                return observe_down(module, source, **kw)
            with patch('cdlno.msar_lno.core.observe_down', side_effect=down_observer):
                diag = model(x, return_aux=True, training_config=replace(cfg, diagnostics=True))
            self.assertEqual(captured, [training and coverage == 'floor'] * 4)
            torch.testing.assert_close(plain.prediction, diag.prediction, atol=0, rtol=0)
            torch.testing.assert_close(before_rng, torch.get_rng_state(), atol=0, rtol=0)
            self.assertEqual(len(diag.diagnostics['encoder']), 4)
            self.assertEqual(len(diag.diagnostics['fusion']), 3)
            for stats in diag.diagnostics['encoder']:
                for value in stats.values():
                    self.assertEqual(value.shape, (2,))
                    self.assertFalse(value.requires_grad)
                    self.assertTrue(torch.isfinite(value).all())
            for stats in diag.diagnostics['fusion']:
                torch.testing.assert_close(stats['alpha_mean'], torch.full((2, 2), .5))
                torch.testing.assert_close(stats['entropy_mean'], torch.full((2,), math.log(2)))
                self.assertFalse(stats['alpha_mean'].requires_grad)
                self.assertFalse(stats['entropy_mean'].requires_grad)

    def test_diagnostics_recomputation_and_nonzero_fusion_reference(self):
        down = LearnedQueryDown(8, 2, 5)
        x = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
        _, a = down(x, valid_mask=mask, return_aux=True, coverage=MSARTrainingConfig())
        reuse = observe_down(down, x, valid_mask=mask, attention=a)
        recompute = observe_down(down, x, valid_mask=mask)
        reference = coverage_diagnostics(a, valid_mask=mask)
        for key in reference:
            torch.testing.assert_close(reuse[key], reference[key], atol=0, rtol=0)
            torch.testing.assert_close(recompute[key], reference[key], atol=0, rtol=0)
            self.assertFalse(recompute[key].requires_grad)
        fusion = PairwiseAttnResFusion(8)
        with torch.no_grad():
            fusion.w.normal_(0, .3)
        e, u = torch.randn(2, 5, 8), torch.randn(2, 5, 8) * 3
        stats = observe_fusion(fusion, e, u)
        p = torch.sigmoid(((primitives.rms(e) - primitives.rms(u)) * fusion.w).sum(-1))
        expected = torch.stack((p.mean(1), (1 - p).mean(1)), -1)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean(1)
        torch.testing.assert_close(stats['alpha_mean'], expected)
        torch.testing.assert_close(stats['entropy_mean'], entropy)

    def test_normal_paths_have_no_cpu_statistics(self):
        model = MSARLNO(small())
        x = torch.randn(1, 11, 8)
        with (patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU synchronization')),
              patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar synchronization')),
              patch('cdlno.msar_lno.core.observe_down', side_effect=AssertionError('unrequested diagnostics')),
              patch('cdlno.msar_lno.core.observe_fusion', side_effect=AssertionError('unrequested diagnostics'))):
            model(x)
            model(x, return_aux=True)

    def test_all_stages_gradients_and_synthetic_optimizer_step(self):
        model = MSARLNO(small(), output_dim=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x = torch.randn(2, 35, 8, requires_grad=True)
        outputs = model(x, return_aux=True, training_config=MSARTrainingConfig(coverage_kappa=.9))
        # Explicit synthetic MSE, not any original task loss/optimizer protocol.
        loss = (outputs.prediction - torch.randn_like(outputs.prediction)).square().mean() + .01 * outputs.coverage_mean
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for group in (model.downs, model.encoders, model.decoders, model.ups, model.fusions):
            for module in group:
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_coverage_only_encoder_down_dependencies_and_raw_mean(self):
        model = MSARLNO(small())
        x = torch.randn(2, 11, 8, requires_grad=True)
        out = model(x, return_aux=True, training_config=MSARTrainingConfig(coverage_kappa=1))
        torch.testing.assert_close(out.coverage_mean, out.coverage_per_level.mean(), atol=0, rtol=0)
        out.coverage_mean.backward()
        for down in model.downs:
            self.assertIsNotNone(down.latent_queries.grad)
            self.assertIsNotNone(down.to_k.weight.grad)
            self.assertTrue(torch.isfinite(down.latent_queries.grad).all())
        for group in (model.decoders, model.ups, model.fusions, [model.final_up, model.output]):
            for module in group:
                self.assertTrue(all(p.grad is None for p in module.parameters()))
        # The last encoder stack follows Down4's attention, so is also outside coverage.
        self.assertTrue(all(p.grad is None for p in model.encoders[3].parameters()))

    def test_variable_n_batch_permutation_mask_and_no_cross_call_state(self):
        model = MSARLNO(small()).eval()
        a = torch.randn(2, 35, 8)
        original = a.clone()
        with torch.no_grad():
            ya = model(a)
            self.assertEqual(model(torch.randn(1, 3, 8)).shape, (1, 3, 1))
            torch.testing.assert_close(model(a), ya, atol=0, rtol=0)
            torch.testing.assert_close(model(a), torch.cat([model(a[i:i+1]) for i in range(2)]), atol=2e-6, rtol=2e-4)
            p = torch.randperm(35)
            torch.testing.assert_close(model(a[:, p]), ya[:, p], atol=2e-6, rtol=2e-4)
            mask = torch.ones(2, 35, dtype=torch.bool); mask[:, -3:] = False
            padded = a.clone(); padded[:, -3:] = 1000
            torch.testing.assert_close(model(a, valid_mask=mask)[:, :-3],
                                       model(padded, valid_mask=mask)[:, :-3], atol=0, rtol=0)
        torch.testing.assert_close(a, original, atol=0, rtol=0)
        for module in model.modules():
            self.assertFalse(any(isinstance(v, torch.Tensor) for v in module.__dict__.values()))
        self.assertEqual(list(model.buffers()), [])

    def test_final_branch_has_no_e0_skip(self):
        model = MSARLNO(small(), output_dim=3)
        with torch.no_grad():
            model.final_up.to_out.weight.zero_(); model.final_up.to_out.bias.zero_()
            model.output_norm.bias.fill_(.3)
            model.output.bias.fill_(.2)
        x = torch.randn(2, 11, 8, requires_grad=True)
        expected = model.output(model.output_norm(torch.zeros_like(x)))
        torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
        self.assertEqual(torch.autograd.grad(model(x).sum(), x)[0].count_nonzero().item(), 0)

    def test_strict_state_roundtrip_and_trusted_object_fresh_cwds(self):
        model = MSARLNO(small(), output_dim=2).eval()
        x = torch.randn(1, 9, 8)
        with tempfile.TemporaryDirectory() as folder, torch.no_grad():
            path = Path(folder)
            torch.save(model.state_dict(), path / 'weights.pt')
            clone = MSARLNO(small(), output_dim=2).eval()
            clone.load_state_dict(torch.load(path / 'weights.pt', weights_only=True), strict=True)
            torch.testing.assert_close(clone(x), model(x), atol=0, rtol=0)
            broken = model.state_dict(); broken.pop('downs.0.latent_queries')
            with self.assertRaises(RuntimeError):
                clone.load_state_dict(broken, strict=True)
            with self.assertRaises(RuntimeError):
                MSARLNO(small(), output_dim=1).load_state_dict(model.state_dict(), strict=True)
            torch.save(model, path / 'whole.pt')
            torch.save([model], path / 'list.pt')
            torch.save(dict(x=x, expected=model(x)), path / 'io.pt')
            script = """import torch,sys
from pathlib import Path
torch.set_num_threads(1)
p=Path(sys.argv[1]); data=torch.load(p/'io.pt',weights_only=True)
for name in ('whole.pt','list.pt'):
    obj=torch.load(p/name,weights_only=False) # Only this test's trusted local captures.
    model=obj[0] if isinstance(obj,list) else obj
    assert type(model).__module__=='cdlno.msar_lno.core'
    with torch.no_grad():torch.testing.assert_close(model(data['x']),data['expected'],atol=0,rtol=0)
"""
            env = os.environ | {'PYTHONPATH': str(ROOT)}
            for directory in ('PDE-Solving-StandardBenchmark', 'Car-Design-ShapeNetCar', 'Airfoil-Design-AirfRANS'):
                result = subprocess.run([sys.executable, '-B', '-c', script, str(path)], cwd=ROOT/directory,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_light_full_shapes_and_full_small_n_strict_roundtrip(self):
        for profile in ('light', 'full'):
            config = resolve_profile(profile)
            model = MSARLNO(config, output_dim=4).eval()
            self.assertEqual([tuple(d.latent_queries.shape) for d in model.downs],
                             [(m, config.d) for m in config.num_latents])
            self.assertEqual([d.heads for d in model.downs], [4, 4, 8, 8])
            self.assertEqual(sum(len(s) for s in model.encoders), 6)
            self.assertEqual(sum(len(s) for s in model.decoders), 6)
            # Full M1=1024 remains 1024, despite only 7 input points.
            x = torch.randn(1, 7, config.d)
            with torch.no_grad():
                y = model(x)
            self.assertEqual(y.shape, (1, 7, 4))
            self.assertTrue(torch.isfinite(y).all())
            if profile == 'full':
                buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
                clone = MSARLNO(config, output_dim=4).eval()
                clone.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(clone(x), y, atol=0, rtol=0)

    def test_invalid_contracts(self):
        for config in ({}, 'light', MSARTrainingConfig()):
            with self.assertRaises(ValueError):
                MSARLNO(config)
        for dim in (0, True, 1.5):
            with self.assertRaises(ValueError):
                MSARLNO(small(), output_dim=dim)
        model = MSARLNO(small())
        for x in (torch.randn(1, 7, 7), torch.empty(1, 0, 8), torch.ones(1, 7, 8, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                model(x)
        x = torch.randn(1, 7, 8)
        with self.assertRaises(ValueError):
            model(x, valid_mask=torch.zeros(1, 7, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(x, return_aux=1)
        with self.assertRaises(ValueError):
            model(x, training_config={})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_core_aux_and_off_prediction(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), sdpa_kernel(SDPBackend.MATH):
                model = MSARLNO(small(), output_dim=4).cuda()
                x = torch.randn(2, 11, 8, device='cuda', requires_grad=True)
                with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                    aux = model(x, return_aux=True)
                    off = model(x, training_config=MSARTrainingConfig(coverage_mode='off'))
                    objective = aux.prediction.float().square().mean() + .01 * aux.coverage_mean
                self.assertEqual(aux.coverage_mean.dtype, torch.float32)
                self.assertEqual(aux.prediction.dtype, dtype)
                atol, rtol = (3e-6, 2e-4) if dtype == torch.float32 else (3e-3, .03)
                torch.testing.assert_close(aux.prediction, off, atol=atol, rtol=rtol)
                objective.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
