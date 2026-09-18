"""L3 complete-model initialization, fixed-source parity and independent oracle."""
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from linearno.attention_support import errors, ROOT, MANIFEST
from linearno.standard_support import (official_class, local_class, release_configs,
    official_kwargs, local_kwargs, small_config, inputs, VARIANTS)
from linearno.standard_reference import reference


class StandardModelParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.backend = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                       torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        cls.rows = []

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = cls.backend
        if path := os.environ.get('LINEARNO_L3_PARITY_REPORT'):
            Path(path).write_text(json.dumps(dict(torch=str(torch.__version__),
                cuda_build=torch.version.cuda, source_commit=MANIFEST['commit'],
                tf32=False, compile=False, scope='synthetic complete Standard model', rows=cls.rows), indent=2)+'\n')

    def checked(self, a, b, name, metrics, atol=1e-6, rtol=1e-5):
        self.assertEqual(a.shape, b.shape, name)
        self.assertTrue(torch.isfinite(a).all(), name)
        self.assertTrue(torch.isfinite(b).all(), name)
        metrics[name] = errors(a, b)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)

    def test_six_release_configurations_and_parameter_counts(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/official_standard_models.json').read_text())
        for task, cfg in release_configs().items():
            with self.subTest(task=task):
                torch.manual_seed(fixture['seed']); local = local_class()(**local_kwargs(cfg))
                rng = torch.get_rng_state()
                expected = fixture['tasks'][task]
                self.assertEqual(sum(p.numel() for p in local.parameters()), expected['parameters'])
                self.assertEqual({k: list(v.shape) for k, v in local.state_dict().items()}, expected['keys'])
                state_hash = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in local.state_dict().values())).hexdigest()
                self.assertEqual(state_hash, expected['init_hash'])
                self.assertEqual(hashlib.sha256(rng.numpy().tobytes()).hexdigest(), expected['rng_hash'])
                # Independent closed-form accounting includes all norms and final head.
                d, h, m, r = cfg['n_hidden'], cfg['n_head'], cfg['key_ratio'], cfg['mlp_ratio']
                conv = 'conv' in cfg['args']['model']; temp = 'temp' in cfg['args']['model'] and cfg['args']['model'] != 'no_temp'
                lift = cfg['fun_dim'] + (cfg['ref']**2 if cfg['unified_pos'] else cfg['space_dim'])
                pre = lift*2*d + 2*d + 2*d*d + d
                block = (9 if conv else 1)*d*d + d + 2*(d//h)*m + (d//h)**2
                block += (2 if conv else 1)*(d*d+d) + (2*h if temp else 0)
                block += 4*d + 2*r*d*d + r*d + d
                count = pre + cfg['n_layers']*block + 2*d + d*cfg['out_dim'] + cfg['out_dim'] + d
                if cfg['Time_Input']: count += 2*(d*d+d)
                self.assertEqual(count, expected['parameters'])
                row = dict(kind='published_config', task=task, count=count, init_hash=state_hash,
                           exact_key_shapes=True, closed_form_count=count, native_official='CPU')
                if cfg['unified_pos'] and not torch.cuda.is_available():
                    row['native_official'] = 'NOT RUN: requires CUDA; captured official fixture checked'
                else:
                    torch.manual_seed(fixture['seed']); official = official_class()(**official_kwargs(cfg))
                    torch.testing.assert_close(local.state_dict(), official.state_dict(), atol=0, rtol=0)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    if cfg['unified_pos']:
                        row['native_official'] = 'CUDA position / CPU parameters, unmodified constructor'
                        metrics = {}; self.checked(local.pos, official.pos.cpu(), 'position', metrics)
                        row['metrics'] = metrics
                self.rows.append(row)

    def test_default_allocations_single_apply_then_placeholder_RNG(self):
        Local, Official = local_class(), official_class()
        for variant in VARIANTS:
            for time in (False, True):
                with self.subTest(variant=variant, time=time):
                    cfg = small_config(variant, time=time)
                    records = []
                    def instrument(cls):
                        class Observed(cls):
                            def initialize_weights(self):
                                before = {k: v.clone() for k, v in self.state_dict().items()}
                                self_rng = torch.get_rng_state().clone()
                                assert 'placeholder' not in before
                                super().initialize_weights()
                                records.append((before, self_rng, {k: v.clone() for k, v in self.state_dict().items()},
                                                torch.get_rng_state().clone()))
                        return Observed
                    torch.manual_seed(314); a = instrument(Local)(**local_kwargs(cfg)); ra = torch.get_rng_state()
                    torch.manual_seed(314); b = instrument(Official)(**official_kwargs(cfg)); rb = torch.get_rng_state()
                    self.assertEqual(len(records), 2)
                    torch.testing.assert_close(records[0], records[1], atol=0, rtol=0)
                    torch.testing.assert_close(a.state_dict(), b.state_dict(), atol=0, rtol=0)
                    self.assertTrue(torch.equal(ra, rb))
                    # The release uses reciprocal multiplication, not division rounding.
                    torch.set_rng_state(records[0][3]); expected = (1/cfg['n_hidden'])*torch.rand(cfg['n_hidden'])
                    torch.testing.assert_close(a.placeholder, expected, atol=0, rtol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), ra))
                    visited = []; callback = Local._init_weights
                    def counted(module):
                        visited.append(id(module)); callback(module)
                    with patch.object(Local, '_init_weights', staticmethod(counted)):
                        c = Local(**local_kwargs(cfg))
                    self.assertEqual(len(visited), len(set(visited)))
                    self.assertEqual(set(visited), {id(m) for m in c.modules()})
                    for name, value in c.named_parameters():
                        if 'temperature' in name: self.assertTrue(torch.equal(value, torch.full_like(value, .5)))
                    self.rows.append(dict(kind='initialization', variant=variant, time=time,
                        allocated_and_initialized_state_equal=True, RNG_equal=True, placeholder_last=True,
                        once_per_module=len(visited)))

    def parity(self, cfg, B, device, dtype, use_time):
        torch.manual_seed(315)
        b = official_class()(**official_kwargs(cfg)).to(device=device, dtype=dtype)
        a = local_class()(**local_kwargs(cfg)).to(device=device, dtype=dtype)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        a.load_state_dict(b.state_dict(), strict=True); b.load_state_dict(a.state_dict(), strict=True)
        a.train(); b.train()
        xa = inputs(cfg, B=B, device=device, dtype=dtype, T=use_time)
        xb = tuple(v.detach().clone().requires_grad_() if v is not None else None for v in xa)
        original = tuple(v.detach().clone() if v is not None else None for v in xa)
        blocks_a, blocks_b = [], []
        hooks = [m.register_forward_hook(lambda m, i, o: blocks_a.append(o.detach().clone())) for m in a.blocks]
        hooks += [m.register_forward_hook(lambda m, i, o: blocks_b.append(o.detach().clone())) for m in b.blocks]
        cpu_rng = torch.get_rng_state(); gpu_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else []
        ya = a(*xa); after_cpu = torch.get_rng_state(); after_gpu = torch.cuda.get_rng_state_all() if gpu_rng else []
        torch.set_rng_state(cpu_rng)
        if gpu_rng: torch.cuda.set_rng_state_all(gpu_rng)
        yb = b(*xb)
        self.assertTrue(torch.equal(after_cpu, torch.get_rng_state()))
        for one, two in zip(after_gpu, torch.cuda.get_rng_state_all() if gpu_rng else []): self.assertTrue(torch.equal(one, two))
        for hook in hooks: hook.remove()
        atol, rtol = ((1e-12, 1e-10) if dtype == torch.float64 else (1e-6, 1e-5) if device == 'cpu' else (1e-5, 1e-4))
        metrics = {}
        for i, (one, two) in enumerate(zip(blocks_a, blocks_b)):
            self.checked(one, two, f'block/{i}', metrics, atol, rtol)
        self.assertEqual(len(blocks_a), cfg['n_layers'])
        self.checked(ya, yb, 'forward', metrics, atol, rtol)
        if cfg['unified_pos']: self.checked(a.pos, b.pos, 'position', metrics, atol, rtol)
        cotangent = torch.randn_like(ya)
        la = (ya*cotangent).mean() + ya.square().mean(); lb = (yb*cotangent).mean() + yb.square().mean()
        self.checked(la, lb, 'loss', metrics, atol, rtol)
        la.backward(); lb.backward()
        inactive = []
        for label, one, two, saved in zip(('x', 'fx', 'T'), xa, xb, original):
            if one is None: continue
            torch.testing.assert_close(one.detach(), saved, atol=0, rtol=0)
            if label == 'x' and cfg['unified_pos']:
                self.assertIsNone(one.grad); self.assertIsNone(two.grad); inactive.append('input/x replacement'); continue
            self.checked(one.grad, two.grad, 'input_grad/'+label, metrics, atol, rtol)
        for name, p in a.named_parameters():
            q = dict(b.named_parameters())[name]
            if (name == 'placeholder' and cfg['fun_dim']) or (name.startswith('time_fc') and not use_time):
                self.assertIsNone(p.grad); self.assertIsNone(q.grad); inactive.append(name); continue
            self.assertIsNotNone(p.grad, name); self.assertIsNotNone(q.grad, name)
            self.checked(p.grad, q.grad, 'grad/'+name, metrics, atol, rtol)
        oa = torch.optim.AdamW(a.parameters(), lr=.001); ob = torch.optim.AdamW(b.parameters(), lr=.001)
        oa.step(); ob.step()
        for name, v in a.state_dict().items(): self.checked(v, b.state_dict()[name], 'step/'+name, metrics, atol, rtol)
        stream = io.BytesIO(); torch.save(a.state_dict(), stream); stream.seek(0)
        loaded = local_class()(**local_kwargs(cfg)).to(device=device, dtype=dtype).eval()
        loaded.load_state_dict(torch.load(stream, map_location=device, weights_only=True), strict=True)
        a.eval()
        with torch.no_grad(): self.checked(loaded(*xa), a(*xa), 'checkpoint', metrics, atol, rtol)
        self.rows.append(dict(kind='official_parity', config=cfg, B=B, device=device, dtype=str(dtype),
            T_present=use_time, atol=atol, rtol=rtol, rng_equal=True, mapping='identity; strict in both directions',
            inactive=inactive, metrics=metrics))

    def test_cpu_official_all_variants_full_forward_gradients_step(self):
        for variant in VARIANTS:
            for B, fun_dim, time, dtype in ((1, 0, False, torch.float64), (2, 1, True, torch.float32)):
                with self.subTest(variant=variant, B=B, dtype=dtype):
                    self.parity(small_config(variant, fun_dim=fun_dim, dropout=.25 if B==2 else 0.), B, 'cpu', dtype, time)
        # Eight FULL blocks in representative structured/time and point/no-time cases.
        for variant in ('conv_temp', 'plain'):
            self.parity(small_config(variant, layers=8, fun_dim=0), 2, 'cpu', torch.float32, True)

    @unittest.skipUnless(torch.cuda.is_available(), 'native official unified Model requires CUDA')
    def test_native_cuda_official_unified_replacement_all_variants(self):
        for variant in VARIANTS:
            for B, fun_dim, time in ((1, 0, True), (2, 1, False)):
                with self.subTest(variant=variant, B=B):
                    self.parity(small_config(variant, fun_dim=fun_dim, unified=True, dropout=.25 if B==2 else 0.),
                                B, 'cuda', torch.float32, time)

    def test_independent_double_complete_equations_and_all_gradients(self):
        for variant in VARIANTS:
            for fun_dim in (0, 1):
                with self.subTest(variant=variant, fun_dim=fun_dim):
                    cfg = small_config(variant, fun_dim=fun_dim, time=False)
                    torch.manual_seed(316); model = local_class()(**local_kwargs(cfg)).double().eval()
                    x = inputs(cfg, dtype=torch.float64, T=False)
                    xr = tuple(v.detach().clone().requires_grad_() if v is not None else None for v in x)
                    state = {k: v.detach().clone().requires_grad_() for k, v in model.state_dict().items()}
                    a = model(*x); b, _ = reference(*xr, state, cfg)
                    metrics = {}; self.checked(a, b, 'forward', metrics, 1e-12, 1e-10)
                    upstream = torch.randn_like(a); (a*upstream).sum().backward(); (b*upstream).sum().backward()
                    for label, one, two in zip(('x', 'fx', 'T'), x, xr):
                        if one is not None: self.checked(one.grad, two.grad, 'input_grad/'+label, metrics, 1e-12, 1e-10)
                    for name, p in model.named_parameters():
                        if name == 'placeholder' and fun_dim:
                            self.assertIsNone(p.grad); self.assertIsNone(state[name].grad); continue
                        self.checked(p.grad, state[name].grad, 'grad/'+name, metrics, 1e-12, 1e-10)
                    self.rows.append(dict(kind='independent_oracle', variant=variant, fun_dim=fun_dim,
                        dtype='float64', atol=1e-12, rtol=1e-10, metrics=metrics))
