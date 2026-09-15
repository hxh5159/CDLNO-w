"""A1: front-only ablations and existing core; no task/experiment imports."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdlno import CDLNO, CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdlno.checkpoint import compare_architecture, save_sidecar, validate_sidecar, SidecarMismatch
from cdlno.config import HISTORY_RULE
from cdlno.modules import LRSAFrontBlock, ConvFFN
from test_modules import explicit_attention, rms, linear
from test_core import trace, explicit_core

MODES = ('full', 'no_sa', 'identity')
CDPA_MODES = ('off', 'entry', 'every_block')
REMOVED = {
    'full': (),
    'no_sa': ('latent_norm_sa', 'latent_sa'),
    'identity': ('latent_norm_1', 'latent_ffn_1', 'latent_norm_sa', 'latent_sa',
                 'latent_norm_2', 'latent_ffn_2'),
}


def config(**kw):
    return replace(CDLNOArchitectureConfig(L=8, F=2, M=4, d_model=8,
                                          num_heads=2, output_dim=3), **kw)


def block(mode='full', structured=False):
    return LRSAFrontBlock(8, 2, 4, structured=structured,
                          grid_shape=(5, 7) if structured else None,
                          front_latent_mode=mode, dropout=0.)


def copy_retained(full, ablated):
    """Test-only common-weight mapping; the destination is always loaded strict."""
    original = full.state_dict()
    retained = ablated.state_dict()
    assert set(retained) <= set(original)
    ablated.load_state_dict({key: original[key] for key in retained}, strict=True)


def reference_front(m, x, mode):
    """Explicit latent residual equations, independent of front/attention forward.

    Frozen point FFN implementation is reused; phase-2 independently tests it.
    All attention here uses QK/softmax/AV, never production SDPA.
    """
    hn = rms(x, m.point_norm.weight)
    s = explicit_attention(m.down, None, hn)
    def ffn(value, norm, module):
        return linear(F.gelu(linear(rms(value, norm.weight), module.fc1)), module.fc2)
    if mode == 'full':
        a = s + ffn(s, m.latent_norm_1, m.latent_ffn_1)
        an = rms(a, m.latent_norm_sa.weight)
        b = a + explicit_attention(m.latent_sa.attn, an, an)
        t = b + ffn(b, m.latent_norm_2, m.latent_ffn_2)
    elif mode == 'no_sa':
        a = s + ffn(s, m.latent_norm_1, m.latent_ffn_1)
        t = a + ffn(a, m.latent_norm_2, m.latent_ffn_2)
    else:
        t = s
    u = x + explicit_attention(m.up, hn, rms(t, m.up_latent_norm.weight))
    un = rms(u, m.point_ffn_norm.weight)
    update = m.point_ffn(un, m.grid_shape) if m.structured else m.point_ffn(un)
    return u + update, t


class FrontAblationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(1601)
        context = sdpa_kernel(SDPBackend.MATH)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def assertFiniteGradients(self, model, x):
        for name, value in [*model.named_parameters(), ('input', x)]:
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all().item(), name)
        # Some score-only parameters legitimately have zero first-step grad.

    def assertIndependent(self, model):
        params = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(params), len({id(p) for _, p in params}))
        self.assertEqual(len(params), len({p.untyped_storage().data_ptr() for _, p in params}))

    def test_implicit_explicit_full_same_construction_and_computation(self):
        self.assertEqual(config(), config(front_latent_mode='full'))
        for make in (lambda explicit: CDLNO(config(**explicit)),
                     lambda explicit: LRSAFrontBlock(8, 2, 4, **explicit)):
            torch.manual_seed(73)
            default = make({})
            torch.manual_seed(73)
            explicit = make(dict(front_latent_mode='full'))
            self.assertEqual([(n,type(m)) for n,m in default.named_modules()],
                             [(n,type(m)) for n,m in explicit.named_modules()])
            self.assertEqual(list(default.state_dict()), list(explicit.state_dict()))
            for key, value in default.state_dict().items():
                torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
            x = torch.randn(2, 11, 8)
            torch.testing.assert_close(default.eval()(x), explicit.eval()(x), atol=0, rtol=0)

    def test_configuration_validation_roundtrip_and_runtime_separation(self):
        for mode in MODES:
            cfg = config(front_latent_mode=mode)
            self.assertEqual(CDLNOArchitectureConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)
            self.assertEqual(pickle.loads(pickle.dumps(cfg)), cfg)
            self.assertEqual(cfg.P, 6)
            self.assertEqual(cfg.history_rule, HISTORY_RULE)
        self.assertEqual(compare_architecture(config(), config(front_latent_mode='no_sa')), ['front_latent_mode'])
        for bad in ('', 'FULL', 'sa', None, True, 0, [], {}, float('nan')):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'front_latent_mode'):
                    config(front_latent_mode=bad).validate()
                with self.assertRaisesRegex(ValueError, 'front_latent_mode'):
                    block(bad)
        with self.assertRaises(TypeError):
            CDLNORuntimeConfig(front_latent_mode='full')
        self.assertNotIn('front_latent_mode', CDLNORuntimeConfig().to_dict())

    def test_old_json_alias_strict_modes_and_sidecar_bytes(self):
        old = config().to_dict()
        old.pop('front_latent_mode')
        old['history_rule'] = 'front-t-after-ffn2-before-up-v1'
        self.assertEqual(CDLNOArchitectureConfig.from_dict(old), config())
        self.assertEqual(compare_architecture(old, config()), [])
        for mode in ('no_sa', 'identity'):
            with self.assertRaises(ValueError):
                CDLNOArchitectureConfig.from_dict(dict(old, front_latent_mode=mode))
        with self.assertRaises(ValueError):
            CDLNOArchitectureConfig.from_dict(dict(old, model_version='future'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'architecture.json'
            path.write_text(json.dumps(dict(schema_version=1, architecture=old, derived={'P':6})))
            before = path.read_bytes()
            for chunk in (0, 1, 2, 99):
                validate_sidecar(path, config(), CDLNORuntimeConfig(source_chunk_size=chunk, device='cuda', dtype='bfloat16'))
            for mode in ('no_sa', 'identity'):
                with self.assertRaises(SidecarMismatch):
                    validate_sidecar(path, config(front_latent_mode=mode))
            with self.assertRaises(FileExistsError):
                save_sidecar(path, config())
            self.assertEqual(path.read_bytes(), before)
            for mode in MODES:
                fresh = Path(directory)/f'{mode}.json'
                cfg = config(front_latent_mode=mode)
                save_sidecar(fresh, cfg)
                self.assertEqual(validate_sidecar(fresh, cfg)['architecture'], cfg.to_dict())

    def test_pickle_layout_rejects_ambiguous_or_unknown_state(self):
        # Genuine legacy pickle execution is separately checked against the
        # pre-edit artifacts. These malformed states test only rejection.
        for bad in ([], [0]*17, [0]*19, [0]*18, {},
                    {'config_pickle_version':True,'architecture':config().to_dict()},
                    {'config_pickle_version':2,'architecture':config().to_dict()},
                    {'config_pickle_version':1,'architecture':{}}):
            with self.subTest(state=bad):
                empty = object.__new__(CDLNOArchitectureConfig)
                with self.assertRaises(ValueError):
                    empty.__setstate__(bad)

    def test_block_formula_outputs_and_gradients_point_and_grid(self):
        for structured in (False, True):
            for mode in MODES:
                with self.subTest(structured=structured, mode=mode):
                    m = block(mode, structured).double().eval()
                    x = torch.randn(2, 35 if structured else 11, 8, dtype=torch.float64, requires_grad=True)
                    actual = m(x)
                    expected = reference_front(m, x, mode)
                    for a,b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
                    cots = tuple(torch.randn_like(y) for y in actual)
                    ga = torch.autograd.grad(actual, (x,*m.parameters()), cots)
                    gb = torch.autograd.grad(expected, (x,*m.parameters()), cots)
                    for a,b in zip(ga,gb):
                        torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-7)

    def test_exact_sublayer_calls_removed_parameters_and_counts(self):
        for structured in (False, True):
            full = block('full', structured)
            for mode in MODES:
                m, calls = block(mode, structured), []
                with ExitStack() as stack:
                    for name, child in m.named_children():
                        stack.callback(child.register_forward_hook(lambda _m,_a,_o,n=name: calls.append(n)).remove)
                    m(torch.randn(2,35 if structured else 11,8))
                expected = ['point_norm','down']
                if mode != 'identity': expected += ['latent_norm_1','latent_ffn_1']
                if mode == 'full': expected += ['latent_norm_sa','latent_sa']
                if mode != 'identity': expected += ['latent_norm_2','latent_ffn_2']
                expected += ['up_latent_norm','up','point_ffn_norm','point_ffn']
                self.assertEqual(calls, expected)
                self.assertEqual(set(m.state_dict()), {k for k in full.state_dict() if k.split('.')[0] not in REMOVED[mode]})
                for removed in REMOVED[mode]:
                    self.assertNotIn(removed, m._modules)
                    self.assertFalse(hasattr(m, removed))
                self.assertIndependent(m)
                removed_count = {'full':0,'no_sa':4*8**2+2*8+2*4,'identity':12*8**2+10*8+2*4}[mode]
                self.assertEqual(sum(p.numel() for p in full.parameters())-sum(p.numel() for p in m.parameters()), removed_count)

    def test_identity_history_is_live_down_before_up_norm_and_point_nonlinearity(self):
        for structured in (False, True):
            m, seen = block('identity', structured), {}
            with ExitStack() as stack:
                def down(_m,_a,out):
                    seen['s'] = out
                    seen['snapshot'] = out.detach().clone()
                    out.retain_grad()
                stack.callback(m.down.register_forward_hook(down).remove)
                stack.callback(m.up_latent_norm.register_forward_pre_hook(lambda _m,args: seen.update(up_input=args[0])).remove)
                if structured:
                    stack.callback(m.point_ffn.conv.register_forward_hook(lambda _m,_a,out: seen.update(conv_shape=out.shape)).remove)
                    stack.callback(m.point_ffn.norm.register_forward_hook(lambda _m,_a,_o: seen.update(point_ln=True)).remove)
                x = torch.randn(2,35 if structured else 11,8,requires_grad=True)
                with patch.object(F,'gelu',wraps=F.gelu) as activation:
                    h,t = m(x)
                self.assertEqual(activation.call_count, 1)
            self.assertIs(t, seen['s'])
            self.assertIs(t, seen['up_input'])
            torch.testing.assert_close(t,seen['snapshot'],atol=0,rtol=0)
            if structured:
                self.assertEqual(seen['conv_shape'],(2,8,5,7))
                self.assertTrue(seen['point_ln'])
                self.assertEqual(m.point_ffn.conv.groups,1)
            h.square().mean().backward()
            self.assertGreater(t.grad.abs().max().item(),0.)
            self.assertGreater(m.down.latent_queries.grad.abs().max().item(),0.)
            self.assertFiniteGradients(m,x)
            torch.testing.assert_close(t,seen['snapshot'],atol=0,rtol=0)

    def test_zero_output_branches_match_ablations_with_copied_weights(self):
        for structured in (False, True):
            for mode in ('no_sa','identity'):
                with self.subTest(mode=mode,structured=structured):
                    full, ablated = block('full',structured).double().eval(), block(mode,structured).double().eval()
                    copy_retained(full,ablated)
                    with torch.no_grad():
                        full.latent_sa.attn.to_out.weight.zero_()
                        full.latent_sa.attn.to_out.bias.zero_()
                        if mode == 'identity':
                            for ffn in (full.latent_ffn_1,full.latent_ffn_2):
                                ffn.fc2.weight.zero_()
                                ffn.fc2.bias.zero_()
                    x = torch.randn(2,35 if structured else 11,8,dtype=torch.float64,requires_grad=True)
                    xa = x.detach().clone().requires_grad_()
                    reference,actual = full(x),ablated(xa)
                    for r,a in zip(reference,actual): torch.testing.assert_close(r,a,atol=0,rtol=0)
                    cots = tuple(torch.randn_like(y) for y in actual)
                    retained_names = list(dict(ablated.named_parameters()))
                    gf = torch.autograd.grad(reference, (x,*(dict(full.named_parameters())[n] for n in retained_names)), cots)
                    ga = torch.autograd.grad(actual, (xa,*ablated.parameters()), cots)
                    for r,a in zip(gf,ga): torch.testing.assert_close(r,a,atol=2e-9,rtol=2e-7)

    def test_core_depth_modes_finite_gradients_independence_and_active_counts(self):
        for mode in MODES:
            for total,front in ((8,0),(8,2),(8,6),(12,2)):
                for cdpa in CDPA_MODES:
                    with self.subTest(mode=mode,L=total,F=front,cdpa=cdpa):
                        cfg = config(L=total,F=front,front_latent_mode=mode,cdpa_mode=cdpa,structured=True,grid_shape=(5,7))
                        m, counts = CDLNO(cfg).train(), Counter()
                        self.assertIndependent(m)
                        self.assertEqual(len(m.front_blocks),front)
                        self.assertEqual(len(m.latent_blocks),total-front)
                        self.assertTrue(all(b.front_latent_mode==mode for b in m.front_blocks))
                        with ExitStack() as stack:
                            groups = {
                                'down': [*(b.down for b in m.front_blocks),m.bridge],
                                'up': [*(b.up for b in m.front_blocks),m.readout.up],
                                'sa': [*(b.latent_sa for b in m.front_blocks if hasattr(b,'latent_sa')),
                                       *(b.self_attention for b in m.latent_blocks)],
                                'conv': [v for v in m.modules() if isinstance(v,ConvFFN)],
                            }
                            for name,children in groups.items():
                                for child in children:
                                    def count(_m,_a,out,label=name):
                                        counts[label] += 1
                                        if label in ('down','sa'): self.assertEqual(out.shape,(2,4,8))
                                    stack.callback(child.register_forward_hook(count).remove)
                            with trace(m) as records:
                                x = torch.randn(2,35,8,requires_grad=True)
                                y = m(x)
                                y.square().mean().backward()
                        self.assertFiniteGradients(m,x)
                        self.assertEqual(y.shape,(2,35,3))
                        self.assertEqual(counts,dict(down=front+1,up=front+1,sa=total if mode=='full' else total-front,conv=front+1))
                        sources = sum(len(r[2]) for r in records['cdpa'])
                        expected = 0 if cdpa=='off' else front if cdpa=='entry' else cfg.P*front+cfg.P*(cfg.P-1)//2
                        self.assertEqual(sources,expected)
                        if front==0:
                            self.assertFalse(any(n.startswith('front_blocks.') for n in m.state_dict()))

    def test_f0_modes_same_weights_outputs_gradients_and_parameter_keys(self):
        for total in (1,8):
            for cdpa in CDPA_MODES:
                cfg = config(L=total,F=0,cdpa_mode=cdpa)
                baseline = CDLNO(cfg)
                x = torch.randn(2,11,8,requires_grad=True)
                y = baseline(x)
                cot = torch.randn_like(y)
                grads = torch.autograd.grad(y,(x,*baseline.parameters()),cot)
                for mode in ('no_sa','identity'):
                    m = CDLNO(replace(cfg,front_latent_mode=mode))
                    m.load_state_dict(baseline.state_dict(),strict=True)
                    self.assertEqual(list(m.state_dict()),list(baseline.state_dict()))
                    xm = x.detach().clone().requires_grad_()
                    ym = m(xm)
                    torch.testing.assert_close(y,ym,atol=0,rtol=0)
                    gm = torch.autograd.grad(ym,(xm,*m.parameters()),cot)
                    for a,b in zip(grads,gm): torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_ablation_history_unchanged_across_consumers_and_new_forwards(self):
        for mode in ('no_sa','identity'):
            for cdpa in CDPA_MODES:
                cfg = config(front_latent_mode=mode,cdpa_mode=cdpa)
                m = CDLNO(cfg)
                snapshots = []
                with ExitStack() as stack, trace(m) as first:
                    for front in m.front_blocks:
                        stack.callback(front.register_forward_hook(
                            lambda _m,_a,out: snapshots.append(out[1].detach().clone())).remove)
                    x = torch.randn(2,11,8,requires_grad=True)
                    y = m(x)
                ts = [r[2] for r in first['front']]
                z0 = first['bridge'][0][1]
                rears = [r[1] for r in first['rear']]
                for j, current, history, _fused in first['cdpa']:
                    expected = ts + [z0,*rears][:j] if cdpa=='every_block' else ts
                    self.assertEqual([id(t) for t in history],[id(t) for t in expected])
                    self.assertNotIn(id(current),[id(t) for t in history])
                for t in ts: t.retain_grad()
                y.square().mean().backward()
                for t,snapshot in zip(ts,snapshots):
                    self.assertIsNotNone(t.grad)
                    self.assertTrue(torch.isfinite(t.grad).all())
                    torch.testing.assert_close(t,snapshot,atol=0,rtol=0)
                with trace(m) as second:
                    x2 = x.detach().clone().requires_grad_()
                    m.zero_grad(set_to_none=True)
                    y2 = m(x2)
                    y2.square().mean().backward()
                self.assertFalse({id(t) for t in ts} & {id(r[2]) for r in second['front']})
                torch.testing.assert_close(y,y2,atol=0,rtol=0)

    def test_core_explicit_history_reference_and_chunk_parity(self):
        for mode in ('no_sa','identity'):
            m = CDLNO(config(front_latent_mode=mode,cdpa_mode='every_block')).eval()
            with torch.no_grad():
                for fusion in m.cdpa_at.values(): fusion.w.uniform_(-.2,.2)
            x = torch.randn(2,11,8,requires_grad=True)
            y = m(x)
            expected = explicit_core(m,x)
            torch.testing.assert_close(y,expected,atol=2e-6,rtol=2e-5)
            cot = torch.randn_like(y)
            base_grads = torch.autograd.grad(y,(x,*m.parameters()),cot)
            for chunk in (1,2,99):
                xc = x.detach().clone().requires_grad_()
                yc = m(xc,source_chunk_size=chunk)
                torch.testing.assert_close(y,yc,atol=2e-6,rtol=2e-5)
                gc = torch.autograd.grad(yc,(xc,*m.parameters()),cot)
                for a,b in zip(base_grads,gc): torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)

    def test_new_core_checkpoint_roundtrip_each_mode_and_wrong_state_rejected(self):
        for mode in MODES:
            cfg = config(front_latent_mode=mode)
            model = CDLNO(cfg).eval()
            x = torch.randn(2,11,8)
            file = io.BytesIO()
            torch.save(model,file)
            file.seek(0)
            restored = torch.load(file,weights_only=False)
            self.assertEqual(restored.config,cfg)
            self.assertTrue(all(b.front_latent_mode==mode for b in restored.front_blocks))
            torch.testing.assert_close(model(x),restored(x),atol=0,rtol=0)
            another = CDLNO(cfg)
            another.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(model(x),another(x),atol=0,rtol=0)
            with self.assertRaises(RuntimeError):
                another.load_state_dict({k:v for k,v in model.state_dict().items() if k!='bridge.latent_queries'},strict=True)
        full = CDLNO(config())
        for mode in ('no_sa','identity'):
            with self.assertRaises(RuntimeError):
                CDLNO(config(front_latent_mode=mode)).load_state_dict(full.state_dict(),strict=True)


if __name__ == '__main__':
    unittest.main()
