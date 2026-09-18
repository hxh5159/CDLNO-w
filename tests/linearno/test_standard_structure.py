"""L3 observable topology, positions, device/shape contracts and import isolation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno.attention import LinearNOAttention
from cdlno.linearno.schema import validate_model_spec
from linearno.attention_support import ROOT
from linearno.standard_support import local_class, local_kwargs, small_config, inputs, VARIANTS
from linearno.standard_reference import reference


class ShapeTrace(TorchDispatchMode):
    """Record shapes only; do not keep model tensors or computation graphs."""
    def __init__(self):
        super().__init__()
        self.shapes = set(); self.bmm = []; self.softmax = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        def record(value):
            if isinstance(value, torch.Tensor): self.shapes.add(tuple(value.shape))
            elif isinstance(value, (list, tuple)):
                for item in value: record(item)
        record(result)
        if func == torch.ops.aten.bmm.default:
            self.bmm.append([list(args[0].shape), list(args[1].shape), list(result.shape)])
        if func == torch.ops.aten._softmax.default:
            self.softmax.append([list(args[0].shape), args[1]])
        return result


class StandardModelStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1); cls.rows = []

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        if target := os.environ.get('LINEARNO_L3_STRUCTURE_REPORT'):
            Path(target).write_text(json.dumps(cls.rows, indent=2)+'\n')

    def test_eighth_block_executes_all_sublayers_no_quadratic_attention(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = small_config(variant, time=False, layers=8)
                cfg.update(H=5, W=7)
                model = local_class()(**local_kwargs(cfg)).eval()
                events = []; hooks = []
                for index, block in enumerate(model.blocks):
                    for name in ('Attn', 'mlp', 'ln_3', 'mlp2'):
                        if hasattr(block, name):
                            hooks.append(getattr(block, name).register_forward_hook(
                                lambda m, i, o, tag=f'{index}.{name}': events.append(tag)))
                trace = ShapeTrace()
                with torch.no_grad(), trace:
                    y = model(*inputs(cfg, T=False))
                for hook in hooks: hook.remove()
                expected = [f'{i}.{n}' for i in range(8) for n in ('Attn', 'mlp')]
                expected += ['7.ln_3', '7.mlp2']
                self.assertEqual(events, expected)
                self.assertEqual(tuple(y.shape), (2, 35, 2))
                atoms = [m for m in model.modules() if isinstance(m, LinearNOAttention)]
                self.assertEqual(len(atoms), 8)
                for atom in atoms:
                    self.assertEqual(tuple(atom.to_q.weight.shape), (4, 4))
                    self.assertIsNot(atom.to_q.weight, atom.to_k.weight)
                self.assertEqual(trace.bmm, [
                    [[6, 4, 35], [6, 35, 4], [6, 4, 4]],  # legal rank x value context
                    [[6, 35, 4], [6, 4, 4], [6, 35, 4]]]*8)
                self.assertEqual(trace.softmax, [[[2, 3, 35, 4], -1], [[2, 3, 35, 4], -2]]*8)
                self.assertFalse(any(len(s) >= 2 and s[-2:] == (35, 35) for s in trace.shapes))
                self.assertFalse(any(isinstance(m, torch.nn.MultiheadAttention) for m in model.modules()))
                self.assertFalse(any('slice' in key.lower() for key in model.state_dict()))
                parameters = [p for block in model.blocks for p in block.parameters()]
                self.assertEqual(len(parameters), len({id(p) for p in parameters}))
                self.assertEqual(len(parameters), len({p.untyped_storage().data_ptr() for p in parameters}))
                largest = max(trace.shapes, key=lambda s: __import__('math').prod(s))
                self.rows.append(dict(kind='eight_block_execution', variant=variant, events=events,
                    bmm=trace.bmm, softmax=trace.softmax, all_intermediate_shapes=sorted(trace.shapes),
                    largest_tensor_shape=largest, largest_tensor_elements=__import__('math').prod(largest),
                    N_squared=False, slice_self_attention=False, independent_block_storage=True))

    def test_batches_point_permutation_variable_N_and_no_inplace_or_cache(self):
        for variant in VARIANTS:
            cfg = small_config(variant, time=True, fun_dim=1)
            model = local_class()(**local_kwargs(cfg)).eval()
            x, fx, t = inputs(cfg)
            original = [v.detach().clone() for v in (x, fx, t)]; state = copy.deepcopy(model.state_dict())
            y = model(x, fx, t)
            separate = torch.cat([model(x[i:i+1], fx[i:i+1], t[i:i+1]) for i in range(2)])
            torch.testing.assert_close(y, separate, atol=1e-6, rtol=1e-5)
            altered = fx.detach().clone(); altered[1].add_(100)
            torch.testing.assert_close(model(x, altered, t)[0], y[0], atol=0, rtol=0)
            y[0].square().sum().backward()
            for v, before in zip((x, fx, t), original):
                self.assertEqual(torch.count_nonzero(v.grad[1]).item(), 0)
                torch.testing.assert_close(v, before, atol=0, rtol=0)
            model.zero_grad(set_to_none=True)
            model(x, fx, t).square().mean().backward()  # no first-graph reuse
            torch.testing.assert_close(model.state_dict(), state, atol=0, rtol=0)
            for module in model.modules():
                self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(module).values()))
            self.assertEqual(list(model.buffers()), [])
            if variant in ('plain', 'temp'):
                order = torch.randperm(15)
                torch.testing.assert_close(model(x[:, order], fx[:, order], t), model(x, fx, t)[:, order], atol=1e-6, rtol=1e-5)
                for N in (1, 7, 19):
                    self.assertEqual(tuple(model(*inputs(cfg, N=N)).shape), (2, N, 2))

    def test_unified_position_replaces_coordinates_nonpersistent_device_buffer(self):
        cfg = small_config('plain', unified=True, fun_dim=1)
        cfg.update(H=5, W=7, ref=4)
        model = local_class()(**local_kwargs(cfg)).eval()
        self.assertEqual(set(dict(model.named_buffers())), {'pos'})
        self.assertNotIn('pos', model.state_dict())
        self.assertEqual(model.preprocess.linear_pre[0].in_features, 17)
        self.assertFalse(model.pos.requires_grad)
        # Independent scalar geometry oracle, including row/column order.
        positions = torch.tensor([[[((i/4-a/3)**2+(j/6-b/3)**2)**.5
                                    for a in range(4) for b in range(4)]
                                   for j in range(7)] for i in range(5)]).unsqueeze(0)
        torch.testing.assert_close(model.pos, positions, atol=2e-7, rtol=1e-6)
        x, fx, t = inputs(cfg)
        torch.testing.assert_close(model(x, fx, t), model(x*100+7, fx, t), atol=0, rtol=0)
        self.assertIsNone(torch.autograd.grad(model(x, fx, t).sum(), x, allow_unused=True)[0])
        initial_keys = list(model.state_dict())
        model.double(); self.assertEqual(model.pos.dtype, torch.float64)
        xd = inputs(cfg, dtype=torch.float64)
        yd = model(*xd)
        expected, _ = reference(*xd, model.state_dict(), cfg, positions=model.pos)
        torch.testing.assert_close(yd, expected, atol=1e-12, rtol=1e-10)
        yd.square().mean().backward()
        self.assertIsNotNone(model.time_fc[0].weight.grad)
        self.assertEqual(initial_keys, list(model.state_dict()))
        if torch.cuda.is_available():
            model.float().cuda(); self.assertEqual(model.pos.device.type, 'cuda')
            gpu_inputs = tuple(v.float().cuda() for v in xd)
            output = model(*gpu_inputs).detach().cpu()
            model.cpu()
            torch.testing.assert_close(output, model(*(v.float() for v in xd)), atol=1e-5, rtol=1e-4)
        self.rows.append(dict(kind='position_device', grid=[5,7], ref=4, replaces_x=True,
            persistent=False, cpu_double_time=True, cuda_cpu_roundtrip=torch.cuda.is_available()))

    def test_rejected_shapes_configuration_and_registration_name_isolation(self):
        Model = local_class(); base = local_kwargs(small_config('conv'))
        for changes, message in (({'n_layers':0}, 'n_layers'), ({'n_hidden':13}, 'divisible'),
                ({'n_head':0}, 'n_head'), ({'linearno_rank':0}, 'linearno_rank'),
                ({'linearno_rank':True}, 'linearno_rank'), ({'H':0}, 'H'),
                ({'linearno_variant':'LinearNO'}, 'linearno_variant'),
                ({'act':'relu'}, 'gelu'), ({'fun_dim':-1}, 'fun_dim'),
                ({'Time_Input':1}, 'bool'), ({'unified_pos':1}, 'bool'),
                ({'n_hidden':9,'n_head':3}, 'even'), ({'dropout':-1}, 'dropout')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                Model(**dict(base, **changes))
        for name in ('args', 'model', 'slice_num', 'key_ratio'):
            with self.assertRaises(TypeError): Model(**dict(base, **{name: 'conv_temp'}))
        m = Model(**base); x, fx, t = inputs(small_config())
        for a, b, c, message in ((x[:,:14],fx[:,:14],t,'N = H'), (x[:,:,0],fx,t,'x must'),
                (x,None,t,'fun_dim'), (x,fx[:,:14],t,'fx must'), (x,fx,t[:,0],'T must'),
                (x,fx,t.double().repeat(1,2),'T must')):
            with self.assertRaisesRegex(ValueError, message): m(a,b,c)
        with self.assertRaisesRegex(ValueError,'Time_Input'):
            Model(**dict(base, Time_Input=False))(x,fx,t)
        # fun_dim=0 with explicit empty fx is valid release behavior: no placeholder.
        m = Model(**dict(base, fun_dim=0)); empty = torch.empty(2,15,0)
        self.assertEqual(tuple(m(x,empty,t).shape), (2,15,2))
        self.assertFalse(torch.equal(m(x,empty,t),m(x,None,t)))
        spec = dict(class_path='model.LinearNO.Model', constructor_kwargs=base)
        validate_model_spec(spec, constructor=Model)
        with self.assertRaisesRegex(ValueError,'unknown constructor'):
            validate_model_spec(dict(spec,constructor_kwargs=dict(base,task='darcy')),constructor=Model)
        with self.assertRaisesRegex(ValueError,'class_path'):
            validate_model_spec(dict(spec,class_path='model.Transolver.Model'),constructor=Model)

    def test_fresh_standard_cwd_import_strict_checkpoint_and_legacy_registry(self):
        code = '''
import io, torch
from types import SimpleNamespace
from model.LinearNO import Model
import model_dict
torch.set_num_threads(1)
config=dict(space_dim=2,fun_dim=1,n_hidden=12,n_head=3,n_layers=2,H=3,W=5,
            linearno_variant='conv_temp',linearno_rank=4,unified_pos=True)
model=Model(**config).eval(); x=torch.rand(2,15,2); fx=torch.rand(2,15,1)
stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
other=Model(**config).eval();other.load_state_dict(torch.load(stream,weights_only=True),strict=True)
torch.testing.assert_close(model(x,fx),other(x,fx),atol=0,rtol=0)
assert Model.__module__ == 'model.LinearNO'
assert model_dict.get_model(SimpleNamespace(model='Transolver_Irregular_Mesh')).Model.__module__ == 'model.Transolver_Irregular_Mesh'
try: model_dict.get_model(SimpleNamespace(model='linearno'))
except KeyError: pass
else: raise AssertionError('L3 must not register LinearNO')
print('stable Model import, pure-state strict roundtrip, original selection intact; linearno intentionally unregistered')
'''
        proc = subprocess.run([sys.executable,'-B','-c',code], cwd=ROOT/'PDE-Solving-StandardBenchmark',
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)),
            text=True,capture_output=True,timeout=60)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.rows.append(dict(kind='fresh_process',cwd='PDE-Solving-StandardBenchmark',stdout=proc.stdout.strip()))
