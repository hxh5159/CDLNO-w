"""L7 task model: fixed official source plus independent functional oracle."""
import ast
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from einops import rearrange
from timm.layers import trunc_normal_

from cdlno.linearno.shapenet import ShapeNetLinearNO
from linearno.attention_support import MANIFEST, SOURCE, errors
from linearno.attention_reference import reference as attention_reference


def official_class():
    record = MANIFEST['files']['shapenet']; path = SOURCE/record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) or
             isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'ACTIVATION']
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
                 rearrange=rearrange, trunc_normal_=trunc_normal_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['LinearAttentionNeuralOperator']


def config(full=False, dropout=0., unified=False):
    return dict(space_dim=7 if unified else 3, fun_dim=0 if unified else 4,
                n_layers=8, n_hidden=256 if full else 16, n_head=8 if full else 4,
                linearno_rank=32 if full else 8, mlp_ratio=2, out_dim=4,
                dropout=dropout, act='gelu', Time_Input=False,
                ref=8, unified_pos=unified, H=3 if unified else 85,
                W=5 if unified else 85, isregular=False)


def official_kwargs(cfg):
    result = dict(cfg); rank = result.pop('linearno_rank')
    result['key_ratio'] = rank//(result['n_hidden']//result['n_head'])
    return result


def graph(n=17, device='cpu', dtype=torch.float32):
    return Data(x=torch.randn(n, 7, device=device, dtype=dtype, requires_grad=True),
                pos=torch.randn(n, 3, device=device, dtype=dtype),
                y=torch.randn(n, 4, device=device, dtype=dtype),
                surf=torch.arange(n, device=device)%2 == 0)


def reference(model, data):
    """No Model/block/MLP forward call; independent pointwise functional path."""
    import torch.nn.functional as F
    p = dict(model.named_parameters())
    def linear(x, key): return F.linear(x, p[key+'.weight'], p.get(key+'.bias'))
    def norm(x, key): return F.layer_norm(x, (x.shape[-1],), p[key+'.weight'], p[key+'.bias'], 1e-5)
    def mlp(x, key): return linear(F.gelu(linear(x, key+'.linear_pre.0')), key+'.linear_post')
    x = mlp(data.x[None], 'preprocess') + p['placeholder'][None, None]
    for i in range(len(model.blocks)):
        key = f'blocks.{i}'
        attn = {k[len(key+'.Attn.'):]: v for k, v in p.items() if k.startswith(key+'.Attn.')}
        branch, _ = attention_reference(norm(x, key+'.ln_1'), attn,
                                       heads=model.blocks[i].Attn.heads, variant='shapenet')
        x = x + branch
        x = x + mlp(norm(x, key+'.ln_2'), key+'.mlp')
        if i == len(model.blocks)-1:
            x = linear(norm(x, key+'.ln_3'), key+'.mlp2')
    return x[0]


class ShapeNetModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1); cls.rows = []
        cls.tf32 = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads); torch.backends.cuda.matmul.allow_tf32 = cls.tf32
        if path := os.environ.get('LINEARNO_L7_MODEL_REPORT'):
            Path(path).write_text(json.dumps(dict(source_commit=MANIFEST['commit'], rows=cls.rows), indent=2)+'\n')

    def checked(self, a, b, label, metrics, atol, rtol):
        self.assertTrue(torch.isfinite(a).all(), label); self.assertTrue(torch.isfinite(b).all(), label)
        metrics[label] = errors(a, b); torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=label)

    def test_formal_count_exact_keys_initialization_and_rng(self):
        torch.manual_seed(971); a = ShapeNetLinearNO(**config(True)); rng = torch.get_rng_state()
        torch.manual_seed(971); b = official_class()(**official_kwargs(config(True)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        torch.testing.assert_close(a.state_dict(), b.state_dict(), atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()), 3852420)
        self.assertEqual(sum(p.numel() for p in b.parameters()), 3852420)
        for block in a.blocks:
            self.assertEqual(block.Attn.to_q.out_features, 32)
            self.assertTrue(torch.equal(block.Attn.tempreature_q, torch.full((1,8,1,1), .5)))
        self.assertFalse(any(isinstance(m, torch.nn.Conv2d) for m in a.modules()))
        self.rows.append(dict(kind='formal_initialization', parameters=3852420, all_values_equal=True,
            RNG_equal=True, keys={k:list(v.shape) for k,v in a.state_dict().items()}))

    def parity(self, device, dtype, dropout, unified=False):
        cfg = config(dropout=dropout, unified=unified); torch.manual_seed(972)
        a = ShapeNetLinearNO(**cfg).to(device=device, dtype=dtype)
        context = patch.object(torch.Tensor, 'cuda', lambda t,*args,**kw:t) if device == 'cpu' and unified else nullcontext()
        with context: b = official_class()(**official_kwargs(cfg)).to(device=device, dtype=dtype)
        if unified:
            # Native release pos isn't registered: align only the reference dtype
            # in double tests. Production buffer's saved key set remains empty.
            b.pos = b.pos.to(device=device, dtype=dtype)
            torch.testing.assert_close(a.pos, b.pos, atol=0, rtol=0)
            self.assertNotIn('pos', a.state_dict())
        a.load_state_dict(b.state_dict(), strict=True)
        da = graph(15 if unified else 17, device, dtype)
        db = Data(x=da.x.detach().clone().requires_grad_())
        ta, tb = [], []
        handles = [m.register_forward_hook(lambda m,i,o:ta.append(o.detach().clone())) for m in a.blocks]
        handles += [m.register_forward_hook(lambda m,i,o:tb.append(o.detach().clone())) for m in b.blocks]
        rng = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all() if device == 'cuda' else []
        ya = a((da, object())); torch.set_rng_state(rng)
        if cuda: torch.cuda.set_rng_state_all(cuda)
        yb = b((db, None))
        for h in handles: h.remove()
        atol, rtol = ((1e-12,1e-10) if dtype == torch.float64 else (1e-6,1e-5) if device == 'cpu' else (1e-5,1e-4))
        metrics = {}; self.assertEqual(len(ta), 8)
        for i,(u,v) in enumerate(zip(ta,tb)): self.checked(u,v,f'block/{i}',metrics,atol,rtol)
        self.checked(ya,yb,'forward',metrics,atol,rtol)
        cotangent = torch.randn_like(ya)
        la = (ya*cotangent).mean()+ya.square().mean(); lb = (yb*cotangent).mean()+yb.square().mean()
        self.checked(la,lb,'loss',metrics,atol,rtol); la.backward(); lb.backward()
        if unified:
            self.assertIsNone(da.x.grad); self.assertIsNone(db.x.grad)
        else: self.checked(da.x.grad,db.x.grad,'input/x',metrics,atol,rtol)
        for name,p in a.named_parameters(): self.checked(p.grad,dict(b.named_parameters())[name].grad,'grad/'+name,metrics,atol,rtol)
        oa = torch.optim.Adam(a.parameters(),lr=.001); ob = torch.optim.Adam(b.parameters(),lr=.001)
        oa.step(); ob.step()
        for name,v in a.state_dict().items(): self.checked(v,b.state_dict()[name],'step/'+name,metrics,atol,rtol)
        stream = io.BytesIO(); torch.save(a.state_dict(),stream); stream.seek(0)
        c = ShapeNetLinearNO(**cfg).to(device=device,dtype=dtype).eval()
        c.load_state_dict(torch.load(stream,weights_only=True,map_location=device),strict=True); a.eval()
        self.checked(a((da,None)),c((da,None)),'checkpoint',metrics,atol,rtol)
        self.rows.append(dict(kind='official_full_model_parity',device=device,dtype=str(dtype),dropout=dropout,
            unified_pos=unified,CPU_reference_cuda_adapter=bool(device=='cpu' and unified),atol=atol,rtol=rtol,metrics=metrics))

    def test_cpu_official_full_forward_gradients_step(self):
        self.parity('cpu',torch.float64,0.)
        self.parity('cpu',torch.float32,.2)
        self.parity('cpu',torch.float32,0.,unified=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_native_cuda_official_full_parity(self):
        self.parity('cuda',torch.float32,.2)
        self.parity('cuda',torch.float32,0.,unified=True)

    def test_independent_double_oracle(self):
        torch.manual_seed(973); model = ShapeNetLinearNO(**config()).double(); data = graph(dtype=torch.float64)
        a = model((data,None)); b = reference(model,data); metrics={}
        self.checked(a,b,'forward',metrics,1e-12,1e-10)
        active = [data.x,*model.parameters()]
        ga = torch.autograd.grad(a.square().sum(),active,retain_graph=True)
        gb = torch.autograd.grad(b.square().sum(),active)
        for i,(u,v) in enumerate(zip(ga,gb)): self.checked(u,v,f'gradient/{i}',metrics,1e-12,1e-10)
        self.rows.append(dict(kind='independent_double_oracle',metrics=metrics))

    def test_PyG_variable_N_masks_geometry_unused_batch_and_input_unchanged(self):
        torch.manual_seed(974); model = ShapeNetLinearNO(**config()).eval()
        for n in (1,13,29):
            data = graph(n); before={k:v.clone() for k,v in data if isinstance(v,torch.Tensor)}
            y = model((data,None)); self.assertEqual(y.shape,(n,4))
            torch.testing.assert_close(y,model((data,torch.randn(3,4))),atol=0,rtol=0)
            y.square().mean().backward()
            for name,p in model.named_parameters(): self.assertTrue(torch.isfinite(p.grad).all(),name)
            for key,value in before.items(): torch.testing.assert_close(data[key],value,atol=0,rtol=0)
            permutation=torch.randperm(n)
            torch.testing.assert_close(model((Data(x=data.x[permutation]),None)),y[permutation],atol=1e-6,rtol=1e-5)
            model.zero_grad(set_to_none=True)
        with self.assertRaisesRegex(ValueError,'one graph'): model((Batch.from_data_list([graph(5),graph(7)]),None))
        self.assertEqual(model((Batch.from_data_list([graph(5)]),None)).shape,(5,4))
        with self.assertRaisesRegex(ValueError,'expects'): model(graph())
        with self.assertRaisesRegex(ValueError,'integer key_ratio'): ShapeNetLinearNO(**dict(config(),linearno_rank=7))
        with self.assertRaisesRegex(ValueError,'fun_dim'): ShapeNetLinearNO(**dict(config(),unified_pos=True))
        self.rows.append(dict(kind='real_PyG',N=[1,13,29],input_unchanged=True,geometry_unused=True,multigraph_rejected=True))

    def test_original_train_AST_MSE_and_fresh_process_whole_object(self):
        root=Path(__file__).resolve().parents[2]; path=root/'Car-Design-ShapeNetCar/train.py'
        tree=ast.parse(path.read_text()); function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        scope=dict(torch=torch,nn=torch.nn,np=np)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path)+':train-only','exec'),scope)
        torch.manual_seed(975); a=ShapeNetLinearNO(**config()); b=ShapeNetLinearNO(**config())
        b.load_state_dict(a.state_dict(),strict=True); data=graph(19)
        oa=torch.optim.Adam(a.parameters(),lr=.001); ob=torch.optim.Adam(b.parameters(),lr=.001)
        sa=torch.optim.lr_scheduler.OneCycleLR(oa,max_lr=.001,total_steps=8,final_div_factor=1000)
        sb=torch.optim.lr_scheduler.OneCycleLR(ob,max_lr=.001,total_steps=8,final_div_factor=1000)
        result=scope['train']('cpu',a,DataLoader([(data,torch.zeros(3,3))],batch_size=1),oa,sa,reg=.5)
        out=b((data,None)); lp=(out[data.surf,3]-data.y[data.surf,3]).square().mean()
        lv=(out[:,:3]-data.y[:,:3]).square().mean(); loss=lv+.5*lp
        loss.backward(); ob.step(); sb.step()
        torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=1e-6,rtol=1e-5)
        self.assertAlmostEqual(result[0],lp.item(),places=6); self.assertAlmostEqual(result[1],lv.item(),places=6)
        with tempfile.TemporaryDirectory(prefix='linearno-car-local-pickle-') as tmp:
            a.eval(); torch.save(a,Path(tmp)/'model.pt')
            torch.save(dict(x=data.x.detach(),prediction=a((data,None)).detach()),Path(tmp)/'input.pt')
            code='''import sys, torch
from pathlib import Path
from torch_geometric.data import Data
from models.LinearNO import Model
torch.set_num_threads(1); p=Path(sys.argv[1])
m=torch.load(p/'model.pt',weights_only=False,map_location='cpu'); assert isinstance(m,Model)
v=torch.load(p/'input.pt',weights_only=True)
torch.testing.assert_close(m((Data(x=v['x']),None)),v['prediction'],atol=0,rtol=0)
'''
            proc=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=root/'Car-Design-ShapeNetCar',
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=60)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.rows.append(dict(kind='original_train_function',objective='all-point velocity MSE + 0.5 surface pressure MSE',
            optimizer_steps=1,scheduler_steps=1,local_trusted_pickle=True,scope='not production CLI/native resume'))
