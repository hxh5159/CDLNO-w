"""L6 full AirfRANS topology/parity independent of unresolved paper objectives."""
import ast
from contextlib import nullcontext
import hashlib
import io
import json
import math
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
from einops import rearrange, repeat
from timm.layers import trunc_normal_

from cdlno.linearno.airfrans import AirfRANSLinearNO
from linearno.attention_support import MANIFEST, SOURCE, errors
from linearno.attention_reference import reference as attention_reference


def official_class():
    record=MANIFEST['files']['airfrans'];path=SOURCE/record['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) or
           isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='ACTIVATION']
    scope=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,np=np,rearrange=rearrange,repeat=repeat,trunc_normal_=trunc_normal_)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['LinearAttentionNeuralOperator']


def config(full=False, dropout=0.):
    return dict(space_dim=7,n_layers=8,n_hidden=256 if full else 16,dropout=dropout,n_head=8 if full else 4,
                act='gelu',mlp_ratio=2,fun_dim=0,out_dim=4,linearno_rank=32 if full else 6,ref=8,unified_pos=True,linear=True)


def official_kwargs(cfg):
    result=dict(cfg);result['slice_num']=result.pop('linearno_rank');return result


def graph(n=17,device='cpu',dtype=torch.float32):
    return Data(x=torch.randn(n,7,device=device,dtype=dtype,requires_grad=True),
                pos=torch.randn(n,2,device=device,dtype=dtype,requires_grad=True),
                y=torch.randn(n,4,device=device,dtype=dtype),surf=torch.arange(n,device=device)%2==0)


def reference(model,data):
    """Functional network oracle; calls no tested Model/block/MLP forward."""
    import torch.nn.functional as F
    weights=dict(model.named_parameters())
    def linear(x,key):return F.linear(x,weights[key+'.weight'],weights.get(key+'.bias'))
    def norm(x,key):return F.layer_norm(x,(x.shape[-1],),weights[key+'.weight'],weights[key+'.bias'],1e-5)
    def mlp(x,key):return linear(F.gelu(linear(x,key+'.linear_pre.0')),key+'.linear_post')
    # Independent nested x-major/y-minor loop; float32 linspace values as release.
    points=[(float(np.float32(x)),float(np.float32(y))) for x in np.linspace(-2,4,8) for y in np.linspace(-1.5,1.5,8)]
    coords=torch.tensor(points,device=data.pos.device,dtype=data.pos.dtype)
    distance=torch.stack([((data.pos[:,0]-a)**2+(data.pos[:,1]-b)**2).sqrt() for a,b in coords],-1)
    x=mlp(torch.cat((data.x,distance),-1)[None],'preprocess')+weights['placeholder'][None,None]
    for i in range(len(model.blocks)):
        key=f'blocks.{i}';attn={k[len(key+'.Attn.'):]:v for k,v in weights.items() if k.startswith(key+'.Attn.')}
        read,_=attention_reference(norm(x,key+'.ln_1'),attn,heads=model.blocks[i].Attn.heads,variant='airfrans')
        x=x+read;x=x+mlp(norm(x,key+'.ln_2'),key+'.mlp')
        if i==len(model.blocks)-1:x=linear(norm(x,key+'.ln_3'),key+'.mlp2')
    return x[0]


class AirfRANSModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1);cls.rows=[]
        cls.tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads);torch.backends.cuda.matmul.allow_tf32=cls.tf32
        if path:=os.environ.get('LINEARNO_L6_MODEL_REPORT'):
            Path(path).write_text(json.dumps(dict(source_commit=MANIFEST['commit'],rows=cls.rows),indent=2)+'\n')

    def checked(self,a,b,label,metrics,atol,rtol):
        self.assertTrue(torch.isfinite(a).all(),label);self.assertTrue(torch.isfinite(b).all(),label)
        metrics[label]=errors(a,b);torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=label)

    def test_formal_count_exact_initialization_rng_and_state(self):
        torch.manual_seed(960);a=AirfRANSLinearNO(**config(True));ra=torch.get_rng_state()
        torch.manual_seed(960);b=official_class()(**official_kwargs(config(True)));rb=torch.get_rng_state()
        self.assertEqual(sum(p.numel() for p in a.parameters()),3358788)
        self.assertEqual(sum(p.numel() for p in b.parameters()),3358788)
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=0,rtol=0)
        self.assertTrue(torch.equal(ra,rb));self.assertNotIn('reference',a.state_dict())
        self.assertEqual(a.preprocess.linear_pre[0].in_features,71)
        self.rows.append(dict(kind='formal_initialization',parameters=3358788,all_values_equal=True,RNG_equal=True,
            keys={k:list(v.shape) for k,v in a.state_dict().items()}))

    def test_reference_grid_golden_and_actual_raw_position(self):
        model=AirfRANSLinearNO(**config())
        pos=torch.tensor([[[-2,-1.5],[4,1.5],[1,0],[-2,1.5],[0,0]]],dtype=torch.float32)
        expected=torch.tensor([[[math.sqrt((float(x)-float(np.float32(a)))**2+(float(y)-float(np.float32(b)))**2)
                                 for a in np.linspace(-2,4,8) for b in np.linspace(-1.5,1.5,8)] for x,y in pos[0]]])
        torch.testing.assert_close(model.get_grid(pos),expected,atol=1e-6,rtol=1e-6)
        self.assertEqual(model.get_grid(pos)[0,0,0].item(),0)
        self.assertEqual(model.get_grid(pos)[0,1,-1].item(),0)
        data=graph(5);data.pos=pos[0];before=data.x.clone();seen=[]
        handle=model.preprocess.register_forward_pre_hook(lambda m,i:seen.append(i[0].clone()))
        model(data);handle.remove()
        torch.testing.assert_close(seen[0][0,:,:7],before,atol=0,rtol=0)
        torch.testing.assert_close(seen[0][0,:,7:],expected[0],atol=1e-6,rtol=1e-6)
        self.rows.append(dict(kind='golden_positions',positions=pos.tolist(),order='x-major/y-minor',input71=True))

    def parity(self,device,dtype,dropout):
        cfg=config(dropout=dropout);torch.manual_seed(961)
        a=AirfRANSLinearNO(**cfg).to(device=device,dtype=dtype)
        b=official_class()(**official_kwargs(cfg)).to(device=device,dtype=dtype)
        a.load_state_dict(b.state_dict(),strict=True);b.load_state_dict(a.state_dict(),strict=True)
        da=graph(device=device,dtype=dtype)
        db=Data(x=da.x.detach().clone().requires_grad_(),pos=da.pos.detach().clone().requires_grad_())
        xa,pa=da.x.detach().clone(),da.pos.detach().clone()
        trace_a=[];trace_b=[]
        hs=[m.register_forward_hook(lambda m,i,o:trace_a.append(o.detach().clone())) for m in a.blocks]
        hs += [m.register_forward_hook(lambda m,i,o:trace_b.append(o.detach().clone())) for m in b.blocks]
        rng=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if device=='cuda' else []
        ya=a(da);torch.set_rng_state(rng)
        if gpu:torch.cuda.set_rng_state_all(gpu)
        # Sole CPU adaptation is the release's hardcoded Tensor.cuda in get_grid.
        context=patch.object(torch.Tensor,'cuda',lambda t,*args,**kw:t) if device=='cpu' else nullcontext()
        with context:yb=b(db)
        for h in hs:h.remove()
        atol,rtol=((1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5) if device=='cpu' else (1e-5,1e-4))
        metrics={};self.assertEqual(len(trace_a),8)
        for i,(u,v) in enumerate(zip(trace_a,trace_b)):self.checked(u,v,f'block/{i}',metrics,atol,rtol)
        self.checked(ya,yb,'forward',metrics,atol,rtol)
        cotangent=torch.randn_like(ya)
        la=(ya*cotangent).mean()+ya.square().mean();lb=(yb*cotangent).mean()+yb.square().mean()
        self.checked(la,lb,'loss',metrics,atol,rtol);la.backward();lb.backward()
        self.checked(da.x.grad,db.x.grad,'input/x',metrics,atol,rtol)
        self.checked(da.pos.grad,db.pos.grad,'input/pos',metrics,atol,rtol)
        dead=[]
        for name,p in a.named_parameters():
            other=dict(b.named_parameters())[name]
            if name.endswith('.temperature'):
                self.assertIsNone(p.grad);self.assertIsNone(other.grad);dead.append(name)
            else:self.checked(p.grad,other.grad,'grad/'+name,metrics,atol,rtol)
        oa=torch.optim.Adam(a.parameters(),lr=.001);ob=torch.optim.Adam(b.parameters(),lr=.001)
        oa.step();ob.step()
        for k,v in a.state_dict().items():self.checked(v,b.state_dict()[k],'step/'+k,metrics,atol,rtol)
        stream=io.BytesIO();torch.save(a.state_dict(),stream);stream.seek(0)
        loaded=AirfRANSLinearNO(**cfg).to(device=device,dtype=dtype).eval()
        loaded.load_state_dict(torch.load(stream,map_location=device,weights_only=True),strict=True);a.eval()
        with torch.no_grad():self.checked(a(da),loaded(da),'checkpoint',metrics,atol,rtol)
        torch.testing.assert_close(da.x,xa,atol=0,rtol=0);torch.testing.assert_close(da.pos,pa,atol=0,rtol=0)
        self.rows.append(dict(kind='official_full_model_parity',device=device,dtype=str(dtype),dropout=dropout,
             CPU_adaptation='Tensor.cuda identity for reference only' if device=='cpu' else None,
             atol=atol,rtol=rtol,metrics=metrics,dead=dead))

    def test_cpu_full_official_parity(self):
        for dtype,drop in [(torch.float64,0.),(torch.float32,.2)]:self.parity('cpu',dtype,drop)

    @unittest.skipUnless(torch.cuda.is_available(),'native release get_grid requires CUDA')
    def test_native_cuda_full_official_parity(self):
        self.parity('cuda',torch.float32,.2)

    def test_independent_double_network_oracle(self):
        torch.manual_seed(962);model=AirfRANSLinearNO(**config()).double();data=graph(dtype=torch.float64)
        y=model(data);r=reference(model,data);metrics={}
        self.checked(y,r,'forward',metrics,1e-12,1e-10)
        active=[data.x,data.pos]+[p for n,p in model.named_parameters() if not n.endswith('.temperature')]
        ga=torch.autograd.grad(y.square().sum(),active,retain_graph=True)
        gb=torch.autograd.grad(r.square().sum(),active)
        for i,(a,b) in enumerate(zip(ga,gb)):self.checked(a,b,f'gradient/{i}',metrics,1e-12,1e-10)
        self.rows.append(dict(kind='independent_double_oracle',metrics=metrics))

    def test_real_PyG_variable_nodes_dead_parameters_and_batch_isolation(self):
        model=AirfRANSLinearNO(**config()).eval()
        for n in (1,13,29):
            data=graph(n);before={k:v.clone() for k,v in data if isinstance(v,torch.Tensor)}
            y=model(data);self.assertEqual(y.shape,(n,4));y.square().mean().backward()
            for k,v in before.items():torch.testing.assert_close(data[k],v,atol=0,rtol=0)
            with torch.no_grad():
                for block in model.blocks:block.Attn.temperature.fill_(900)
                torch.testing.assert_close(model(data),y,atol=0,rtol=0)
            for name,p in model.named_parameters():
                if name.endswith('.temperature'):self.assertIsNone(p.grad)
                else:self.assertTrue(torch.isfinite(p.grad).all(),name)
            model.zero_grad(set_to_none=True)
        batch=Batch.from_data_list([graph(5),graph(7)])
        with self.assertRaisesRegex(ValueError,'one graph'):model(batch)
        sampled=Batch.from_data_list([graph(9)])
        sampled.x=sampled.x[:5];sampled.pos=sampled.pos[:5];sampled.batch=sampled.batch[:5]
        self.assertEqual(model(sampled).shape,(5,4))  # original Infer_test retains ptr[1]=9
        bad=graph();bad.pos=torch.randn(17,3)
        with self.assertRaisesRegex(ValueError,'pos'):model(bad)
        self.rows.append(dict(kind='real_PyG',variable_N=[1,13,29],multi_graph_rejected=True,sampled_ptr_preserved=True))

    def test_original_normalized_MSE_training_function_and_fresh_pickle(self):
        # Safe exact function AST: does not import main, read data or construct a graph.
        root=Path(__file__).resolve().parents[2]
        path=root/'Airfoil-Design-AirfRANS/train.py';tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        scope=dict(torch=torch,nn=torch.nn,np=np)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path)+':train-only','exec'),scope)
        torch.manual_seed(963);a=AirfRANSLinearNO(**config());b=AirfRANSLinearNO(**config())
        b.load_state_dict(a.state_dict(),strict=True)
        data=graph(19);optimizer=torch.optim.Adam(a.parameters(),lr=.001)
        other=torch.optim.Adam(b.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,total_steps=8)
        scheduler_other=torch.optim.lr_scheduler.OneCycleLR(other,max_lr=.001,total_steps=8)
        result=scope['train']('cpu',a,DataLoader([data],batch_size=1),optimizer,scheduler,criterion='MSE_weighted',reg=1)
        prediction=b(data);squared=(prediction-data.y).square()
        objective=squared[~data.surf].mean()+squared[data.surf].mean()
        objective.backward();other.step();scheduler_other.step()
        torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=1e-6,rtol=1e-5)
        self.assertEqual(scheduler.last_epoch,1)
        self.assertAlmostEqual(float(result[0]),float(squared.detach().mean()),places=6)
        self.assertAlmostEqual(float(result[4]+result[5]),float(objective.detach()),places=6)
        with tempfile.TemporaryDirectory(prefix='linearno-air-local-pickle-') as tmp:
            a.eval();torch.save(a,Path(tmp)/'local-trusted.pt')
            torch.save(dict(x=data.x.detach(),pos=data.pos.detach(),prediction=a(data).detach()),Path(tmp)/'input.pt')
            code='''import sys, torch
from pathlib import Path
from torch_geometric.data import Data
from models.LinearNO import Model
p=Path(sys.argv[1]); torch.set_num_threads(1)
model=torch.load(p/'local-trusted.pt',map_location='cpu',weights_only=False)
assert isinstance(model,Model)
v=torch.load(p/'input.pt',weights_only=True)
with torch.no_grad(): torch.testing.assert_close(model(Data(x=v['x'],pos=v['pos'])),v['prediction'],atol=0,rtol=0)
'''
            process=subprocess.run([sys.executable,'-B','-c',code,tmp],cwd=root/'Airfoil-Design-AirfRANS',
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root)),capture_output=True,text=True,timeout=60)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.rows.append(dict(kind='original_train_function',actual_PyG=True,objective='normalized four-channel volume MSE + surface MSE',
            optimizer_steps=1,scheduler_steps=1,fresh_cwd_local_pickle=True,scope='isolated function, not production CLI/native resume'))
