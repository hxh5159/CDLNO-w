import copy
import importlib
import json
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

from cdlno.linearno_loop.body import LinearNOBlockBody
from cdlno.linearno.airfrans import AirfRANSBlock
from cdlno.linearno.shapenet import ShapeNetBlock
from loop_linearno.test_point_attnres import errors

ROOT=Path(__file__).resolve().parents[2]
VARIANTS=('plain','temp','conv','conv_temp','airfrans','shapenet')


def make_block(variant,last,dropout=0.):
    if variant=='airfrans':return AirfRANSBlock(8,2,4,2,dropout,3,last)
    if variant=='shapenet':return ShapeNetBlock(8,2,4,2,dropout,3,last)
    project=str(ROOT/'PDE-Solving-StandardBenchmark')
    if project not in sys.path:sys.path.insert(0,project)
    cls=importlib.import_module('model.LinearNO').LinearNOBlock
    return cls(hidden=8,heads=2,rank=4,variant=variant,dropout=dropout,
               mlp_ratio=2,H=3,W=5,last_layer=last,out_dim=3)


class BlockBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL2_BODY_REPORT'):
            Path(path).write_text(json.dumps(dict(torch=str(torch.__version__),rows=cls.rows),indent=2)+'\n')

    def parity(self,variant,last,dtype,training,device):
        torch.manual_seed(710)
        native=make_block(variant,last,.25).to(device=device,dtype=dtype).train(training)
        candidate=copy.deepcopy(native)  # Same actual weights, copied by TEST only.
        before=torch.get_rng_state().clone()
        keys={k:list(v.shape) for k,v in candidate.state_dict().items()}
        ids={k:id(v) for k,v in candidate.named_parameters()}
        adapter=LinearNOBlockBody(candidate)
        self.assertIs(adapter.block,candidate)
        self.assertFalse(isinstance(adapter,nn.Module))
        self.assertEqual(keys,{k:list(v.shape) for k,v in candidate.state_dict().items()})
        self.assertEqual(ids,{k:id(v) for k,v in candidate.named_parameters()})
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.randn(2,15,8,device=device,dtype=dtype,requires_grad=True)
        xcopy=x.detach().clone().requires_grad_()
        cpu_state=torch.get_rng_state().clone()
        gpu_state=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        expected=native(x)
        end_cpu=torch.get_rng_state().clone()
        end_gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        torch.set_rng_state(cpu_state)
        if gpu_state is not None:torch.cuda.set_rng_state(gpu_state)
        body=adapter.native(xcopy)
        self.assertEqual(body.shape,x.shape)
        actual=adapter.finalize(body) if last else body
        self.assertTrue(torch.equal(end_cpu,torch.get_rng_state()))
        if end_gpu is not None:self.assertTrue(torch.equal(end_gpu,torch.cuda.get_rng_state()))
        atol,rtol=((1e-12,1e-10) if dtype==torch.float64 else ((1e-5,1e-4) if device=='cuda' else (1e-6,1e-5)))
        row=dict(variant=variant,last_layer=last,dtype=str(dtype),device=device,training=training,
                 B=2,N=15,H=3,W=5,hidden=8,heads=2,rank=4,atol=atol,rtol=rtol,errors={},unused=[])
        def check(a,b,name):
            torch.testing.assert_close(a,b,atol=atol,rtol=rtol,msg=name)
            self.assertTrue(torch.isfinite(a).all())
            row['errors'][name]=errors(a,b)
        check(actual,expected,'forward')
        target=torch.linspace(-.7,.8,actual.numel(),device=device,dtype=dtype).reshape_as(actual)
        loss=(actual-target).square().mean();ref_loss=(expected-target).square().mean()
        check(loss,ref_loss,'loss');loss.backward();ref_loss.backward()
        check(xcopy.grad,x.grad,'input_gradient')
        for (name,p),(other,reference) in zip(candidate.named_parameters(),native.named_parameters()):
            self.assertEqual(name,other)
            if reference.grad is None:
                self.assertEqual((variant,name),('airfrans','Attn.temperature'))
                self.assertIsNone(p.grad);row['unused'].append(name)
            else:check(p.grad,reference.grad,'gradient.'+name)
        opt=torch.optim.AdamW(candidate.parameters(),lr=.001,weight_decay=.01)
        ropt=torch.optim.AdamW(native.parameters(),lr=.001,weight_decay=.01)
        opt.step();ropt.step()
        for name,value in candidate.state_dict().items():check(value,native.state_dict()[name],'step.'+name)
        self.assertEqual(ids,{k:id(v) for k,v in candidate.named_parameters()})
        self.rows.append(row)

    def test_six_variants_native_and_final_fp64_fp32_eval_train(self):
        for variant in VARIANTS:
            for last in (False,True):
                for dtype in (torch.float64,torch.float32):
                    for training in (False,True):
                        with self.subTest(variant=variant,last=last,dtype=dtype,train=training):
                            self.parity(variant,last,dtype,training,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_six_variants_cuda_float32(self):
        settings=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                  torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
        try:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            for variant in VARIANTS:
                for last in (False,True):
                    with self.subTest(variant=variant,last=last):self.parity(variant,last,torch.float32,True,'cuda')
        finally:
            (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)=settings

    def test_raw_branches_scaled_residual_and_no_identity_scaling(self):
        for variant in VARIANTS:
            block=make_block(variant,False).double().eval();adapter=LinearNOBlockBody(block)
            x=torch.randn(2,15,8,dtype=torch.float64,requires_grad=True)
            before=x.clone()
            u=block.Attn(block.ln_1(x))
            torch.testing.assert_close(adapter.operator(x),u,atol=0,rtol=0)
            torch.testing.assert_close(adapter.mlp(x),block.mlp(block.ln_2(x)),atol=0,rtol=0)
            y=x+u/3;expected=y+block.mlp(block.ln_2(y))/3
            torch.testing.assert_close(adapter.scaled(x,1/3),expected,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(adapter.scaled(x,1),block(x),atol=0,rtol=0)
            torch.testing.assert_close(x,before,atol=0,rtol=0)
            # With both branch outputs zero, any identity scaling is observable.
            with torch.no_grad():
                for p in block.Attn.parameters():p.zero_()
                for p in block.mlp.parameters():p.zero_()
            torch.testing.assert_close(adapter.scaled(x,.5),x,atol=0,rtol=0)

    def test_head_only_on_explicit_finalize_and_branch_submodule_counts(self):
        for variant in VARIANTS:
            block=make_block(variant,True).eval();adapter=LinearNOBlockBody(block)
            calls={name:0 for name in ('ln_1','Attn','ln_2','mlp','ln_3','mlp2')}
            def count(name):
                def hook(*args):calls[name]+=1
                return hook
            handles=[getattr(block,name).register_forward_hook(count(name)) for name in calls]
            try:
                x=torch.randn(2,15,8);body=adapter.native(x)
                self.assertEqual(calls,dict(ln_1=1,Attn=1,ln_2=1,mlp=1,ln_3=0,mlp2=0))
                output=adapter.finalize(body)
                self.assertEqual(output.shape,(2,15,3))
                self.assertEqual(calls,dict(ln_1=1,Attn=1,ln_2=1,mlp=1,ln_3=1,mlp2=1))
            finally:
                for handle in handles:handle.remove()

    def test_nonowning_repeated_calls_and_errors_do_not_change_block(self):
        block=make_block('plain',False);adapter=LinearNOBlockBody(block)
        modules=list(block.named_modules());params=list(block.named_parameters());state=copy.deepcopy(block.state_dict())
        for N,B in ((7,2),(3,1),(11,3)):
            self.assertEqual(adapter.native(torch.randn(B,N,8)).shape,(B,N,8))
        self.assertEqual(list(block.named_modules()),modules)
        self.assertEqual(list(block.named_parameters()),params)
        for k,v in block.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
        self.assertEqual(LinearNOBlockBody.__slots__,('block',))
        with self.assertRaisesRegex(ValueError,'last_layer'):adapter.finalize(torch.randn(1,2,8))
        with self.assertRaises(TypeError):LinearNOBlockBody(object())
        with self.assertRaisesRegex(ValueError,'ln_1'):LinearNOBlockBody(nn.Identity())
        for scale in (True,0,-1,'0.5',float('nan'),float('inf')):
            with self.assertRaises(ValueError):adapter.scaled(torch.randn(1,2,8),scale)
        for variant in ('conv','conv_temp'):
            with self.assertRaisesRegex(ValueError,'N = H'):
                LinearNOBlockBody(make_block(variant,False)).native(torch.randn(2,14,8))


if __name__=='__main__':unittest.main()
