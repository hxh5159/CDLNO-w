"""L2 equation, fixed-source, parameter/gradient/optimizer and mixed-precision parity."""
import copy
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import unittest

import torch

from cdlno.linearno.attention import initialize_release_weights, VARIANTS
from linearno.attention_reference import reference
from linearno.attention_support import load_classes, kwargs, errors, MANIFEST, SOURCE


class AttentionParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.rows=[]
        cls.tf32=torch.backends.cuda.matmul.allow_tf32;cls.cudnn_tf32=torch.backends.cudnn.allow_tf32
        cls.benchmark=torch.backends.cudnn.benchmark;cls.deterministic=torch.backends.cudnn.deterministic
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.backends.cuda.matmul.allow_tf32=cls.tf32;torch.backends.cudnn.allow_tf32=cls.cudnn_tf32
        torch.backends.cudnn.benchmark=cls.benchmark;torch.backends.cudnn.deterministic=cls.deterministic
        target=os.environ.get('LINEARNO_L2_PARITY_REPORT')
        if target:
            Path(target).write_text(json.dumps(dict(source_commit=MANIFEST['commit'],source_root=str(SOURCE),
                torch=str(torch.__version__),cuda_build=torch.version.cuda,
                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                scope='synthetic attention primitives only',tf32=False,compile=False,rows=cls.rows),indent=2)+'\n')

    def checked(self, actual, expected, category, metrics, atol,rtol):
        self.assertEqual(actual.shape,expected.shape)
        self.assertTrue(torch.isfinite(actual).all(),category)
        self.assertTrue(torch.isfinite(expected).all(),category)
        metrics[category]=errors(actual,expected)
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol,msg=category)

    def test_release_constructor_and_outer_initialization_RNG_keys(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                local,official,init=load_classes(variant)
                torch.manual_seed(102);a=local(**kwargs(variant));rng_a=torch.get_rng_state()
                torch.manual_seed(102);b=official(**kwargs(variant));rng_b=torch.get_rng_state()
                self.assertTrue(torch.equal(rng_a,rng_b))
                self.assertEqual(list(a.state_dict()),list(b.state_dict()))
                self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
                torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=0,rtol=0)
                torch.manual_seed(103);a.apply(initialize_release_weights);rng_a=torch.get_rng_state()
                torch.manual_seed(103);b.apply(lambda m:init(None,m));rng_b=torch.get_rng_state()
                torch.testing.assert_close(a.state_dict(),b.state_dict(),atol=0,rtol=0)
                self.assertTrue(torch.equal(rng_a,rng_b))
                self.rows.append(dict(kind='initialization',variant=variant,parameter_count=sum(p.numel() for p in a.parameters()),
                    keys={k:list(v.shape) for k,v in a.state_dict().items()},constructor_max_abs=0,initialized_max_abs=0,rng_equal=True))
        # Norm callback is part of the later outer initialization, not attention math.
        norm=torch.nn.LayerNorm(5);norm.weight.data.fill_(9);norm.bias.data.fill_(9)
        norm.apply(initialize_release_weights)
        self.assertTrue(torch.equal(norm.weight,torch.ones(5)));self.assertTrue(torch.equal(norm.bias,torch.zeros(5)))

    def oracle_case(self,variant,dtype,B,H,W,dim,heads,rank):
        local,_,_=load_classes(variant)
        torch.manual_seed(110)
        model=local(**kwargs(variant,dim=dim,heads=heads,dim_head=dim//heads,rank=rank,H=H,W=W)).to(dtype).eval()
        model.apply(initialize_release_weights)
        # Moderate nonzero projections exercise both normalization derivatives.
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.ndim==2:p.mul_(8)
        x=torch.randn(B,H*W,dim,dtype=dtype,requires_grad=True)
        xr=x.detach().clone().requires_grad_()
        state={k:v.detach().clone().requires_grad_() for k,v in model.state_dict().items()}
        actual=model(x);expected,_=reference(xr,state,heads=heads,variant=variant,H=H,W=W)
        atol,rtol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
        metrics={};self.checked(actual,expected,'forward',metrics,atol,rtol)
        cotangent=torch.randn_like(actual)
        (actual*cotangent).sum().backward();(expected*cotangent).sum().backward()
        self.checked(x.grad,xr.grad,'input_grad',metrics,atol,rtol)
        for name,p in model.named_parameters():
            if name=='temperature' and variant=='airfrans':
                self.assertIsNone(p.grad);self.assertIsNone(state[name].grad);continue
            self.assertIsNotNone(p.grad,name);self.assertIsNotNone(state[name].grad,name)
            self.checked(p.grad,state[name].grad,'grad/'+name,metrics,atol,rtol)
        self.rows.append(dict(kind='oracle',variant=variant,device='cpu',dtype=str(dtype),B=B,N=H*W,dim=dim,heads=heads,rank=rank,
                              atol=atol,rtol=rtol,metrics=metrics))

    def test_cpu_double_and_float_oracle_all_variants(self):
        for dtype in (torch.float64,torch.float32):
            for variant in VARIANTS:
                for B,H,W,dim,heads,rank in ((1,1,7,8,2,4),(2,3,5,12,3,8)):
                    with self.subTest(dtype=dtype,variant=variant,B=B):
                        self.oracle_case(variant,dtype,B,H,W,dim,heads,rank)

    def release_case(self,variant,device,dtype,amp=None,dropout=0.):
        local,official,init=load_classes(variant)
        torch.manual_seed(120)
        a=local(**kwargs(variant,dropout=dropout)).to(device=device,dtype=dtype)
        a.apply(initialize_release_weights)
        b=official(**kwargs(variant,dropout=dropout)).to(device=device,dtype=dtype)
        b.load_state_dict(a.state_dict(),strict=True)  # identity mapping, no unknown/missing keys
        a.load_state_dict(b.state_dict(),strict=True)  # explicitly check inverse coverage
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        a.train();b.train()
        x=torch.randn(2,15,12,device=device,dtype=dtype,requires_grad=True)
        xb=x.detach().clone().requires_grad_();before=x.detach().clone()
        weights={k:v.clone() for k,v in a.state_dict().items()}
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []
        def restore():
            torch.set_rng_state(cpu_rng)
            if gpu_rng:torch.cuda.set_rng_state_all(gpu_rng)
        def context():return torch.autocast('cuda',dtype=amp) if amp else nullcontext()
        restore()
        with context():ya=a(x)
        after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state_all() if gpu_rng else []
        restore()
        with context():yb=b(xb)
        self.assertTrue(torch.equal(after_cpu,torch.get_rng_state()))
        for one,two in zip(after_gpu,torch.cuda.get_rng_state_all() if gpu_rng else []):self.assertTrue(torch.equal(one,two))
        self.assertTrue(torch.equal(x,before));torch.testing.assert_close(a.state_dict(),weights,atol=0,rtol=0)
        self.assertEqual(ya.dtype,yb.dtype)
        atol,rtol=((1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5) if device=='cpu' else (1e-5,1e-4))
        metrics={};self.checked(ya,yb,'forward',metrics,atol,rtol)
        cotangent=torch.randn_like(ya)
        # Use stable squared + linear synthetic objective, identical upstream gradient.
        ((ya.float()*cotangent.float()).mean()+ya.float().square().mean()).backward() if amp else ((ya*cotangent).mean()+ya.square().mean()).backward()
        ((yb.float()*cotangent.float()).mean()+yb.float().square().mean()).backward() if amp else ((yb*cotangent).mean()+yb.square().mean()).backward()
        self.checked(x.grad,xb.grad,'input_grad',metrics,atol,rtol)
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other)
            if name=='temperature' and variant=='airfrans':
                self.assertIsNone(p.grad);self.assertIsNone(q.grad);continue
            self.assertIsNotNone(p.grad,name);self.assertIsNotNone(q.grad,name)
            self.checked(p.grad,q.grad,'grad/'+name,metrics,atol,rtol)
        oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
        oa.step();ob.step()
        for name,value in a.state_dict().items():self.checked(value,b.state_dict()[name],'step/'+name,metrics,atol,rtol)
        # Strict pure-state roundtrip, not an external pickle conversion.
        stream=io.BytesIO();torch.save(a.state_dict(),stream);stream.seek(0)
        c=local(**kwargs(variant,dropout=dropout)).to(device=device,dtype=dtype)
        c.load_state_dict(torch.load(stream,weights_only=True,map_location=device),strict=True)
        a.eval();c.eval()
        with torch.no_grad(),context():
            self.checked(c(x.detach()),a(x.detach()),'checkpoint_forward',metrics,atol,rtol)
        self.rows.append(dict(kind='official_parity',variant=variant,device=device,dtype=str(dtype),amp=str(amp),dropout=dropout,
                              B=2,N=15,dim=12,heads=3,rank=8,atol=atol,rtol=rtol,metrics=metrics,
                              rng_equal=True,strict_key_mapping='identity both directions'))

    def test_fixed_official_cpu_forward_all_grads_step_checkpoint(self):
        for dtype in (torch.float64,torch.float32):
            for variant in VARIANTS:
                for dropout in (0.,.25):
                    with self.subTest(dtype=dtype,variant=variant,dropout=dropout):
                        self.release_case(variant,'cpu',dtype,dropout=dropout)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available; GPU FP32/AMP NOT RUN')
    def test_fixed_official_cuda_fp32_fp16_bfloat16(self):
        for amp in (None,torch.float16,torch.bfloat16):
            if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported():
                self.rows.append(dict(kind='NOT RUN',device='cuda',amp='bfloat16',reason='hardware unsupported'));continue
            for variant in VARIANTS:
                with self.subTest(variant=variant,amp=amp):
                    self.release_case(variant,'cuda',torch.float32,amp=amp,dropout=.25)
