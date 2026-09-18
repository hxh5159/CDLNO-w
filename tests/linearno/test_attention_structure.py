"""L2 invariants independent of official implementations and production math."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno.attention import LinearNOAttention, initialize_release_weights, VARIANTS
from linearno.attention_reference import reference
from linearno.attention_support import load_classes, kwargs, ROOT


class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.bmms=[];self.softmax=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        if func==torch.ops.aten.bmm.default:
            self.bmms.append((tuple(args[0].shape),tuple(args[1].shape),tuple(result.shape)))
        if func==torch.ops.aten._softmax.default:
            self.softmax.append((tuple(args[0].shape),args[1],result.detach().clone()))
        return result


class AttentionStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def build(self,variant,**options):
        cls,_,_=load_classes(variant)
        torch.manual_seed(203)
        return cls(**kwargs(variant,**options)).double().eval()

    def test_normalization_factorized_dense_and_execution_shapes(self):
        for variant in VARIANTS:
            # M==dh=4 creates a legal square context. N is deliberately different.
            with self.subTest(variant=variant):
                model=self.build(variant,rank=4)
                x=torch.randn(2,15,12,dtype=torch.float64)
                trace=Trace()
                with trace:y=model(x)
                fact,tensors=reference(x,model.state_dict(),heads=3,variant=variant,H=3,W=5)
                dense,_=reference(x,model.state_dict(),heads=3,variant=variant,H=3,W=5,dense=True)
                torch.testing.assert_close(y,fact,atol=1e-12,rtol=1e-10)
                torch.testing.assert_close(fact,dense,atol=1e-12,rtol=1e-10)
                self.assertEqual(len(trace.softmax),2)
                for (shape,axis,out),expected_axis,expected in zip(trace.softmax,(-1,-2),(tensors['Q'],tensors['K'])):
                    self.assertEqual(shape,(2,3,15,4));self.assertEqual(axis%4,expected_axis%4)
                    torch.testing.assert_close(out.sum(expected_axis),torch.ones_like(out.sum(expected_axis)),atol=1e-12,rtol=1e-12)
                    torch.testing.assert_close(out,expected,atol=1e-12,rtol=1e-10)
                self.assertEqual(trace.bmms,[((6,4,15),(6,15,4),(6,4,4)),((6,15,4),(6,4,4),(6,15,4))])
                self.assertNotIn((6,15,15),[out for _,_,out in trace.bmms])
                self.assertEqual(tensors['C'].shape,(2,3,4,4))
                # No attention matrices/history retained by the model after forward.
                self.assertFalse(any(isinstance(v,torch.Tensor) for v in vars(model).values()))
                self.assertEqual(list(model.buffers()),[])

    def test_batch_isolation_gradients_and_point_equivariance(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model=self.build(variant)
                x=torch.randn(2,15,12,dtype=torch.float64,requires_grad=True)
                state=copy.deepcopy(model.state_dict());old=x.detach().clone()
                y=model(x)
                torch.testing.assert_close(y,torch.cat([model(x[i:i+1]) for i in range(2)]),atol=1e-12,rtol=1e-10)
                altered=x.detach().clone();altered[1]*=7
                torch.testing.assert_close(model(altered)[0],y[0],atol=0,rtol=0)
                y[0].square().sum().backward()
                self.assertEqual(torch.count_nonzero(x.grad[1]).item(),0)
                self.assertGreater(x.grad[0].abs().sum().item(),0)
                torch.testing.assert_close(x.detach(),old,atol=0,rtol=0)
                torch.testing.assert_close(model.state_dict(),state,atol=0,rtol=0)
                if variant not in ('conv','conv_temp'):
                    permutation=torch.randperm(15)
                    torch.testing.assert_close(model(x[:,permutation]),y[:,permutation],atol=1e-12,rtol=1e-10)

    def test_shared_head_projections_and_independent_parameters(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model=self.build(variant)
                other=self.build(variant)
                params=list(model.parameters())+list(other.parameters())
                self.assertEqual(len(params),len({p.data_ptr() for p in params}))
                self.assertEqual(model.to_q.weight.shape,(8,4))
                self.assertEqual(model.to_k.weight.shape,(8,4))
                self.assertEqual(model.to_v.weight.shape,(4,4))
                self.assertIsNone(model.to_q.bias);self.assertIsNone(model.to_k.bias);self.assertIsNone(model.to_v.bias)
                self.assertIsNot(model.to_q.weight,model.to_k.weight)
                # All heads receive identical vectors, hence the one shared layer
                # must compute identical projected values, with no per-head weights.
                features=torch.randn(2,1,7,4,dtype=torch.float64).expand(-1,3,-1,-1)
                q=model.to_q(features)
                torch.testing.assert_close(q[:,0],q[:,2],atol=0,rtol=0)
                linear_names=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)]
                self.assertEqual(sum(n=='to_q' for n in linear_names),1)
                self.assertEqual(sum(n=='to_k' for n in linear_names),1)
                self.assertIsInstance(model.to_out[-1],torch.nn.Dropout)
                self.assertEqual(model.to_out[-1].p,0.)
                self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in model.modules()),int(variant.startswith('conv')))
                self.assertFalse(any('slice' in n or 'in_project_fx' in n for n in model.state_dict()))

    def test_temperature_clamp_boundary_gradients_and_air_dead_parameter(self):
        for variant in ('temp','conv_temp','shapenet'):
            with self.subTest(variant=variant):
                model=self.build(variant)
                prefix='tempreature' if variant=='shapenet' else 'temperature'
                lo,hi=(.1,2.) if variant=='shapenet' else (.01,1.)
                x=torch.randn(2,15,12,dtype=torch.float64)
                for qv,kv in ((lo-1,hi+1),(lo,hi),(.5,.7)):
                    model.zero_grad(set_to_none=True)
                    with torch.no_grad():getattr(model,prefix+'_q').fill_(qv);getattr(model,prefix+'_k').fill_(kv)
                    states={k:v.detach().clone().requires_grad_() for k,v in model.state_dict().items()}
                    result=model(x);expected,_=reference(x,states,heads=3,variant=variant,H=3,W=5)
                    torch.testing.assert_close(result,expected,atol=1e-12,rtol=1e-10)
                    result.square().sum().backward();expected.square().sum().backward()
                    for axis in ('q','k'):
                        p=getattr(model,prefix+'_'+axis)
                        torch.testing.assert_close(p.grad,states[prefix+'_'+axis].grad,atol=1e-12,rtol=1e-10)
                        if qv<lo:self.assertEqual(torch.count_nonzero(p.grad).item(),0)
        air=self.build('airfrans');x=torch.randn(2,15,12,dtype=torch.float64)
        before=air(x)
        with torch.no_grad():air.temperature.fill_(1234.)
        after=air(x);torch.testing.assert_close(before,after,atol=0,rtol=0)
        after.square().sum().backward();self.assertIsNone(air.temperature.grad)
        self.assertIn('temperature',air.state_dict())
        self.assertNotIn('temperature_q',self.build('shapenet').state_dict())
        default_air,_,_=load_classes('airfrans')
        self.assertEqual(default_air(dim=16,heads=4,dim_head=4).rank,32)

    def test_conv_grid_order_projection_and_noncontiguous_input(self):
        model=self.build('conv',H=5,W=7)
        self.assertEqual(model.in_project_x.groups,1)
        self.assertEqual(model.in_project_x.stride,(1,1))
        self.assertEqual(model.in_project_x.padding,(1,1))
        # Independent Python indexing establishes row-major n=h*W+w explicitly.
        x=torch.arange(2*35*12,dtype=torch.float64).reshape(2,35,12)/100
        for variant in ('conv','conv_temp'):
            model=self.build(variant,H=5,W=7)
            with torch.no_grad():
                model.in_project_x.weight.zero_();model.in_project_x.bias.zero_()
                model.in_project_x.weight[0,0,0,2]=1.
            captured=[]
            handle=model.to_q.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
            model(x);handle.remove()
            expect=torch.zeros(2,35,dtype=torch.float64)
            for h in range(5):
                for w in range(7):
                    if h>0 and w<6:expect[:,h*7+w]=x[:,(h-1)*7+(w+1),0]
            torch.testing.assert_close(captured[0][:,0,:,0],expect,atol=0,rtol=0)
            noncontiguous=x.transpose(1,2).contiguous().transpose(1,2)
            self.assertFalse(noncontiguous.is_contiguous())
            torch.testing.assert_close(model(noncontiguous),model(x),atol=0,rtol=0)
            with self.assertRaisesRegex(ValueError,'N = H'):model(x[:,:34])

    def test_dropout_only_on_output_and_forward_no_debug_state(self):
        for variant in VARIANTS:
            model=self.build(variant,dropout=.5).train();x=torch.randn(2,15,12,dtype=torch.float64)
            calls=[];handles=[]
            for name,module in model.named_modules():
                if isinstance(module,torch.nn.Dropout):
                    handles.append(module.register_forward_hook(lambda m,a,o,n=name:calls.append((n,tuple(a[0].shape)))))
            before=set(vars(model));out=model(x)
            for handle in handles:handle.remove()
            self.assertIsInstance(out,torch.Tensor);self.assertEqual(set(vars(model)),before)
            self.assertEqual(calls,[(('to_out.3' if variant in ('conv','conv_temp','shapenet') else 'to_out.1'),(2,15,12))])
            self.assertFalse(model._forward_hooks)

    def test_illegal_shapes_types_and_constructors(self):
        args=dict(dim=12,heads=3,dim_head=4,rank=8,variant='plain')
        for changes in ({'dim':11},{'heads':0},{'dim_head':True},{'rank':0},{'rank':2.5},{'variant':'LinearNO'},
                        {'dropout':float('nan')},{'dropout':-1},{'dropout':1.1},{'H':3},
                        {'variant':'conv','H':3,'W':5,'kernel':2},{'variant':'conv','H':0,'W':5}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):LinearNOAttention(**(args|changes))
        model=LinearNOAttention(**args)
        for x in (None,torch.ones(4,12),torch.ones(2,3,13),torch.ones(0,3,12),torch.ones(2,0,12)):
            with self.assertRaises(ValueError):model(x)
        with self.assertRaises(TypeError):model(torch.ones(2,3,12,dtype=torch.int64))
        car,_,_=load_classes('shapenet')
        for value in (0,True,1.5):
            with self.assertRaises(ValueError):car(dim=12,heads=3,dim_head=4,key_ratio=value)

    def test_double_gradcheck_independent_reference(self):
        # Finite differences on oracle input AND every active parameter. Temperature
        # is interior to clamp; oracle retains double and calls no tested forward.
        for variant in ('plain','temp','conv_temp','airfrans','shapenet'):
            model=self.build(variant,dim=4,heads=2,dim_head=2,rank=2,H=1,W=3)
            x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
            state={k:v.detach().clone().requires_grad_() for k,v in model.state_dict().items()}
            names=[n for n in state if n!='temperature']
            def fn(*values):
                local=dict(state);local.update(zip(names,values[1:]))
                return reference(values[0],local,heads=2,variant=variant,H=1,W=3)[0]
            self.assertTrue(torch.autograd.gradcheck(fn,(x,*(state[n] for n in names)),fast_mode=True,atol=1e-6,rtol=1e-4))

    def test_no_old_transolver_or_history_dependencies(self):
        source=(ROOT/'cdlno/linearno/attention.py').read_text();tree=ast.parse(source)
        imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('kcdno' in s or 'msar' in s or 'Physics_Attention' in s for s in imports))
        call_names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for forbidden in ('scaled_dot_product_attention','.cpu','.numpy','.item','retain_grad'):
            self.assertFalse(any(forbidden in n for n in call_names),forbidden)

    def test_task_local_imports_from_three_fresh_workdirs(self):
        for project,package,names in (
            ('PDE-Solving-StandardBenchmark','model',('LinearNO','LinearNO_temp','LinearNO_Conv','LinearNO_Conv_temp')),
            ('Airfoil-Design-AirfRANS','models',('LinearNO',)),
            ('Car-Design-ShapeNetCar','models',('LinearNO',))):
            code = '''
import importlib,io,torch,sys
from cdlno.linearno.attention import initialize_release_weights
torch.set_num_threads(1)
module=importlib.import_module(sys.argv[1]+'.LinearNO_Attention')
for name in sys.argv[2:]:
    cls=getattr(module,name)
    kw=dict(dim=16,heads=4,dim_head=4)
    if 'Conv' in name:kw.update(H=3,W=5)
    model=cls(**kw).eval();model.apply(initialize_release_weights)
    x=torch.randn(2,15,16);expected=model(x)
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    clone=cls(**kw).eval();clone.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    torch.testing.assert_close(clone(x),expected,atol=0,rtol=0)
assert not any(s.startswith('exp_') or s in ('main','main_evaluation') for s in sys.modules)
'''
            with self.subTest(project=project):
                p=subprocess.run([sys.executable,'-B','-c',code,package,*names],cwd=ROOT/project,
                    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1'),
                    capture_output=True,text=True,timeout=60)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
