import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cdlno.linearno_loop.construction import build_from_config
from cdlno.linearno_loop.core import LinearNOLoopCore
from linearno_loop.config import resolve_config
from loop_linearno.lb_oracle import lb_reference
from loop_linearno.lb_support import mode_config,excite,LBTrace
from loop_linearno.sr_support import TOPOLOGIES,inputs
from loop_linearno.test_point_attnres import errors
from loop_linearno.point_attnres_oracle import point_attnres


class LBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=[];cls.adamw=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL5_REPORT'):
            Path(path).write_text(json.dumps(dict(rows=cls.rows,adamw=cls.adamw),indent=2)+'\n')

    def test_independent_full_core_oracle_all_variants_and_topologies(self):
        for variant in ('plain','temp','conv','conv_temp','airfrans','shapenet'):
            task='airfrans' if variant=='airfrans' else 'car' if variant=='shapenet' else 'darcy'
            for preset in TOPOLOGIES:
                for dtype in (torch.float64,torch.float32):
                    with self.subTest(variant=variant,preset=preset,dtype=dtype):
                        core=build_from_config(mode_config(task,preset,variant=variant if task=='darcy' else None)).loop.to(dtype)
                        excite(core);g=torch.Generator().manual_seed(873)
                        x=torch.randn(2,15,8,generator=g,dtype=dtype,requires_grad=True);rx=x.detach().clone().requires_grad_()
                        state={k:v.detach().clone().requires_grad_() for k,v in core.named_parameters()}
                        with LBTrace(core) as t:y=core(x)
                        ref,rt=lb_reference(rx,state,P=core.prefix_blocks,C=core.recurrent_core_blocks,R=core.loop_repeats,
                            S=core.suffix_blocks,variant=variant,heads=2,H=3,W=5)
                        atol,rtol=(1e-12,1e-10) if dtype==torch.float64 else (1e-6,1e-5)
                        row=dict(variant=variant,topology=preset,device='cpu',dtype=str(dtype),atol=atol,rtol=rtol,
                            propagated_weight_tolerance=dict(atol=1e-5,rtol=1e-4) if dtype==torch.float32 else dict(atol=atol,rtol=rtol),
                            errors={},unused=[])
                        self.rows.append(row)
                        def check(a,b,name):
                            row['errors'][name]=errors(a,b)
                            # Only propagated FP32 router weights get a separate
                            # budget: Y-H loses significant digits then RMS has
                            # derivative up to 1/sqrt(eps)=1000. Same-source AR
                            # below retains the original primitive tolerance.
                            wa,wr=(1e-5,1e-4) if dtype==torch.float32 and name.startswith('weights') else (atol,rtol)
                            torch.testing.assert_close(a,b,atol=wa,rtol=wr,msg=name)
                            self.assertTrue(torch.isfinite(a).all())
                        for i,r in enumerate(rt['rounds']):
                            check(t.entries[i],r['entry'],f'H{i+1}');check(t.ends[i],r['end'],f'Y{i+1}')
                            check(t.routes[i]['sources'][-1],r['delta'],f'Delta{i+1}')
                            torch.testing.assert_close(t.routes[i]['sources'][-1],t.ends[i]-t.entries[i],atol=0,rtol=0)
                            check(t.routes[i]['weights'],r['weights'],f'weights{i+1}')
                            router=core.lb_boundaries[i] if i<core.loop_repeats-1 else core.lb_output
                            _,local_weights=point_attnres(t.routes[i]['sources'],router.query,router.norm_scale)
                            check(t.routes[i]['weights'],local_weights,f'same_source_weights{i+1}')
                            check(t.routes[i]['routed'],r['routed'],f'route{i+1}')
                            for j,(a,b) in enumerate(zip(t.routes[i]['sources'],r['sources'])):check(a,b,f'route{i+1}.source{j}')
                            if i+1<core.loop_repeats:self.assertIs(t.entries[i+1],t.routes[i]['routed'])
                        for i,(a,b) in enumerate(zip(t.visits,rt['visits'])):check(a,b,f'visit{i}')
                        check(y,ref,'final');target=torch.randn(y.shape,generator=g,dtype=dtype)
                        loss=(y-target).square().mean();rloss=(ref-target).square().mean()
                        check(loss,rloss,'loss');loss.backward();rloss.backward();check(x.grad,rx.grad,'input_gradient')
                        for name,p in core.named_parameters():
                            if name.endswith('Attn.temperature'):
                                self.assertEqual(variant,'airfrans');self.assertIsNone(p.grad);self.assertIsNone(state[name].grad);row['unused'].append(name)
                            else:
                                check(p.grad,state[name].grad,'gradient.'+name)
                                if name.startswith('lb_'):self.assertGreater(p.grad.abs().sum().item(),0)
                        torch.optim.SGD(core.parameters(),lr=.001).step();torch.optim.SGD(state.values(),lr=.001).step()
                        for name,p in core.named_parameters():check(p,state[name],'step.'+name)

    def test_zero_query_exact_mean_and_router_timing(self):
        for preset in TOPOLOGIES:
            core=build_from_config(mode_config(preset=preset)).loop.double()
            with LBTrace(core) as t:core(torch.randn(2,15,8,dtype=torch.float64))
            self.assertEqual(len(t.routes),core.loop_repeats)
            expected=[f'prefix.{i}' for i in range(core.prefix_blocks)]
            for r in range(core.loop_repeats):
                expected += [f'core.{i}' for i in range(core.recurrent_core_blocks)]+[f'route.{r}']
                self.assertEqual(len(t.routes[r]['sources']),r+2)
                self.assertIs(t.routes[r]['sources'][0],t.entries[0])
                for j in range(r):self.assertIs(t.routes[r]['sources'][j+1],t.routes[j]['sources'][-1])
                weights=t.routes[r]['weights'];torch.testing.assert_close(weights,torch.full_like(weights,1/(r+2)),atol=0,rtol=0)
                # Mathematical mean; reciprocal-multiply and division may differ
                # by roundoff, so use the established FP64 numerical bound.
                torch.testing.assert_close(t.routes[r]['routed'],torch.stack(t.routes[r]['sources']).mean(0),atol=1e-12,rtol=1e-10)
            expected += [f'suffix.{i}' for i in range(core.suffix_blocks)]
            self.assertEqual(t.events,expected)
        # Binary-exact scalar fixtures make both named R2 identities bitwise.
        core=self.toy(R=2,k1=2,k2=2)
        x=torch.full((1,2,8),3.,dtype=torch.float64)
        with LBTrace(core) as t:y=core(x)
        torch.testing.assert_close(t.entries[1],(x+t.routes[0]['sources'][1])/2,atol=0,rtol=0)
        torch.testing.assert_close(y,(x+t.routes[1]['sources'][1]+t.routes[1]['sources'][2])/3,atol=0,rtol=0)
        torch.testing.assert_close(y,torch.full_like(x,10),atol=0,rtol=0)

    @staticmethod
    def toy(R,k1,k2):
        from loop_linearno.test_block_body import make_block
        class Scale(nn.Module):
            def __init__(self,k):
                super().__init__();self.k=k;self.dim=8;self.heads=2;self.dim_head=4;self.rank=4;self.variant='plain'
            def forward(self,x):return self.k*x
        def factory(*,last_layer):
            b=make_block('plain',last_layer);b.ln_1=nn.Identity();b.ln_2=nn.Identity()
            b.Attn=Scale(0 if last_layer else k1);b.mlp=Scale(0 if last_layer else k2)
            if last_layer:b.ln_3=nn.Identity();b.mlp2=nn.Identity()
            return b
        return LinearNOLoopCore(prefix_blocks=0,recurrent_core_blocks=1,loop_repeats=R,suffix_blocks=1,
            residual_mode='lb_attnres_1_over_r',block_factory=factory).double()

    def test_r1_is_output_receiver_only_and_not_sr(self):
        core=self.toy(R=1,k1=2,k2=3);x=torch.ones(1,3,8,dtype=torch.float64)
        with LBTrace(core) as t:y=core(x)
        self.assertEqual(len(core.lb_boundaries),0);self.assertEqual(len(t.routes),1)
        # Phi(a)=12a, Delta=11a, output=(a+11a)/2=6a, not12a.
        torch.testing.assert_close(t.ends[0],12*x,atol=0,rtol=0)
        torch.testing.assert_close(t.routes[0]['sources'][1],11*x,atol=0,rtol=0)
        torch.testing.assert_close(y,6*x,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in core.parameters()),16)

    def test_delta_is_actual_subtraction_not_reconstructed_branch_sum(self):
        core=build_from_config(mode_config('elasticity')).loop.double()
        calls=[];original=torch.Tensor.__sub__
        def subtract(a,b):
            result=original(a,b);calls.append((a,b,result));return result
        with patch.object(torch.Tensor,'__sub__',subtract),LBTrace(core) as t:
            core(torch.randn(2,9,8,dtype=torch.float64))
        for i,row in enumerate(t.routes):
            self.assertTrue(any(a is t.ends[i] and b is t.entries[i] and value is row['sources'][-1] for a,b,value in calls))

    def test_local_history_vjp_and_exception_recovery(self):
        model=build_from_config(mode_config('elasticity')).double();excite(model.loop)
        known={n:set(vars(m)) for n,m in model.named_modules()}
        x=torch.randn(2,9,2,dtype=torch.float64,requires_grad=True)
        with LBTrace(model.loop) as t:y=model(x,None)
        grads=torch.autograd.grad(t.routes[-1]['routed'].square().sum(),(t.routes[0]['sources'][1],x),retain_graph=True)
        for g in grads:self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)
        def fail(*args):raise RuntimeError('after delta exists')
        h=model.loop.lb_output.register_forward_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError,'delta'):model(x,None)
        h.remove();fresh=torch.randn(1,7,2,dtype=torch.float64,requires_grad=True)
        with LBTrace(model.loop) as new:z=model(fresh,None)
        self.assertEqual(new.routes[0]['sources'][0].shape,(1,7,8))
        self.assertIsNone(torch.autograd.grad(z.sum(),x,allow_unused=True)[0])
        torch.testing.assert_close(model(x,None),y,atol=0,rtol=0)
        for n,m in model.named_modules():
            self.assertEqual(set(vars(m)),known[n]);self.assertFalse(any(isinstance(v,torch.Tensor) for v in vars(m).values()))

    def test_native_unroll_train_dropout_adamw(self):
        for task in ('darcy','ns','plasticity','elasticity','airfrans','car'):
            for preset in TOPOLOGIES[:2]:
                for dtype in (torch.float64,torch.float32):
                    c=mode_config(task,preset);r=c['request']
                    c=resolve_config(task,options=r['options'],profile_overrides={**r['profile_overrides'],'model.dropout':.2})
                    core=build_from_config(c).loop.to(dtype);excite(core);ref=copy.deepcopy(core)
                    g=torch.Generator().manual_seed(222);x=torch.randn(2,15,8,generator=g,dtype=dtype,requires_grad=True)
                    rx=x.detach().clone().requires_grad_();rng=torch.get_rng_state().clone();y=core(x);end=torch.get_rng_state().clone()
                    torch.set_rng_state(rng);z=rx
                    for b in ref.prefix:z=b.block(z)
                    a=z;deltas=[]
                    for r in range(ref.loop_repeats):
                        entry=z
                        for physical in ref.core:
                            b=physical.block;z=z+(1/ref.loop_repeats)*b.Attn(b.ln_1(z));z=z+(1/ref.loop_repeats)*b.mlp(b.ln_2(z))
                        deltas.append(z-entry)
                        router=ref.lb_boundaries[r] if r<ref.loop_repeats-1 else ref.lb_output
                        z=router((a,*deltas))
                    for b in ref.suffix:z=b.block(z)
                    torch.testing.assert_close(y,z,atol=0,rtol=0);self.assertTrue(torch.equal(end,torch.get_rng_state()))
                    y.square().mean().backward();z.square().mean().backward();torch.testing.assert_close(x.grad,rx.grad,atol=0,rtol=0)
                    rp=dict(ref.named_parameters());opt=torch.optim.AdamW(core.parameters(),lr=.001);ropt=torch.optim.AdamW(ref.parameters(),lr=.001)
                    for n,p in core.named_parameters():
                        if p.grad is None:self.assertIsNone(rp[n].grad)
                        else:torch.testing.assert_close(p.grad,rp[n].grad,atol=0,rtol=0)
                    opt.step();ropt.step()
                    for n,p in core.named_parameters():
                        torch.testing.assert_close(p,rp[n],atol=0,rtol=0)
                        for key,value in opt.state[p].items():torch.testing.assert_close(value,ropt.state[rp[n]][key],atol=0,rtol=0)
                    self.adamw.append(dict(task=task,preset=preset,dtype=str(dtype),dropout=.2,atol=0,rtol=0,max_error=0,exact_rng=True))


if __name__=='__main__':unittest.main()
