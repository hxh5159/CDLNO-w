"""M8: actual Light/Full task profiles; reuse safe M5-M7 helpers and losses.

No exp/main imports, data downloads, model edits or new training framework.
Temporal wrappers keep their accepted N=4096/3131. Other tasks use small N.
Optional MSAR_M8_RESULTS writes per-case evidence; MSAR_M8_DEVICE=cpu|cuda
selects the formal-profile test device without changing production defaults.
"""
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import ast
import copy
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend

import test_msar_static as static
import test_msar_temporal as temporal
import test_msar_industrial as industrial
from cdlno.modules import _SelfAttention
from cdlno.msar_lno.core import MSARLNO
from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.objective import training_forward, training_objective, ObjectiveMetrics
from cdlno.msar_lno.modules import (LearnedQueryDown, QueryAlignedUpCross,
    LatentFFNSAFFNBlock, PairwiseAttnResFusion, CoverageFloorLoss)
from cdlno.msar_lno.profiles import resolve_profile

TASKS=tuple(static.TASKS)+tuple(temporal.TEMPORAL)+tuple(industrial.PROJECTS)
DEVICE=os.environ.get('MSAR_M8_DEVICE','cpu')
ROWS=[]


def record(row):
    ROWS.append(row)
    if os.environ.get('MSAR_M8_RESULTS'):
        Path(os.environ['MSAR_M8_RESULTS']).write_text(json.dumps(ROWS,indent=2)+'\n')


def arguments(task,profile='light',mode='floor'):
    flags=['--profile',profile,'--coverage-mode',mode]
    if task in static.TASKS:return static.arguments(task,flags)
    if task in temporal.TEMPORAL:return temporal.arguments(task,flags)
    return industrial.arguments(task,flags)


def construct(task,args):
    if task in static.TASKS:return static.construct(task,args,grid=(7,7) if task=='darcy' else (5,7))
    if task in temporal.TEMPORAL:return temporal.construct(task,args)
    return industrial.CLASSES[task](**industrial.model_kwargs(args))


def inputs(task,model):
    device=next(model.parameters()).device
    if task in static.TASKS:
        x,fx=static.sample(model,b=1,device=device)
        return (x,fx.unsqueeze(-1) if task=='darcy' else None),{}
    if task in temporal.TEMPORAL:
        x,fx,_,tim=temporal.sample(task,b=1,steps=1,device=device)
        return (x,fx),dict(T=tim[:,:1]) if task=='plasticity' else {}
    graph=industrial.graph(task,19)
    if task=='car':graph=tuple(g.to(device) for g in graph)
    else:graph=graph.to(device)
    return (graph,),{}


def original_step_and_eval(task,model,args):
    """Original losses/time loops; only safe helper AST/data tensors, no trainer rewrite."""
    device=next(model.parameters()).device
    if task in static.TASKS:
        x,fx,target,norm,scope=static.batch(task,model,args,device=device)
        loop=next(n for n in ast.walk(static.source(task)) if isinstance(n,ast.For) and ast.unparse(n.iter)=='test_loader')
        nodes=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('out','tl')]
        scope.update(y=target);model.eval()
        with torch.no_grad():exec(compile(ast.Module(nodes,type_ignores=[]),'<original-static-eval>','exec'),scope)
        return scope['msar_epoch'].values(),dict(B=2,N=x.shape[1],forwards=1,optimizer_steps=1,eval_metric=scope['tl'])
    if task in temporal.TEMPORAL:
        data=temporal.sample(task,b=1,device=device)
        scope=temporal.train_batch(task,model,args,data)
        metrics=scope['msar_epoch'].values()
        # Release training graph references before ten/twenty independent eval calls.
        del scope
        result=temporal.evaluate(task,model,data)
        return metrics,dict(B=1,N=data[0].shape[1],forwards=10 if task=='ns' else 20,
                            optimizer_steps=1 if task=='ns' else 20,eval_metric=result['test_l2_full'])
    graph=industrial.graph(task,19);trainer=industrial.trainer(task)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,total_steps=20)
    metrics=ObjectiveMetrics();kwargs=dict(reg=args.weight,msar_metrics=metrics)
    if task=='airfrans':kwargs['criterion']='MSE_weighted'
    trainer.train(device,model,[graph],optimizer,scheduler,**kwargs)
    values=trainer.test(device,model,[graph])
    return metrics.values(),dict(B=1,N=19,forwards=1,optimizer_steps=1,eval_metric=float(values[0]))


class FormalTaskAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DEVICE not in ('cpu','cuda'):raise ValueError('MSAR_M8_DEVICE must be cpu or cuda')
        if DEVICE=='cuda' and not torch.cuda.is_available():raise unittest.SkipTest('requested CUDA unavailable')
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        cls.math=sdpa_kernel(SDPBackend.MATH);cls.math.__enter__()
    @classmethod
    def tearDownClass(cls):
        cls.math.__exit__(None,None,None);torch.set_num_threads(cls.threads)
        torch.backends.cuda.matmul.allow_tf32=cls.tf32

    def run_light(self,task):
        if task in industrial.PROJECTS and not industrial.HAS_PYG:self.skipTest('real PyG unavailable')
        torch.manual_seed(9180+TASKS.index(task));base=construct(task,arguments(task))
        initial={k:v.detach().clone() for k,v in base.state_dict().items()}
        del base
        for mode in ('off','floor'):
            with self.subTest(mode=mode):
                started=time.perf_counter();args=arguments(task,mode=mode)
                model=construct(task,args).to(DEVICE);model.load_state_dict(initial,strict=True)
                self.assertEqual(model.config,resolve_profile('light'))
                call,kwargs=inputs(task,model)
                # One model, unchanged parameters, same tensors. Train-mode floor
                # uses explicit A, off uses SDPA. Neither pass performs an update.
                with torch.no_grad():
                    on=training_forward(model,*call,training_config=replace(model.training_config,coverage_mode='floor'),**kwargs)
                    outputs=[];handles=[d.register_forward_hook(lambda m,a,out:outputs.append(out)) for d in model.core.downs]
                    with patch.object(CoverageFloorLoss,'forward',side_effect=AssertionError('off computed coverage')):
                        off=training_forward(model,*call,training_config=replace(model.training_config,coverage_mode='off'),**kwargs)
                    for h in handles:h.remove()
                    self.assertTrue(all(isinstance(out,torch.Tensor) for out in outputs));self.assertIsNone(off.auxiliary)
                    maxdiff=(on.prediction-off.prediction).abs().max().item()
                    torch.testing.assert_close(on.prediction,off.prediction,atol=2e-6,rtol=1e-4)
                del call,kwargs,on,off,outputs,handles
                # Identical fixture RNG per mode, explicit common parameter copy above.
                torch.manual_seed(9190+TASKS.index(task))
                metrics,details=original_step_and_eval(task,model,args)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
                self.assertAlmostEqual(metrics['objective_total'],metrics['objective_pde']+metrics['coverage_weighted'],places=5)
                self.assertAlmostEqual(metrics['coverage_weighted'],.01*metrics['coverage_raw'],places=6)
                if mode=='off':self.assertEqual(metrics['objective_total'],metrics['objective_pde'])
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(torch.isfinite(torch.as_tensor(details['eval_metric'])))
                record(dict(task=task,profile='light',mode=mode,device=DEVICE,dtype='float32',backend='MATH',
                    architecture=model.config.to_dict(),parameters=sum(p.numel() for p in model.parameters()),
                    original_loss=True,backward=True,optimizer=True,eval=True,real_PyG=task in industrial.PROJECTS,
                    same_weight_off_floor_max_abs_diff=maxdiff,atol=2e-6,rtol=1e-4,metrics=metrics,
                    seconds=time.perf_counter()-started,**details))
                del model;gc.collect()

    def run_full(self,task):
        if task in industrial.PROJECTS and not industrial.HAS_PYG:self.skipTest('real PyG unavailable')
        started=time.perf_counter();torch.manual_seed(9200+TASKS.index(task))
        args=arguments(task,'full');model=construct(task,args).to(DEVICE)
        self.assertEqual(model.config,resolve_profile('full'))
        call,kwargs=inputs(task,model)
        # Full numerical test deliberately one forward, synthetic MSE, not a
        # claim of full original temporal loops or industrial sampling epochs.
        f=training_forward(model,*call,**kwargs)
        target=torch.randn_like(f.prediction)
        loss=training_objective((f.prediction-target).square().mean(),f)
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        shape=list(f.prediction.shape);params=sum(p.numel() for p in model.parameters())
        del f,loss,target;model.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():expected=model(*call,**kwargs).detach()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt';torch.save(model.state_dict(),path)
            saved=torch.load(path,weights_only=True,map_location=DEVICE)
            keys_shapes={k:list(v.shape) for k,v in saved.items()}
            fresh=construct(task,args).to(DEVICE).eval()
            fresh.load_state_dict(saved,strict=True)
            self.assertEqual({k:list(v.shape) for k,v in fresh.state_dict().items()},keys_shapes)
            with torch.no_grad():actual=fresh(*call,**kwargs)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            broken=dict(saved);broken.pop(next(iter(broken)))
            with self.assertRaises(RuntimeError):fresh.load_state_dict(broken,strict=True)
            del fresh,saved,broken
        record(dict(task=task,profile='full',mode='floor',device=DEVICE,dtype='float32',backend='MATH',
            architecture=model.config.to_dict(),parameters=params,shape=shape,B=1,
            N=shape[-2],latent_expansion=model.config.num_latents[0]>shape[-2],
            forward=True,backward=True,loss='explicit synthetic MSE + coverage, NOT original task loss',
            optimizer=False,strict_state_dict=True,eval=True,checkpoint_atol=0,checkpoint_rtol=0,
            seconds=time.perf_counter()-started))
        del model,call,kwargs,actual,expected;gc.collect()


def install_task_methods():
    for task in TASKS:
        for profile in ('light','full'):
            def check(self,task=task,profile=profile):getattr(self,'run_'+profile)(task)
            setattr(FormalTaskAcceptance,'test_'+profile+'_'+task,check)
install_task_methods()


class AblationClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_formal_light_execution_counts_slot_fusion_and_storage(self):
        torch.manual_seed(9280);model=MSARLNO(resolve_profile('light'))
        counts=Counter();lengths=[];seen_fusion=[];handles=[]
        classes=(LearnedQueryDown,QueryAlignedUpCross,PairwiseAttnResFusion,LatentFFNSAFFNBlock)
        for layer in model.modules():
            if isinstance(layer,classes):
                handles.append(layer.register_forward_hook(lambda m,a,o:counts.update([type(m).__name__])))
            if isinstance(layer,_SelfAttention):handles.append(layer.register_forward_pre_hook(lambda m,a:lengths.append(a[0].shape[1])))
        def fusion_hook(m,a,out):
            torch.testing.assert_close(out,a[0]+a[1],atol=0,rtol=0);seen_fusion.append(a[0].shape[1])
        for m in model.fusions:handles.append(m.register_forward_hook(fusion_hook))
        with torch.no_grad():model(torch.randn(2,35,96))
        for h in handles:h.remove()
        self.assertEqual(counts,Counter(LearnedQueryDown=4,QueryAlignedUpCross=4,PairwiseAttnResFusion=3,LatentFFNSAFFNBlock=12))
        self.assertEqual(Counter(lengths),Counter({512:6,256:2,128:2,64:2}))
        self.assertEqual(seen_fusion,[128,256,512])
        self.assertFalse(any(isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d)) for m in model.modules()))
        self.assertFalse(any(s in (type(m).__name__+' '+n).lower() for n,m in model.named_modules() for s in ('history','cdpa','kernel','bridge')))
        named=list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(len(named),len({id(p) for _,p in named}))
        self.assertEqual(len(named),len({p.untyped_storage().data_ptr() for _,p in named}))
        record(dict(check='formal_light_execution',counts=dict(counts),SA_tokens=lengths,point_N=35,fusion_w0_exact=True,independent_storage=True))

    def test_three_fusions_source_axis_token_specific_weights_and_gradients(self):
        torch.manual_seed(9281);model=MSARLNO(resolve_profile('light'))
        for index,fusion in enumerate(model.fusions):
            e=torch.randn(2,9,96,requires_grad=True);u=torch.randn(2,9,96,requires_grad=True)
            self.assertEqual(list(dict(fusion.named_parameters())),['w'])
            y=fusion(e,u);torch.testing.assert_close(y,e+u,atol=0,rtol=0)
            ge,gu,gw=torch.autograd.grad(y.sum(),(e,u,fusion.w))
            torch.testing.assert_close(ge,torch.ones_like(e),atol=0,rtol=0)
            torch.testing.assert_close(gu,torch.ones_like(u),atol=0,rtol=0)
            self.assertGreater(gw.abs().max().item(),0)
            with torch.no_grad():fusion.w.copy_(torch.linspace(-.25,.3,96))
            # Independent per-source formula; no production forward/reference helper.
            scores=[]
            for raw in (e,u):scores.append(torch.einsum('bmd,d->bm',raw/torch.sqrt(raw.square().mean(-1,keepdim=True)+1e-6),fusion.w))
            alpha=torch.softmax(torch.stack(scores,dim=-1),dim=-1)
            self.assertEqual(alpha.shape,(2,9,2))
            torch.testing.assert_close(alpha.sum(-1),torch.ones(2,9))
            self.assertGreater((alpha[:,1:]-alpha[:,:-1]).abs().max().item(),.01)
            expected=2*(alpha[...,0,None]*e+alpha[...,1,None]*u)
            actual=fusion(e,u);torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
            self.assertFalse(torch.allclose(fusion(e,u.flip(1)),actual))
            gradients=torch.autograd.grad(actual.square().mean(),(e,u,fusion.w))
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().max()>0 for g in gradients))
        record(dict(check='three_fusions',zero_exact=True,source_axis=2,per_token=True,raw_values=True,gradients=['w','E','U'],extra_parameters=False))

    def test_boundary_aux_isolation_and_coverage_dependency(self):
        torch.manual_seed(9282)
        model=MSARLNO(resolve_profile('light',d=8,num_latents=[7,5,3,2],heads=[2,2,4,4]),output_dim=2)
        original_keys={name:set(vars(m)) for name,m in model.named_modules()}
        gradient_evidence=[]
        # Complete first backward followed by a new graph with no retain_graph.
        for index,(b,n,masked) in enumerate(((1,3,False),(2,11,True),(1,35,False))):
            x=torch.randn(b,n,8,requires_grad=True);mask=torch.ones(b,n,dtype=torch.bool)
            if masked:mask[:,-3:]=False
            mask=mask if masked else None
            cfg=MSARTrainingConfig(coverage_kappa=1,diagnostics=True)
            model.zero_grad(set_to_none=True)
            f=training_forward(model,x,training_config=cfg,valid_mask=mask)
            self.assertTrue(f.auxiliary.coverage_active)
            self.assertTrue(torch.isfinite(f.auxiliary.coverage_per_level).all())
            down_leaves=tuple(p for down in model.downs for p in (down.latent_queries,down.to_k.weight))
            down_grads=torch.autograd.grad(f.auxiliary.coverage_mean,down_leaves,retain_graph=True)
            # Every Down is connected. At initialization the deepest encoder
            # slots can be identical: its floor is then satisfied exactly, so
            # a zero gradient is valid. Test effective gradients below using
            # each real Down with independently varying source features.
            self.assertTrue(all(torch.isfinite(g).all() for g in down_grads))
            gradient_evidence.append(dict(B=b,N=n,raw=f.auxiliary.coverage_per_level.detach().tolist(),
                query_key_grad_max=[g.detach().abs().max().item() for g in down_grads]))
            forbidden=tuple(p for group in (model.decoders,model.ups,model.fusions,model.final_up,model.output) for p in group.parameters())
            grads=torch.autograd.grad(f.auxiliary.coverage_mean,forbidden,allow_unused=True,retain_graph=True)
            self.assertTrue(all(g is None for g in grads))
            total=training_objective(f.prediction.square().mean(),f)
            torch.testing.assert_close(total.total,total.pde+.01*f.auxiliary.coverage_mean,atol=0,rtol=0)
            total.total.backward();self.assertTrue(torch.isfinite(x.grad).all())
            for stats in f.auxiliary.diagnostics.values():
                for row in stats:
                    self.assertTrue(all(not v.requires_grad and v.numel()<=2*b for v in row.values()))
            prior=x
            y=torch.randn(b,n,8,requires_grad=True)
            next_f=training_forward(model,y,valid_mask=mask)
            self.assertIsNone(torch.autograd.grad(next_f.prediction.sum(),prior,allow_unused=True,retain_graph=True)[0])
            training_objective(next_f.prediction.square().mean(),next_f).total.backward()
            self.assertEqual(original_keys,{name:set(vars(m)) for name,m in model.named_modules()})
        isolated_gradients=[]
        for index,down in enumerate(model.downs):
            source=torch.randn(2,11 if index==0 else model.config.num_latents[index-1],8,requires_grad=True)
            cfg=MSARTrainingConfig(coverage_kappa=1)
            _,attention=down(source,return_aux=True,coverage=cfg)
            raw=CoverageFloorLoss(cfg)(attention)
            self.assertGreater(raw.item(),0)
            gradients=torch.autograd.grad(raw,(source,down.latent_queries,down.to_k.weight))
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().max()>0 for g in gradients))
            isolated_gradients.append([g.abs().max().item() for g in gradients])
        for cfg in (MSARTrainingConfig(coverage_kappa=0),MSARTrainingConfig(coverage_weight=0),MSARTrainingConfig(coverage_mode='off')):
            x=torch.randn(2,11,8)
            out=training_forward(model,x,training_config=cfg)
            objective=training_objective(out.prediction.square().mean(),out)
            self.assertEqual(objective.coverage_raw.item(),0)
            if not cfg.coverage_enabled:self.assertIs(objective.total,objective.pde)
            objective.total.backward()
        with self.assertRaises(ValueError):model(torch.randn(2,11,8),valid_mask=torch.zeros(2,11,dtype=torch.bool))
        record(dict(check='coverage_graph_boundaries',batches=[1,2],Ns=[3,11,35],mask=['none','partial','empty_rejected'],kappa0=True,weight0=True,off=True,
                    all_four_Down_connected_finite=True,core_gradients=gradient_evidence,
                    isolated_Down_source_query_key_nonzero_gradients=isolated_gradients,
                    decoder_fusion_aux_gradients=None,second_backward_fresh=True,retained_model_state=False))


if __name__=='__main__':unittest.main(verbosity=2)
