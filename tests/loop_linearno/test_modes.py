import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from cdlno.linearno_loop.attnres import PointDepthAttnRes
from cdlno.linearno_loop.construction import build_from_config
from linearno_loop.config import resolve_config
from linearno_loop.contracts import RESIDUAL_MODES,seal,digest
from linearno_loop.schema import read_metadata,restore_config,write_metadata
from loop_linearno.lb_support import mode_config,excite
from loop_linearno.sr_support import inputs,TOPOLOGIES
from loop_linearno.support import metadata,ROOT

TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')


def routers(model):return [(n,m) for n,m in model.named_modules() if isinstance(m,PointDepthAttnRes)]


class ModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.counts=[];cls.costs=[];cls.roundtrips=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL5_MODES_REPORT'):
            Path(path).write_text(json.dumps(dict(counts=cls.counts,costs=cls.costs,roundtrips=cls.roundtrips),indent=2)+'\n')

    def test_all_tasks_three_modes_common_weights_rng_and_real_parameter_counts(self):
        for task in TASKS:
            for preset in TOPOLOGIES[:2]:
                baseline=None;before=None
                for mode in RESIDUAL_MODES:
                    # Actual full profile constructors, no full-N forward.
                    c=resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode))
                    path,name=c['model_spec']['class_path'].rsplit('.',1);cls=getattr(importlib.import_module(path),name)
                    torch.manual_seed(672);m=cls(**c['model_spec']['constructor_kwargs'])
                    common={n:v for n,v in m.state_dict().items() if not n.startswith(('loop.rb_','loop.lb_'))}
                    if baseline is None:baseline=common;before=torch.get_rng_state().clone()
                    else:
                        self.assertTrue(torch.equal(before,torch.get_rng_state()))
                        self.assertEqual(set(common),set(baseline))
                        for n,v in common.items():torch.testing.assert_close(v,baseline[n],atol=0,rtol=0)
                    H=m.loop.core[0].block.Attn.dim;C=m.loop.recurrent_core_blocks;R=m.loop.loop_repeats
                    expected=0 if mode=='sr_1_over_r' else 2*C*R+1 if mode=='rb_attnres' else R
                    rs=routers(m);self.assertEqual(len(rs),expected)
                    added=sum(p.numel() for _,r in rs for p in r.parameters());self.assertEqual(added,2*H*expected)
                    all_params=list(m.named_parameters(remove_duplicate=False))
                    self.assertEqual(len(all_params),len({id(p) for _,p in all_params}))
                    for _,r in rs:
                        torch.testing.assert_close(r.query,torch.zeros_like(r.query),atol=0,rtol=0)
                        torch.testing.assert_close(r.norm_scale,torch.ones_like(r.norm_scale),atol=0,rtol=0)
                    wrong='loop.lb_' if mode=='rb_attnres' else 'loop.rb_'
                    self.assertFalse(any(n.startswith(wrong) for n in m.state_dict()))
                    self.counts.append(dict(task=task,preset=preset,mode=mode,hidden=H,actual_rank=c['loop_spec']['resolved_rank'],
                        receivers=len(rs),added_parameters=added,unique_parameters=sum(p.numel() for p in m.parameters()),
                        executed_depth=m.loop.executed_depth,unique_depth=m.loop.unique_depth))

    def test_three_modes_full_steps_metadata_and_fresh_process_strict(self):
        archives=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for task in TASKS:
                for mode in RESIDUAL_MODES:
                    for preset in TOPOLOGIES[:2]:
                        c=mode_config(task,preset,mode);m=build_from_config(c);excite(m.loop);args=inputs(c)
                        opt=torch.optim.AdamW(m.parameters(),lr=.001)
                        out=m(*args);out.square().mean().backward()
                        for p in m.parameters():
                            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
                        opt.step();m.eval();expected=m(*args).detach()
                        restored=build_from_config(json.loads(json.dumps(c))).eval()
                        restored.load_state_dict(m.state_dict(),strict=True)
                        torch.testing.assert_close(restored(*args),expected,atol=0,rtol=0)
                        self.roundtrips.append(dict(task=task,mode=mode,preset=preset,optimizer='AdamW',exact_reload=True))
                        if task in ('darcy','airfrans','car') and preset==TOPOLOGIES[0]:
                            p=root/(task+'_'+mode);p.mkdir()
                            meta=metadata(c)
                            # Real source fingerprints; protocol fixture stays
                            # explicitly synthetic, not a production resume.
                            sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                for base in ('linearno_loop','cdlno/linearno_loop') for f in sorted((ROOT/base).glob('*.py'))}
                            meta['provenance_spec'].update(code_version='LL5-synthetic-state-roundtrip',
                                source_sha256=digest(sources),normalized_patch_sha256=digest({'scope':'LL5 fixture source set','sources':sources}))
                            meta=seal(meta,'metadata_hash');write_metadata(p/'metadata.json',meta)
                            torch.save(m.state_dict(),p/'weights.pt');torch.save(expected,p/'expected.pt');archives.append(str(p))
            # Single entirely fresh interpreter reads all nine metadata files
            # before each construction/load; no pickle model / strict fallback.
            code='''import sys,torch
from pathlib import Path
from linearno_loop.schema import read_metadata,restore_config
from cdlno.linearno_loop.construction import build_from_config
from loop_linearno.sr_support import inputs
for item in sys.argv[1:]:
 p=Path(item);c=restore_config(read_metadata(p/'metadata.json'))['config']
 m=build_from_config(c).eval();m.load_state_dict(torch.load(p/'weights.pt',weights_only=True),strict=True)
 torch.testing.assert_close(m(*inputs(c)),torch.load(p/'expected.pt',weights_only=True),atol=0,rtol=0)
print('9/9 metadata-first strict reloads exact')
'''
            proc=subprocess.run([sys.executable,'-B','-c',code,*archives],cwd=directory,capture_output=True,text=True,
                env={**os.environ,'PYTHONPATH':str(ROOT/'tests')+':'+str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'})
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertIn('9/9',proc.stdout)

    def test_mode_conflicts_fail_before_load_and_do_not_add_other_routers(self):
        states={};configs={}
        for mode in RESIDUAL_MODES:
            c=mode_config(mode=mode);m=build_from_config(c);states[mode]=m.state_dict();configs[mode]=c
            meta=metadata(c)
            for wrong in RESIDUAL_MODES:
                if wrong==mode:continue
                with patch('torch.load',side_effect=AssertionError('weights read early')),patch(
                        'cdlno.linearno_loop.construction.importlib.import_module',side_effect=AssertionError('model imported early')):
                    with self.assertRaisesRegex(ValueError,'residual_mode'):restore_config(meta,explicit={'residual_mode':wrong})
            with self.assertRaisesRegex(ValueError,'strict'):restore_config(meta,strict=False)
            for invalid in (True,False,None,0,[],{},'unknown','sr_1_over_r+rb_attnres'):
                kw={**c['model_spec']['constructor_kwargs'],'residual_mode':invalid}
                before=torch.get_rng_state().clone()
                with self.assertRaises(ValueError):type(m)(**kw)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
            with self.assertRaises(TypeError):type(m)(**c['model_spec']['constructor_kwargs'],lb_enabled=True)
        for mode in RESIDUAL_MODES:
            m=build_from_config(configs[mode])
            for wrong,state in states.items():
                if mode!=wrong:
                    with self.assertRaises(RuntimeError):m.load_state_dict(state,strict=True)
        lb=build_from_config(configs['lb_attnres_1_over_r']).loop
        lb.lb_output=lb.lb_boundaries[0]
        with self.assertRaisesRegex(ValueError,'independent'):lb(torch.zeros(2,15,8))
        lb=build_from_config(configs['lb_attnres_1_over_r']).loop
        lb.rb_output=PointDepthAttnRes(8)
        with self.assertRaisesRegex(ValueError,'RB receiver'):lb(torch.zeros(2,15,8))
        rb=build_from_config(configs['rb_attnres']).loop;rb.lb_output=PointDepthAttnRes(8)
        with self.assertRaisesRegex(ValueError,'only LB'):rb(torch.zeros(2,15,8))

    def test_measured_extra_contractions_and_elementwise_work(self):
        # Exact shapes observed at reduction kernels; don't report the absence
        # of matmul inside AR as zero compute. Dense backbone MACs separately.
        class Work(TorchDispatchMode):
            def __init__(self):self.active=None;self.ops={};self.contractions=[];self.dense=0;self.point_subtractions=[]
            def __torch_dispatch__(self,fn,types,args=(),kwargs=None):
                result=fn(*args,**(kwargs or {}));name=str(fn)
                if self.active is not None:
                    row=self.ops.setdefault(name,dict(calls=0,output_elements=0));row['calls']+=1
                    if isinstance(result,torch.Tensor):row['output_elements']+=result.numel()
                    if name=='aten.sum.dim_IntList' and args[0].ndim==4:
                        dims=tuple(args[1])
                        if dims in ((-1,),(0,)):
                            self.contractions.append(dict(receiver=self.active,shape=list(args[0].shape),axis=dims[0],macs=args[0].numel()))
                elif name=='aten.bmm.default':self.dense+=args[0].shape[0]*args[0].shape[1]*args[0].shape[2]*args[1].shape[2]
                elif name=='aten.sub.Tensor' and isinstance(result,torch.Tensor) and tuple(result.shape)==(2,15,8):
                    self.point_subtractions.append(result.numel())
                return result
        for preset in TOPOLOGIES[:2]:
            dense_base=None
            for mode in RESIDUAL_MODES:
                c=mode_config(preset=preset,mode=mode);m=build_from_config(c).eval();counter=Work();handles=[];sources=[]
                def linear(mod,args,out):
                    if isinstance(mod,nn.Linear):counter.dense+=out.numel()*mod.in_features
                    else:counter.dense+=out.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
                for mod in m.modules():
                    if isinstance(mod,(nn.Linear,nn.Conv2d)):handles.append(mod.register_forward_hook(linear))
                for name,mod in routers(m):
                    def enter(mod,args,name=name):counter.active=name;sources.append(len(args[0]))
                    def leave(*args):counter.active=None
                    handles.extend((mod.register_forward_pre_hook(enter),mod.register_forward_hook(leave)))
                try:
                    with torch.no_grad(),counter:m(*inputs(c))
                finally:
                    for h in handles:h.remove()
                measured=sum(row['macs'] for row in counter.contractions)
                expected=2*2*15*8*sum(s for s in sources if s>1)
                self.assertEqual(measured,expected)
                if dense_base is None:dense_base=counter.dense
                else:self.assertEqual(counter.dense,dense_base)
                self.assertEqual(len(counter.contractions),2*sum(s>1 for s in sources))
                self.assertEqual(counter.point_subtractions,[240,240] if mode=='lb_attnres_1_over_r' else [])
                self.costs.append(dict(preset=preset,mode=mode,B=2,N=15,hidden=8,source_counts=sources,
                    measured_dense_backbone_macs=counter.dense,measured_router_contraction_macs=measured,
                    measured_total_matrix_and_contraction_macs=counter.dense+measured,
                    receiver_reductions=counter.contractions,router_operator_inventory=counter.ops,
                    actual_delta_subtraction_elements=counter.point_subtractions,
                    note='Contraction MACs count input terms in actual score/value multiply-sums; RMS/softmax/elementwise separately listed, not included in 2*MAC FLOPs. No real timing.'))


if __name__=='__main__':unittest.main()
