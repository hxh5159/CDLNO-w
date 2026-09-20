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
from loop_linearno.sr_support import config, counts, inputs, TOPOLOGIES


TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')


class SRAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.profile_rows=[];cls.hook_rows=[]

    @classmethod
    def tearDownClass(cls):
        if path:=os.environ.get('LOOP_LL3_ACCOUNTING_REPORT'):
            Path(path).write_text(json.dumps(dict(profile='paper_table8_on_release_model',
                profile_rows=cls.profile_rows,hook_rows=cls.hook_rows,
                limits='B=1 full-profile counts use canonical N, constructor only; no real data/forward/timing. '
                       'Executed parameter uses count shared and dead slots per visit, not extra storage. '
                       'Dense MACs exclude nonlinear/norm/softmax/bias/residual/position/sinusoid costs.'),indent=2)+'\n')

    def test_full_profile_unique_parameters_eight_tasks_presets_rank_controls(self):
        for task in TASKS:
            for preset in TOPOLOGIES[:2]:
                for multiplier in (1,2):
                    with self.subTest(task=task,preset=preset,multiplier=multiplier):
                        c=resolve_config(task,options=dict(topology_preset=preset,
                            residual_mode='sr_1_over_r',rank_multiplier=multiplier))
                        model=build_from_config(c);expected=counts(c)
                        actual=sum(p.numel() for p in model.parameters())
                        self.assertEqual(actual,expected['unique_parameters'])
                        # Multiplicity accounting, not allocation or measured FLOPs.
                        core_parameters=sum(p.numel() for b in model.loop.core for p in b.parameters())
                        executed=actual+(model.loop.loop_repeats-1)*core_parameters
                        self.assertEqual(executed,expected['executed_parameter_uses'])
                        self.assertLess(actual,executed)
                        self.assertLess(expected['one_visit_per_physical_block_macs'],expected['executed_macs'])
                        self.profile_rows.append(dict(task=task,topology=preset,rank_multiplier=multiplier,
                            actual_rank=c['loop_spec']['resolved_rank'],measured_unique_parameters=actual,
                            **expected))

    def test_dense_mac_hooks_actual_executed_and_unique_visits(self):
        for task in TASKS:
            for preset in TOPOLOGIES:
                with self.subTest(task=task,preset=preset):
                    c=config(task,preset);model=build_from_config(c).eval();args=inputs(c)
                    B,N=(1,13) if task in ('airfrans','car') else (2,15)
                    expected=counts(c,B=B,N=N);observed={};ctx=[None];handles=[]
                    def matrix(mod,inputs,output):
                        if isinstance(mod,nn.Linear):value=output.numel()*mod.in_features
                        else:value=output.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
                        observed.setdefault(ctx[0],[]).append(value)
                    for mod in model.modules():
                        if isinstance(mod,(nn.Linear,nn.Conv2d)):handles.append(mod.register_forward_hook(matrix))
                    # Each visit gets its own list. Group repeats afterward to
                    # count one physical visit without hiding repeated matmuls.
                    visits={};physical={}
                    for group in ('prefix','core','suffix'):
                        for i,block in enumerate(getattr(model.loop,group)):
                            key=f'{group}.{i}'
                            def enter(mod,args,key=key):
                                j=visits.get(key,0);visits[key]=j+1;ctx[0]=key+':'+str(j)
                                physical[ctx[0]]=key
                            def leave(*args):ctx[0]=None
                            handles.append(block.register_forward_pre_hook(enter))
                            handles.append(block.register_forward_hook(leave))
                    original=torch.einsum
                    def einsum(eq,*values,**kw):
                        if eq in ('bhnm,bhnd->bhmd','bhnm,bhmd->bhnd'):
                            q=values[0];v=values[1]
                            observed.setdefault(ctx[0],[]).append(q.numel()*v.shape[-1])
                        return original(eq,*values,**kw)
                    try:
                        with torch.no_grad(),patch('torch.einsum',side_effect=einsum):model(*args)
                    finally:
                        for handle in handles:handle.remove()
                    actual=sum(sum(values) for values in observed.values())
                    one_visit=sum(sum(values) for key,values in observed.items() if key is None or key.endswith(':0'))
                    self.assertEqual(actual,expected['executed_macs'])
                    self.assertEqual(one_visit,expected['one_visit_per_physical_block_macs'])
                    self.assertEqual(sum(visits.values()),expected['executed_depth'])
                    self.hook_rows.append(dict(task=task,topology=preset,measured_executed_macs=actual,
                        measured_one_visit_macs=one_visit,visits=visits,**expected))

    def test_factory_reused_modules_and_illegal_core_head_rejected(self):
        from loop_linearno.test_block_body import make_block
        kw=dict(prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=2,suffix_blocks=1,residual_mode='sr_1_over_r')
        reused=make_block('plain',False)
        with self.assertRaisesRegex(ValueError,'must be distinct'):
            LinearNOLoopCore(**kw,block_factory=lambda *,last_layer:make_block('plain',True) if last_layer else reused)
        with self.assertRaisesRegex(ValueError,'only final suffix'):
            LinearNOLoopCore(**kw,block_factory=lambda *,last_layer:make_block('plain',True))
        model=build_from_config(config())
        model.loop.core[0].block.Attn.rank=5
        with self.assertRaisesRegex(ValueError,'matching LinearNO'):model.loop(torch.zeros(1,15,8))


if __name__=='__main__':unittest.main()
