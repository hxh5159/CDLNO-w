from collections import defaultdict
import unittest
from unittest.mock import patch

import torch
from torch import nn
from linearno_loop.v3.costs import analytic_cost
from .core_support import COST_ROWS,MODES,configuration,inputs,make


def parameter_group(name):
    if '.latent_processor.' in name:return 'latent'
    if '.adapter.' in name:return 'adapter'
    if name.startswith(('rb_','lb_')):return 'router'
    if '.ln_3.' in name or '.mlp2.' in name:return 'head'
    return 'shared_core' if name.startswith('core.') else name.split('.')[0]


class AccountingTests(unittest.TestCase):
    def test_eight_tasks_two_presets_modes_ablations_parameters_and_executed_macs(self):
        for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'):
            for preset in ('p1_c3_r2_s1','p2_c2_r2_s2'):
                for mode in MODES:
                    for latent,adapter in ((False,False),(False,True),(True,False),(True,True)):
                        with self.subTest(task=task,preset=preset,mode=mode,z=latent,a=adapter):
                            c=configuration(task,preset,mode,latent,adapter);m=make(c,dtype=torch.float32)
                            params=defaultdict(int)
                            for name,p in m.named_parameters():params[parameter_group(name)]+=p.numel()
                            expected=analytic_cost(c,batch=2,points=6)
                            for key in ('prefix','shared_core','suffix','head','latent','adapter','router'):
                                self.assertEqual(params[key],expected['parameter_groups'][key],key)
                            macs=defaultdict(int);parts=defaultdict(lambda:defaultdict(int));router=[];handles=[];current=[''];adapter_active=[False];adapter_ops=[]
                            def matrix(name,module,args,out):
                                group=parameter_group(name+'.weight')
                                n=out.numel()*(module.in_features if isinstance(module,nn.Linear) else module.weight[0].numel())
                                macs[group]+=n
                                if group in ('prefix','shared_core','suffix'):
                                    component=('point_ffn' if '.mlp.' in name else
                                               'input_projection' if '.in_project_x' in name else
                                               'q_projection' if '.to_q' in name else
                                               'k_projection' if '.to_k' in name else
                                               'v_projection' if '.to_v' in name else 'output_projection')
                                    parts[group][component]+=n
                            for name,module in m.named_modules():
                                if isinstance(module,(nn.Linear,nn.Conv2d)):
                                    handles.append(module.register_forward_hook(lambda mod,args,out,name=name:matrix(name,mod,args,out)))
                                if name.endswith('.Attn'):
                                    handles.append(module.register_forward_pre_hook(lambda mod,args,name=name:current.__setitem__(0,parameter_group(name+'.weight'))))
                                if name.endswith('.adapter'):
                                    handles.append(module.register_forward_pre_hook(lambda mod,args:adapter_active.__setitem__(0,True)))
                                    handles.append(module.register_forward_hook(lambda mod,args,out:adapter_active.__setitem__(0,False)))
                                if name.startswith(('rb_','lb_')) and hasattr(module,'source_weights'):
                                    handles.append(module.register_forward_pre_hook(lambda mod,args:router.append((len(args[0]),args[0][0].numel()))))
                            original=torch.einsum
                            original_linear=torch.nn.functional.linear
                            def linear(x,w,bias=None):
                                result=original_linear(x,w,bias)
                                if adapter_active[0]:
                                    n=result.numel()*w.shape[-1];macs['adapter']+=n
                                    adapter_ops.append(dict(input=list(x.shape),weight=list(w.shape),macs=n))
                                return result
                            def einsum(eq,*args,**kw):
                                out=original(eq,*args,**kw)
                                if eq in ('bhnm,bhnd->bhmd','bhnm,bhmd->bhnd'):
                                    q=args[0];B,h,N,M=q.shape;dh=args[1].shape[-1];n=B*h*N*M*dh
                                    macs[current[0]]+=n;parts[current[0]]['KtV' if 'bhnd->' in eq else 'Q_readout']+=n
                                return out
                            try:
                                with patch('torch.einsum',side_effect=einsum),patch('torch.nn.functional.linear',side_effect=linear):m(inputs(c,dtype=torch.float32))
                            finally:
                                for h in handles:h.remove()
                            for key in ('prefix','shared_core','suffix','head','latent','adapter'):
                                self.assertEqual(macs[key],expected['matrix_mac_groups'][key],key)
                            for group in ('prefix','shared_core','suffix'):
                                self.assertEqual(dict(parts[group]),expected['matrix_mac_parts'][group])
                            self.assertEqual([s for s,_ in router],expected['router_source_counts'])
                            self.assertEqual(len(adapter_ops),4*m.recurrent_core_blocks if adapter else 0)
                            logical=sum(2*s*n for s,n in router);actual=sum(2*s*n for s,n in router if s>1)
                            self.assertEqual(logical,expected['router_contraction_macs'])
                            saved=2*inputs(c).numel() if mode=='rb_attnres' else 0
                            self.assertEqual(logical-actual,saved)
                            COST_ROWS.append(dict(task=task,preset=preset,mode=mode,latent=latent,adapter=adapter,
                                unique_depth=m.unique_depth,executed_depth=m.executed_depth,parameters=dict(params),matrix_macs=dict(macs),
                                matrix_parts={k:dict(v) for k,v in parts.items()},matrix_flops=2*sum(macs.values()),router_sources=[s for s,_ in router],
                                router_logical_macs=logical,router_executed_macs=actual,singleton_identity_saved_macs=saved,adapter_ops=adapter_ops,status='PASS'))
