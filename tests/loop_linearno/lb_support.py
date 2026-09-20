"""LL5 test-only live tracing; never installed into production forward."""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from linearno_loop.config import resolve_config
from loop_linearno.sr_support import config


def mode_config(task='darcy',preset='p1_c3_r2_s1',mode='lb_attnres_1_over_r',**kwargs):
    c=config(task,preset,**kwargs);r=c['request']
    return resolve_config(task,options={**r['options'],'residual_mode':mode},profile_overrides=r['profile_overrides'])


def excite(core):
    g=torch.Generator().manual_seed(741)
    with torch.no_grad():
        for n,p in core.named_parameters():
            if n.startswith(('lb_','rb_')):
                p.copy_(torch.randn(p.shape,generator=g,dtype=p.dtype)*.2+(1 if n.endswith('norm_scale') else 0))


class LBTrace:
    def __init__(self,core):
        self.core=core;self.entries=[];self.ends=[];self.routes=[];self.visits=[];self.events=[]
        self.handles=[];self.stack=ExitStack()

    def __enter__(self):
        for group in ('prefix','core','suffix'):
            for i,block in enumerate(getattr(self.core,group)):
                def before(m,args,group=group,i=i):
                    self.events.append(f'{group}.{i}')
                    if group=='core' and i==0:self.entries.append(args[0])
                def after(m,args,out,group=group,i=i):
                    self.visits.append(out)
                    if group=='core' and i==len(self.core.core)-1:self.ends.append(out)
                self.handles.extend((block.register_forward_pre_hook(before),block.register_forward_hook(after)))
        for i,router in enumerate([*self.core.lb_boundaries,self.core.lb_output]):
            actual=router._weights_and_values
            def weights(sources,actual=actual):
                result=actual(sources);self.routes[-1]['weights']=result[0].permute(1,2,0);return result
            self.stack.enter_context(patch.object(router,'_weights_and_values',side_effect=weights))
            def before(m,args,i=i):
                self.events.append('route.'+str(i));self.routes.append(dict(sources=args[0]))
            def after(m,args,out):self.routes[-1]['routed']=out
            self.handles.extend((router.register_forward_pre_hook(before),router.register_forward_hook(after)))
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.stack.close()
