"""LL4 test-only configs, live hooks and routing traces; no runtime observer."""
from contextlib import ExitStack
from unittest.mock import patch

import torch

from linearno_loop.config import resolve_config
from loop_linearno.sr_support import config


def rb_config(task='darcy',preset='p1_c3_r2_s1',**kw):
    c=config(task,preset,**kw);r=c['request']
    return resolve_config(task,options={**r['options'],'residual_mode':'rb_attnres'},profile_overrides=r['profile_overrides'])


def excite_routers(core):
    gen=torch.Generator().manual_seed(440)
    with torch.no_grad():
        for name,p in core.named_parameters():
            if name.startswith(('rb_receivers.','rb_output.')):
                p.copy_(torch.randn(p.shape,generator=gen,dtype=p.dtype)*.2+(1 if name.endswith('norm_scale') else 0))


class Trace:
    """Capture the actual weights and live sources with removable test hooks."""
    def __init__(self,core):
        self.core=core;self.routes=[];self.raw=[];self.norm_inputs=[];self.events=[];self.qkv=[]
        self.head=[];self.handles=[];self.stack=ExitStack()

    def __enter__(self):
        for name,router in [(f'{r}.{j}',v) for r,row in enumerate(self.core.rb_receivers) for j,v in enumerate(row)]+[('out',self.core.rb_output)]:
            actual=router._weights_and_values
            def weights(sources,actual=actual):
                result=actual(sources);self.routes[-1]['weights']=result[0].permute(1,2,0)
                return result
            self.stack.enter_context(patch.object(router,'_weights_and_values',side_effect=weights))
            def before(m,args,name=name):
                self.events.append('route.'+name)
                self.routes.append(dict(name=name,sources=args[0]))
            def after(m,args,out):
                row=self.routes[-1];row['h']=out
                if len(args[0])==1:row['weights']=torch.ones((*out.shape[:2],1),device=out.device,dtype=out.dtype)
            self.handles.extend((router.register_forward_pre_hook(before),router.register_forward_hook(after)))
        for i,physical in enumerate(self.core.core):
            b=physical.block
            for kind,norm,branch in [('operator',b.ln_1,b.Attn),('mlp',b.ln_2,b.mlp)]:
                def before(m,args,i=i,kind=kind):
                    self.events.append(f'{i}.{kind}.ln');self.norm_inputs.append(args[0])
                def after(m,args,out,i=i,kind=kind):
                    self.events.append(f'{i}.{kind}.raw');self.raw.append(out)
                self.handles.extend((norm.register_forward_pre_hook(before),branch.register_forward_hook(after)))
            for factor in ('to_q','to_k','to_v'):
                def project(m,args,out,i=i,factor=factor):self.qkv.append((i,factor,out))
                self.handles.append(getattr(b.Attn,factor).register_forward_hook(project))
        for name in ('ln_3','mlp2'):
            self.handles.append(getattr(self.core.suffix[-1].block,name).register_forward_hook(
                lambda m,a,y,name=name:self.head.append(name)))
        return self

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.stack.close()

    def partial_after(self,index):
        r,j=divmod(index,2*self.core.recurrent_core_blocks)
        if j+1<2*self.core.recurrent_core_blocks:return self.routes[index+1]['sources'][-1]
        return self.routes[index+1]['sources'][r+1]
