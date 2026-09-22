"""Independent full V3 loop equations; no production core/body/router imports.

LAA3's independent attention equations are reused. Residual control, LN,
point MLP, AttnRes and round trace are derived directly here.
"""
import torch
from .attention_oracle import attention_reference


def route(sources, query, scale):
    if len(sources)==1:
        return sources[0],torch.ones((*sources[0].shape[:2],1),dtype=sources[0].dtype,device=sources[0].device)
    values=torch.stack(sources,dim=2)
    normalized=values/(values.square().sum(-1,keepdim=True)/values.shape[-1]+1e-6).sqrt()
    scores=((normalized*scale)*query).sum(-1)
    exp=(scores-scores.amax(-1,keepdim=True)).exp();weights=exp/exp.sum(-1,keepdim=True)
    return (values*weights.unsqueeze(-1)).sum(2),weights


def loop_equations(anchor, *, P,C,R,S,mode,raw,native,receiver,finalize):
    """Generic arithmetic oracle; callbacks in full_reference are independent."""
    visits=[];steps=[];rounds=[];routes=[]
    def native_visit(x,group,i):
        y=native(x,group,i);visits.append((group,i,None,x,y));return y
    for i in range(P):anchor=native_visit(anchor,'prefix',i)
    x=anchor;completed=[anchor];deltas=[]
    for r in range(R):
        entry=x;terms=[]
        for i in range(C):
            visit_entry=x
            for branch in (0,1):
                if mode=='rb_attnres':
                    sources=[*completed]
                    if terms:sources.append(sum(terms[1:],terms[0]))
                    key=f'rb_receivers.{r}.{2*i+branch}'
                    h,w=receiver(sources,key);routes.append(dict(key=key,sources=tuple(sources),weights=w,h=h))
                else:h=x;sources=();w=None
                u=raw(h,i,r,branch)
                if mode=='rb_attnres':
                    terms.append(u);partial=sum(terms[1:],terms[0])
                else:x=x+u/R;partial=None
                steps.append(dict(round=r,position=i,branch=branch,h=h,u=u,partial=partial,
                                  after=None if mode=='rb_attnres' else x,scale=None if mode=='rb_attnres' else 1/R))
            if mode!='rb_attnres':visits.append(('core',i,r,visit_entry,x))
        if mode=='rb_attnres':
            completed.append(partial);rounds.append(dict(summary=partial))
        elif mode=='lb_attnres_1_over_r':
            delta=x-entry;deltas.append(delta);sources=(anchor,*deltas)
            key=f'lb_boundaries.{r}' if r<R-1 else 'lb_output'
            routed,w=receiver(sources,key);routes.append(dict(key=key,sources=sources,weights=w,h=routed))
            rounds.append(dict(entry=entry,end=x,delta=delta,routed=routed));x=routed
        else:rounds.append(dict(entry=entry,end=x))
    if mode=='rb_attnres':
        x,w=receiver(completed,'rb_output');routes.append(dict(key='rb_output',sources=tuple(completed),weights=w,h=x))
    final=x
    for i in range(S):x=native_visit(x,'suffix',i)
    return finalize(x),dict(anchor=anchor,steps=steps,rounds=rounds,routes=routes,visits=visits,final=final)


def full_reference(x,state,spec):
    P,C,R,S=(spec[k] for k in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks'))
    def norm(z,key):
        centered=z-z.mean(-1,keepdim=True)
        return centered/(centered.square().mean(-1,keepdim=True)+1e-5).sqrt()*state[key+'.weight']+state[key+'.bias']
    def linear(z,key):return z@state[key+'.weight'].T+state[key+'.bias']
    def point(z,key):
        h=linear(norm(z,key+'.ln_2'),key+'.mlp.linear_pre.0')
        return linear(.5*h*(1+torch.erf(h/(2**.5))),key+'.mlp.linear_post')
    def op(z,key,r=None):
        a={k[len(key+'.Attn.'):]:v for k,v in state.items() if k.startswith(key+'.Attn.')}
        return attention_reference(norm(z,key+'.ln_1'),a,variant=spec['variant'],heads=spec['heads'],
            round_index=r,latent=r is not None and spec['latent_enabled'],
            adapter=r is not None and spec['adapter_mode']!='none',alpha=spec['adapter_alpha'],
            grid=(spec['grid_height'],spec['grid_width']))[0]
    def raw(z,i,r,branch):
        key=f'core.{i}.block';return op(z,key,r) if branch==0 else point(z,key)
    def native(z,group,i):
        key=f'{group}.{i}.block';z=z+op(z,key);return z+point(z,key)
    def receiver(sources,key):return route(sources,state[key+'.query'],state[key+'.norm_scale'])
    def finalize(z):
        key=f'suffix.{S-1}.block';return linear(norm(z,key+'.ln_3'),key+'.mlp2')
    return loop_equations(x,P=P,C=C,R=R,S=S,mode=spec['residual_mode'],raw=raw,native=native,receiver=receiver,finalize=finalize)
