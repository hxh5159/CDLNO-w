"""LL4 independent equations: point-source routing plus complete PCRS core.

No production loop/body/receiver/block forward is called. Raw branches use
explicit LN/erf GELU and the already independent LinearNO factor equations.
"""
import torch

from linearno.attention_reference import reference as attention_reference
from loop_linearno.point_attnres_oracle import point_attnres


def rb_reference(x,state,*,P,C,R,S,variant,heads,H,W):
    trace=[];visits=[]

    def norm(z,key):
        centered=z-z.mean(-1,keepdim=True)
        return centered/(centered.square().mean(-1,keepdim=True)+1e-5).sqrt()*state[key+'.weight']+state[key+'.bias']

    def linear(z,key):return z@state[key+'.weight'].T+state[key+'.bias']

    def raw(z,key,kind):
        if kind=='operator':
            a={k[len(key+'.Attn.'):]:v for k,v in state.items() if k.startswith(key+'.Attn.')}
            return attention_reference(norm(z,key+'.ln_1'),a,heads=heads,variant=variant,H=H,W=W)[0]
        v=linear(norm(z,key+'.ln_2'),key+'.mlp.linear_pre.0')
        return linear(.5*v*(1+torch.erf(v/(2**.5))),key+'.mlp.linear_post')

    def native(z,group,i):
        key=f'{group}.{i}.block';z=z+raw(z,key,'operator');z=z+raw(z,key,'mlp')
        visits.append(z)
        return z

    for i in range(P):x=native(x,'prefix',i)
    anchor=x;completed=[anchor]
    for r in range(R):
        terms=[]
        for j in range(2*C):
            # Build the raw partial independently from the terms accumulated
            # so far. No receiver output/anchor is a term.
            sources=list(completed)
            if terms:sources.append(sum(terms[1:],terms[0]))
            key=f'rb_receivers.{r}.{j}'
            h,alpha=point_attnres(sources,state[key+'.query'],state[key+'.norm_scale'])
            u=raw(h,f'core.{j//2}.block','operator' if j%2==0 else 'mlp')
            terms.append(u);partial=sum(terms[1:],terms[0])
            trace.append(dict(sources=tuple(sources),weights=alpha,h=h,u=u,partial=partial,round=r,sublayer=j))
        completed.append(partial)
    x,weights=point_attnres(completed,state['rb_output.query'],state['rb_output.norm_scale'])
    final_route=dict(sources=tuple(completed),weights=weights,h=x)
    for i in range(S):x=native(x,'suffix',i)
    key=f'suffix.{S-1}.block'
    output=linear(norm(x,key+'.ln_3'),key+'.mlp2');visits[-1]=output
    return output,dict(anchor=anchor,steps=trace,completed=tuple(completed),final_route=final_route,visits=visits)
