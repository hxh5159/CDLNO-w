"""Independent LL5 full-core equations; no production loop/block forward."""
import torch
from linearno.attention_reference import reference as attention_reference
from loop_linearno.point_attnres_oracle import point_attnres


def lb_reference(x,state,*,P,C,R,S,variant,heads,H,W):
    rounds=[];visits=[]

    def norm(z,key):
        z=z-z.mean(-1,keepdim=True)
        return z/(z.square().mean(-1,keepdim=True)+1e-5).sqrt()*state[key+'.weight']+state[key+'.bias']

    def linear(z,key):return z@state[key+'.weight'].T+state[key+'.bias']

    def visit(z,group,i,scale):
        key=f'{group}.{i}.block'
        a={k[len(key+'.Attn.'):]:v for k,v in state.items() if k.startswith(key+'.Attn.')}
        op=attention_reference(norm(z,key+'.ln_1'),a,heads=heads,H=H,W=W,variant=variant)[0]
        z=z+scale*op
        v=linear(norm(z,key+'.ln_2'),key+'.mlp.linear_pre.0')
        v=.5*v*(1+torch.erf(v/(2**.5)))
        z=z+scale*linear(v,key+'.mlp.linear_post')
        visits.append(z)
        return z

    for i in range(P):x=visit(x,'prefix',i,1)
    anchor=x;deltas=[]
    for r in range(R):
        entry=x
        for i in range(C):x=visit(x,'core',i,1/R)
        end=x;delta=end-entry;deltas.append(delta)
        key=f'lb_boundaries.{r}' if r<R-1 else 'lb_output'
        sources=(anchor,*deltas)
        x,weights=point_attnres(sources,state[key+'.query'],state[key+'.norm_scale'])
        rounds.append(dict(entry=entry,end=end,delta=delta,sources=sources,weights=weights,routed=x))
    final=x
    for i in range(S):x=visit(x,'suffix',i,1)
    key=f'suffix.{S-1}.block'
    output=linear(norm(x,key+'.ln_3'),key+'.mlp2');visits[-1]=output
    return output,dict(anchor=anchor,rounds=rounds,final=final,visits=visits)
