"""Independent full SR equations; never invokes loop/body/attention forward."""
import torch
from linearno.attention_reference import reference as attention_reference


def core_reference(x, state, *, P, C, R, S, variant, heads, H, W):
    """Explicit physical-key reuse, independent LN/MLP/operator equations."""
    trace=[]
    def norm(z,key):
        z=z-z.mean(-1,keepdim=True)
        return z/torch.sqrt(z.square().mean(-1,keepdim=True)+1e-5)*state[key+'.weight']+state[key+'.bias']
    def linear(z,key):return z@state[key+'.weight'].T+state[key+'.bias']
    def mlp(z,key):
        z=linear(z,key+'.linear_pre.0')
        z=.5*z*(1+torch.erf(z/(2**.5)))
        return linear(z,key+'.linear_post')
    def visit(z,group,index,scale):
        key=f'{group}.{index}.block'
        attn={k[len(key+'.Attn.'):]:v for k,v in state.items() if k.startswith(key+'.Attn.')}
        out,_=attention_reference(norm(z,key+'.ln_1'),attn,heads=heads,variant=variant,H=H,W=W)
        z=z+scale*out
        z=z+scale*mlp(norm(z,key+'.ln_2'),key+'.mlp')
        trace.append(z)
        return z
    for i in range(P):x=visit(x,'prefix',i,1)
    for _ in range(R):
        for i in range(C):x=visit(x,'core',i,1/R)
    for i in range(S):x=visit(x,'suffix',i,1)
    key=f'suffix.{S-1}.block'
    output=linear(norm(x,key+'.ln_3'),key+'.mlp2')
    trace[-1]=output
    return output,trace
