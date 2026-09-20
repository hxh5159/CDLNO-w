"""Reproduce the original FP32 math-oracle AdamW sensitivity, no data/GPU."""
import json
from pathlib import Path
import torch

from cdlno.linearno_loop.construction import build_from_config
from loop_linearno.sr_support import config
from loop_linearno.sr_oracle import core_reference


def main():
    core=build_from_config(config('car','p2_c2_r2_s2')).loop
    gen=torch.Generator().manual_seed(44)
    x=torch.randn(2,15,8,generator=gen,requires_grad=True)
    rx=x.detach().clone().requires_grad_()
    state={k:v.detach().clone().requires_grad_() for k,v in core.named_parameters()}
    actual=core(x)
    reference,_=core_reference(rx,state,P=2,C=2,R=2,S=2,variant='shapenet',heads=2,H=3,W=5)
    target=torch.randn(actual.shape,generator=gen)
    (actual-target).square().mean().backward();(reference-target).square().mean().backward()
    key='core.0.block.mlp.linear_pre.0.weight';p=dict(core.named_parameters())[key];q=state[key]
    initial=p.detach().clone();a=p.grad.detach().clone();b=q.grad.detach().clone()
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    torch.optim.AdamW(core.parameters(),lr=.001).step()
    torch.optim.AdamW(state.values(),lr=.001).step()
    diff=(p-q).abs();idx=tuple(int(v) for v in torch.unravel_index(diff.argmax(),diff.shape))
    first_step=lambda grad:initial*(1-.001*.01)-.001*grad/(grad.abs()+1e-8)
    predicted=(first_step(a)-first_step(b)).abs()[idx]
    torch.testing.assert_close(diff[idx],predicted,atol=0,rtol=0)
    result=dict(case='car/P2/FP32',field=key,index=idx,initial=initial[idx].item(),
        native_gradient=a[idx].item(),oracle_gradient=b[idx].item(),gradient_abs_error=(a-b).abs()[idx].item(),
        native_after=p[idx].item(),oracle_after=q[idx].item(),step_abs_error=diff[idx].item(),
        scalar_formula_step_difference=predicted.item(),forward_max=(actual-reference).abs().max().item(),
        gradient_range=[a.min().item(),a.max().item()],
        explanation='FP32 explicit LN/erf/attention summation roundoff amplified by AdamW g/(abs(g)+eps) near a small gradient; no tolerance changed')
    (Path(__file__).parent/'adamw-oracle-roundoff.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
