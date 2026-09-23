"""Capture/replay genuine pre-V4 state and gradients from an isolated git archive."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import torch


def digest(state):
    value=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if tensor is not None:value.update(name.encode());value.update(tensor.detach().contiguous().numpy().tobytes())
    return value.hexdigest()


def cases():
    from linearno_loop.config import resolve_config as v1
    from linearno_loop.v2.config import resolve_config as v2
    from linearno_loop.v3.config import resolve_config as v3
    shared=dict(topology_preset='p1_c3_r2_s1',residual_mode='sr_1_over_r',linearno_rank=4)
    overrides={'model.hidden':8,'model.heads':2,'model.ref':3,'runtime.seed':17}
    yield 'v1',v1('elasticity',options=shared,profile_overrides=overrides)
    for mode in ('round_specific','round_specific_latent'):
        yield 'v2_'+mode,v2('elasticity',options={**shared,'core_ffn_mode':mode},profile_overrides=overrides)
    yield 'v3',v3('elasticity',options=dict(architecture='operator_latent_adapter_v3',cost_profile='custom',topology_preset='custom',
        prefix_blocks=1,recurrent_core_blocks=3,loop_repeats=2,suffix_blocks=1,residual_mode='sr_1_over_r',
        hidden_width=16,heads=2,actual_M=4,latent_width=16,latent_enabled=True,
        adapter_mode='bilateral_qk_lowrank_second_visit',adapter_rank=4,adapter_alpha=4.),profile_overrides={'runtime.seed':17})


def main(action,path,report):
    torch.set_num_threads(1)
    from cdlno.linearno_loop.versioning import construct
    from cdlno.linearno.checkpoint import strict_load
    rows=[];saved=torch.load(path,weights_only=True) if action=='replay' else {}
    for name,config in cases():
        torch.manual_seed(173)
        model=construct(config)
        if action=='replay':
            assert config==saved[name]['config'];strict_load(model,saved[name]['state'])
        x=torch.linspace(-.7,.9,18).reshape(1,9,2).requires_grad_();output=model(x,None)
        output.square().mean().backward()
        grads={key:None if parameter.grad is None else parameter.grad.detach().clone() for key,parameter in model.named_parameters()}
        record=dict(config=config,state={key:value.detach().clone() for key,value in model.state_dict().items()},
                    output=output.detach(),input_grad=x.grad.clone(),gradients=grads)
        if action=='capture':saved[name]=record
        else:
            for key in ('state','gradients'):
                assert saved[name][key].keys()==record[key].keys()
                for k,v in record[key].items():assert v is None and saved[name][key][k] is None or torch.equal(v,saved[name][key][k]),(name,key,k)
            assert torch.equal(output,saved[name]['output']);assert torch.equal(x.grad,saved[name]['input_grad'])
        rows.append(dict(version=name,state_hash=digest(record['state']),gradient_hash=digest(grads),strict=True,bitwise=True))
    if action=='capture':torch.save(saved,path)
    report.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('state',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();main(a.action,a.state,a.report)
