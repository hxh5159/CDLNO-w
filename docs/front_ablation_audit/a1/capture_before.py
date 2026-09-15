"""Run against the isolated PRE-A1 source only; never import experiment entries."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import cdlno
from cdlno import CDLNO, CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdlno.checkpoint import save_sidecar
from cdlno.modules import LRSAFrontBlock

p = argparse.ArgumentParser()
p.add_argument('baseline', type=Path)
p.add_argument('group', choices=('core', 'car', 'airfrans'))
a = p.parse_args()
assert Path(cdlno.__file__).resolve().is_relative_to(a.baseline / 'source')
assert 'front_latent_mode' not in CDLNOArchitectureConfig.__dataclass_fields__
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SEED = 20260914
out = a.baseline / a.group
out.mkdir(exist_ok=False)
records = []


def capture(name, model, sample, description):
    model.eval()
    params = dict(model.named_parameters())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cotangents = None

    def once():
        nonlocal cotangents
        model.zero_grad(set_to_none=True)
        x = sample.detach().clone().requires_grad_()
        if a.group == 'car':
            from torch_geometric.data import Data, Batch
            data = Batch.from_data_list([Data(x=x, y=torch.zeros(x.shape[0], 4))])
            prediction = model((data, Data()))
        elif a.group == 'airfrans':
            from torch_geometric.data import Data, Batch
            data = Batch.from_data_list([Data(x=x, pos=x[:, :2], y=torch.zeros(x.shape[0], 4))])
            prediction = model(data)
        else:
            prediction = model(x)
        ys = prediction if isinstance(prediction, tuple) else (prediction,)
        if cotangents is None:
            g = torch.Generator().manual_seed(909)
            cotangents = tuple(torch.randn(y.shape, generator=g) for y in ys)
        sum((y * cot).sum() for y, cot in zip(ys, cotangents)).backward()
        grads = {k: v.grad.detach().clone() for k, v in params.items()}
        grads['input'] = x.grad.detach().clone()
        assert all(torch.isfinite(v).all() for v in grads.values())
        return tuple(y.detach().clone() for y in ys), grads

    with sdpa_kernel(SDPBackend.MATH):
        first, grads = once()
        second, repeated = once()
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert all(torch.equal(grads[k], repeated[k]) for k in grads)
    directory = out / name
    directory.mkdir()
    fixture = dict(description=description, input=sample, output=first,
                   cotangents=cotangents, gradients=grads, state=state,
                   buffers={k: v.detach().clone() for k,v in model.named_buffers()},
                   state_keys=list(state), seed=SEED, atol=0., rtol=0.)
    if hasattr(model, 'config'):
        fixture['config'] = model.config.to_dict()
        save_sidecar(directory/'architecture.json', model.config, CDLNORuntimeConfig(device='cpu'))
    torch.save(fixture, directory/'fixture.pt')
    torch.save(model, directory/'legacy-model.pt')
    if a.group == 'airfrans':
        torch.save([model, copy.deepcopy(model)], directory/'legacy-list.pt')
    files = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in directory.iterdir()}
    records.append(dict(name=name, description=description, files=files,
                        state_keys=list(state), gradient_zero=[k for k,v in grads.items() if not torch.count_nonzero(v)],
                        repeat_max_output_error=0., repeat_max_gradient_error=0.))


g = torch.Generator().manual_seed(101)
if a.group == 'core':
    for structured in (False, True):
        torch.manual_seed(SEED)
        options = dict(dim=8, heads=2, num_latents=4, structured=structured,
                       grid_shape=(5,7) if structured else None, dropout=0.)
        capture('front_'+str(structured), LRSAFrontBlock(**options),
                torch.randn(2,35 if structured else 11,8,generator=g), dict(kind='front', kwargs=options))
    for structured, f, mode in [(s,2,m) for s in (False,True) for m in ('off','entry','every_block')] + [(False,f,'entry') for f in (0,3,4)]:
        torch.manual_seed(SEED)
        cfg = CDLNOArchitectureConfig(L=8,F=f,M=4,d_model=8,num_heads=2,output_dim=3,
              cdpa_mode=mode,structured=structured,grid_shape=(5,7) if structured else None)
        capture(f'core_{structured}_{f}_{mode}', CDLNO(cfg),
                torch.randn(2,35 if structured else 11,8,generator=g), dict(kind='core', nonzero_scorer=False))
    torch.manual_seed(SEED)
    model = CDLNO(cfg := CDLNOArchitectureConfig(L=8,F=2,M=4,d_model=8,num_heads=2,output_dim=3,cdpa_mode='every_block'))
    with torch.no_grad():
        for fusion in model.cdpa_at.values():
            fusion.w.copy_(torch.linspace(-.2,.2,8))
    capture('core_nonzero', model, torch.randn(2,11,8,generator=g), dict(kind='core', nonzero_scorer=True))
else:
    if a.group == 'car':
        from models.CDLNO import Model
    else:
        from cdlno.airfrans import AirfRANSModel as Model
    for n in (11,17):
        torch.manual_seed(SEED)
        kwargs=dict(n_hidden=8,n_head=2,slice_num=4,n_layers=8,front_blocks=2)
        capture(str(n),Model(**kwargs),torch.randn(n,7,generator=g),dict(kind=a.group,kwargs=kwargs))

(out/'manifest.json').write_text(json.dumps(dict(
    source=str(Path(cdlno.__file__).resolve()), cwd=os.getcwd(), python=sys.version.split()[0],
    torch=torch.__version__, cuda_build=torch.version.cuda, device='cpu', dtype='float32',
    sdpa='MATH', threads=1, deterministic=True, dropout=0, eval=True, atol=0., rtol=0.,
    capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records), indent=2)+'\n')
print(a.group, len(records), 'pre-change fixtures: exact repeat outputs/gradients PASS')
