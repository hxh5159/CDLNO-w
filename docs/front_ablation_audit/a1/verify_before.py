"""Compare current full against immutable, genuinely pre-edit A1 fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import cdlno
from cdlno import CDLNO, CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdlno.checkpoint import validate_sidecar
from cdlno.modules import LRSAFrontBlock

p = argparse.ArgumentParser()
p.add_argument('baseline', type=Path)
p.add_argument('group', choices=('core','car','airfrans'))
p.add_argument('--results', type=Path, required=True)
a = p.parse_args()
assert not Path(cdlno.__file__).resolve().is_relative_to(a.baseline/'source')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
manifest = json.loads((a.baseline/a.group/'manifest.json').read_text())
results = []


def check_output_and_gradients(model, fixture):
    model.eval().zero_grad(set_to_none=True)
    x = fixture['input'].clone().requires_grad_()
    if a.group in ('car','airfrans'):
        from torch_geometric.data import Data, Batch
        kwargs = dict(x=x,y=torch.zeros(x.shape[0],4))
        if a.group == 'airfrans': kwargs['pos'] = x[:,:2]
        data = Batch.from_data_list([Data(**kwargs)])
        args = (data,Data()) if a.group == 'car' else data
    else:
        args = x
    with sdpa_kernel(SDPBackend.MATH):
        prediction = model(args)
        ys = prediction if isinstance(prediction,tuple) else (prediction,)
        for actual, expected in zip(ys, fixture['output']):
            torch.testing.assert_close(actual,expected,atol=fixture['atol'],rtol=fixture['rtol'])
        sum((y*c).sum() for y,c in zip(ys,fixture['cotangents'])).backward()
    grads = {k:v.grad for k,v in model.named_parameters()}
    grads['input'] = x.grad
    assert grads.keys() == fixture['gradients'].keys()
    for key, value in grads.items():
        torch.testing.assert_close(value,fixture['gradients'][key],atol=fixture['atol'],rtol=fixture['rtol'])
    buffers = dict(model.named_buffers())
    assert buffers.keys() == fixture['buffers'].keys()
    for key,value in buffers.items():
        torch.testing.assert_close(value,fixture['buffers'][key],atol=0,rtol=0)


for row in manifest['records']:
    directory = a.baseline/a.group/row['name']
    for name, sha in row['files'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == sha
    fixture = torch.load(directory/'fixture.pt',weights_only=True)
    assert fixture['atol'] == fixture['rtol'] == 0.
    torch.manual_seed(fixture['seed'])
    kind = fixture['description']['kind']
    if kind == 'front':
        model = LRSAFrontBlock(**fixture['description']['kwargs'])
    elif kind == 'core':
        cfg = CDLNOArchitectureConfig.from_dict(fixture['config'])
        assert cfg.front_latent_mode == 'full'
        model = CDLNO(cfg)
    else:
        if kind == 'car':
            from models.CDLNO import Model
        else:
            from cdlno.airfrans import AirfRANSModel as Model
        model = Model(**fixture['description']['kwargs'])
    assert list(model.state_dict()) == fixture['state_keys']
    for key,value in model.state_dict().items():
        if fixture['description'].get('nonzero_scorer') and key.endswith('.w'):
            continue
        torch.testing.assert_close(value,fixture['state'][key],atol=0,rtol=0)
    model.load_state_dict(fixture['state'],strict=True)
    check_output_and_gradients(model,fixture)
    if 'config' in fixture:
        sidecar = directory/'architecture.json'
        original = sidecar.read_bytes()
        validate_sidecar(sidecar,model.config,CDLNORuntimeConfig(device='cpu',source_chunk_size=2))
        assert sidecar.read_bytes() == original
    # Locally generated trusted object, from before production edits. No global
    # torch.load changes; this includes the genuine old 18-slot config state.
    legacy = torch.load(directory/'legacy-model.pt',map_location='cpu',weights_only=False)
    fronts = [legacy] if kind == 'front' else list((legacy if kind == 'core' else legacy.core).front_blocks)
    assert all(not hasattr(block,'front_latent_mode') for block in fronts)
    check_output_and_gradients(legacy,fixture)
    if kind == 'airfrans':
        members = torch.load(directory/'legacy-list.pt',map_location='cpu',weights_only=False)
        assert type(members) is list and len(members) == 2 and members[0] is not members[1]
        for member in members:
            check_output_and_gradients(member,fixture)
    results.append(dict(name=row['name'],state_keys=len(fixture['state_keys']),
                        initialization='exact (controlled nonzero scorer excluded)',
                        output_max_error=0.,gradient_max_error=0.,buffers='exact',
                        legacy_object='passed',atol=0.,rtol=0.))
a.results.write_text(json.dumps(dict(group=a.group,python=sys.version.split()[0],
    torch=torch.__version__,source=cdlno.__file__,baseline=str(a.baseline),results=results),indent=2)+'\n')
print(a.group,len(results),'full pre-edit strict weights/init/output/gradients/legacy-object PASS (atol=rtol=0)')
