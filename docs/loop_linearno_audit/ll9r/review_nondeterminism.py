"""Repeat unchanged pre-repair CUDA SR to isolate backend roundoff."""
import json
from pathlib import Path
import torch
from capture_numeric import capture, configuration

torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
c=configuration('airfoil','p1_c3_r2_s1','sr_1_over_r',small=True)
records=[]
for deterministic in (False,True):
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic=deterministic
    values=[capture(c,'cuda','FP32')[0] for _ in range(3)]
    first=values[0]
    for value in values[1:]:
        fields={}
        for field in ('initial','output','loss','input_gradients','gradients','step_state'):
            a,b=first[field],value[field]
            pairs=[(field,a,b)] if isinstance(a,torch.Tensor) else [(k,a[k],b[k]) for k in a]
            diffs={k:float((x-y).abs().max()) for k,x,y in pairs if x is not None and not torch.equal(x,y)}
            fields[field]=dict(exact=not diffs,max_abs=max(diffs.values(),default=0),nonexact_keys=list(diffs))
        records.append(dict(deterministic=deterministic,fields=fields))
import cdlno.linearno_loop.core as core
assert 'loop-ll9r-before-' in core.__file__,core.__file__
assert any(not r['fields']['gradients']['exact'] for r in records if not r['deterministic'])
assert all(all(f['exact'] for f in r['fields'].values()) for r in records if r['deterministic'])
Path(__file__).with_name('nondeterminism-review.json').write_text(json.dumps(dict(
    unchanged_pre_repair_core=core.__file__,torch=str(torch.__version__),
    backend='CUDA/cuDNN nondeterministic vs deterministic; TF32 off, no tolerance relaxation',rows=records),indent=2)+'\n')
print('Unchanged old SR reproduces non-bitwise gradients; deterministic old SR exact')
