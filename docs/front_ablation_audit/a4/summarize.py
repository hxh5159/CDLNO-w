"""Summarize measured outputs and count actual preset model parameters."""
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tools.cdlno_perf.models import Case, TASKS, build

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
before = json.loads((OUT/'full-before.json').read_text())
after = json.loads((OUT/'cpu-full.json').read_text())
baseline_checks = []
for old in before['results']:
    new = next(r for r in after['results'] if r['model']==old['model'] and r['chunk']==old['chunk'])
    assert old['initial_weights_sha256']==new['initial_weights_sha256']
    for key in ('matrix_macs','matrix_flops_2_per_mac','matrix_macs_by_component','history_sizes'):
        assert old['cost'][key]==new['cost'][key], (old['model'],key)
    for key,value in old['cost']['counts'].items():
        assert new['cost']['counts'][key]==value
    assert new['cost']['parameters']['registered']==old['cost']['parameters']['registered']
    baseline_checks.append(old['model'])
counts=[]
for mode in ('full','no_sa','identity'):
    data=json.loads((OUT/f'cpu-{mode}.json').read_text())
    assert not data['errors']
    for row in data['results']:
        counts.append(dict(model=row['model'],mode=row['config']['front_latent_mode'],chunk=row['chunk'],
                           counts=row['cost']['counts'],matrix_macs=row['cost']['matrix_macs'],
                           parameters=row['cost']['parameters']['registered'],chunk_equivalence=row['chunk_equivalence']))
presets=[]
for task in TASKS:
    for mode in ('full','no_sa','identity'):
        model,case,_=build(Case.preset(task,front_latent_mode=mode),'cdlno_entry')
        presets.append(dict(task=task,mode=mode,config=case.description(),
            registered_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        del model
    gc.collect()
timings=[]
for path in sorted(OUT.glob('gpu-*.json')):
    data=json.loads(path.read_text());assert not data['errors']
    for row in data['results']:
        timings.append(dict(file=path.name,model=row['model'],mode=row['config']['front_latent_mode'],
            case=row['config'],parameters=row['cost']['parameters']['registered'],
            macs=row['cost']['matrix_macs'],counts=row['cost']['counts'],
            forward=row['measurement']['forward'],train_step=row['measurement']['train_step'],
            optimizer=row['measurement']['optimizer'],
            aten_training_calls=row['measurement']['untimed_train_profiler']['aten_calls'],
            actual_sdpa_ops=row['untimed_eager_profiler']['observed_sdpa_ops'],
            settings=data['settings'],environment=data['environment']))
result=dict(status='passed',unchanged_full_baseline_models=baseline_checks,
            cpu_rows=counts,preset_parameters=presets,gpu_rows=timings)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('Full before/after identical cost/initial weights:',baseline_checks)
print('CPU rows',len(counts),'GPU measured rows',len(timings),'Actual preset parameter counts',len(presets))
for task in TASKS:
    print(task, [(r['mode'],r['registered_parameters']) for r in presets if r['task']==task])
for r in timings:
    print(r['file'],r['model'],round(r['macs']/1e9,6),
          [round(r[k]['peak_allocated_bytes']/2**20,3) for k in ('forward','train_step')],
          'ATen train calls',r['aten_training_calls'])
