"""A4's fixed, finite synthetic evidence set using the existing CLI.

No data/task trainer invocation, no automatic width/depth search. Existing
output JSONs are never overwritten. GPU cases run sequentially for timing.
"""
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
commands = []
for mode in ('full', 'no_sa', 'identity'):
    commands.append((f'cpu-{mode}', ['--task','airfoil','--grid','5','7','--B','2',
        '--d','16','--h','4','--M','4','--front-latent-mode',mode,'--audit-only']))
if torch.cuda.is_available():
    for mode in ('full', 'no_sa', 'identity'):
        # Elasticity presets match dimensions for original and new models.
        models = ['transolver','lrsa_matched','cdlno_off','cdlno_entry','cdlno_every_block'] if mode=='full' else ['cdlno_entry']
        commands.append((f'gpu-elasticity-task-{mode}', ['--task','elasticity','--comparison','task',
            '--models',*models,'--chunks','0','--front-latent-mode',mode,'--device','cuda:0',
            '--precision','fp32','--backend','math','--warmup','5','--iterations','20']))
        commands.append((f'gpu-airfoil-matched-{mode}', ['--task','airfoil','--comparison','matched',
            '--grid','17','23','--B','2','--models','cdlno_entry','--chunks','0',
            '--front-latent-mode',mode,'--device','cuda:0','--precision','fp32','--backend','math',
            '--warmup','5','--iterations','20']))
record = {'gpu_available':torch.cuda.is_available(), 'runs':[]}
for name, flags in commands:
    cmd = [sys.executable,'-B',str(ROOT/'tools/cdlno_benchmark.py'),*flags,'--output',str(OUT/(name+'.json'))]
    print('Running', name, flush=True)
    with (OUT/(name+'.log')).open('x') as log:
        status = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    record['runs'].append(dict(name=name,command=cmd,exit_code=status))
    (OUT/'performance-commands.json').write_text(json.dumps(record,indent=2)+'\n')
    if status:
        raise RuntimeError(f'{name} failed: inspect its retained log/JSON')
print('All finite synthetic cases passed', flush=True)
