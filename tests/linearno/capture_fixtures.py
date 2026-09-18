"""Explicit fixture generation, kept out of automatic tests. Refuses overwrite.

Usage (from repository root): python -B tests/linearno/capture_fixtures.py /NEW/output.json
The generator is for small synthetic original-Transolver references only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main(destination):
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    rows = {}
    for task in ('darcy','elasticity','airfoil','pipe','ns','plasticity','car','airfrans'):
        project = ('Car-Design-ShapeNetCar' if task=='car' else 'Airfoil-Design-AirfRANS' if task=='airfrans'
                   else 'PDE-Solving-StandardBenchmark')
        p = subprocess.run([sys.executable,'-B',str(HERE/'legacy_worker.py'),task],cwd=ROOT/project,
                           capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
        rows[task] = json.loads(p.stdout)
        print(task, rows[task]['parameters'], rows[task]['output_shape'], flush=True)
    result = dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                  generator_sha256=hashlib.sha256((HERE/'legacy_worker.py').read_bytes()).hexdigest(),
                  scope='original models, synthetic tensors; no trained checkpoints', tasks=rows)
    with path.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    main(sys.argv[1])
