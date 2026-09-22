"""Compile in memory and syntax-check shells without running task entrypoints."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
start = time.perf_counter()
files = sorted({p for directory in (ROOT/'linearno_loop/v3', ROOT/'tests/loop_linearno_latent_adapter', OUT)
                for p in directory.rglob('*.py')})
compiled = []
for path in files:
    compile(path.read_bytes(), str(path), 'exec')
    compiled.append(str(path.relative_to(ROOT)))
compile_time = time.perf_counter()-start
shells = sorted(p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p.endswith('.sh'))
syntax = []
for path in shells:
    start=time.perf_counter()
    p=subprocess.run(['bash','-n',path],cwd=ROOT,capture_output=True,text=True)
    syntax.append(dict(path=path,exit_code=p.returncode,seconds=time.perf_counter()-start,output=p.stdout+p.stderr))
start=time.perf_counter()
p=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
diff=dict(command=['git','diff','--check'],exit_code=p.returncode,seconds=time.perf_counter()-start,output=p.stdout+p.stderr)
assert all(row['exit_code']==0 for row in syntax) and diff['exit_code']==0
result=dict(status='PASS',python=dict(count=len(compiled),files=compiled,seconds=compile_time,no_pyc=True),
            shell=dict(count=len(shells),commands=syntax),diff=diff)
with (OUT/'static-checks.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(status='PASS',python=len(compiled),shell=len(shells),diff=diff)))
