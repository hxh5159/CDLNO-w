from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
artifact=Path(json.loads((OUT/'pre-edit.json').read_text())['artifact_root'])
env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
def run(job):
 task,family=job;project='Airfoil-Design-AirfRANS' if task=='air' else 'Car-Design-ShapeNetCar'
 worker=ROOT/'tests/linearno'/f'{"history_" if family=="history" else ""}{task}{"" if family=="history" else "_entry"}_worker.py'
 rows=[]
 for action in ('train','interrupt'):
  name=f'{task}-{family}-{action}';directory=artifact/name
  directory=directory/(f'{"airfrans" if task=="air" else "car"}__paper_table8_on_release_model__L4__A1K0__seed{901 if task=="air" else 19}' if family=='history' else 'pure')
  cmd=[sys.executable,'-B',str(worker),action,str(directory),str(artifact/(name+'.json'))]+(['A1K0'] if family=='history' else [])
  with (OUT/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=ROOT/project,env=env,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,(name,r.returncode)
  rows.append(dict(task=task,family=family,action=action,directory=str(directory),report=str(artifact/(name+'.json')),command=cmd))
  print(name,'pass',flush=True)
 return rows
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[(t,f) for t in ('air','car') for f in ('pure','history')]))
(OUT/'pre-checkpoints.json').write_text(json.dumps([r for group in rows for r in group],indent=2)+'\n')
