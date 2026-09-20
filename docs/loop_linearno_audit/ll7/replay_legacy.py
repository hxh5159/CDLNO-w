"""Replay truly pre-LL7 pure/history checkpoints in fresh current processes."""
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
old=json.loads((OUT/'pre-checkpoints.json').read_text())
env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
def run(row):
 task,family=row['task'],row['family'];reports={}
 full=next(r for r in old if (r['task'],r['family'],r['action'])==(task,family,'train'))
 for action in ('resume','eval'):
  cmd=list(row['command']);cmd[3]=action;cmd[5]=str(OUT/f'{task}-{family}-post-{action}.json')
  with (OUT/f'{task}-{family}-post-{action}.log').open('w') as f:p=subprocess.run(cmd,cwd=ROOT/('Airfoil-Design-AirfRANS' if task=='air' else 'Car-Design-ShapeNetCar'),env=env,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,(task,family,action)
  reports[action]=json.loads(Path(cmd[5]).read_text())
 expected=json.loads(Path(full['report']).read_text());interrupted=json.loads(Path(row['report']).read_text())
 for key in ('state_hash','resume_hash','prediction'):
  assert expected[key]==reports['resume'][key]==reports['eval'][key],(task,family,key)
 assert expected['batches']==interrupted['batches']+reports['resume']['batches']
 print(task,family,'exact',flush=True)
 return dict(task=task,family=family,exact_weights_optimizer_rng_output=True)
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[r for r in old if r['action']=='interrupt']))
(OUT/'pre-checkpoint-replay.json').write_text(json.dumps(rows,indent=2)+'\n')
