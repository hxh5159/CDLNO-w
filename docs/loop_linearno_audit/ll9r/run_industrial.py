"""Real PyG/native industrial train, interrupted and fresh-process continuation."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from loop_linearno.industrial_support import config
from linearno_loop.config import run_directory_id
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
artifact=Path(tempfile.mkdtemp(prefix='loop-ll9r-industrial-',dir='/home/hwz/CDLNO-artifacts'))
env=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')

def job(task,preset,mode):
 start=time.monotonic();air=task=='airfrans';member_count=2 if air else 1
 identifier=run_directory_id(config(task,preset,mode,nmodel=member_count))
 base=artifact/task/preset/mode;base.mkdir(parents=True);reports={}
 actions=[('ensemble' if air else 'train','full'),('ensemble_later_interrupt' if air else 'interrupt','split'),('resume','split'),('eval','split')]
 if air:actions+=[('ensemble_interrupt','boundary'),('resume','boundary'),('eval','boundary')]
 for action,group in actions:
  key=group+'-'+action;report=base/(key+'.json');log=base/(key+'.log')
  cmd=[sys.executable,'-B',str(ROOT/f'tests/loop_linearno/{"air" if air else "car"}_worker.py'),action,str(base/group/identifier),str(report),preset,mode]
  with log.open('w') as f:p=subprocess.run(cmd,cwd=ROOT/('Airfoil-Design-AirfRANS' if air else 'Car-Design-ShapeNetCar'),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  if p.returncode:raise RuntimeError(str(log)+'\n'+log.read_text()[-4500:])
  reports[key]=json.loads(report.read_text())
 reference=reports['full-'+('ensemble' if air else 'train')];resumed=reports['split-resume'];evaluated=reports['split-eval']
 for k in ('state_hash','resume_hash','prediction'):
  assert reference[k]==resumed[k]==evaluated[k],(task,preset,mode,k)
 assert reference['batches']==reports['split-'+('ensemble_later_interrupt' if air else 'interrupt')]['batches']+resumed['batches']
 if air:
  assert reference['distinct_members']
  for k in ('member_hashes','resume_hash','prediction'):assert reference[k]==reports['boundary-resume'][k]==reports['boundary-eval'][k]
  assert reference['batches']==reports['boundary-ensemble_interrupt']['batches']+reports['boundary-resume']['batches']
 return dict(task=task,preset=preset,mode=mode,artifact=str(base),seconds=round(time.monotonic()-start,3),exact=True,reports=reports)

def main():
 tasks=os.environ.get('LL9_TASKS','airfrans car').split();presets=os.environ.get('LL9_PRESETS',' '.join(PRESETS)).split();modes=os.environ.get('LL9_MODES',' '.join(RESIDUAL_MODES)).split()
 name=os.environ.get('LL9_REPORT','industrial-matrix.json')
 rows=json.loads((OUT/name).read_text())['rows'] if (OUT/name).exists() else []
 complete={(r['task'],r['preset'],r['mode']) for r in rows}
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(job,t,p,m) for t in tasks for p in presets for m in modes if (t,p,m) not in complete]
  for f in as_completed(futures):
   row=f.result();rows.append(row);print(json.dumps({k:row[k] for k in ('task','preset','mode','seconds','exact')}),flush=True)
   (OUT/name).write_text(json.dumps(dict(artifact_root=str(artifact),expected=len(tasks)*len(presets)*len(modes),completed=len(rows),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
