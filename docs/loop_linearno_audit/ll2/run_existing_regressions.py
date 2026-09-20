"""LL2 rerun of the unchanged LL0 test inventory: execute unchanged CPU tests in isolated processes."""
import concurrent.futures, json, os, pathlib, re, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
names=['attention_parity','attention_structure','standard_model','standard_structure','airfrans_model','shapenet_model','profiles','schema','converter','legacy','rng','static_integration','temporal_integration','air_entry','car_entry','car_metrics','air_objective_decision','car_objective_decision','history_schema','history_core','latent_attnres','history_k','history_integration']
jobs=[('linearno.test_'+n,['linearno.test_'+n]) for n in names]+[('monitor',['discover','-s','monitor','-p','test_*.py'])]
env=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
for p in (ROOT/'tests/linearno').glob('test_*.py'):
 for key in re.findall(r"os.environ.get\(['\"]([^'\"]*REPORT)['\"]",p.read_text()): env[key]=str(OUT/(key.lower()+'.json'))
(OUT/'test-environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('LINEARNO_') or k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLBACKEND')},indent=2)+'\n')
worker='''import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
r=unittest.TextTestRunner(verbosity=2).run(suite)
print('LL2_RESULT_JSON='+json.dumps(dict(tests=r.testsRun,failures=[str(t) for t,_ in r.failures],errors=[str(t) for t,_ in r.errors],skipped=[(str(t),why) for t,why in r.skipped])))
sys.exit(not r.wasSuccessful())
'''
def run(job):
 name,args=job; cmd=[sys.executable,'-B','-c',worker,name] if name!='monitor' else [sys.executable,'-B','-m','unittest',*args,'-v']; start=time.monotonic()
 with (OUT/(name+'.log')).open('w') as f:
  try: p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200); code=p.returncode
  except subprocess.TimeoutExpired:code=124
 log=(OUT/(name+'.log')).read_text(); match=re.search(r'LL2_RESULT_JSON=(.*)',log)
 summary=json.loads(match.group(1)) if match else {'unittest_summary':log[-1800:]}
 row=dict(module=name,command=cmd,exit_code=code,seconds=round(time.monotonic()-start,3),**summary)
 print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(run,jobs))
(OUT/'regression-results.json').write_text(json.dumps(rows,indent=2)+'\n')
