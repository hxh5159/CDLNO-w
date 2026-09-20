"""Complete unchanged CPU suites in isolated processes; never real datasets."""
import concurrent.futures,json,os,pathlib,re,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent
paths=sorted((ROOT/'tests').glob('test_*.py'))+sorted((ROOT/'tests/linearno').glob('test_*.py'))+sorted((ROOT/'tests/loop_linearno').glob('test_*.py'))+sorted((ROOT/'monitor').glob('test_*.py'))
# Preserve the pre-LL9 inventory; new LL9 tools tests run separately.
paths=[p for p in paths if p.name != 'test_performance_tools.py']
env=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
if pathlib.Path('/home/hwz/LRSA-Operator').is_dir():env['CDLNO_LRSA_ROOT']='/home/hwz/LRSA-Operator'
for p in paths:
 for key in re.findall(r"os.environ.get\(['\"]([^'\"]*REPORT)['\"]",p.read_text()):env[key]=str(OUT/(key.lower()+'.json'))
(OUT/'test-environment.json').write_text(json.dumps(env,indent=2))
# Store only relevant non-secret environment in the report.
(OUT/'test-environment.json').write_text(json.dumps({k:v for k,v in env.items() if k.endswith('REPORT') or k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLBACKEND','CDLNO_LRSA_ROOT')},indent=2))
worker='''import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
r=unittest.TextTestRunner(verbosity=2).run(suite)
print('LL9_RESULT_JSON='+json.dumps(dict(tests=r.testsRun,failures=[(str(t),msg) for t,msg in r.failures],errors=[(str(t),msg) for t,msg in r.errors],skipped=[(str(t),why) for t,why in r.skipped])))
sys.exit(not r.wasSuccessful())
'''
def run(path):
 name=str(path.relative_to(ROOT/'tests')).removesuffix('.py').replace('/','.') if path.is_relative_to(ROOT/'tests') else 'monitor.'+path.stem
 cmd=[sys.executable,'-B','-c',worker,name];start=time.monotonic()
 with (OUT/(name+'.log')).open('w') as f:
  try:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=2400).returncode
  except subprocess.TimeoutExpired:code=124
 log=(OUT/(name+'.log')).read_text();match=re.search(r'LL9_RESULT_JSON=(.*)',log)
 summary=json.loads(match.group(1)) if match else {'incomplete_log_tail':log[-3000:]}
 row=dict(module=name,command=cmd,exit_code=code,seconds=round(time.monotonic()-start,3),**summary)
 print(json.dumps({k:v for k,v in row.items() if k not in ('command','failures','errors')},ensure_ascii=False),flush=True)
 return row
(OUT/'regression-inventory.json').write_text(json.dumps([str(p.relative_to(ROOT)) for p in paths],indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,paths))
(OUT/'regression-results.json').write_text(json.dumps(rows,indent=2)+'\n')
