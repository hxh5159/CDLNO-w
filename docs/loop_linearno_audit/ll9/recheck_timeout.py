"""Repeat unchanged history K suite after initial subprocess timeout; retain both."""
import json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
worker='''import json,sys,unittest
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(sys.argv[1]))
print('LL9_RESULT_JSON='+json.dumps(dict(tests=r.testsRun,failures=[(str(t),msg) for t,msg in r.failures],errors=[(str(t),msg) for t,msg in r.errors],skipped=[(str(t),why) for t,why in r.skipped])))
sys.exit(not r.wasSuccessful())
'''
env=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LINEARNO_R4_REPORT=str(OUT/'linearno_r4_recheck_report.json'))
name='linearno.test_history_k';cmd=[sys.executable,'-B','-c',worker,name];start=time.monotonic()
with (OUT/'history-k-recheck.log').open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
text=(OUT/'history-k-recheck.log').read_text();summary=json.loads(re.search(r'LL9_RESULT_JSON=(.*)',text).group(1))
(OUT/'regression-rechecks.json').write_text(json.dumps([dict(module=name,command=cmd,exit_code=r.returncode,seconds=time.monotonic()-start,**summary)],indent=2))
sys.exit(r.returncode)
