"""LL10 evidence runner. Existing checks only; outputs are isolated in LL10."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PY=sys.executable
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',
    CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
if Path('/home/hwz/LRSA-Operator').is_dir():ENV['CDLNO_LRSA_ROOT']='/home/hwz/LRSA-Operator'

DELEGATE='''import importlib.util, pathlib, sys
p=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('_ll10_existing_check',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=pathlib.Path(sys.argv[2]);m.main()
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('name',choices=('regressions','repair_targets','counts','matrix','cuda','native_standard','native_industrial','old_archives'))
    ap.add_argument('--attempt',type=int,default=1)
    ap.add_argument('--continue-from',type=int,default=None,
                    help='Reuse individually completed evidence from an interrupted attempt')
    args=ap.parse_args();name=args.name
    attempt=OUT if args.attempt==1 else OUT/('attempt'+str(args.attempt))
    attempt.mkdir(exist_ok=True)
    evidence=attempt if args.continue_from is None else OUT/('attempt'+str(args.continue_from))
    env=dict(ENV)
    if name=='regressions':
        env['LL9R_REGRESSION_DIR']=str(evidence/'regressions')
        cmd=[PY,'-B','docs/loop_linearno_audit/ll9r/run_regressions.py']
    elif name=='repair_targets':
        cmd=[PY,'-B','-m','unittest','loop_linearno.test_ll9r','-v']
    elif name in ('counts','matrix','cuda'):
        cmd=[PY,'-B','tools/linearno_loop_performance.py',name,'--output',str(attempt/(name+'.json'))]
        if name=='matrix':cmd.append('--diagnostics')
        if name=='cuda':env['CUDA_VISIBLE_DEVICES']='0'
    else:
        script={'native_standard':'run_native_matrix.py','native_industrial':'run_industrial.py','old_archives':'replay_old_archives.py'}[name]
        cmd=[PY,'-B','-c',DELEGATE,str(ROOT/'docs/loop_linearno_audit/ll9r'/script),str(OUT)]
        if args.attempt!=1:
            # Keep the delegate's OUT.parent pointing at the historical audit
            # root; redirect only writable matrix reports to a fresh attempt.
            if name=='native_standard':env['LL9_STANDARD_MATRIX_REPORT']=str(evidence/'native-matrix.json')
            elif name=='native_industrial':env['LL9_REPORT']=str(evidence/'industrial-matrix.json')
            else:raise SystemExit('old archive replay has already been isolated; do not overwrite it')
    log=attempt/(name+'.log');report=attempt/(name+'-command.json')
    if log.exists() or report.exists():raise SystemExit('refusing to overwrite prior LL10 command evidence')
    start=time.monotonic()
    with log.open('x') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    row=dict(name=name,argv=cmd,cwd=str(ROOT),environment={k:env[k] for k in
        ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLBACKEND')},
        exit_code=result.returncode,seconds=round(time.monotonic()-start,3),log=log.name)
    row['evidence_directory']=str(evidence)
    row['continued_from_interrupted_attempt']=args.continue_from
    if name=='regressions' and (evidence/'regressions/regression-results.json').exists():
        results=json.loads((evidence/'regressions/regression-results.json').read_text())
        row.update(modules=len(results),tests=sum(r.get('tests',0) for r in results),
            failures=sum(len(r.get('failures',[])) for r in results),errors=sum(len(r.get('errors',[])) for r in results),
            skipped=sum(len(r.get('skipped',[])) for r in results),failed_modules=[r['module'] for r in results if r['exit_code']])
        row['passed']=row['tests']-row['failures']-row['errors']-row['skipped']
        row['exit_code']=int(bool(row['exit_code'] or row['failed_modules']))
    if name=='repair_targets':
        text=log.read_text();m=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text)
        if m:row['tests']=int(m[1])
    row['status']='PASS' if row['exit_code']==0 else 'FAIL'
    report.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    raise SystemExit(row['exit_code'])


if __name__=='__main__':main()
