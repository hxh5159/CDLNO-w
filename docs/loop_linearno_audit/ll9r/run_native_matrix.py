"""36 native CPU synthetic cycles: continuous vs fresh-process resume/eval."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from loop_linearno.native_worker import fixture_config
from linearno_loop.config import run_directory_id
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
from cdlno.linearno_loop.checkpoint import inspect_checkpoint,read_pair
from cdlno.training_state import _same


def main():
    tasks=os.environ.get('LL9_STANDARD_TASKS','airfoil darcy elasticity pipe ns plasticity').split()
    modes=os.environ.get('LL9_STANDARD_MODES',' '.join(RESIDUAL_MODES)).split()
    presets=os.environ.get('LL9_STANDARD_PRESETS',' '.join(PRESETS)).split()
    artifact=Path(tempfile.mkdtemp(prefix='loop-ll9r-standard-',dir='/home/hwz/CDLNO-artifacts'))
    report_name=os.environ.get('LL9_STANDARD_MATRIX_REPORT','native-matrix.json')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    rows=json.loads((OUT/report_name).read_text())["rows"] if (OUT/report_name).exists() else []
    complete={(r["task"],r["preset"],r["mode"]) for r in rows}
    def job(task,preset,mode):
        then=time.monotonic();config=fixture_config(task,preset,mode);identifier=run_directory_id(config)
        base=artifact/task/preset/mode;base.mkdir(parents=True)
        continuous=base/'continuous'/identifier;split=base/'split'/identifier;reports={}
        for action,directory in [('train',continuous),('interrupt',split),('resume',split),('eval',split)]:
            path=base/(action+'.json');log=base/(action+'.log')
            command=[sys.executable,'-B',str(ROOT/'tests/loop_linearno/native_worker.py'),task,action,str(directory),str(path),preset,mode]
            with log.open('w') as stream:
                proc=subprocess.run(command,cwd=ROOT/'PDE-Solving-StandardBenchmark',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            if proc.returncode:raise RuntimeError(f'{task}/{preset}/{mode}/{action}: {log}\n'+log.read_text()[-3500:])
            reports[action]=json.loads(path.read_text())
        _,pa=inspect_checkpoint(continuous,'final');_,pb=inspect_checkpoint(split,'final')
        ma,wa=read_pair(pa);mb,wb=read_pair(pb)
        assert _same(wa,wb),'final weights differ'
        assert ma['resume_state']==mb['resume_state'],'optimizer/scheduler/RNG/generators differ'
        assert reports['train']['batches']==reports['interrupt']['batches']+reports['resume']['batches']
        assert reports['train']['prediction_hash']==reports['resume']['prediction_hash']==reports['eval']['prediction_hash']
        if task in ('ns','plasticity'):
            assert reports['train']['time_queries']==reports['interrupt']['time_queries']+reports['resume']['time_queries']
            assert reports['train']['numpy_next']==reports['resume']['numpy_next']
        assert all(r['forward_local_checks']>0 for r in reports.values())
        assert reports['train']['recorded_artifacts'] and reports['eval']['recorded_artifacts']
        return dict(task=task,preset=preset,mode=mode,artifact=str(base),seconds=round(time.monotonic()-then,3),
            exact_weights=True,exact_optimizer_scheduler_rng=True,exact_batches=True,exact_eval=True,
            metadata_first=True,sidecar_immutable=True,normalizer_no_refit=True,
            real_spatial_shape=True,synthetic_samples='4train/2test',epochs=3,hidden=8,heads=2,actual_M=4,
            reports=reports)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,t,p,m) for t in tasks for p in presets for m in modes if (t,p,m) not in complete]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            print(json.dumps({k:row[k] for k in ('task','preset','mode','seconds','exact_weights')}),flush=True)
            (OUT/report_name).write_text(json.dumps(dict(artifact_root=str(artifact),completed=len(rows),
                expected=len(tasks)*len(presets)*len(modes),rows=rows),indent=2)+'\n')
    assert len(rows)==len(tasks)*len(presets)*len(modes)


if __name__=='__main__':main()
