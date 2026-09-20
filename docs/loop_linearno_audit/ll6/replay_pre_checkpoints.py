"""Resume/evaluate genuine pre-LL6 pure/history archives using new routing."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT),str(ROOT/'PDE-Solving-StandardBenchmark')]
from cdlno.training_state import _same
from cdlno.linearno import checkpoint as pure
from cdlno.linearno_history import checkpoint as history
from model.LinearNO import Model

rows=json.loads((OUT/'pre-checkpoints.json').read_text())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
results=[]
for family in ('pure','history'):
    full=Path(next(r['directory'] for r in rows if r['family']==family and r['action']=='train'))
    split=Path(next(r['directory'] for r in rows if r['family']==family and r['action']=='interrupt'))
    for action in ('resume','eval'):
        report=OUT/(family+'-post-'+action+'.json')
        command=[sys.executable,'-B',str(ROOT/'tests/linearno'/('static_worker.py' if family=='pure' else 'history_static_worker.py')),
                 'darcy',action,str(split),str(report)]
        if family=='history':command+=['A1K0']
        with (OUT/(family+'-post-'+action+'.log')).open('w') as stream:
            p=subprocess.run(command,cwd=ROOT/'PDE-Solving-StandardBenchmark',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        assert p.returncode==0,(family,action)
    if family=='pure':
        ma,pa=pure.inspect_checkpoint(full,'final',Model);mb,pb=pure.inspect_checkpoint(split,'final',Model)
        wa=pure.read_pair(pa,Model)[1];wb=pure.read_pair(pb,Model)[1]
    else:
        ma,pa=history.inspect_checkpoint(full,'final');mb,pb=history.inspect_checkpoint(split,'final')
        wa=history._read_pair(pa,ma);wb=history._read_pair(pb,mb)
    assert _same(wa,wb) and ma['resume_state']==mb['resume_state']
    pre=Path(json.loads((OUT/'pre-edit.json').read_text())['artifacts'])
    continuous=json.loads((pre/(family+'-train.json')).read_text())
    interrupt=json.loads((pre/(family+'-interrupt.json')).read_text())
    resumed=json.loads((OUT/(family+'-post-resume.json')).read_text());evaluated=json.loads((OUT/(family+'-post-eval.json')).read_text())
    assert continuous['batches']==interrupt['batches']+resumed['batches']
    assert continuous['prediction_hash']==resumed['prediction_hash']==evaluated['prediction_hash']
    results.append(dict(family=family,pre_edit_full=str(full),pre_edit_interrupted=str(split),
        exact_final_weights=True,exact_optimizer_scheduler_rng=True,exact_batches=True,exact_eval=True))
    print(json.dumps(results[-1]),flush=True)
(OUT/'pre-checkpoint-replay.json').write_text(json.dumps(results,indent=2)+'\n')
