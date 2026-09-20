"""Replay genuine LL9 epoch1 bytes in fresh copies; never edit old runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from cdlno.linearno_loop.checkpoint import read_pair
from cdlno.training_state import _same


def copy_epoch(source,dest):
    dest.mkdir(parents=True)
    for name in ('architecture.json','config.json'):
        if (source/name).exists():shutil.copy2(source/name,dest/name)
    manifest=json.loads((source/'checkpoints/epoch_0001.json').read_text())
    for kind in ('checkpoint','weights','metadata'):
        path=manifest[kind]['path'];(dest/path).parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(source/path,dest/path)
    raw=(source/'checkpoints/epoch_0001.json').read_bytes()
    (dest/'checkpoints/epoch_0001.json').write_bytes(raw)
    (dest/'checkpoints/latest.json').write_text(json.dumps(dict(manifest='epoch_0001.json',sha256=hashlib.sha256(raw).hexdigest())))


def main():
    artifact=Path(tempfile.mkdtemp(prefix='loop-ll9r-old-replay-',dir='/home/hwz/CDLNO-artifacts'))
    standard=json.loads((OUT.parent/'ll9/native-matrix.json').read_text())['rows']
    industrial=json.loads((OUT.parent/'ll9/industrial-matrix.json').read_text())['rows']
    jobs=[r for r in standard if r['task']=='darcy']+industrial
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT),CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    def run(row):
        task,preset,mode=row['task'],row['preset'],row['mode'];old=Path(row['artifact'])
        source=next((old/('continuous' if task=='darcy' else 'full')).iterdir())
        frozen={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file()}
        dest=artifact/task/preset/mode/source.name
        if task=='airfrans':
            dest.mkdir(parents=True)
            for name in ('architecture.json','config.json'):shutil.copy2(source/name,dest/name)
            copy_epoch(source/'member_000',dest/'member_000')
        else:copy_epoch(source,dest)
        reports={};base=dest.parent
        for action in ('resume','eval'):
            report=base/(action+'.json');log=base/(action+'.log')
            if task=='darcy':
                cmd=[sys.executable,'-B',str(ROOT/'tests/loop_linearno/native_worker.py'),task,action,str(dest),str(report),preset,mode]
            else:
                cmd=[sys.executable,'-B',str(ROOT/f'tests/loop_linearno/{"air" if task=="airfrans" else "car"}_worker.py'),action,str(dest),str(report),preset,mode]
            project={'darcy':'PDE-Solving-StandardBenchmark','airfrans':'Airfoil-Design-AirfRANS','car':'Car-Design-ShapeNetCar'}[task]
            with log.open('w') as f:p=subprocess.run(cmd,cwd=ROOT/project,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            assert p.returncode==0,str(log)+'\n'+log.read_text()[-2500:]
            reports[action]=json.loads(report.read_text())
        reference=row['reports']['train' if task=='darcy' else 'full-ensemble' if task=='airfrans' else 'full-train']
        keys=('prediction_hash',) if task=='darcy' else ('state_hash','resume_hash','prediction')
        for key in keys:assert reference[key]==reports['resume'][key]==reports['eval'][key],(task,mode,key)
        members=['member_000','member_001'] if task=='airfrans' else ['']
        for member in members:
            a,w=read_pair(source/member/'checkpoints/epoch_0003.json');b,z=read_pair(dest/member/'checkpoints/epoch_0003.json')
            assert _same(w,z) and a['resume_state']==b['resume_state']
        assert frozen=={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob('*') if p.is_file()}
        print(task,mode,'old archive exact',flush=True)
        return dict(task=task,preset=preset,mode=mode,old_archive=str(source),copy=str(dest),exact=True,
                    old_bytes_retained=True,new_processes=2,strict_weights=True,
                    optimizer_scheduler_rng_exact=True,output_exact=True,reports=reports)
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,row) for row in jobs]):
            rows.append(future.result())
            (OUT/'old-checkpoint-replay.json').write_text(json.dumps(dict(
                expected=len(jobs),completed=len(rows),rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
