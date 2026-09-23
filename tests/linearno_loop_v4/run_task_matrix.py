"""Run explicit tiny synthetic native epochs, retaining each fresh-process log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',default=['airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'])
    p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    a.evidence.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='v4-native-matrix-'))
    rows=[]
    for task in a.tasks:
        worker='industrial_worker.py' if task in ('car','airfrans') else 'standard_worker.py'
        for mode in ('latent_k_point_q','point_k_point_q'):
            run=temp/f'{task}-{mode}'
            for action in ('interrupt','resume','eval'):
                label=f'{task}-{mode}-{action}';log=a.evidence/(label+'.txt');report=a.evidence/(label+'.json')
                command=[sys.executable,'-B',str(Path(__file__).with_name(worker)),task,mode,action,str(run),str(report)]
                with log.open('w') as stream:
                    process=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':'.:cdlno:PDE-Solving-StandardBenchmark',
                        'OMP_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT)
                row=dict(task=task,mode=mode,action=action,exit_code=process.returncode,command=command)
                rows.append(row);print(label,process.returncode,flush=True)
                (a.evidence/'matrix-progress.json').write_text(json.dumps(dict(synthetic_root=str(temp),rows=rows),indent=2)+'\n')
                if process.returncode: return 1
            restored=json.loads((a.evidence/f'{task}-{mode}-resume.json').read_text())
            evaluated=json.loads((a.evidence/f'{task}-{mode}-eval.json').read_text())
            assert restored['weights_hash']==evaluated['weights_hash']
            if mode=='latent_k_point_q':
                label=f'{task}-{mode}-uninterrupted';report=a.evidence/(label+'.json')
                command=[sys.executable,'-B',str(Path(__file__).with_name(worker)),task,mode,'train',str(temp/(label)),str(report)]
                with (a.evidence/(label+'.txt')).open('w') as stream:
                    process=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':'.:cdlno:PDE-Solving-StandardBenchmark','OMP_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT)
                rows.append(dict(task=task,mode=mode,action='uninterrupted',exit_code=process.returncode,command=command))
                if process.returncode:return 1
                direct=json.loads(report.read_text());assert restored['weights_hash']==direct['weights_hash'],task
                print(label,'exact_weights_equal',flush=True)
    (a.evidence/'matrix-final.json').write_text(json.dumps(dict(synthetic_root=str(temp),rows=rows,exact_resume=True),indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
