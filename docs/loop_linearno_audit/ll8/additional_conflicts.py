"""Saved head/width/variant and each topology assertion through new launchers."""
import contextlib,io,json,os,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from tran_evaluate.linearno_loop import launch
import torch
matrix=json.loads((OUT/'dry-run-matrix.json').read_text());rows=[]
env=dict(os.environ,CDLNO_REPO_ROOT=str(ROOT),CDLNO_PYTHON=sys.executable,CUDA_VISIBLE_DEVICES='')
for task in launch.TASKS:
    for action in ('resume','eval'):
        record=next(r['plan'] for r in matrix['rows'] if r['plan']['task']==task and r['plan']['action']==action and
                    r['plan']['config']['loop_spec']['topology_preset']=='p1_c3_r2_s1' and
                    r['plan']['config']['loop_spec']['residual_mode']=='sr_1_over_r')
        entries=(('--n-hidden' if task in launch.ENTRIES else '--linearno-hidden','16'),
                 ('--n-heads' if task in launch.ENTRIES else '--linearno-heads','4'),
                 ('--linearno-loop-prefix-blocks','2'),('--linearno-loop-core-blocks','2'),
                 ('--linearno-loop-repeats','3'),('--linearno-loop-suffix-blocks','2'),
                 ('--linearno-loop','0'),('--linearno_latent_attnres','0'),
                 ('--linearno-variant','temp' if task=='ns' else 'plain'))
        for flag,value in entries:
            with contextlib.redirect_stderr(io.StringIO()),patch('torch.load',side_effect=AssertionError('early tensor load')):
                try:launch.plan(task,action,['--experiment-dir',record['run'],flag,value],env)
                except (ValueError,SystemExit):rows.append(dict(task=task,action=action,flag=flag,value=value,rejected=True))
                else:raise AssertionError((task,action,flag))
(OUT/'additional-conflicts.json').write_text(json.dumps(rows,indent=2)+'\n')
print(len(rows),'saved structural assertions rejected before tensor load')
