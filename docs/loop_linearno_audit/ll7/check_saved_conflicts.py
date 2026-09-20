"""Every saved industrial topology/mode: conflicts fail before weights/model."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
code=r'''
import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1]+'/tests')
task,directory=sys.argv[2:]
if task=='airfrans':
 from loop_linearno.air_worker import parser_for,parse_args
else:
 from loop_linearno.car_worker import parser_for,parse_args
from cdlno.linearno_loop import construction,industrial_state
metadata=json.loads((Path(directory)/'architecture.json').read_text())
s=metadata['loop_spec']
wrong_mode='rb_attnres' if s['residual_mode']!='rb_attnres' else 'sr_1_over_r'
wrong_preset='p2_c2_r2_s2' if s['topology_preset']=='p1_c3_r2_s1' else 'p1_c3_r2_s1'
fields=[['--linearno-loop-residual-mode',wrong_mode],['--linearno-loop-topology',wrong_preset],
 ['--linearno-rank','8'],['--linearno-heads','4'],['--linearno-hidden','16'],
 ['--linearno-profile','official_release'],['--linearno_latent_attnres','0'],['--linearno-loop','0'],
 ['--model' if task=='airfrans' else '--cfd_model','Transolver']]
fields.append(['--task','full'] if task=='airfrans' else ['--fold_id','0'])
for flags in fields:
 with patch('torch.load',side_effect=AssertionError('early weight load')),patch.object(construction,'build_from_config',side_effect=AssertionError('early construction')),patch.object(industrial_state,'construct',side_effect=AssertionError('early construction')):
  try:parse_args(parser_for(True),argv=['--experiment-dir',directory,*flags],evaluation=True)
  except SystemExit:pass
  else:raise AssertionError('accepted '+str(flags))
print('CONFLICT_RESULT='+json.dumps(dict(task=task,checks=len(fields),passed=True)))
'''
matrix=json.loads((OUT/'industrial-matrix.json').read_text());assert matrix['completed']==matrix['expected']==12
rows=[]
for row in matrix['rows']:
 directory=next((Path(row['artifact'])/'split').iterdir())
 env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),row['task'],str(directory)],cwd=ROOT,env=env,capture_output=True,text=True)
 with (OUT/'saved-conflicts.log').open('a') as f:f.write(p.stdout+p.stderr)
 assert p.returncode==0,(row['task'],p.stdout,p.stderr)
 result=json.loads(next(s[len('CONFLICT_RESULT='):] for s in p.stdout.splitlines() if s.startswith('CONFLICT_RESULT=')))
 rows.append(dict(task=row['task'],preset=row['preset'],mode=row['mode'],**{k:v for k,v in result.items() if k!='task'}))
(OUT/'saved-conflicts.json').write_text(json.dumps(rows,indent=2)+'\n')
print('120 actual saved-run conflicts rejected before model/torch.load')
