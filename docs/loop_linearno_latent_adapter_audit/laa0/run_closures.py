"""Reuse historical controlled workers, redirect all evidence to LAA0."""
from pathlib import Path
import os,runpy,time,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
os.environ.update(LL9_STANDARD_PRESETS='p2_c2_r2_s2',LL9_STANDARD_MODES='sr_1_over_r',LL9_PRESETS='p2_c2_r2_s2',LL9_MODES='sr_1_over_r')
rows=[]
for file in ('run_native_matrix.py','run_industrial.py'):
 start=time.monotonic()
 namespace=runpy.run_path(str(ROOT/'docs/loop_linearno_audit/ll9r'/file),run_name='_laa0_reuse')
 namespace['main'].__globals__['OUT']=OUT
 namespace['main']()
 rows.append({'script':file,'seconds':time.monotonic()-start,'status':'PASS','evidence_output_override':str(OUT)})
 (OUT/'native-closure-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
