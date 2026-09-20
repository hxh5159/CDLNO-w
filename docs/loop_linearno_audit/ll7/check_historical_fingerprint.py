"""Explain the unmodified R8 test failure; check its remaining assertions separately."""
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from cdlno.linearno.standard_entry import provenance
from cdlno.linearno import air_entry as air,car_entry as car
from cdlno.linearno_history import air_entry as hist_air,car_entry as hist_car
from cdlno.linearno_history.provenance import baseline_source

old=json.loads((ROOT/'docs/linearno_history_audit/r8/baseline-provenance.json').read_text())
pre=json.loads((OUT/'pre-edit.json').read_text())
ll0=json.loads((OUT.parent/'ll0/baseline-provenance.json').read_text())['current']
before={'standard':ll0,'airfrans':pre['air'],'car':pre['car']}
functions={'standard':provenance,'airfrans':air._provenance,'car':car.task_provenance}
actual={task:fn() for task,fn in functions.items()}
real_check_output=subprocess.check_output
old_head=old['standard']['base_commit']
def historical_base_only(command,*args,**kwargs):
    if command[:2]==['git','show'] and command[2].startswith('HEAD:'):
        command=[*command[:2],old_head+command[2][4:],*command[3:]]
    return real_check_output(command,*args,**kwargs)

# Only git show's read-only comparison base changes; no files/HEAD are replaced.
with patch('subprocess.check_output',side_effect=historical_base_only):
    historical={task:fn() for task,fn in functions.items()}
hashes={}
for task in functions:
    hashes[task]={}
    for key in ('source_sha256','normalized_patch_sha256'):
        now=actual[task][key]
        row=dict(current=now,pre_LL7=before[task][key],R8=old[task][key],
                 recomputed_with_R8_git_base=historical[task][key])
        assert now==row['pre_LL7'],(task,key,row)
        assert row['recomputed_with_R8_git_base']==row['R8'],(task,key,row)
        if key=='source_sha256':assert now==row['R8']
        hashes[task][key]=row

snapshot=Path(json.loads((ROOT/'docs/linearno_history_audit/r8/baseline.json').read_text())['snapshot'])/'source'
sources={}
for name in ('Airfoil-Design-AirfRANS/train.py','cdlno/linearno/air_entry.py','cdlno/linearno/car_entry.py'):
    projected=baseline_source(name,(ROOT/name).read_text())
    assert projected==(snapshot/name).read_text(),name
    sources[name]=dict(exact_bytes=True,sha256=hashlib.sha256(projected.encode()).hexdigest())
launchers=[]
for task in ('car','airfrans'):
    for action in ('train','resume','eval'):
        command=['bash',str(ROOT/f'tran_evaluate/linearno_history/{task}.sh'),action,'--dry-run','--experiment-dir','test-run']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        launchers.append(dict(task=task,action=action,command=command,exit_code=result.returncode,stdout=result.stdout))
helpers=[]
for pure,hist,names in ((car,hist_car,('evaluate','inspect_data','load_data','restore_coef_norm')),
                       (air,hist_air,('_load_dataset','_data_spec','_pressure_rL2_dataset','_restore_coef_norm'))):
    for name in names:
        assert getattr(pure,name) is getattr(hist,name),name
        helpers.append(pure.__name__+'.'+name)
assert ast.dump(ast.parse(inspect.getsource(car.CarRun.objective).strip()))==ast.dump(ast.parse(inspect.getsource(hist_car.CarRun.objective).strip()))
result=dict(status='PASS: historical failure explained, original test remains failed and unchanged',
    cause='normalized_patch_sha256 includes git HEAD before-AST; R8 used d5abe014, current checkout uses 5b991226. Changing only git-show comparison base reproduces all three R8 hashes exactly.',
    original_suite=dict(methods=5,passed=4,failures=1,errors=0,seconds=688.606),
    hashes=hashes,projected_sources=sources,launcher_checks=launchers,identical_helpers=helpers,car_objective_ast_equal=True,
    production_or_golden_changed=False)
(OUT/'historical-fingerprint-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],source_hashes_preserved=3,patch_hashes_explained=3,
                     source_bodies=3,launchers=6,identical_helpers=8,car_objective_ast_equal=True),ensure_ascii=False))
