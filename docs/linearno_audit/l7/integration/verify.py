"""Read-only L7 integration preservation audit. No imports of task entries."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tests'))
from linearno_entry_projection import strip_linearno
HERE=Path(__file__).parent
baseline=json.loads((HERE/'baseline.json').read_text())
current={}
for kind,args in [('tracked',['ls-files','-z']),('untracked',['ls-files','--others','--exclude-standard','-z']),('ignored',['ls-files','--others','--ignored','--exclude-standard','-z'])]:
    for name in subprocess.check_output(['git',*args],cwd=ROOT).decode().split('\0'):
        p=ROOT/name
        if name and p.is_file():current[name]=dict(kind=kind,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
allowed={'Car-Design-ShapeNetCar/'+p for p in ('main.py','main_evaluation.py','train.py','models/cdlno_run.py')}
allowed|={'cdlno/linearno/profiles.py','tests/linearno_entry_projection.py','tests/linearno/test_legacy.py',
    'tests/test_shapenet_car.py','docs/LINEARNO_L7_REPORT.md','docs/LINEARNO_IMPLEMENTATION_STATUS.md',
    'docs/LINEARNO_REPRODUCTION_MATRIX.md','tran_evaluate/linearno/README.md'}
changed=[]
for row in baseline['files']:
    name=row['path'];now=current.get(name)
    assert now is not None,('missing baseline file',name)
    assert now['kind']==row['kind'],('changed classification',name)
    if now['sha256']!=row['sha256'] or now['size']!=row['size']:
        assert name in allowed,('unexpected edit',name)
        assert row['kind']!='ignored',('changed user cache/artifact',name)
        changed.append(name)
old={r['path'] for r in baseline['files']}
new=sorted(current.keys()-old)
new_allowed={'cdlno/linearno/car_entry.py','tests/linearno/car_entry_worker.py','tests/linearno/test_car_entry.py',
    'tran_evaluate/linearno/car_train.sh','tran_evaluate/linearno/car_eval.sh'}
for name in new:
    assert name in new_allowed or name.startswith('docs/linearno_audit/l7/integration/'),('unexpected new file',name)
    assert current[name]['kind']!='ignored',('new ignored output',name)
for name in ('main.py','main_evaluation.py','train.py','models/cdlno_run.py'):
    rel='Car-Design-ShapeNetCar/'+name
    before=Path(baseline['snapshot'])/'source'/rel
    assert ast.dump(strip_linearno(ast.parse((ROOT/rel).read_text())))==ast.dump(ast.parse(before.read_text())),rel
l0=json.loads((ROOT/'docs/linearno_audit/l0/freeze-manifest.json').read_text())
frozen_changes=[r['path'] for r in l0['files'] if current[r['path']]['sha256']!=r['sha256']]
prior={r['path']:r for r in baseline['files']}
historical=[r['path'] for r in l0['files'] if prior[r['path']]['sha256']!=r['sha256']]
new_frozen=sorted(set(frozen_changes)-set(historical))
assert new_frozen==['Car-Design-ShapeNetCar/models/cdlno_run.py','tests/test_shapenet_car.py'],new_frozen
math_files=[r for r in l0['files'] if any(s in Path(r['path']).name for s in ('Transolver','Physics_Attention','Embedding'))]
assert all(current[r['path']]['sha256']==r['sha256'] for r in math_files)
for name in l0['absent']:assert not (ROOT/name).exists(),name
assert subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','--','LINEARNO/'],cwd=ROOT)==b''
for name in new+changed:
    if name.endswith('.py'):ast.parse((ROOT/name).read_text(),feature_version=(3,10))
print(json.dumps(dict(check='PASS',stage='BLOCKED pending Car objective decision',baseline_files=len(old),
    changed=changed,new_files=new,new_ignored=[],frozen_changes_from_L0=frozen_changes,
    historical_broad_manifest_changes=historical,new_broad_manifest_changes=new_frozen,
    original_transolver_math_changes=0,car_complete_legacy_AST_equal=True,absent_paths=l0['absent'],
    hashes={name:current[name]['sha256'] for name in new+changed if name.endswith(('.py','.sh'))}),indent=2))
