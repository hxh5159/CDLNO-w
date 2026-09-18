"""Read-only comparison to actual review start and immutable L0 model hashes."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tests'))
base=json.loads((HERE/'baseline.json').read_text())
allowed={'cdlno/linearno/air_entry.py','Airfoil-Design-AirfRANS/train.py','tests/linearno/air_entry_worker.py',
 'tests/linearno/test_air_entry.py','tran_evaluate/linearno/airfrans_train.sh','tran_evaluate/linearno/airfrans_eval.sh',
 'docs/LINEARNO_L6_REPORT.md','docs/LINEARNO_IMPLEMENTATION_STATUS.md','docs/LINEARNO_REPRODUCTION_MATRIX.md',
 'docs/LINEARNO_AIRFRANS_OBJECTIVE_DECISION.md','tran_evaluate/linearno/README.md'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changed=[p for p,r in base['files'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=r['sha256']]
assert set(changed)<=allowed, sorted(set(changed)-allowed)
from test_airfrans import AirFrozenChecks
path='Airfoil-Design-AirfRANS/train.py'
old=Path(base['artifact'])/'source'/path
assert ast.dump(AirFrozenChecks._strip_linearno_air(ast.parse(old.read_text())))==ast.dump(AirFrozenChecks._strip_linearno_air(ast.parse((ROOT/path).read_text())))
l0=json.loads((ROOT/'docs/linearno_audit/l0/freeze-manifest.json').read_text())
math=[r for r in l0['files'] if any(x in r['path'] for x in ('Physics_Attention','Embedding','/Transolver'))]
assert all(sha(ROOT/r['path'])==r['sha256'] for r in math)
ignored=[]
previous=json.loads((ROOT/'docs/linearno_audit/l7/integration/baseline.json').read_text())
for row in previous['files']:
 if row['kind']=='ignored':
  assert (ROOT/row['path']).exists() and sha(ROOT/row['path'])==row['sha256'],row['path']
  ignored.append(row['path'])
result=dict(changed=changed,old_air_train_AST_equal=True,transolver_math_hashes=len(math),unchanged_ignored=len(ignored),LINEARNO_status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','--','LINEARNO/'],cwd=ROOT).decode())
print(json.dumps(result,indent=2))
