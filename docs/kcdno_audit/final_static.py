"""Read-only final freeze/AST/source inventory against the actual pre-K4 snapshot."""
import ast
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from test_kcdno_tasks import strip_new
import yaml

before=json.loads((ROOT/'docs/kcdno_audit/k4/before.json').read_text())
snapshot=Path(before['source_snapshot'])
projects=('PDE-Solving-StandardBenchmark','Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS')
mutable={f'PDE-Solving-StandardBenchmark/{x}' for x in ('cdlno_entry.py','model_dict.py','exp_darcy.py','exp_elas.py','exp_airfoil.py','exp_pipe.py','exp_ns.py','exp_plas.py')}
mutable.update(f'{p}/{name}' for p in projects[1:] for name in ('main.py','main_evaluation.py'))
mutable.update(('Car-Design-ShapeNetCar/models/cdlno_run.py','Airfoil-Design-AirfRANS/cdlno_entry.py','Airfoil-Design-AirfRANS/params.yaml'))
frozen=[]
for name,digest in before['hashes'].items():
    if name in mutable:continue
    if (name.startswith(projects) or (name.startswith('cdlno/') and not name.startswith('cdlno/kcdno/') and name!='cdlno/experiment.py')
            or (name.startswith('tran_evaluate/') and not name.startswith('tran_evaluate/kcdno/'))
            or name in ('AGENTS.md','pyproject.toml','requirements.txt','Physics_Attention.py','path.sh')):
        p=ROOT/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        frozen.append(name)

class ParserProjection(ast.NodeTransformer):
    def visit_Expr(self,n):
        if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument':
            a=n.value.args
            if a and isinstance(a[0],ast.Constant) and a[0].value in ('--profile','--kernel-rank','--history-mode','--kcdno-run-dir'):
                return None
        return self.generic_visit(n)
    def visit_keyword(self,n):
        if n.arg=='choices' and isinstance(n.value,ast.Tuple):
            n.value.elts=[x for x in n.value.elts if not isinstance(x,ast.Constant) or x.value not in ('kcdno','lrsa_matched')]
        return self.generic_visit(n)

ast_checks=[]
for name in sorted(mutable):
    if name.endswith('.yaml'):
        old=yaml.safe_load((snapshot/name).read_text());new=yaml.safe_load((ROOT/name).read_text())
        assert {k:v for k,v in new.items() if k not in ('kcdno','lrsa_matched')}==old
        assert new['kcdno']==new['lrsa_matched']==old['Transolver']
    else:
        tree=ParserProjection().visit(strip_new(ast.parse((ROOT/name).read_text())))
        original=ast.parse((snapshot/name).read_text())
        assert ast.dump(tree)==ast.dump(original),name
    ast_checks.append(name)
versions={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() for name,root in [('integrated',ROOT),('lrsa',Path('/home/hwz/LRSA-Operator'))]}
result=dict(baseline_snapshot=str(snapshot),commits=versions,frozen_files=frozen,frozen_count=len(frozen),
            exact_ast_or_yaml_projection=ast_checks,projection_count=len(ast_checks),
            status='passed',scope='source-only; not real data/training acceptance')
(ROOT/'docs/kcdno_audit/k10/freeze.json').write_text(json.dumps(result,indent=2)+'\n')
print('Frozen byte-identical files:',len(frozen),'Exact whole-entry/helper/model-registry AST or YAML:',len(ast_checks))
