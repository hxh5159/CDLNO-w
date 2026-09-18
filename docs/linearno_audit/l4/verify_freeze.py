"""Read-only L4 content/classification/AST freeze audit; JSON to stdout only."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests'))
from linearno_entry_projection import strip_linearno


def load(path):return json.loads((ROOT/path).read_text())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args]).decode()
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


baseline=load('docs/linearno_audit/l4/baseline.json')
current={}
for kind,args in [('tracked',('ls-files','-z')),('untracked',('ls-files','--others','--exclude-standard','-z')),
                  ('ignored',('ls-files','--others','--ignored','--exclude-standard','-z'))]:
    for p in filter(None,git(*args).split('\0')):
        if (ROOT/p).is_file():current[p]=dict(kind=kind,size=(ROOT/p).stat().st_size,sha256=sha(p))
integrated={f'PDE-Solving-StandardBenchmark/{name}' for name in
            ('exp_darcy.py','exp_elas.py','exp_airfoil.py','exp_pipe.py','model_dict.py','cdlno_entry.py')}
integrated.add('cdlno/experiment.py')
additional={'tests/msar_entry_projection.py','tests/linearno/test_legacy.py','cdlno/linearno/profiles.py',
            'docs/LINEARNO_IMPLEMENTATION_STATUS.md','docs/LINEARNO_REPRODUCTION_MATRIX.md'}
changed=[]
for row in baseline['files']:
    path=row['path'];now=current[path]
    assert now['kind']==row['kind'],path
    if now['sha256']!=row['sha256'] or now['size']!=row['size']:changed.append(path)
assert set(changed)==integrated|additional,changed
for path in integrated:
    before=ast.parse((Path(baseline['snapshot'])/'source'/path).read_text())
    actual=strip_linearno(ast.parse((ROOT/path).read_text()))
    assert ast.dump(actual)==ast.dump(before),path

l0_files=load('docs/linearno_audit/l0/manifest.json')
freeze=load('docs/linearno_audit/l0/freeze-manifest.json')
l0_changed=[];frozen_changed=[]
for rows,changed_list in ((l0_files,l0_changed),(freeze['files'],frozen_changed)):
    for row in rows:
        path=row['path'];now=current[path]
        assert now['kind']==row['classification'],path
        if now['sha256']!=row['sha256'] or now['size']!=row['size']:
            assert path in integrated|{'tests/msar_entry_projection.py'},path
            changed_list.append(path)
    for path in freeze['absent']:assert not (ROOT/path).exists(),path

core=[]
for row in baseline['files']:
    p=row['path']
    if (p=='Physics_Attention.py' or ('/model/' in p or '/models/' in p) and
            p.endswith('.py') and ('Transolver' in p or 'Physics_Attention' in p or 'Embedding' in p)):
        assert sha(p)==row['sha256'],p
        core.append(p)
math_sources=['cdlno/linearno/attention.py','PDE-Solving-StandardBenchmark/model/LinearNO.py',
              'PDE-Solving-StandardBenchmark/model/LinearNO_Attention.py',
              'Car-Design-ShapeNetCar/models/LinearNO_Attention.py','Airfoil-Design-AirfRANS/models/LinearNO_Attention.py']
for p in math_sources:assert p not in changed,p
for p in ['PDE-Solving-StandardBenchmark/exp_ns.py','PDE-Solving-StandardBenchmark/exp_plas.py']:
    assert p not in changed,p

new=sorted(set(current)-{row['path'] for row in baseline['files']})
new_ignored=[p for p in new if current[p]['kind']=='ignored'];assert not new_ignored,new_ignored
allowed_prefixes=('docs/linearno_audit/l4/','tran_evaluate/linearno/')
allowed={'PDE-Solving-StandardBenchmark/linearno_entry.py','cdlno/linearno/checkpoint.py',
         'cdlno/linearno/standard_entry.py','tests/linearno/static_worker.py',
         'tests/linearno/test_static_integration.py','tests/linearno_entry_projection.py','docs/LINEARNO_L4_REPORT.md'}
assert all(p in allowed or p.startswith(allowed_prefixes) for p in new),new
for p in [*changed,*new]:
    if p.endswith('.py'):ast.parse((ROOT/p).read_text(),feature_version=(3,10))
assert git('rev-parse','HEAD').strip()==baseline['head']
summary=load('docs/linearno_audit/l4/results-summary.json')
assert not summary['unresolved_failures']
assert all(not r['failures'] and not r['errors'] for r in summary['legacy_regression'])
print(json.dumps(dict(status='PASS',head=baseline['head'],baseline_files=len(baseline['files']),
    unchanged=len(baseline['files'])-len(changed),changed=changed,new_files=new,new_ignored=new_ignored,
    complete_legacy_AST_equal=sorted(integrated),l0_original=len(l0_files),l0_original_changes=l0_changed,
    l0_broad_freeze=len(freeze['files']),l0_broad_freeze_changes=frozen_changed,
    transolver_core_byte_equal=core,linearno_math_byte_equal=math_sources,
    absent_frozen_paths=freeze['absent'],ns_plasticity_production_unchanged=True,
    new_and_modified_source_hashes={p:sha(p) for p in [*changed,*new] if p.endswith(('.py','.sh'))},
    git_status=git('status','--porcelain=v1','--untracked-files=all'),
    frozen_path_status=git('status','--porcelain=v1','--untracked-files=all','--',*freeze['paths']),
    diff_stat=git('diff','--stat')),indent=2))
