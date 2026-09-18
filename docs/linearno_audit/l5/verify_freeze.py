"""L5 read-only classification/hash/AST preservation and Python3.10 syntax."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests'))
from linearno_entry_projection import strip_linearno

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args]).decode()
def load(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

baseline=load('docs/linearno_audit/l5/baseline.json')
current={}
for kind,args in [('tracked',('ls-files','-z')),('untracked',('ls-files','--others','--exclude-standard','-z')),
                  ('ignored',('ls-files','--others','--ignored','--exclude-standard','-z'))]:
    for path in filter(None,git(*args).split('\0')):
        if (ROOT/path).is_file():current[path]=dict(kind=kind,size=(ROOT/path).stat().st_size,sha256=sha(path))
allowed={'PDE-Solving-StandardBenchmark/exp_ns.py','PDE-Solving-StandardBenchmark/exp_plas.py',
         'PDE-Solving-StandardBenchmark/linearno_entry.py','cdlno/linearno/profiles.py',
         'cdlno/linearno/standard_entry.py','tests/linearno/test_legacy.py',
         'tests/linearno/test_static_integration.py','tests/test_front_task_modes.py',
         'tran_evaluate/linearno/_static.sh','tran_evaluate/linearno/README.md',
         'docs/LINEARNO_IMPLEMENTATION_STATUS.md','docs/LINEARNO_REPRODUCTION_MATRIX.md'}
changed=[]
for row in baseline['files']:
    path=row['path'];now=current[path]
    assert now['kind']==row['kind'],path
    if now['sha256']!=row['sha256'] or now['size']!=row['size']:
        assert path in allowed,path
        changed.append(path)
assert set(changed)==allowed,set(changed)^allowed
for path in ['PDE-Solving-StandardBenchmark/exp_ns.py','PDE-Solving-StandardBenchmark/exp_plas.py']:
    old=ast.parse((Path(baseline['snapshot'])/'source'/path).read_text())
    assert ast.dump(strip_linearno(ast.parse((ROOT/path).read_text())))==ast.dump(old),path
l0=load('docs/linearno_audit/l0/manifest.json')
integrated={f'PDE-Solving-StandardBenchmark/{p}' for p in
            ('exp_darcy.py','exp_elas.py','exp_airfoil.py','exp_pipe.py','exp_ns.py','exp_plas.py','model_dict.py','cdlno_entry.py')}
integrated.add('cdlno/experiment.py')
old_tests={'tests/msar_entry_projection.py','tests/test_front_task_modes.py'}
l0_changed=[]
for row in l0:
    path=row['path'];now=current[path]
    assert now['kind']==row['classification'],path
    if now['sha256']!=row['sha256'] or now['size']!=row['size']:
        assert path in integrated|old_tests,path
        l0_changed.append(path)
freeze=load('docs/linearno_audit/l0/freeze-manifest.json')
frozen_changed=[r['path'] for r in freeze['files'] if sha(r['path'])!=r['sha256']]
assert set(frozen_changed)=={'cdlno/experiment.py',*old_tests},frozen_changed
for path in freeze['absent']:assert not (ROOT/path).exists(),path
core=[]
for row in baseline['files']:
    path=row['path']
    if path.endswith('.py') and (path=='Physics_Attention.py' or '/model/' in path or '/models/' in path
                                 or path in ('cdlno/linearno/attention.py','cdlno/linearno/checkpoint.py')):
        assert sha(path)==row['sha256'],path
        core.append(path)
new=sorted(set(current)-{r['path'] for r in baseline['files']})
new_allowed={'tests/linearno/temporal_worker.py','tests/linearno/test_temporal_integration.py','docs/LINEARNO_L5_REPORT.md',
    *(f'tran_evaluate/linearno/{task}_{action}.sh' for task in ('ns','plasticity') for action in ('train','eval'))}
concurrent=load('docs/linearno_audit/l5/concurrent-files.json')
for row in concurrent:
    assert current[row['path']]=={k:row[k] for k in ('kind','size','sha256')},row['path']
assert all(p in new_allowed or p.startswith('docs/linearno_audit/l5/') or p in {r['path'] for r in concurrent} for p in new),new
assert not [p for p in new if current[p]['kind']=='ignored']
for p in changed+new:
    if p.endswith('.py'):ast.parse((ROOT/p).read_text(),feature_version=(3,10))
assert git('rev-parse','HEAD').strip()==baseline['head']
print(json.dumps(dict(status='PASS',head=baseline['head'],baseline_files=len(baseline['files']),
    unchanged=len(baseline['files'])-len(changed),changed=changed,new_files=new,new_ignored=[],concurrent_preserved=concurrent,
    temporal_legacy_complete_AST_equal=True,all_benchmark_model_files_byte_equal=core,
    l0_original=len(l0),l0_changes=l0_changed,l0_broad_freeze=len(freeze['files']),
    l0_broad_freeze_changes=frozen_changed,absent_frozen_paths=freeze['absent'],
    source_hashes={p:sha(p) for p in changed+new if p.endswith(('.py','.sh'))},
    git_status=git('status','--porcelain=v1','--untracked-files=all'),diff_stat=git('diff','--stat')),indent=2))
