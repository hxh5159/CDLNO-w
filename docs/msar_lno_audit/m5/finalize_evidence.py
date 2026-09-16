"""M5 source/fixture audit; writes evidence only, never runs task/data entries."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from msar_entry_projection import strip_msar


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):(E/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def main():
    before=read(E/'before.json');snapshot=Path(before['archive'])/'source'
    expected={'PDE-Solving-StandardBenchmark/cdlno_entry.py','PDE-Solving-StandardBenchmark/model_dict.py',
        *(f'PDE-Solving-StandardBenchmark/exp_{s}.py' for s in ('darcy','elas','airfoil','pipe')),
        'cdlno/experiment.py','cdlno/msar_lno/objective.py','tests/test_front_training.py',
        'tests/test_kcdno_tasks.py','tests/test_static_standard.py','tests/output_recording_projection.py',
        'tests/test_periodic_visualization.py','docs/MSAR_LNO_IMPLEMENTATION_STATUS.md',
        'docs/CDLNO_IMPLEMENTATION_STATUS.md','memory/current-state.md'}
    changed=[];unchanged=[]
    for name,digest in before['files'].items():
        assert (ROOT/name).is_file(),name
        (unchanged if sha(ROOT/name)==digest else changed).append(name)
    assert set(changed)==expected,changed
    added=['cdlno/msar_lno/standard.py','cdlno/msar_lno/standard_entry.py',
        'PDE-Solving-StandardBenchmark/msar_entry.py','PDE-Solving-StandardBenchmark/model/MSAR_Standard.py',
        *(f'PDE-Solving-StandardBenchmark/configs/msar_lno/{s}.json' for s in ('darcy','elasticity','airfoil','pipe')),
        *(f'tran_evaluate/msar_lno/{s}.sh' for s in ('darcy','elasticity','airfoil','pipe','_static')),
        'tran_evaluate/msar_lno/README.md','tests/test_msar_static.py','tests/msar_entry_projection.py',
        'docs/MSAR_LNO_M5_STATIC_TASKS.md','docs/msar_lno_audit/m5/finalize_evidence.py']
    patch=[]
    for name in sorted(changed+added):
        old=(snapshot/name).read_text() if name in before['files'] else ''
        now=(ROOT/name).read_text()
        patch.extend(difflib.unified_diff(old.splitlines(True),now.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        if name.endswith('.py'):ast.parse(now,feature_version=(3,10))
        if name.endswith('.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True)
    (E/'stage.patch').write_text(''.join(patch))
    projected=[]
    for stem in ('darcy','elas','airfoil','pipe'):
        name=f'PDE-Solving-StandardBenchmark/exp_{stem}.py'
        a=ast.dump(strip_msar(ast.parse((ROOT/name).read_text())))
        b=ast.dump(ast.parse((snapshot/name).read_text()))
        assert a==b,name
        projected.append(dict(file=name,pre_M5_ast=hashlib.sha256(b.encode()).hexdigest(),status='exact'))
    name='PDE-Solving-StandardBenchmark/model_dict.py'
    now=ast.parse((ROOT/name).read_text());f=next(n for n in now.body if isinstance(n,ast.FunctionDef))
    nested=f.body[0].body.pop(0)
    wanted=ast.parse("if getattr(args, 'msar_task', None) in ('darcy', 'elasticity', 'airfoil', 'pipe'):\n    from model import MSAR_Standard\n    return MSAR_Standard").body[0]
    assert ast.dump(nested)==ast.dump(wanted)
    assert ast.dump(now)==ast.dump(ast.parse((snapshot/name).read_text()))
    suites=[]
    for file,count,skip in [('msar-tests-final.log',82,0),('old-tests-final.log',79,2)]:
        text=(E/file).read_text();m=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
        assert m and int(m[1])==count and '\nOK' in text and 'FAILED' not in text,file
        suites.append(dict(log=file,tests=count,seconds=float(m[2]),failures=0,errors=0,skip_records=skip))
    rows=[]
    for name in ('k0-replay','wrapper-replay'):
        for group in read(E/(name+'.json')):
            assert group['returncode']==0
            for row in group.get('rows',group.get('summary',{}).get('rows',[])):
                assert row['status']=='passed',row
                rows.append(dict(group=name,name=row['name'],status='passed',atol=0,rtol=0))
    assert len(rows)==65
    oldindex=read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for f in oldindex['integrity_files']:assert sha(f['path'])==f['sha256'],f['path']
    coreindex=read(ROOT/'docs/msar_lno_audit/m3/fixture-index.json')
    for f in coreindex['records']:assert sha(f['fixture'])==f['sha256'],f['fixture']
    assert all(r['status']=='passed' for r in read(E/'msar-core-replay.json'))
    write('fixture-index.json',dict(old_index='docs/msar_lno_audit/m0/fixture-index.json',
        msar_core_index='docs/msar_lno_audit/m3/fixture-index.json',old_files_verified=182,
        old_replayed_now=65,msar_core_replayed_now=2,rows=rows,
        retained_evidence=dict(count=10,source='docs/msar_lno_audit/m4/fixture-index.json',
            reason='6 K/matched standalone core + 4 other models unchanged; retained M4 exact evidence'),
        limits='Fixed synthetic same-weight references, not real dataset results or universal historical checkpoint compatibility'))
    cases=read(E/'task-cases.json')
    assert sum(bool(r.get('strict_checkpoint')) for r in cases)==8
    assert sum(r.get('device')=='cuda' for r in cases)==4
    write('freeze.json',dict(snapshot=str(snapshot),files_at_start=len(before['files']),
        unchanged_count=len(unchanged),changed=changed,added=added,entries=projected,
        old_factory_ast='exact after removing validated nested static-task selection',
        unchanged_sha256={p:before['files'][p] for p in unchanged},
        changed_sha256={p:sha(ROOT/p) for p in changed+added},
        frozen='Old model math, MSAR M1/M2/M3 math, data/dependencies, temporal and industrial entries unchanged.'))
    write('summary.json',dict(stage='M5',environment=read(E/'environment.json'),suites=suites,
        msar_new_methods=15,old_same_weight_replays=65,msar_core_same_weight_replays=2,
        preserved_old_artifact_hashes=182,retained_old_evidence=10,
        static_task_modes=8,shell_train_previews=12,shell_eval_previews=12,
        fresh_cwd='four real static wrapper strict loads, identical weights, CPU FP32 MATH atol=rtol=0',
        gpu='4 small-task original loss/optimizer steps, FP32 MATH; limited preexisting M2/M3 AMP tests reran',
        profile_forward='Real Light Elasticity N35 and Full N972/M1024, CPU, no token clipping, no backward in these two formal cases',
        checkpoint='bare state_dict plus strict architecture/task sidecars, original save frequency, no optimizer/RNG resume',
        initial_test_corrections=['one-tree AST identity for Elasticity','separate Darcy rectangular wrapper from square derivative entry',
                                  'same MATH backend in fresh process without loosening zero tolerance',
                                  'new-family AST projection in old recording/visualization checks'],
        not_run=['real data/download/training/convergence/accuracy','remote Python3.10/torch2.11/cu128',
                 'MSAR temporal/industrial/PyG tasks','formal full-size task training/AMP matrix',
                 'existing AirfRANS sampled graph checks: torch_cluster missing'],next_stage_executed=False))
    for doc in ('docs/MSAR_LNO_M5_STATIC_TASKS.md','tran_evaluate/msar_lno/README.md'):
        path=ROOT/doc
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in target and not target.startswith('#'):assert (path.parent/target.split('#')[0]).exists(),target
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    print(json.dumps(dict(changed=changed,unchanged=len(unchanged),tests=[82,79],old_replayed=65,core_replayed=2)))


if __name__=='__main__':main()
