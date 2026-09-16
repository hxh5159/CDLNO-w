"""M6 final source/fixture integrity and executed-result index; no task imports."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tests'))
from msar_entry_projection import strip_msar


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, data): (E/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def main():
    before=read(E/'before.json');snapshot=Path(before['archive'])/'source'
    expected={'PDE-Solving-StandardBenchmark/exp_ns.py','PDE-Solving-StandardBenchmark/exp_plas.py',
        'PDE-Solving-StandardBenchmark/model_dict.py','PDE-Solving-StandardBenchmark/msar_entry.py',
        'cdlno/msar_lno/standard_entry.py','cdlno/msar_lno/objective.py',
        'tests/test_msar_static.py','tests/test_static_standard.py',
        'tran_evaluate/msar_lno/_static.sh','tran_evaluate/msar_lno/README.md',
        'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md','docs/CDLNO_IMPLEMENTATION_STATUS.md','memory/current-state.md'}
    changed=[];unchanged=[]
    for name,digest in before['files'].items():
        assert (ROOT/name).is_file(),name
        (unchanged if sha(ROOT/name)==digest else changed).append(name)
    assert set(changed)==expected,changed
    added=['cdlno/msar_lno/temporal.py','PDE-Solving-StandardBenchmark/model/MSAR_Temporal.py',
        'PDE-Solving-StandardBenchmark/configs/msar_lno/ns.json',
        'PDE-Solving-StandardBenchmark/configs/msar_lno/plasticity.json',
        'tran_evaluate/msar_lno/ns.sh','tran_evaluate/msar_lno/plasticity.sh',
        'tests/test_msar_temporal.py','docs/MSAR_LNO_M6_TEMPORAL_TASKS.md',
        'docs/msar_lno_audit/m6/finalize_evidence.py']
    patch=[]
    for name in sorted(changed+added):
        old=(snapshot/name).read_text() if name in before['files'] else ''
        now=(ROOT/name).read_text()
        patch.extend(difflib.unified_diff(old.splitlines(True),now.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        if name.endswith('.py'):ast.parse(now,feature_version=(3,10))
        if name.endswith('.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True)
    (E/'stage.patch').write_text(''.join(patch))
    projected=[]
    for stem in ('ns','plas'):
        name=f'PDE-Solving-StandardBenchmark/exp_{stem}.py'
        a=ast.dump(strip_msar(ast.parse((ROOT/name).read_text())))
        b=ast.dump(ast.parse((snapshot/name).read_text()))
        assert a==b,name
        projected.append(dict(file=name,pre_M6_ast=hashlib.sha256(b.encode()).hexdigest(),status='exact'))
    name='PDE-Solving-StandardBenchmark/model_dict.py'
    now=ast.parse((ROOT/name).read_text());factory=next(n for n in now.body if isinstance(n,ast.FunctionDef))
    branch=factory.body[0].body.pop(0)
    wanted=ast.parse("if getattr(args, 'msar_task', None) in ('ns', 'plasticity'):\n    from model import MSAR_Temporal\n    return MSAR_Temporal").body[0]
    assert ast.dump(branch)==ast.dump(wanted)
    assert ast.dump(now)==ast.dump(ast.parse((snapshot/name).read_text()))
    suites=[]
    for name,count in [('msar-tests.log',93),('old-tests.log',93)]:
        log=(E/name).read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
        assert match and int(match[1])==count and '\nOK' in log and 'FAILED' not in log,name
        skipped=re.search(r'OK \(skipped=(\d+)\)',log)
        suites.append(dict(log=name,tests=count,seconds=float(match[2]),failures=0,errors=0,
                           skip_records=int(skipped[1]) if skipped else 0))
    rows=[]
    for name,expected_count in [('k0-temporal-replay',8),('wrapper-replay',18)]:
        group=read(E/(name+'.json'));assert len(group)==expected_count
        for row in group:
            assert row['status']=='passed',row
            rows.append(dict(group=name,name=row['name'],status='passed',atol=0,rtol=0))
    oldindex=read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for item in oldindex['integrity_files']:assert sha(item['path'])==item['sha256'],item['path']
    statics=read(E/'static-fixtures.json')
    for item in statics:assert sha(item['path'])==item['sha256'],item['path']
    write('fixture-index.json',dict(old_index='docs/msar_lno_audit/m0/fixture-index.json',
        old_artifact_files_preserved=len(oldindex['integrity_files']),old_replayed_now=len(rows),
        temporal_old_replayed=14,static_K_matched_replayed=12,rows=rows,
        accepted_M5_pre_M6_new_captures=statics,static_M5_replays=4,
        retained_evidence='Unaffected old math/core/industrial fixtures retain M4/M5 evidence; not all rerun.',
        limits='Same saved weights and fixed synthetic inputs; not real trained checkpoints or universal historical compatibility.'))
    cases=read(E/'task-cases.json')
    assert sum(bool(row.get('strict_checkpoint')) for row in cases)==4
    assert sum(row.get('device')=='cuda' for row in cases)==2
    assert sum(row.get('profile') in ('light','full') for row in cases)==4
    write('freeze.json',dict(snapshot=str(snapshot),files_at_start=len(before['files']),
        unchanged_count=len(unchanged),changed=changed,added=added,entries=projected,
        old_factory_ast='exact after removing the explicitly validated temporal MSAR branch',
        unchanged_sha256={p:before['files'][p] for p in unchanged},
        changed_sha256={p:sha(ROOT/p) for p in changed+added},
        frozen='All old model math, M1/M2/M3 math and M5 static wrapper/entries, data/dependencies and industrial entries unchanged.'))
    write('summary.json',dict(stage='M6',environment=read(E/'environment.json'),suites=suites,
        new_methods=11,temporal_task_modes=4,old_same_weight_replays=26,M5_same_weight_replays=4,
        preserved_old_fixture_files=len(oldindex['integrity_files']),
        actual_temporal_loops='NS 10 forwards/1 backward/optimizer/scheduler; Plasticity 20 forwards/backwards/optimizer, scheduler1',
        coverage='NS original PDE sum + weight * time mean of four-level raw coverage; Plasticity each PDE + weight * its four-level coverage',
        shell_train_previews=6,shell_eval_previews=6,
        fresh_cwd='two actual wrappers strict load from original PDE cwd, CPU FP32 MATH, atol=rtol=0',
        gpu='two temporal tasks, actual N, B1/d8/M7,5,3,2, complete original time-loop loss steps, FP32 MATH',
        profile_forward='actual Light/Full, B1/N4096 NS and N3131 Plasticity, CPU forward only',
        checkpoint='bare state_dict model.pt + strict architecture/task metadata, unchanged save cadence, no new resume',
        initial_failure='New negative test used history-mode=2, triggering argparse SystemExit before architecture validation; corrected to valid-but-inapplicable off. No production fix/tolerance change.',
        not_run=['real data reading/download/training/convergence/accuracy','remote Python3.10/torch2.11/cu128',
                 'formal Light/Full temporal backward/optimizer matrix','new temporal AMP/backends/compile matrix',
                 'MSAR industrial tasks/PyG integration (M7 not authorized)',
                 'existing AirfRANS sampled tests skipped for absent torch_cluster',
                 'new long-horizon NS experiment (no existing 20/40 CLI/data protocol)'],next_stage_executed=False))
    for doc in ('docs/MSAR_LNO_M6_TEMPORAL_TASKS.md','tran_evaluate/msar_lno/README.md'):
        p=ROOT/doc
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in target and not target.startswith('#'):assert (p.parent/target.split('#')[0]).exists(),target
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    (E/'status-after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    print(json.dumps(dict(tests=suites,changed=len(changed),added=len(added),unchanged=len(unchanged),
                         old_replayed=26,M5_replayed=4),ensure_ascii=False))


if __name__=='__main__':main()
