"""Read-only source/fixture verification, writing M4 evidence only."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def write(name,value):(EVIDENCE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before=read(EVIDENCE/'before.json');archive=Path(before['archive'])/'source'
    expected_changes={'PDE-Solving-StandardBenchmark/model_dict.py','cdlno/msar_lno/registry.py',
        'tests/test_static_standard.py','docs/MSAR_LNO_IMPLEMENTATION_STATUS.md',
        'docs/CDLNO_IMPLEMENTATION_STATUS.md','memory/current-state.md'}
    changed=[];unchanged=[]
    for name,digest in before['files'].items():
        p=ROOT/name
        assert p.exists(),f'deleted existing file: {name}'
        (unchanged if sha(p)==digest else changed).append(name)
    assert set(changed)==expected_changes,changed
    added=['PDE-Solving-StandardBenchmark/model/MSAR_LNO.py','cdlno/msar_lno/entry.py',
        'cdlno/msar_lno/objective.py','cdlno/msar_lno/checkpoint.py',
        'tests/test_msar_integration.py','docs/MSAR_LNO_M4_INTEGRATION.md',
        'docs/msar_lno_audit/m4/construct_example.py','docs/msar_lno_audit/m4/finalize_evidence.py']
    patch=[]
    for name in sorted(changed+added):
        previous=(archive/name).read_text() if name in before['files'] else ''
        current=(ROOT/name).read_text()
        patch.extend(difflib.unified_diff(previous.splitlines(True),current.splitlines(True),
                                        fromfile='before/'+name,tofile='after/'+name))
        if name.endswith('.py'):
            ast.parse(current,feature_version=(3,10))
        assert all(line.rstrip()==line for line in current.splitlines()),name
    (EVIDENCE/'stage.patch').write_text(''.join(patch))
    old_index=read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for record in old_index['integrity_files']:
        assert sha(record['path'])==record['sha256'],record['path']
    core_index=read(ROOT/'docs/msar_lno_audit/m3/fixture-index.json')
    for record in core_index['records']:
        assert sha(record['fixture'])==record['sha256'],record['fixture']
    replay=[];not_run=[]
    for name in ('k0-replay','wrapper-replay','other-replay'):
        for group in read(EVIDENCE/(name+'.json')):
            assert group['returncode']==0,group
            for row in group.get('rows',group.get('summary',{}).get('rows',[])):
                if row['status']=='not_run':not_run.append(row);continue
                assert row['status']=='passed',row
                replay.append(dict(group=name,name=row['name'],status=row['status']))
    for row in read(EVIDENCE/'kcore-replay.json'):
        assert row['status']=='passed',row
        replay.append(dict(group='kcore-replay',name=row['name'],status=row['status']))
    assert len(replay)==75,len(replay)
    core_rows=read(EVIDENCE/'msar-core-replay.json')
    assert len(core_rows)==2 and all(r['status']=='passed' for r in core_rows)
    suites=[]
    for file,expected in (('msar-tests.log',66),('old-entry-tests.log',33),('integration-tests-final.log',18)):
        text=(EVIDENCE/file).read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
        assert match and int(match[1])==expected and '\nOK\n' in text and 'FAILED' not in text,file
        suites.append(dict(log=file,tests=int(match[1]),seconds=float(match[2]),failures=0,errors=0,skips=0))
    write('fixture-index.json',dict(old_index='docs/msar_lno_audit/m0/fixture-index.json',
        core_index='docs/msar_lno_audit/m3/fixture-index.json',old_fixture_files_verified=len(old_index['integrity_files']),
        core_fixture_files_verified=2,old_same_weight_passed=75,core_same_weight_passed=2,
        tolerance=dict(old_cpu=dict(dtype='float32',backend='MATH',atol=0,rtol=0),
                       core_cpu=dict(dtype='float32',backend='MATH',atol=0,rtol=0)),
        rows=replay,not_run=not_run,limits='Existing synthetic fixed-weight fixtures; not all historical trained checkpoints.'))
    factory=ast.parse((ROOT/'PDE-Solving-StandardBenchmark/model_dict.py').read_text())
    fn=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='get_model')
    expected=ast.parse("if args.model == 'msar_lno':\n    from model import MSAR_LNO\n    return MSAR_LNO").body[0]
    assert ast.dump(fn.body.pop(0))==ast.dump(expected)
    assert ast.dump(factory)==ast.dump(ast.parse((archive/'PDE-Solving-StandardBenchmark/model_dict.py').read_text()))
    freeze=dict(snapshot=str(archive),files_at_start=len(before['files']),unchanged_count=len(unchanged),
                changed=changed,added_source_test_report=added,factory_old_ast='exact after validating/removing one explicit new branch',
                unchanged_sha256={p:before['files'][p] for p in unchanged},
                current_changed_sha256={p:sha(ROOT/p) for p in changed+added},
                frozen='All existing task entry/train/eval/data/model math/config/dependency files unchanged; only listed factory/registry/test/docs changed.')
    write('freeze.json',freeze)
    write('summary.json',dict(stage='M4',environment=read(EVIDENCE/'environment.json'),suites=suites,
        distinct_test_methods=100,distinct_count_note='49 M1-M3 + 18 final M4 + 33 old entries; logs overlap, not a single 100-test invocation',
        old_same_weight_passed=75,msar_core_same_weight_passed=2,old_fixture_hashes=182,
        factory='Real PDE factory exports lifted MSARLNO only; eight production task train/eval wrappers not wired',
        loss='Original safe relative-L2 callable on synthetic lifted core + explicit coverage; not full Darcy/task loop',
        checkpoint='Strict bare state_dict, trusted whole/list/member core snapshots; no optimizer/RNG resume archive',
        gpu='Small CUDA MATH FP32 objective/AdamW/strict checkpoint passed, atol=1e-6 rtol=1e-5; existing M2/M3 limited AMP tests reran',
        not_run=['MSAR task/PyG wrappers and full task loss/time/decode/physical metrics',
                 'Real datasets/training/convergence/accuracy','Remote Python3.10/torch2.11/cu128',
                 'Task resume integration','Existing GUNet graph path: torch_cluster missing'],next_stage_executed=False))
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    print(json.dumps(dict(changed=changed,unchanged=len(unchanged),tests=100,old_fixtures=75,core_fixtures=2)))


if __name__=='__main__':main()
