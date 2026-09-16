"""M7 source/fixture integrity and actual result index, no task/data imports."""
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
def write(name,data):(E/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    before=read(E/'before.json');snapshot=Path(before['archive'])/'source'
    expected={
        'Car-Design-ShapeNetCar/main.py','Car-Design-ShapeNetCar/main_evaluation.py',
        'Car-Design-ShapeNetCar/train.py','Car-Design-ShapeNetCar/models/cdlno_run.py',
        'Airfoil-Design-AirfRANS/main.py','Airfoil-Design-AirfRANS/main_evaluation.py',
        'Airfoil-Design-AirfRANS/train.py','Airfoil-Design-AirfRANS/cdlno_entry.py','Airfoil-Design-AirfRANS/params.yaml',
        'tests/msar_entry_projection.py','tests/test_airfrans.py','tests/test_experiment_records.py',
        'tests/test_kcdno_tasks.py','tests/test_kcdno_car.py','tests/test_kcdno_airfrans.py',
        'tran_evaluate/msar_lno/README.md','docs/MSAR_LNO_IMPLEMENTATION_STATUS.md',
        'docs/CDLNO_IMPLEMENTATION_STATUS.md','memory/current-state.md'}
    changed=[];unchanged=[]
    for name,digest in before['files'].items():
        assert (ROOT/name).is_file(),name
        (unchanged if sha(ROOT/name)==digest else changed).append(name)
    assert set(changed)==expected,changed
    added=['cdlno/msar_lno/industrial.py','cdlno/msar_lno/industrial_entry.py','cdlno/msar_lno/car_preflight.py',
           'Car-Design-ShapeNetCar/models/MSAR_LNO.py','Car-Design-ShapeNetCar/msar_entry.py',
           'Car-Design-ShapeNetCar/configs/msar_lno/car.json',
           'Airfoil-Design-AirfRANS/models/MSAR_LNO.py','Airfoil-Design-AirfRANS/msar_entry.py',
           'Airfoil-Design-AirfRANS/configs/msar_lno/airfrans.json',
           'tran_evaluate/msar_lno/car.sh','tran_evaluate/msar_lno/airfrans.sh','tran_evaluate/msar_lno/_dispatch.sh',
           'tests/test_msar_industrial.py','docs/MSAR_LNO_M7_INDUSTRIAL_TASKS.md','docs/msar_lno_audit/m7/finalize_evidence.py']
    patch=[]
    for name in sorted(changed+added):
        old=(snapshot/name).read_text() if name in before['files'] else ''
        now=(ROOT/name).read_text()
        patch.extend(difflib.unified_diff(old.splitlines(True),now.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        if name.endswith('.py'):ast.parse(now,feature_version=(3,10))
        if name.endswith('.sh'):subprocess.run(['bash','-n',str(ROOT/name)],check=True)
    (E/'stage.patch').write_text(''.join(patch))
    projections=[]
    for project in ('Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS'):
        for file in ('main.py','main_evaluation.py','train.py'):
            name=project+'/'+file
            a=ast.dump(strip_msar(ast.parse((ROOT/name).read_text())))
            b=ast.dump(ast.parse((snapshot/name).read_text()))
            assert a==b,name
            projections.append(dict(file=name,pre_M7_AST_sha256=hashlib.sha256(b.encode()).hexdigest(),status='exact'))
    suites=[]
    for name,count,status in [('msar-tests.log',107,'OK (skipped=1)'),
                              ('old-tests.log',90,'FAILED (failures=1, errors=1, skipped=2)'),
                              ('old-tests-corrections.log',2,'OK')]:
        log=(E/name).read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
        assert match and int(match[1])==count and '\n'+status in log,name
        suites.append(dict(log=name,methods=count,seconds=float(match[2]),reported_status=status))
    # The only old suite failures must be the two corrected test adaptations.
    oldlog=(E/'old-tests.log').read_text();corrections=(E/'old-tests-corrections.log').read_text()
    problems=re.findall(r'^(?:FAIL|ERROR): (\w+) ',oldlog,re.M)
    assert set(problems)=={'test_cli_yaml_inheritance_and_explicit_overrides',
                          'test_air_weighted_loss_and_real_epoch_record_fragment_without_sampling'}
    for method in problems:assert re.search(r'^'+method+r'.* \.\.\. ok$',corrections,re.M),method
    replays=[]
    for name,size in [('k0-car-replay',4),('k0-airfrans-replay',4),('car-wrapper-replay',3),('airfrans-wrapper-replay',3)]:
        rows=read(E/(name+'.json'));assert len(rows)==size
        for row in rows:
            assert row['status']=='passed',row
            replays.append(dict(group=name,name=row['name'],status='passed',atol=0,rtol=0))
    oldindex=read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for item in oldindex['integrity_files']:assert sha(item['path'])==item['sha256'],item['path']
    write('fixture-index.json',dict(previous_index='docs/msar_lno_audit/m0/fixture-index.json',
        preserved_artifact_files=len(oldindex['integrity_files']),replayed=len(replays),results=replays,
        unchanged_other_evidence='Pre-M1 core / K0 standard / M0 standard wrappers retain accepted prior evidence; not rerun here.',
        limits='Fixed synthetic inputs and the same saved weights. No real trained historical industrial checkpoint supplied; not universal historical compatibility.'))
    cases=read(E/'task-cases.json')
    assert sum(bool(c.get('strict_checkpoint')) for c in cases)==4
    assert sum(c.get('device')=='cuda' for c in cases)==2
    assert sum(bool(c.get('complete_synthetic_epoch')) for c in cases)==2
    assert sum(bool(c.get('epoch_record_fragment')) for c in cases)==2
    assert sum(c.get('profile') in ('light','full') for c in cases)==4
    write('freeze.json',dict(snapshot=str(snapshot),head=before['head'],files_at_start=len(before['files']),
        changed=changed,added=added,unchanged_count=len(unchanged),industrial_complete_ast=projections,
        unchanged_sha256={name:before['files'][name] for name in unchanged},
        current_sha256={name:sha(ROOT/name) for name in changed+added},
        frozen='All old model math, MSAR M1-M6 mathematics, six PDE entries, all loaders/metrics/dependencies remain byte-identical. Six complete industrial entry/train ASTs equal pre-M7 after exact new-family projection.'))
    write('summary.json',dict(stage='M7',environment=read(E/'environment.json'),suites=suites,
        new_test_methods=14,MSAR_passed=106,MSAR_skipped=1,
        old_final_unresolved_failures=0,old_initial_failures_corrected_in_tests=problems,old_skip_records=2,
        old_same_weight_exact=14,old_fixture_hashes_preserved=len(oldindex['integrity_files']),
        original_loss='Car velocity-all MSE + reg*surface pressure MSE; Air MSE_weighted volume+reg*surface',
        coverage='Per actual graph forward: weight * four-encoder-Down raw mean. Epoch logging mean per optimizer step.',
        checkpoint='Car whole object; Air whole member and model list, original trusted local boundary; no new resume',
        preview_actual_parser=dict(train=6,eval=6),fresh_cwd_graph_loads=4,
        GPU='2 real PyG synthetic graph original-loss steps, reduced d8/M7,5,3,2 FP32 MATH, no AMP task acceptance',
        profiles='Actual Light and Full parameters/construction/CPU B1 N11 forward, permitted M1>N expansion',
        not_run=['real data reads/download/training/convergence/accuracy','remote Python3.10/torch2.11/cu128',
                 'Air full sampled epoch/radius_graph/VTK/physical metrics (torch_cluster absent)',
                 'Car real drag postprocessing','formal full-size industrial profile backward/performance',
                 'industrial AMP/other backend/compile matrix'],next_stage_executed=False))
    for doc in ('docs/MSAR_LNO_M7_INDUSTRIAL_TASKS.md','tran_evaluate/msar_lno/README.md'):
        path=ROOT/doc
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in target and not target.startswith('#'):assert (path.parent/target.split('#')[0]).exists(),target
        for block in re.findall(r'```bash\n(.*?)```',path.read_text(),re.S):
            subprocess.run(['bash','-n'],input=block,text=True,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    (E/'status-after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    print(json.dumps(dict(stage='M7',changed=len(changed),added=len(added),unchanged=len(unchanged),
        MSAR_passed=106,MSAR_skipped=1,old_exact=14,old_unresolved=0),ensure_ascii=False))


if __name__=='__main__':main()
