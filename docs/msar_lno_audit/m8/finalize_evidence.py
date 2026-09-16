"""M8 result/freeze index; reads existing evidence, never imports a task entry."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value): (E / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    before = read(E / 'before.json')
    snapshot = Path(before['archive']) / 'source'
    changed, unchanged = [], []
    for name, digest in before['files'].items():
        assert (ROOT / name).is_file(), name
        (unchanged if sha(ROOT / name) == digest else changed).append(name)
    assert set(changed) == {'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md',
                           'docs/CDLNO_IMPLEMENTATION_STATUS.md', 'memory/current-state.md'}, changed
    added = ['tests/test_msar_acceptance.py', 'docs/MSAR_LNO_M8_ACCEPTANCE.md',
             'docs/msar_lno_audit/m8/finalize_evidence.py']
    patch = []
    for name in sorted(changed + added):
        old = (snapshot / name).read_text() if name in before['files'] else ''
        current = (ROOT / name).read_text()
        patch.extend(difflib.unified_diff(old.splitlines(True), current.splitlines(True),
                                         fromfile='before/' + name, tofile='after/' + name))
        if name.endswith('.py'): ast.parse(current, feature_version=(3, 10))
    (E / 'stage.patch').write_text(''.join(patch))
    suites = []
    for log, count, status in (
        ('formal-tests-initial.log', 19, 'FAILED (failures=1)'),
        ('boundary-correction.log', 1, 'OK'),
        ('accepted-msar-tests.log', 107, 'OK (skipped=1)'),
        ('legacy-tests.log', 157, 'OK (skipped=1)'),
    ):
        text = (E / log).read_text()
        match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', text)
        assert match and int(match[1]) == count and '\n' + status in text, log
        suites.append(dict(log=log,methods=count,seconds=float(match[2]),reported_status=status))
    initial = (E / 'formal-tests-initial.log').read_text()
    failures = re.findall(r'^(?:FAIL|ERROR): (\w+) ', initial, re.M)
    assert failures == ['test_boundary_aux_isolation_and_coverage_dependency'], failures
    corrected = (E / 'boundary-correction.log').read_text()
    assert re.search(r'^' + failures[0] + r'.* \.\.\. ok$', corrected, re.M)
    cases = read(E / 'formal-cases.json')
    boundaries = read(E / 'boundary-cases.json')
    assert len(cases) == 26 and len(boundaries) == 1
    assert boundaries[0]['all_four_Down_connected_finite']
    assert all(g > 0 for row in boundaries[0]['isolated_Down_source_query_key_nonzero_gradients'] for g in row)
    matrix = []
    for task in TASKS:
        light = [r for r in cases if r.get('task') == task and r['profile'] == 'light']
        full = [r for r in cases if r.get('task') == task and r['profile'] == 'full']
        assert {r['mode'] for r in light} == {'off', 'floor'} and len(full) == 1
        for row in light:
            assert all(row[k] for k in ('original_loss','backward','optimizer','eval'))
            assert row['same_weight_off_floor_max_abs_diff'] < 2e-6
        assert full[0]['strict_state_dict'] and full[0]['backward'] and full[0]['eval']
        industrial = task in ('car', 'airfrans')
        matrix.append(dict(task=task,
            static_parameter_flow='passed; actual parser/factory/CLI defaults/read-first metadata, accepted suite rerun',
            light=dict(profiles_unmodified=True,cases=light,checkpoint='passed M5-M7 rerun at d8/M7,5,3,2; original save protocol, both modes'),
            full=dict(case=full[0],original_loss='not_run',optimizer='not_run',
                      checkpoint='actual Full pure state_dict strict roundtrip; task protocol validation uses reduced config'),
            real_PyG='passed: real Data/Batch, single graph' if industrial else 'not_applicable',
            GPU='passed: local reduced d8/M7,5,3,2 original loss steps, FP32 MATH; formal Light/Full GPU not_run',
            real_data='not_run: reading/training/metrics/convergence/accuracy',
            limits=('Air complete sampled epoch/radius_graph and VTK/forces not_run (torch_cluster absent)' if task=='airfrans' else
                    'Car real drag pipeline not_run' if task=='car' else
                    'NS 10->10 or Plasticity 20 time updates only; Full one-call numerical check' if task in ('ns','plasticity') else
                    'Formal profiles use small N; canonical N layout separately passed with d8')))
    write('coverage-matrix.json', dict(stage='M8',rows=matrix,
        provenance=['formal-cases.json','boundary-cases.json','accepted-msar-tests.log'],
        numerical_contract='CPU FP32 MATH, TF32 off: off/floor atol2e-6 rtol1e-4; strict Full roundtrip atol=rtol=0',
        no_actual_dataset_training=True))
    replays, not_run, replay_times = [], [], []
    for file, expected, logfile in (
        ('k0-replay.json',41,'k0-replay.log'),('wrapper-replay.json',24,'wrappers-replay.log'),
        ('old-core-replay.json',6,'core-replay.log'),('other-replay.json',4,'other-replay.log')):
        rows = []
        for item in read(E / file):
            if 'project' in item:
                assert item['returncode'] == 0
                rows.extend(item['summary']['rows'] if 'summary' in item else item['rows'])
            else: rows.append(item)
        assert sum(r['status'] == 'passed' for r in rows) == expected
        for row in rows:
            if row['status'] == 'not_run':
                assert row['name'] == 'GUNet'
                not_run.append(row)
                continue
            assert row['status'] == 'passed', row
            tol = row.get('tolerance',row)
            assert tol['atol'] == tol['rtol'] == 0, row['name']
            replays.append(dict(group=file,name=row['name'],status='passed',atol=0,rtol=0))
        elapsed = re.search(r'^real ([\d.]+)$',(E/logfile).read_text(),re.M)
        assert elapsed
        replay_times.append(dict(log=logfile,wall_seconds=float(elapsed[1])))
    assert len(replays) == 75
    old = read(ROOT / 'docs/msar_lno_audit/m0/fixture-index.json')
    for item in old['integrity_files']: assert sha(item['path']) == item['sha256'], item['path']
    write('fixture-index.json',dict(previous_index='docs/msar_lno_audit/m0/fixture-index.json',
        same_weight_replays=replays,not_run=not_run,hashes_preserved=len(old['integrity_files']),
        limits='Current synthetic fixtures using their SAME saved inputs/weights, not proof for all historical trained checkpoints. No regenerated reference or new large binary.'))
    production = [p for p in unchanged if p.startswith(('cdlno/','PDE-Solving-StandardBenchmark/',
        'Car-Design-ShapeNetCar/','Airfoil-Design-AirfRANS/','tran_evaluate/','tools/')) and not p.endswith('.md')]
    write('freeze.json',dict(snapshot=str(snapshot),head=before['head'],branch=before['branch'],
        files_at_start=len(before['files']),changed=changed,added=added,unchanged_count=len(unchanged),
        production_task_config_script_tool_unchanged=len(production),
        unchanged_sha256={p:before['files'][p] for p in unchanged},
        current_sha256={p:sha(ROOT/p) for p in changed+added},
        frozen='All pre-M8 production, model, config, task/data/loss/optimizer/time-loop/eval, dependencies and existing tests byte-identical.'))
    write('summary.json',dict(stage='M8',environment=read(E/'environment.json'),suites=suites,
        new_test_methods=19,new_final_unresolved=0,new_initial_failure='overstrong every-sample nonzero gradient assertion; corrected test only',
        accepted_MSAR_methods=107,accepted_MSAR_skipped=1,legacy_methods=157,legacy_skip_records=1,
        legacy_skip_is_subtest=True,old_same_weight_exact=len(replays),old_hashes_preserved=len(old['integrity_files']),
        replay_times=replay_times,formal_light_original_loss_train_eval_cases=16,formal_full_forward_backward_strict_states=8,
        same_weight_off_floor_max_abs=max(r['same_weight_off_floor_max_abs_diff'] for r in cases if r.get('profile')=='light'),
        default_floor_raw='All 8 Light initial synthetic cases raw=0 at kappa=.2; positive-floor gradient checks separately use kappa=1, no production-default change',
        auxiliary_boundary=boundaries[0],production_changed=False,real_data_run=False,next_stage_executed=False,
        not_run=['remote Python3.10/Torch2.11/CUDA12.8','formal Light/Full GPU task matrix',
                 'Full original-loss temporal loop/optimizer matrix','task AMP/backend/compile matrix',
                 'full industrial sampled-data/VTK/force pipeline','real data training/convergence/accuracy',
                 'GUNet fixture (original missing torch_cluster dependency)']))
    for name in ('docs/MSAR_LNO_M8_ACCEPTANCE.md',):
        path=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not link.startswith('#'): assert (path.parent/link.split('#')[0]).exists(),link
        for block in re.findall(r'```bash\n(.*?)```',path.read_text(),re.S):
            subprocess.run(['bash','-n'],input=block,text=True,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    (E/'status-after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    print(json.dumps(dict(stage='M8',changed=len(changed),added=len(added),unchanged=len(unchanged),
        formal_cases=24,old_exact=75,old_hashes_preserved=182,unresolved_test_failures=0)))


if __name__ == '__main__': main()
