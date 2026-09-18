"""L6 read-only source/classification preservation check against L6 and L0."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode()
def load(path):
    return json.loads((ROOT/path).read_text())
def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

baseline = load('docs/linearno_audit/l6/baseline.json')
current = {}
for kind, args in [('tracked', ('ls-files', '-z')),
                   ('untracked', ('ls-files', '--others', '--exclude-standard', '-z')),
                   ('ignored', ('ls-files', '--others', '--ignored', '--exclude-standard', '-z'))]:
    for path in filter(None, git(*args).split('\0')):
        if (ROOT/path).is_file():
            current[path] = dict(kind=kind, size=(ROOT/path).stat().st_size, sha256=sha(path))

allowed_edits = {'docs/LINEARNO_IMPLEMENTATION_STATUS.md', 'docs/LINEARNO_REPRODUCTION_MATRIX.md'}
changed = []
for row in baseline['files']:
    path = row['path']
    assert path in current, ('missing', path)
    assert current[path]['kind'] == row['kind'], ('classification', path)
    if current[path]['sha256'] != row['sha256'] or current[path]['size'] != row['size']:
        assert path in allowed_edits, ('unexpected edit', path)
        changed.append(path)
new = sorted(set(current) - {r['path'] for r in baseline['files']})
new_allowed = {'cdlno/linearno/airfrans.py', 'Airfoil-Design-AirfRANS/models/LinearNO.py',
               'tests/linearno/test_airfrans_model.py', 'docs/LINEARNO_L6_REPORT.md'}
for path in new:
    assert path in new_allowed or path.startswith('docs/linearno_audit/l6/'), ('unexpected new file', path)
    assert current[path]['kind'] != 'ignored', ('new ignored file', path)
for path in new + changed:
    if path.endswith('.py'):
        ast.parse((ROOT/path).read_text(), feature_version=(3, 10))

l0 = load('docs/linearno_audit/l0/manifest.json')
l0_allowed = {f'PDE-Solving-StandardBenchmark/{p}' for p in
              ('exp_darcy.py', 'exp_elas.py', 'exp_airfoil.py', 'exp_pipe.py', 'exp_ns.py', 'exp_plas.py',
               'model_dict.py', 'cdlno_entry.py')}
l0_allowed |= {'cdlno/experiment.py', 'tests/msar_entry_projection.py', 'tests/test_front_task_modes.py'}
l0_changes = []
for row in l0:
    path = row['path']
    assert current[path]['kind'] == row['classification'], path
    if current[path]['sha256'] != row['sha256'] or current[path]['size'] != row['size']:
        assert path in l0_allowed, ('unexpected L0 change', path)
        l0_changes.append(path)
freeze = load('docs/linearno_audit/l0/freeze-manifest.json')
frozen_changed = [r['path'] for r in freeze['files'] if sha(r['path']) != r['sha256']]
assert set(frozen_changed) == {'cdlno/experiment.py', 'tests/msar_entry_projection.py', 'tests/test_front_task_modes.py'}
for path in freeze['absent']:
    assert not (ROOT/path).exists(), path
assert git('status', '--porcelain=v1', '--untracked-files=all', '--', 'LINEARNO/') == ''
assert git('rev-parse', 'HEAD').strip() == baseline['head']
assert git('diff') == (ROOT/'docs/linearno_audit/l6/before.diff').read_text()
old_air = [r['path'] for r in baseline['files'] if r['path'].startswith('Airfoil-Design-AirfRANS/')
           and r['path'].endswith(('.py', '.yaml', '.sh'))]
print(json.dumps(dict(check_status='PASS', stage_status='BLOCKED pending C08',
    baseline_files=len(baseline['files']), unchanged=len(baseline['files'])-len(changed),
    changed=changed, new_files=new, new_ignored=[], tracked_diff_equal_L6_start=True,
    old_AirfRANS_source_byte_identical=old_air, l0_files=len(l0), l0_changes=l0_changes,
    l0_broad_frozen_files=len(freeze['files']), cumulative_L4_L5_frozen_changes=frozen_changed,
    absent_frozen_paths=freeze['absent'], python310_syntax=True,
    new_source_hashes={p:sha(p) for p in new if p.endswith('.py')},
    git_status=git('status','--porcelain=v1','--untracked-files=all')), indent=2))
