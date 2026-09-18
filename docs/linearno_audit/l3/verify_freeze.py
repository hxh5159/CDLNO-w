"""Read-only L3 delivery check, including baseline untracked/ignored files.

Run from any cwd; writes JSON only to stdout. Never restores or changes files.
The explicit additions below are the L3 scope, not a blanket directory exemption.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode()


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT/path).read_text())


baseline = load('docs/linearno_audit/l3/baseline.json')
current = {}
for classification, args in (
    ('tracked', ('ls-files', '-z')),
    ('untracked', ('ls-files', '--others', '--exclude-standard', '-z')),
    ('ignored', ('ls-files', '--others', '--ignored', '--exclude-standard', '-z')),
):
    for path in filter(None, git(*args).split('\0')):
        if (ROOT/path).is_file():
            current[path] = dict(classification=classification, size=(ROOT/path).stat().st_size, sha256=digest(path))

status_path = 'docs/LINEARNO_IMPLEMENTATION_STATUS.md'
changed = []
for row in baseline['files']:
    path = row['path']; actual = current[path]
    assert actual['classification'] == row['kind'], path
    if actual['sha256'] != row['sha256'] or actual['size'] != row['size']:
        changed.append(path)
assert changed == [status_path], changed

counts = {}
for label, rows in (
    ('l0_original', load('docs/linearno_audit/l0/manifest.json')),
    ('l0_frozen', load('docs/linearno_audit/l0/freeze-manifest.json')['files']),
):
    for row in rows:
        path = row['path']
        assert current[path] == {k: row[k] for k in ('classification', 'size', 'sha256')}, path
    counts[label+'_unchanged'] = len(rows)
for stage in ('l1', 'l2'):
    for path, expected in load(f'docs/linearno_audit/{stage}/delivery-check.json')['new_source_hashes'].items():
        assert digest(path) == expected, path
    counts[stage+'_new_sources_unchanged'] = True

new = sorted(set(current) - {row['path'] for row in baseline['files']})
allowed = {
    'PDE-Solving-StandardBenchmark/model/LinearNO.py',
    'tests/linearno/standard_support.py',
    'tests/linearno/standard_reference.py',
    'tests/linearno/test_standard_model.py',
    'tests/linearno/test_standard_structure.py',
    'tests/linearno/fixtures/official_standard_models.json',
    'docs/LINEARNO_L3_REPORT.md',
}
assert all(p in allowed or p.startswith('docs/linearno_audit/l3/') for p in new), new
ignored = [p for p in new if current[p]['classification'] == 'ignored']
assert not ignored, ignored
for path in load('docs/linearno_audit/l0/freeze-manifest.json')['absent']:
    assert not (ROOT/path).exists(), path
assert git('rev-parse', 'HEAD').strip() == baseline['head']
assert git('diff') == (ROOT/'docs/linearno_audit/l3/tracked.diff').read_text()
assert git('diff', '--cached') == (ROOT/'docs/linearno_audit/l3/staged.diff').read_text()

syntax = [p for p in new if p.endswith('.py')]
for path in syntax:
    ast.parse((ROOT/path).read_text(), feature_version=(3, 10))
results = load('docs/linearno_audit/l3/regression-results.json')
existing = load('docs/linearno_audit/l3/existing-regression-results.json')
assert all(r['failures'] == r['errors'] == 0 for r in [*results, existing])
report = dict(status='PASS', head=baseline['head'], baseline_files=len(baseline['files']),
    unchanged=len(baseline['files'])-len(changed), changed=changed, new_files=new,
    new_ignored=ignored, python310_syntax=syntax,
    new_source_hashes={p: digest(p) for p in new if p.endswith(('.py', 'official_standard_models.json'))},
    **counts, results=results, existing_regression=existing,
    git_status=git('status', '--porcelain=v1', '--untracked-files=all'),
    frozen_path_status=git('status', '--porcelain=v1', '--untracked-files=all', '--',
                          *load('docs/linearno_audit/l0/freeze-manifest.json')['paths']),
    monitor='N/A: absent at L0 and delivery', tracked_diff=git('diff'))
print(json.dumps(report, indent=2))
