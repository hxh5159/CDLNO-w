"""M2 evidence/freeze checks only; no model/data/entry imports or writes."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    before = read(OUT / 'before.json')
    allowed = {'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md', 'docs/CDLNO_IMPLEMENTATION_STATUS.md',
               'memory/current-state.md'}
    unchanged, changes = [], []
    for row in before['records']:
        path = ROOT / row['path']
        actual = digest(path)
        if row['sha256'] == actual:
            unchanged.append(row['path'])
        else:
            assert row['path'] in allowed, f'unexpected existing edit: {path}'
            old_text = (Path(before['snapshot']) / 'source' / row['path']).read_text()
            preserved = old_text if path.name == 'current-state.md' else old_text.split('\n', 1)[1]
            assert preserved in path.read_text(), f'history removed: {path}'
            changes.append(dict(path=row['path'], before=row['sha256'], after=actual,
                                historical_text_preserved=True))
    new_sources = ('cdlno/msar_lno/modules.py', 'tests/msar_reference.py', 'tests/test_msar_modules.py')
    for name in (*new_sources, 'docs/msar_lno_audit/m2/verify_delivery.py'):
        source = (ROOT / name).read_text()
        ast.parse(source, feature_version=(3, 10))
        assert all(line.rstrip() == line for line in source.splitlines()), name
    oracle = ast.parse((ROOT / 'tests/msar_reference.py').read_text())
    imports = [ast.unparse(n) for n in ast.walk(oracle) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == ['import math', 'import torch']
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('forward', 'scaled_dot_product_attention')
                   for n in ast.walk(oracle))

    index = read(ROOT / 'docs/msar_lno_audit/m0/fixture-index.json')
    for row in index['integrity_files']:
        assert digest(row['path']) == row['sha256'], f'old fixture changed: {row["path"]}'
    rows = []
    for name in ('k0-replay', 'current-core-replay', 'wrappers-replay', 'other-replay'):
        result = read(OUT / f'{name}.json')
        if name == 'current-core-replay':
            rows.extend(result)
        else:
            for project in result:
                assert project['returncode'] == 0
                rows.extend(project['summary']['rows'] if name == 'k0-replay' else project['rows'])
    assert sum(r['status'] == 'passed' for r in rows) == 75
    assert [(r['name'], r['status']) for r in rows if r['status'] != 'passed'] == [('GUNet', 'not_run')]
    suites = []
    for logname, expected in (('modules-tests.log', 20), ('shared-tests.log', 18), ('config-tests.log', 13)):
        text = (OUT / logname).read_text()
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
        assert match and int(match[1]) == expected and '\nOK\n' in text, logname
        suites.append(dict(log=logname, tests=int(match[1]), seconds=float(match[2]),
                           failures=0, errors=0, skipped=0))
    for name in ('docs/MSAR_LNO_M2_PRIMITIVES.md', 'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md'):
        path = ROOT / name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not target.startswith(('https:', 'http:', '#')):
                target = target.split('#')[0]
                # The two generated records are written immediately below.
                if target.endswith(('m2/freeze.json', 'm2/summary.json')):
                    continue
                assert (path.parent / target).exists(), (name, target)
    check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    (OUT / 'status-after.txt').write_bytes(subprocess.check_output(
        ['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT))
    save('freeze.json', dict(snapshot=before['snapshot'], baseline_files=len(before['records']),
        unchanged_count=len(unchanged), unchanged=unchanged, changes=changes,
        new_sources=[dict(path=p, sha256=digest(ROOT / p)) for p in new_sources],
        existing_production_and_tests_changed=False, python310_grammar='passed',
        oracle_independence='only math/torch; no production imports/forward/SDPA',
        fixture_files_unchanged=len(index['integrity_files']), git_diff_check='passed'))
    save('summary.json', dict(stage='M2', scope='atomic modules and independent reference only',
        suites=suites, test_total=sum(s['tests'] for s in suites),
        same_weight_replay_passed=75, fixture_files_unchanged=len(index['integrity_files']),
        gpu='Local CUDA MATH SDPA FP32/FP16 AMP/BF16 AMP finite forward/backward and FP32 aux checks passed',
        environment=read(OUT / 'environment.json'),
        not_run=['GUNet graph chain (torch_cluster absent)', 'Remote torch2.11/cu128',
                 'Full MSAR core/task/loss integration', 'Real data/training/convergence/accuracy',
                 'Other GPU backends/full-size performance'],
        old_full_suite='M0 316-test evidence retained; only affected module/config tests replayed in M2',
        next_stage_executed=False))
    print(json.dumps(dict(tests=sum(s['tests'] for s in suites), old_model_replay=75,
        fixture_hashes=len(index['integrity_files']), unchanged=len(unchanged),
        documentation_changes=len(changes))))


if __name__ == '__main__':
    main()
