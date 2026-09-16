"""Recheck existing evidence without importing models, entries, or loading data.

Only writes this M0 audit's fixture index and source freeze. Preserves external
weights, earlier evidence and all production files. Run after documentation edits.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    indexed, checked_files, not_run = [], [], []

    def verify(path, expected):
        path = Path(path)
        actual = sha(path)
        if actual != expected:
            raise AssertionError(f'fixture changed: {path}')
        checked_files.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size))

    k0 = read(ROOT / 'docs/kcdno_audit/fixture_index.json')
    k0_results = {}
    for process in read(OUT / 'k0-replay.json'):
        assert process['returncode'] == 0
        k0_results.update({r['name']: r for r in process['summary']['rows']})
    for row in k0['records']:
        parent = Path(row['metadata']).parent
        for name, entry in row['artifacts'].items():
            verify(parent / name, entry['sha256'])
        result = k0_results[row['name']]
        assert result['status'] == 'passed' and result['max_output_abs'] == 0
        indexed.append(dict(name=row['name'], chronology='genuine pre-K1 / pre-M1',
            fixture=str(parent / 'fixture.pt'), metadata=row['metadata'],
            configuration=row['configuration'], reference_index='docs/kcdno_audit/fixture_index.json',
            replay='docs/msar_lno_audit/m0/k0-replay.json', status='passed',
            device=row['device'], dtype='float32', atol=0, rtol=0))
    k0_file_count = len(checked_files)

    m1 = read(ROOT / 'docs/msar_lno_audit/m1/fixture-index.json')
    current_results = {r['name']: r for r in read(OUT / 'current-core-replay.json')}
    for row in m1['current_records']:
        verify(row['fixture'], row['sha256'])
        result = current_results[row['name']]
        assert result['status'] == 'passed' and result['sha256'] == row['sha256']
        indexed.append(row | dict(chronology='genuine pre-M1 current core', status='passed',
            reference_index='docs/msar_lno_audit/m1/fixture-index.json',
            replay='docs/msar_lno_audit/m0/current-core-replay.json'))

    for group in ('wrappers', 'other'):
        captures = {}
        for process in read(OUT / f'{group}-capture.json'):
            assert process['returncode'] == 0
            captures.update({r['name']: r for r in process['rows']})
        for process in read(OUT / f'{group}-replay.json'):
            assert process['returncode'] == 0
            for row in process['rows']:
                if row['status'] != 'passed':
                    not_run.append(row)
                    continue
                captured = captures[row['name']]
                hash_key = 'sha256' if group == 'wrappers' else 'hashes'
                assert row[hash_key] == captured[hash_key]
                parent = Path(row['fixture']).parent
                for name, digest in captured[hash_key].items():
                    verify(parent / name, digest)
                indexed.append({k:v for k,v in row.items()
                    if k not in ('keys_shapes', 'state_keys_shapes')} | dict(
                    chronology='post-M1, pre-MSAR model; captured during retroactive M0',
                    reference_index=f'docs/msar_lno_audit/m0/{group}-capture.json',
                    replay=f'docs/msar_lno_audit/m0/{group}-replay.json'))

    assert len(indexed) == 75
    write('fixture-index.json', dict(schema='msar-retroactive-m0-v1',
        chronology='M1 already existed; this M0 never reconstructs or backdates an old baseline',
        passed_fixtures=len(indexed), groups=dict(k0=41, pre_m1_core=6, wrappers=24, other_models=4),
        k0_files_rechecked=k0_file_count, checked_file_count=len(checked_files),
        records=indexed, integrity_files=checked_files, not_run=not_run,
        limits=['Temporary same-weight real-model references, not trained historical checkpoints.',
                'Only inherited K0 references include diagnostic gradients.',
                'No real data, convergence, full graph sampling, or remote runtime acceptance.',
                'State keys/shapes, saved inputs/outputs and full configs reside in linked evidence and fixtures.']))

    inventory = read(OUT / 'inventory.json')
    allowed = {'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md',
               'docs/CDLNO_IMPLEMENTATION_STATUS.md', 'memory/current-state.md'}
    unchanged, changed = [], []
    for row in inventory['records']:
        path = ROOT / row['path']
        actual = sha(path)
        if actual == row['sha256']:
            unchanged.append(row['path'])
        else:
            assert row['path'] in allowed, f'unexpected existing file change: {path}'
            before = (Path(inventory['snapshot']) / 'source' / row['path']).read_text()
            after = path.read_text()
            if path.suffix == '.md' and path.name != 'current-state.md':
                # New phase is inserted after the existing document title.
                _, original_tail = before.split('\n', 1)
                assert original_tail in after, f'history removed: {path}'
            else:
                assert before in after, f'history removed: {path}'
            changed.append(dict(path=row['path'], before=row['sha256'], after=actual,
                                original_text_preserved=True))

    # Syntax-check new audit helpers only; do not alter existing tests.
    for path in OUT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))
    pre_m1 = Path('/home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/source')
    stable_prefixes = ('cdlno/', 'PDE-Solving-StandardBenchmark/', 'Car-Design-ShapeNetCar/',
                       'Airfoil-Design-AirfRANS/', 'tests/', 'tools/', 'tran_evaluate/')
    pre_m1_unchanged = []
    for path in pre_m1.rglob('*'):
        if path.is_file() and path.relative_to(pre_m1).as_posix().startswith(stable_prefixes):
            rel = path.relative_to(pre_m1)
            assert sha(path) == sha(ROOT / rel), f'pre-M1 source changed: {rel}'
            pre_m1_unchanged.append(str(rel))
    status = subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT)
    (OUT / 'status-after.txt').write_bytes(status)
    check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    write('freeze.json', dict(snapshot=inventory['snapshot'], baseline_files=len(inventory['records']),
        unchanged_count=len(unchanged), unchanged=unchanged, changed=changed,
        pre_m1_source_compared=len(pre_m1_unchanged), pre_m1_source_unchanged=pre_m1_unchanged,
        syntax='new audit scripts parse with Python 3.10 grammar', git_diff_check='passed',
        boundary='No production/config/factory/data/dependency or existing test changes during M0.'))
    print(json.dumps(dict(fixtures=len(indexed), fixture_files=len(checked_files),
        not_run=[r['name'] for r in not_run], unchanged=len(unchanged),
        documentation_changes=len(changed), pre_m1_source_unchanged=len(pre_m1_unchanged))))


if __name__ == '__main__':
    main()
