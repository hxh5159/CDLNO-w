"""Verify LL9R delivery against the preserved starting worktree, never reset it."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    start = json.loads((OUT/'start-manifest.json').read_text())
    snapshot = Path(start['snapshot'])/'source'
    allowed = {
        'Airfoil-Design-AirfRANS/cdlno_entry.py',
        'Car-Design-ShapeNetCar/models/cdlno_run.py',
        'PDE-Solving-StandardBenchmark/cdlno_entry.py',
        'cdlno/linearno_loop/core.py',
        'cdlno/linearno_loop/industrial_state.py',
        'cdlno/linearno_loop/ll7_projection.py',
        'tests/linearno/test_history_industrial.py',
        'tests/linearno/test_history_static.py',
        'tests/linearno/test_legacy.py',
        'tests/linearno/test_static_integration.py',
        'tests/linearno_entry_projection.py',
        'tests/test_kcdno_delivery.py',
        'tests/loop_entry_projection.py',
        'tests/loop_linearno/test_isolation.py',
        'tests/loop_linearno/test_launchers.py',
        'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md',
    }
    added = {
        'cdlno/linearno_loop/ll9r_projection.py',
        'docs/LOOP_LINEARNO_LL9R.md',
        'memory/2026-09-20-loop-linearno-ll9r.md',
        'tests/frozen_revisions.py',
        'tests/historical_git_base.py',
        'tests/linearno_air_projection.py',
        'tests/ll9r_projection.py',
        'tests/ll9r_test_source_projection.py',
        'tests/loop_linearno/test_ll9r.py',
    }
    current = {}
    for label, args in [('tracked', []), ('untracked', ['-o', '--exclude-standard']),
                        ('ignored', ['-oi', '--exclude-standard'])]:
        for raw in git('ls-files', '-z', *args).split(b'\0'):
            if not raw:
                continue
            name = raw.decode(); path = ROOT/name
            if path.is_file():
                data = path.read_bytes()
                current[name] = dict(path=name, classification=label, size=len(data),
                                     sha256=hashlib.sha256(data).hexdigest())
    changes = []; runtime = []; unchanged = 0; patches = []
    for old in start['files']:
        name = old['path']
        assert name in current, ('missing', name)
        now = current[name]
        assert now['classification'] == old['classification'], name
        if old['sha256'] == now['sha256']:
            unchanged += 1
            continue
        if name == '.pytest_cache/v/cache/lastfailed':
            runtime.append(dict(before=old, after=now))
            continue
        assert name in allowed, ('unexpected changed file', name)
        changes.append(dict(before=old, after=now))
        patches.extend(difflib.unified_diff((snapshot/name).read_text().splitlines(True),
            (ROOT/name).read_text().splitlines(True), fromfile='LL9R-before/'+name, tofile='current/'+name))
    original = {r['path'] for r in start['files']}
    new = [row for name, row in current.items() if name not in original]
    for row in new:
        name = row['path']
        assert (name in added or name.startswith('docs/loop_linearno_audit/ll9r/') or
                (row['classification'] == 'ignored' and
                 (name.startswith('output/') or '__pycache__' in Path(name).parts))), name
        if name in added:
            patches.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),
                fromfile='/dev/null', tofile='current/'+name))
    syntax = []
    for name in sorted(allowed | added):
        if name.endswith('.py'):
            ast.parse((ROOT/name).read_text(), filename=name, feature_version=(3, 10))
            syntax.append(name)
    rows = json.loads((OUT/'final-regression/regression-results-final.json').read_text())
    assert len(rows) == 80 and sum(r['tests'] for r in rows) == 689
    assert not any(r['failures'] or r['errors'] or r['exit_code'] for r in rows)
    assert sum(len(r['skipped']) for r in rows) == 36
    cuda = json.loads((OUT/'cuda.json').read_text())['rows']
    assert len(cuda) == 144 and all(r['status']=='PASS' for r in cuda)
    assert Counter(r['precision'] for r in cuda) == dict(FP32=48, AMP_FP16=48, AMP_BF16=48)
    for name, total in [('native-matrix.json', 18), ('industrial-matrix.json', 6),
                        ('old-checkpoint-replay.json', 9)]:
        obj = json.loads((OUT/name).read_text())
        assert obj['expected'] == obj['completed'] == len(obj['rows']) == total
    for name, total in [('counts.json', 96), ('matrix.json', 48), ('cpu.json', 48)]:
        assert len(json.loads((OUT/name).read_text())['rows']) == total
    assert json.loads((OUT/'before-provenance.json').read_text()) == json.loads((OUT/'after-provenance.json').read_text())
    assert git('rev-parse', 'HEAD').decode().strip() == start['head']
    assert git('rev-parse', 'HEAD^{tree}').decode().strip() == start['tree']
    assert git('diff', '--cached') == (OUT/'start-staged.diff').read_bytes()
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout+diff.stderr
    (OUT/'source.diff').write_text(''.join(patches))
    (OUT/'end-status.txt').write_bytes(git('status', '--short', '--branch', '--untracked-files=all'))
    save('end-freeze.json', dict(start_snapshot=start['snapshot'], start_files=len(start['files']),
        unchanged=unchanged, changed_sources=changes, changed_runtime_artifacts=runtime,
        new_files=new, current_counts=dict(Counter(r['classification'] for r in current.values())),
        missing=[], classifications_unchanged=True, HEAD_identical=True, tree_identical=True,
        staged_identical=True, diff_check=dict(exit_code=diff.returncode, output=diff.stdout+diff.stderr),
        self_inventory_note='Generated audit evidence may update after enumeration; original files are all compared.',
        policy='Preserve all pre-existing files; no reset/clean/stash/checkout/rebase/commit/push.'))
    save('syntax-review.json', dict(python310_ast_parse=syntax, status='PASS'))
    save('delivery-review.json', dict(status='PASS', checks=[
        'RB-only dtype assembly; shared primitive, SR/LB bodies, baseline models frozen',
        '192 precision captures: 172 existing exact, 20 repaired, 128 SR/LB exact',
        '144 CUDA cases pass without AMP fallback; 96 counts unchanged',
        'legacy lazy imports, mutation-sensitive source/AST guards, 689 final regression results verified',
        '24 native synthetic closures and 9 immutable old-archive replays complete',
        'HEAD/tree/staged unchanged; all original files accounted for; Python 3.10 grammar and diff check pass'],
        not_run=['real data/training/convergence/accuracy/paper results', 'remote Torch2.11/cu128',
                 'full-width GPU training', 'compile/distributed', 'LL10']))
    print(json.dumps(dict(status='PASS', unchanged=unchanged, changed_sources=len(changes),
        runtime_changes=len(runtime), new_files=len(new), regression='653 pass / 36 skip / 0 fail',
        cuda='144/144', old_archives='9/9'), ensure_ascii=False))


if __name__ == '__main__':
    main()
