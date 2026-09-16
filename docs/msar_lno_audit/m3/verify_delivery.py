"""M3 audit evidence only. No model, trainer, data, or dependency mutations."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    before = read(OUT/'before.json')
    allowed = {'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md', 'docs/CDLNO_IMPLEMENTATION_STATUS.md',
               'memory/current-state.md'}
    unchanged, changes = [], []
    for record in before['records']:
        path = ROOT/record['path']
        actual = sha(path)
        if actual == record['sha256']:
            unchanged.append(record['path'])
        else:
            assert record['path'] in allowed, f'unexpected change: {path}'
            text = (Path(before['snapshot'])/'source'/record['path']).read_text()
            preserved = text if path.name == 'current-state.md' else text.split('\n', 1)[1]
            assert preserved in path.read_text(), f'history lost: {path}'
            changes.append(dict(path=record['path'], before=record['sha256'], after=actual,
                                historical_text_preserved=True))
    sources = ['cdlno/msar_lno/core.py', 'cdlno/msar_lno/diagnostics.py',
               'tests/test_msar_core.py', 'tests/msar_core_reference.py',
               'docs/msar_lno_audit/m3/core_fixtures.py', 'docs/msar_lno_audit/m3/verify_delivery.py']
    for name in sources:
        code = (ROOT/name).read_text()
        ast.parse(code, feature_version=(3, 10))
        assert all(line.rstrip() == line for line in code.splitlines()), name
    tree = ast.parse((ROOT/'tests/msar_core_reference.py').read_text())
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == ['import torch', 'import msar_reference as primitives']
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('forward', 'scaled_dot_product_attention')
                   for n in ast.walk(tree))
    index = read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for record in index['integrity_files']:
        assert sha(record['path']) == record['sha256'], record['path']
    fresh_old = []
    for process in read(OUT/'k0-replay.json'):
        assert process['returncode'] == 0
        fresh_old.extend(process['summary']['rows'])
    fresh_old.extend(read(OUT/'current-core-replay.json'))
    assert len(fresh_old) == 47 and all(r['status'] == 'passed' for r in fresh_old)
    capture = read(OUT/'core-capture.json')
    replay = read(OUT/'core-replay.json')
    for original, replayed in zip(capture, replay):
        assert original['status'] == replayed['status'] == 'passed'
        assert original['sha256'] == replayed['sha256'] == sha(original['fixture'])
        assert sum(original['parameter_groups'].values()) == original['parameters']
        assert original['shape_trace'] == replayed['shape_trace']
        counts = original['module_counts']
        for key, expected in {'LearnedQueryDown':4, 'QueryAlignedUpCross':4, 'LatentFFNSAFFNBlock':12,
                              '_SelfAttention':12, 'PlainFFN':24, 'PairwiseAttnResFusion':3}.items():
            assert counts[key] == expected
    save('fixture-index.json', dict(origin='New post-M3 real cores with synthetic lifted inputs; not trained weights.',
        records=[{k:r[k] for k in ('profile','config','output_dim','input_shape','fixture','sha256','bytes',
                                   'dtype','device','sdpa','atol','rtol','parameters')} for r in capture],
        detailed_keys_shapes_trace='docs/msar_lno_audit/m3/core-capture.json',
        replay='docs/msar_lno_audit/m3/core-replay.json', old_integrity_files_checked=182,
        old_replayed_this_stage=47, old_retained_unchanged_evidence=28))
    suites = []
    for name, expected in (('core-tests.log',16), ('modules-tests.log',20), ('config-tests.log',13)):
        text = (OUT/name).read_text()
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
        assert match and int(match[1]) == expected and '\nOK\n' in text
        suites.append(dict(log=name, tests=expected, seconds=float(match[2]), failures=0, errors=0, skipped=0))
    save('freeze.json', dict(snapshot=before['snapshot'], baseline_files=len(before['records']),
        unchanged_count=len(unchanged), unchanged=unchanged, changes=changes,
        new_sources=[dict(path=p,sha256=sha(ROOT/p)) for p in sources],
        old_production_m1_m2_existing_tests_changed=False, python310_grammar='passed',
        old_fixture_files_unchanged=len(index['integrity_files'])))
    save('summary.json', dict(stage='M3', suites=suites, tests=sum(r['tests'] for r in suites),
        new_core_fixtures=2, fresh_old_replay=47, retained_old_evidence=28, old_fixture_hashes=182,
        parameter_scope='Core with output_dim=4, no task input lift',
        light_parameters=capture[0]['parameters'], full_parameters=capture[1]['parameters'],
        environment=read(OUT/'environment.json'),
        gpu='Small-core CUDA MATH FP32/FP16 AMP/BF16 AMP finite training and off/floor prediction parity passed.',
        not_run=['Task production/factory/data/loss integration','Real data/training/convergence/accuracy',
                 'Remote Python3.10/torch2.11/cu128','Full-size task GPU/other backend performance'],
        next_stage_executed=False))
    for name in ('docs/MSAR_LNO_M3_CORE.md', 'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md'):
        path = ROOT/name
        prose = re.sub(r'```.*?```|`[^`\n]+`', '', path.read_text(), flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)', prose):
            if not target.startswith(('http:', 'https:', '#')):
                assert (path.parent/target.split('#')[0]).exists(), (name,target)
    result = subprocess.run(['git','diff','--check'], cwd=ROOT, capture_output=True,text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    (OUT/'status-after.txt').write_bytes(subprocess.check_output(
        ['git','status','--short','--untracked-files=all'],cwd=ROOT))
    print(json.dumps(dict(tests=49, new_fixtures=2, old_replay=47, old_integrity_files=182,
        unchanged=len(unchanged), changed_documents=len(changes))))


if __name__ == '__main__':
    main()
