"""Reproduce A1 frozen-file checks and its production/test review patch.

Writes audit evidence only. Run from the repository root; no task imports.
The source snapshot and resumed-start hashes predate A1 production edits.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0')
AUDIT = Path(__file__).resolve().parent
START = json.loads((BASE / 'resume-start.json').read_text())
SOURCE_TESTS = [
    'cdlno/config.py', 'cdlno/modules.py', 'cdlno/core.py',
    'tests/test_shapenet_car.py', 'tests/test_airfrans.py',
]
DOC_CHANGES = [
    'AGENTS.md', 'docs/CDLNO_IMPLEMENTATION_STATUS.md', 'memory/current-state.md',
    'docs/CDLNO_FRONT_ABLATION.md', 'docs/CDLNO_PHASE2_MODULES.md',
    'docs/CDLNO_PHASE4_CORE.md', 'tran_evaluate/README.md',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def parsed(name):
    return [ast.parse((r / name).read_text()) for r in (BASE / 'source', ROOT)]


def cls(tree, name):
    return next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == name)


def fn(tree, name):
    return next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == name)


def same_ast(a, b):
    assert ast.dump(a) == ast.dump(b)


allowed = set(SOURCE_TESTS + DOC_CHANGES)
changed, frozen, rows = [], {}, []
for name, before in sorted(START['sha256'].items()):
    path = ROOT / name
    assert path.is_file(), f'missing original file: {name}'
    after = sha(path)
    if after != before:
        assert name in allowed, f'unexpected change: {name}'
        changed.append(name)
        rows.append(dict(file=name, before_sha256=before, after_sha256=after))
    else:
        frozen[name] = after
assert set(changed) == allowed

# HEAD is valid for these five files only: all match the pre-edit snapshot.
assert git('rev-parse', 'HEAD').decode().strip() == START['head']
for name in SOURCE_TESTS:
    original = (BASE / 'source' / name).read_bytes()
    assert hashlib.sha256(original).hexdigest() == START['sha256'][name]
    assert git('show', 'HEAD:' + name) == original
patch = git('diff', '--', *SOURCE_TESTS).decode()
new_test = 'tests/test_front_ablation.py'
assert new_test not in START['sha256']
patch += f'diff --git a/{new_test} b/{new_test}\nnew file mode 100644\n'
patch += ''.join(difflib.unified_diff(
    [], (ROOT / new_test).read_text().splitlines(keepends=True),
    fromfile='/dev/null', tofile='b/' + new_test,
))
(AUDIT / 'a1-changes.patch').write_text(patch)

old, new = parsed('cdlno/core.py')
same_ast(fn(cls(old, 'CDLNO'), 'forward'), fn(cls(new, 'CDLNO'), 'forward'))
old_text = (BASE / 'source/cdlno/core.py').read_text()
new_text = (ROOT / 'cdlno/core.py').read_text()
assert ast.get_source_segment(old_text, fn(cls(old, 'CDLNO'), 'forward')) == ast.get_source_segment(
    new_text, fn(cls(new, 'CDLNO'), 'forward'))
fixed_construction = []
for attr in ('bridge', 'latent_blocks', 'readout', 'cdpa_at'):
    nodes = []
    for tree in (old, new):
        nodes.append(next(v for v in fn(cls(tree, 'CDLNO'), '__init__').body
                          if isinstance(v, ast.Assign) and any(
                              isinstance(t, ast.Attribute) and t.attr == attr for t in v.targets)))
    same_ast(*nodes)
    fixed_construction.append(attr)
old, new = parsed('cdlno/config.py')
same_ast(cls(old, 'CDLNORuntimeConfig'), cls(new, 'CDLNORuntimeConfig'))
old, new = parsed('cdlno/modules.py')
fixed_modules = []
for node in old.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'LRSAFrontBlock':
        current = next(n for n in new.body if isinstance(n, type(node)) and n.name == node.name)
        same_ast(node, current)
        fixed_modules.append(node.name)

# NUL-delimited names preserve the Chinese PDF path without git quote escaping.
current_names = set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')) - {''}
new_names = sorted(current_names - set(START['sha256']))
for name in new_names:
    assert (name == new_test or name == 'docs/CDLNO_FRONT_ABLATION_A1.md'
            or name.startswith('docs/front_ablation_audit/a1/')), f'unexpected added path: {name}'
projects = ('PDE-Solving-StandardBenchmark/', 'Car-Design-ShapeNetCar/', 'Airfoil-Design-AirfRANS/')
counts = {p: sum(name.startswith(p) for name in frozen) for p in projects}
counts['shell_scripts'] = sum(name.endswith('.sh') for name in frozen)
counts['tools'] = sum(name.startswith('tools/') for name in frozen)
for name in ('cdlno/cdpa.py', 'cdlno/checkpoint.py', 'cdlno/standard.py', 'cdlno/airfrans.py',
             'pyproject.toml', 'path.sh', 'tran_evaluate/train_eval.sh'):
    assert name in frozen, name
result = dict(
    status='passed', head=START['head'], baseline=str(BASE / 'resume-start.json'),
    baseline_sha256=sha(BASE / 'resume-start.json'), snapshot_file_count=len(START['sha256']),
    unchanged_file_count=len(frozen), changed_file_count=len(changed), changes=rows,
    new_files_checked=new_names, frozen_group_counts=counts, missing_files=[], unexpected_changes=[],
    ast_checks=dict(core_forward='AST and source text identical',
                    core_nonfront_construction=fixed_construction, runtime_config='AST identical',
                    unchanged_module_definitions=fixed_modules),
    patch=dict(path='a1-changes.patch', sha256=sha(AUDIT / 'a1-changes.patch'),
               scope='Three production files, two inherited-test inventory fixes and one new test file; documentation tracked separately'),
    frozen_sha256=frozen,
    command='python -B docs/front_ablation_audit/a1/audit_scope.py',
    initial_inline_audit='File-addition assertion failed on git-quoted Chinese PDF name; corrected audit enumeration to NUL delimiters; no actual out-of-scope file was added',
)
(AUDIT / 'freeze.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'snapshot_file_count', 'unchanged_file_count',
                                      'changed_file_count', 'frozen_group_counts', 'ast_checks')}, indent=2))
