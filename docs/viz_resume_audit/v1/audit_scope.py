"""V1 incremental audit. No task imports or dataset access."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
start = json.loads((OUT/'start.json').read_text())
before = Path(start['source'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
allowed = {'AGENTS.md', 'memory/current-state.md', 'docs/CDLNO_IMPLEMENTATION_STATUS.md',
           'docs/CDLNO_VISUALIZATION_RESUME_PLAN.md'}
new_code = {'cdlno/training_state.py', 'cdlno/training_observer.py', 'cdlno/visualization.py',
            'tests/test_training_state.py', 'tests/test_visualization.py'}
changed, unchanged = [], {}
for name, digest in start['sha256'].items():
    assert sha(before/name) == digest, f'immutable snapshot changed: {name}'
    assert (ROOT/name).is_file(), f'preexisting file removed: {name}'
    if sha(ROOT/name) != digest:
        assert name in allowed, f'out-of-scope change: {name}'
        changed.append(name)
    else:
        unchanged[name] = digest
inventory = set(subprocess.check_output(
    ['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT
).decode().split('\0'))-{''}
added = sorted(inventory-start['sha256'].keys())
for name in added:
    assert name in new_code|{'docs/CDLNO_VISUALIZATION_RESUME_V1.md'} or name.startswith('docs/viz_resume_audit/v1/'), name
frozen = [n for n in start['sha256'] if n.startswith(('cdlno/', 'PDE-Solving-StandardBenchmark/',
          'Car-Design-ShapeNetCar/', 'Airfoil-Design-AirfRANS/', 'tools/', 'tran_evaluate/'))
          or n in ('Physics_Attention.py','pyproject.toml','path.sh')]
assert all(n in unchanged for n in frozen)
syntax = []
for name in sorted(new_code):
    tree = ast.parse((ROOT/name).read_text(), feature_version=(3,10))
    # No task-entry imports, training loop rewrites, model definitions or pickle bypass.
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(not a.name.startswith('exp_') and a.name!='main' for a in node.names), name
        if isinstance(node,ast.ImportFrom):
            assert not (node.module or '').startswith('exp_') and node.module!='main', name
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load':
            if isinstance(node.func.value,ast.Name) and node.func.value.id=='torch':
                assert any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords),name
    syntax.append(name)
patch = ''
for name in sorted(changed + [n for n in added if not n.startswith('docs/viz_resume_audit/v1/')]):
    old = (before/name).read_text().splitlines(keepends=True) if (before/name).exists() else []
    new = (ROOT/name).read_text().splitlines(keepends=True)
    patch += f'diff --git a/{name} b/{name}\n'
    if not old:
        patch += 'new file mode 100644\n'
    patch += ''.join(difflib.unified_diff(old,new,fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
(OUT/'v1-changes.patch').write_text(patch)
result = dict(status='passed', head=start['head'], source=str(before),snapshot_files=len(start['sha256']),
              changed_existing=changed,added=added,frozen_files=len(frozen),
              frozen_sha256={n:unchanged[n] for n in frozen},python310_syntax=syntax,
              no_existing_production_edits=True,unexpected_changes=[],
              patch_sha256=sha(OUT/'v1-changes.patch'))
(OUT/'freeze.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','snapshot_files','changed_existing','frozen_files',
                                      'no_existing_production_edits','python310_syntax')},indent=2))
