"""A2 hash/AST audit and incremental review patch; never imports task code."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
START = json.loads((OUT / 'start.json').read_text())
BEFORE = Path(START['source'])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
core = ['cdlno/standard.py', 'cdlno/airfrans.py', 'cdlno/checkpoint.py',
        'PDE-Solving-StandardBenchmark/cdlno_entry.py', 'Car-Design-ShapeNetCar/models/CDLNO.py',
        'Car-Design-ShapeNetCar/models/cdlno_run.py', 'Airfoil-Design-AirfRANS/cdlno_entry.py']
presets = [name for name in START['sha256'] if '/configs/CDLNO/' in name and name.endswith('.json')]
shell = ['tran_evaluate/_common.sh', 'tran_evaluate/_standard.sh', 'tran_evaluate/car.sh', 'tran_evaluate/airfrans.sh']
tests = ['tests/test_static_standard.py', 'tests/test_shapenet_car.py', 'tests/test_airfrans.py']
docs = ['AGENTS.md', 'memory/current-state.md', 'docs/CDLNO_IMPLEMENTATION_STATUS.md',
        'docs/CDLNO_FRONT_ABLATION.md', 'docs/CDLNO_TASK_LAUNCHERS.md', 'tran_evaluate/README.md']
allowed = set(core + presets + shell + tests + docs)
changed, unchanged = [], {}
for name, h in START['sha256'].items():
    assert (ROOT / name).is_file(), f'missing preexisting file: {name}'
    assert sha(BEFORE / name) == h, f'snapshot altered: {name}'
    current = sha(ROOT / name)
    if current != h:
        assert name in allowed, f'unexpected change: {name}'
        changed.append(dict(file=name, before_sha256=h, after_sha256=current))
    else:
        unchanged[name] = current
assert {row['file'] for row in changed} == allowed
assert len(presets) == 8
for name in presets:
    original = json.loads((BEFORE / name).read_text())
    current = json.loads((ROOT / name).read_text())
    assert current['model'].pop('front_latent_mode') == 'full'
    assert current == original, name


class RemoveOnlyMode(ast.NodeTransformer):
    def visit_arguments(self, node):
        # Positional/keyword-only arguments appended by the thin adapters.
        first_default = len(node.args) - len(node.defaults)
        for i in range(len(node.args) - 1, -1, -1):
            if node.args[i].arg == 'front_latent_mode':
                node.args.pop(i)
                if i >= first_default: node.defaults.pop(i - first_default)
        for i in range(len(node.kwonlyargs) - 1, -1, -1):
            if node.kwonlyargs[i].arg == 'front_latent_mode':
                node.kwonlyargs.pop(i); node.kw_defaults.pop(i)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.args = [a for a in node.args if not (isinstance(a, ast.Name) and a.id == 'front_latent_mode')]
        node.keywords = [k for k in node.keywords if k.arg != 'front_latent_mode']
        return self.generic_visit(node)


wrapper_checks = []
for name in ('cdlno/standard.py', 'cdlno/airfrans.py', 'Car-Design-ShapeNetCar/models/CDLNO.py'):
    before = ast.parse((BEFORE / name).read_text())
    after = RemoveOnlyMode().visit(ast.parse((ROOT / name).read_text()))
    assert ast.dump(before) == ast.dump(after), name
    wrapper_checks.append(name)
# All actual entry/loop/data/metrics code stays byte-identical; no AST deletion
# is allowed in those files, even for CDLNO branches.
entry_names = [n for n in unchanged if (Path(n).name.startswith('exp_') and n.endswith('.py'))
              or (n.startswith(('Car-Design-ShapeNetCar/', 'Airfoil-Design-AirfRANS/'))
                  and Path(n).name in ('main.py', 'main_evaluation.py', 'train.py'))]
assert len(entry_names) == 12
for name in ('cdlno/config.py', 'cdlno/modules.py', 'cdlno/core.py', 'cdlno/cdpa.py',
             'Airfoil-Design-AirfRANS/params.yaml', 'tran_evaluate/train_eval.sh', 'pyproject.toml', 'path.sh'):
    assert name in unchanged
for name in START['sha256']:
    if name.startswith('tools/') or (('/scripts/' in name) and name.endswith('.sh')):
        assert name in unchanged
# Track all A2 additions, excluding ignored __pycache__ etc.
current = set(subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().split('\0')) - {''}
added = sorted(current - START['sha256'].keys())
new_test = 'tests/test_front_task_modes.py'
for name in added:
    assert name in (new_test, 'docs/CDLNO_FRONT_ABLATION_A2.md', 'docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md') or name.startswith('docs/front_ablation_audit/a2/'), name
patch = ''
for name in sorted(set(core + presets + shell + tests)) + [new_test]:
    before = (BEFORE / name).read_text().splitlines(keepends=True) if name != new_test else []
    after = (ROOT / name).read_text().splitlines(keepends=True)
    patch += f'diff --git a/{name} b/{name}\n'
    if name == new_test: patch += 'new file mode 100644\n'
    patch += ''.join(difflib.unified_diff(before, after, fromfile='a/' + name if before else '/dev/null', tofile='b/' + name))
(OUT / 'a2-changes.patch').write_text(patch)
result = dict(status='passed', baseline=str(BEFORE), head=START['head'],
              snapshot_files=len(START['sha256']), unchanged_files=len(unchanged), changed_files=len(changed),
              changes=changed, added_files_checked=added, frozen_sha256=unchanged,
              wrapper_AST_equal_after_removing_only_mode=wrapper_checks,
              byte_identical_entry_and_training_files=entry_names,
              eight_presets_only_added_full=True,
              unchanged_shells=sum(n.endswith('.sh') for n in unchanged),
              source_patch_sha256=sha(OUT / 'a2-changes.patch'),
              command='python -B docs/front_ablation_audit/a2/audit_scope.py', unexpected_changes=[])
(OUT / 'freeze.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'snapshot_files', 'unchanged_files', 'changed_files',
                                       'unchanged_shells', 'wrapper_AST_equal_after_removing_only_mode',
                                       'byte_identical_entry_and_training_files')}, indent=2))
