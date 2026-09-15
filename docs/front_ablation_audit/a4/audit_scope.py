"""A4 incremental freeze + A1-A4 model/protocol review; no task imports."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
start=json.loads((OUT/'start.json').read_text());before=Path(start['source'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tool_changes={'tools/cdlno_benchmark.py','tools/cdlno_perf/models.py','tools/cdlno_perf/costs.py'}
allowed=tool_changes|{'tests/test_performance.py','README.md','AGENTS.md','memory/current-state.md',
    'docs/CDLNO_FRONT_ABLATION.md','docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md',
    'docs/CDLNO_IMPLEMENTATION_STATUS.md','docs/CDLNO_IMPLEMENTATION_REPORT.md',
    'docs/CDLNO_PERFORMANCE_TOOLS.md','docs/CDLNO_PHASE4_CORE.md','docs/CDLNO_PHASE9_PERFORMANCE.md',
    'docs/CDLNO_REFERENCE_AUDIT.md','docs/CDLNO_REQUIREMENTS_MATRIX.md'}
changed=[];unchanged={}
for name,digest in start['sha256'].items():
    assert sha(before/name)==digest, f'snapshot changed: {name}'
    assert (ROOT/name).is_file(),name
    if sha(ROOT/name)!=digest:
        assert name in allowed,name
        changed.append(name)
    else:unchanged[name]=digest
inventory=set(subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0'))-{''}
added=sorted(inventory-start['sha256'].keys())
assert all(n=='docs/CDLNO_FRONT_ABLATION_A4.md' or n.startswith('docs/front_ablation_audit/a4/') for n in added)
frozen=[n for n in start['sha256'] if (n.startswith(('cdlno/','PDE-Solving-StandardBenchmark/',
    'Car-Design-ShapeNetCar/','Airfoil-Design-AirfRANS/','tools/','tran_evaluate/'))
    or n in ('Physics_Attention.py','pyproject.toml','path.sh')) and n not in tool_changes]
assert all(n in unchanged for n in frozen)

# Recheck cumulative model math, not just A4's unchanged hashes.
oldbase=Path('/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0')
oldstart=json.loads((oldbase/'resume-start.json').read_text())
oldsource=oldbase/'source'
def tree(base,name):return ast.parse((base/name).read_text())
def cls(t,name):return next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name)
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def equal(a,b):assert ast.dump(a)==ast.dump(b)
fixed=[]
for node in tree(oldsource,'cdlno/modules.py').body:
    if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name!='LRSAFrontBlock':
        other=next(n for n in tree(ROOT,'cdlno/modules.py').body if isinstance(n,type(node)) and n.name==node.name)
        equal(node,other);fixed.append(node.name)
old,new=[cls(tree(r,'cdlno/core.py'),'CDLNO') for r in (oldsource,ROOT)]
equal(fn(old,'forward'),fn(new,'forward'))
for attr in ('bridge','latent_blocks','readout','cdpa_at'):
    values=[next(n for n in fn(c,'__init__').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr==attr for t in n.targets)) for c in (old,new)]
    equal(*values)
assert (oldsource/'cdlno/cdpa.py').read_bytes()==(ROOT/'cdlno/cdpa.py').read_bytes()

class RemoveOnlyMode(ast.NodeTransformer):
    def visit_arguments(self,node):
        first=len(node.args)-len(node.defaults)
        for i in range(len(node.args)-1,-1,-1):
            if node.args[i].arg=='front_latent_mode':
                node.args.pop(i)
                if i>=first:node.defaults.pop(i-first)
        for i in range(len(node.kwonlyargs)-1,-1,-1):
            if node.kwonlyargs[i].arg=='front_latent_mode':
                node.kwonlyargs.pop(i);node.kw_defaults.pop(i)
        return self.generic_visit(node)
    def visit_Call(self,node):
        node.args=[n for n in node.args if not (isinstance(n,ast.Name) and n.id=='front_latent_mode')]
        node.keywords=[k for k in node.keywords if k.arg!='front_latent_mode']
        return self.generic_visit(node)
wrappers=('cdlno/standard.py','cdlno/airfrans.py','Car-Design-ShapeNetCar/models/CDLNO.py')
for name in wrappers:equal(tree(oldsource,name),RemoveOnlyMode().visit(tree(ROOT,name)))
presets=[n for n in oldstart['sha256'] if '/configs/CDLNO/' in n and n.endswith('.json')]
for name in presets:
    original=json.loads((oldsource/name).read_text());current=json.loads((ROOT/name).read_text())
    assert current['model'].pop('front_latent_mode')=='full'
    assert current==original,name
original_protocols=[]
for name in oldstart['sha256']:
    p=Path(name)
    if (p.name.startswith('exp_') and p.suffix=='.py') or (name.startswith(('Car-Design-ShapeNetCar/','Airfoil-Design-AirfRANS/')) and p.name in ('main.py','main_evaluation.py','train.py')) or '/dataset/' in name or '/utils/' in name or ('/scripts/' in name and 'Transolver' in p.name) or 'requirements' in p.name or p.name=='params.yaml':
        assert sha(ROOT/name)==oldstart['sha256'][name],name
        original_protocols.append(name)
runtime=lambda n:n.startswith(('cdlno/','tools/','PDE-Solving-StandardBenchmark/','Car-Design-ShapeNetCar/','Airfoil-Design-AirfRANS/','tran_evaluate/')) and not n.endswith('.md')
cumulative=[n for n,h in oldstart['sha256'].items() if runtime(n) and sha(ROOT/n)!=h]
expected={'cdlno/config.py','cdlno/core.py','cdlno/modules.py','cdlno/checkpoint.py',*wrappers,*presets,*tool_changes,
    'PDE-Solving-StandardBenchmark/cdlno_entry.py','Car-Design-ShapeNetCar/models/cdlno_run.py','Airfoil-Design-AirfRANS/cdlno_entry.py',
    'tran_evaluate/_common.sh','tran_evaluate/_standard.sh','tran_evaluate/car.sh','tran_evaluate/airfrans.sh'}
assert set(cumulative)==expected
syntax=[]
for name in sorted(tool_changes|{'tests/test_performance.py'}):
    ast.parse((ROOT/name).read_text(),feature_version=(3,10));syntax.append(name)
shells=[n for n in frozen if n.endswith('.sh')]
for name in shells:subprocess.run(['bash','-n',str(ROOT/name)],check=True)

def patch_for(source,names):
    patch=''
    for name in sorted(names):
        original=(source/name).read_text().splitlines(keepends=True) if (source/name).is_file() else []
        current=(ROOT/name).read_text().splitlines(keepends=True)
        patch+=f'diff --git a/{name} b/{name}\n'
        if not original:patch+='new file mode 100644\n'
        patch+=''.join(difflib.unified_diff(original,current,fromfile='a/'+name if original else '/dev/null',tofile='b/'+name))
    return patch
(OUT/'a4-changes.patch').write_text(patch_for(before,changed+[n for n in added if not n.startswith('docs/front_ablation_audit/a4/')]))
(OUT/'front-ablation-runtime.patch').write_text(patch_for(oldsource,cumulative))
result=dict(status='passed',source=str(before),snapshot_files=len(start['sha256']),changed_existing=changed,added=added,
    frozen_files=len(frozen),frozen_sha256={n:unchanged[n] for n in frozen},python310_syntax=syntax,bash_syntax=len(shells),
    cumulative_runtime_changed=cumulative,cumulative_cdpa_unchanged=True,cumulative_core_history_and_rear_construction_unchanged=True,
    cumulative_modules_unchanged_except_front=fixed,cumulative_wrapper_only_mode=list(wrappers),eight_presets_only_added_default_full=True,
    cumulative_original_protocols_unchanged=original_protocols,
    unrelated_V1_files_unchanged=['cdlno/training_state.py','cdlno/training_observer.py','cdlno/visualization.py'],
    unexpected_changes=[])
(OUT/'freeze.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','snapshot_files','changed_existing','frozen_files','bash_syntax','unexpected_changes')},indent=2))
