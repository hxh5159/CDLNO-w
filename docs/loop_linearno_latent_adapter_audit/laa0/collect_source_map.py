"""Read-only source/document index; hashes are not a substitute for numerical tests."""
from pathlib import Path
import ast,hashlib,json,re
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
files=set()
for folder in ('cdlno/linearno','cdlno/linearno_loop','linearno_loop','tran_evaluate/linearno_loop','tests/loop_linearno_ffn'):
 files.update((ROOT/folder).glob('**/*.py'));files.update((ROOT/folder).glob('*.sh'))
for folder in ('PDE-Solving-StandardBenchmark','Airfoil-Design-AirfRANS','Car-Design-ShapeNetCar'):
 for pattern in ('exp_*.py','main*.py','*entry.py','model_dict.py','train.py','models/cdlno_run.py','model/LinearNO*.py','models/LinearNO*.py'):
  files.update((ROOT/folder).glob(pattern))
files.update(ROOT/x for x in ('cdlno/experiment.py','cdlno/periodic_visualization.py','cdlno/training_state.py','tools/linearno_loop_accounting.py','linearno_loop/versioning.py','path.sh'))
source=[]
for p in sorted(files):
 text=p.read_text();record={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':len(text.splitlines())}
 if p.suffix=='.py':
  tree=ast.parse(text)
  record['symbols']=[{'name':n.name,'line':n.lineno,'end':n.end_lineno,'type':type(n).__name__} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
  record['imports']=[{'line':n.lineno,'statement':ast.unparse(n)} for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
 source.append(record)
(OUT/'source-map.json').write_text(json.dumps(source,indent=2)+'\n')
ledger=[]
for p in sorted({ROOT/'AGENTS.md',ROOT/'memory/current-state.md',*ROOT.glob('docs/LOOP_LINEARNO*.md'),*ROOT.glob('PLAN_Looped_LinearNO/*.md')}):
 text=p.read_text();titles=[{'line':i,'heading':line[:250]} for i,line in enumerate(text.splitlines(),1) if line.startswith('#')]
 ledger.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':len(text.splitlines()),'headings':titles,'method':'document read/section index; final frozen decisions take precedence over historical proposals'})
# AGENTS links may refer to user-deleted historical materials. Never recreate them.
agents=(ROOT/'AGENTS.md').read_text();missing=[]
for target in re.findall(r'\]\(([^)]+)\)',agents):
 if not target.startswith(('http','/')) and not (ROOT/target).exists():missing.append(target)
(OUT/'reading-ledger.json').write_text(json.dumps({'documents':ledger,'source_inventory':'source-map.json','missing_AGENTS_references':sorted(set(missing)),'discussion_review':'25,424-line export: historical evolution indexed; final V3 decision sections 22449-25424 cross-checked against the complete 837-line staged specification; discarded MoE/Hourglass/history proposals confer no authorization','prior_evidence':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'docs/loop_linearno_ffn_audit').glob('lf*/*.json'))]},ensure_ascii=False,indent=2)+'\n')
print('source files',len(source),'docs',len(ledger),'missing historical AGENTS targets',len(set(missing)))
