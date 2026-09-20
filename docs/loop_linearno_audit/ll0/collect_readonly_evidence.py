"""LL0 inventory/provenance collection; never imports runnable task entries."""
import ast, collections, hashlib, importlib.metadata, json, pathlib, platform, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]; OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def save(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=json.loads((OUT/'start-manifest.json').read_text());inventory=[];modules=[];binary=[]
for row in start['files']:
 if row['classification']=='ignored':continue
 p=ROOT/row['path'];b=p.read_bytes();r=dict(row)
 try: t=b.decode('utf-8'); assert '\x00' not in t
 except (UnicodeError,AssertionError):r.update(coverage='binary_hash_only');binary.append(r);inventory.append(r);continue
 r.update(coverage='inventory_only',lines=len(t.splitlines()))
 if p.suffix=='.py':
  try:
   tree=ast.parse(t,filename=str(p));ast.parse(t,feature_version=(3,10))
   symbols=[dict(name=n.name,kind=type(n).__name__,line=n.lineno,end=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
   imports=[dict(line=n.lineno,module=n.module or '',names=[a.name for a in n.names],level=n.level) if isinstance(n,ast.ImportFrom) else dict(line=n.lineno,names=[a.name for a in n.names]) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
   modules.append(dict(path=row['path'],symbols=symbols,imports=imports,python310_parse=True));r['coverage']='AST_indexed_not_full_semantic_review'
  except SyntaxError as e:r.update(coverage='AST_error',error=str(e))
 elif p.suffix=='.ipynb':
  cells=[dict(cell=i,lines=len(''.join(c.get('source',[])).splitlines()),sha256=hashlib.sha256(''.join(c.get('source',[])).encode()).hexdigest()) for i,c in enumerate(json.loads(t).get('cells',[])) if c.get('cell_type')=='code'];r.update(coverage='notebook_code_indexed',code_cells=cells)
 inventory.append(r)
save('file-coverage.json',dict(disclaimer='Inventory/AST coverage is not a claim of semantic reading of every historical report or source file. Targeted semantic reading is separately enumerated in REFERENCE_AUDIT.',counts=dict(collections.Counter(r['coverage'] for r in inventory)),files=inventory))
save('python-symbol-import-index.json',modules)
packages={}
for name in ('torch','numpy','torch-geometric','timm','einops','matplotlib','pytest','torch-cluster'):
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
save('environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,test_device='cpu',cuda_mask='',gpu_not_probed=True,remote_target='user-reported Python3.10 / torch2.11 / cu128; NOT RUN'))
from cdlno.linearno.profiles import resolve_config,TASKS,PROFILES
configs=[]
for task in TASKS:
 for profile in PROFILES:
  options={} if task in ('car','airfrans') else dict(contract='standard_temporal_l5' if task in ('ns','plasticity') else 'standard_static_l4')
  c=resolve_config(task,profile,**options);m=c['values']['model']
  configs.append(dict(task=task,profile=profile,current=c,proposed_loop=dict(base_rank=m['linearno_rank'],rank_multiplier=2,resolved_rank=2*m['linearno_rank'],hidden=m['hidden'],head_dim=m['hidden']//m['heads'],status='DESIGN ONLY; no loop model constructed')))
save('resolved-profiles.json',configs)
old={r['path']:r for r in json.loads((ROOT/'docs/linearno_history_audit/r0/start-manifest.json').read_text())}
protected=json.loads((ROOT/'docs/linearno_history_audit/r10/protected-source-recheck.json').read_text())['paths']
rows=[dict(path=p,baseline=old[p]['sha256'],current=sha(ROOT/p),equal=old[p]['sha256']==sha(ROOT/p)) for p in protected]
save('historical-protected-recheck.json',dict(checked=len(rows),all_byte_identical=all(r['equal'] for r in rows),files=rows))
from cdlno.linearno_history.provenance import baseline_source,ROUTING_REPLACEMENTS
b=json.loads((ROOT/'docs/linearno_history_audit/r6/baseline.json').read_text());base=pathlib.Path(b['snapshot'])/'source';pr=[]
for p in ROUTING_REPLACEMENTS:
 projected=baseline_source(p,(ROOT/p).read_text()); expected=(base/p).read_text();pr.append(dict(path=p,equal=projected==expected,projected_sha256=hashlib.sha256(projected.encode()).hexdigest(),baseline_sha256=hashlib.sha256(expected.encode()).hexdigest()))
save('routing-projection.json',dict(all_equal=all(r['equal'] for r in pr),baseline=str(base),files=pr))
from cdlno.linearno.standard_entry import provenance
current=provenance();prev=json.loads((ROOT/'docs/linearno_history_audit/r6/baseline-provenance.json').read_text())
save('baseline-provenance.json',dict(current=current,comparison={k:dict(current=current[k],historical=prev[k],equal=current[k]==prev[k]) for k in ('source_sha256','normalized_patch_sha256')}))
# Re-verify cached fixed git blobs without consulting a moving upstream branch.
ref=pathlib.Path('/home/hwz/CDLNO-artifacts/linearno-history-r0-20260918T055437Z/references');sources=[]
for record in json.loads((ref/'local-sources.json').read_text()):
 files=[];root=pathlib.Path(record['export'])
 for f in record['files']:
  p=root/f['path'];b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  files.append(dict(path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob=blob,equals_pin=blob==f['blob']))
 sources.append(dict(name=record['name'],commit=record['sha'],tree=record['tree'],last_commit=record['last_commit'],licenses=record['licenses'],all_verified=all(f['equals_pin'] for f in files),files=files))
for kind in ('attnres','k3'):
 commit=json.loads((ref/(kind+'_commit.json')).read_text());tree=json.loads((ref/(kind+'_tree.json')).read_text());files=[]
 for d in json.loads((ref/'downloaded-sources.json').read_text()):
  if pathlib.Path(d['path']).parent.name!=kind:continue
  b=pathlib.Path(d['path']).read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();files.append(dict(**d,actual_blob=blob,equals_pin=blob==d['git_blob']))
 sources.append(dict(name=kind,commit=commit['sha'],tree=commit['commit']['tree']['sha'],last_commit=commit['commit']['message'],tree_files=[r['path'] for r in tree['tree'] if r['type']=='blob'],files=files,all_verified=all(f['equals_pin'] for f in files)))
papers=[('linearno_v3','https://arxiv.org/html/2511.06294v3',pathlib.Path('/home/hwz/CDLNO-artifacts/linearno-l0-before-65a2wvms/paper-v3.html')),('attnres_v1','https://arxiv.org/pdf/2603.15031v1',ref/'attnres_pdf.pdf'),('kimi_k3_v2','https://arxiv.org/html/2607.24653v2',ref/'k3_paper.html'),('residual_scaling_v1','https://arxiv.org/html/2606.18524v1',pathlib.Path('/home/hwz/CDLNO-artifacts/loop-linearno-ll0-20260919/references/residual_scaling_v1.html')),('smelt_v2','https://arxiv.org/html/2609.01343v2',pathlib.Path('/home/hwz/CDLNO-artifacts/loop-linearno-ll0-20260919/references/k3_v2.html'))]
save('reference-ledger.json',dict(repositories=sources,papers=[dict(name=n,url=u,path=str(p),size=p.stat().st_size,sha256=sha(p)) for n,u,p in papers],note='The external download filename k3_v2.html for 2609.01343 is a local naming alias; its actual title is SMELT, not Kimi K3. Kimi K3 is 2607.24653v2. No external code vendored.'))
print(json.dumps(dict(inventory_files=len(inventory),python_modules=len(modules),binary=len(binary),profile_rows=len(configs),protected_identical=all(r['equal'] for r in rows),routing_equal=all(r['equal'] for r in pr),reference_repositories_verified=[(r['name'],r['all_verified']) for r in sources]),indent=2))
