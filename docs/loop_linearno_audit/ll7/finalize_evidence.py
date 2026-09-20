"""LL7 result review and complete tracked/untracked/ignored preservation."""
from collections import Counter
import ast,datetime,difflib,hashlib,importlib.metadata,json,os,platform,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from cdlno.linearno_loop.ll7_projection import REPLACEMENTS,project
AUDIT='docs/loop_linearno_audit/ll7/'
ALLOWED=set(REPLACEMENTS)|{'tests/loop_entry_projection.py','tests/loop_linearno/test_isolation.py','docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md'}
NEW={f'cdlno/linearno_loop/{s}.py' for s in ('industrial_entry','industrial_state','air_entry','car_entry','ll7_projection')}
NEW|={f'tests/loop_linearno/{s}.py' for s in ('industrial_support','air_worker','car_worker','test_industrial')}
NEW.add('docs/LOOP_LINEARNO_LL7_INDUSTRIAL.md')
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def inventory():
 rows={}
 for kind,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),('ignored',['--others','--ignored','--exclude-standard'])]:
  for raw in git('ls-files','-z',*args).split(b'\0'):
   if raw:
    name=os.fsdecode(raw);p=ROOT/name
    if p.is_file():rows[name]=dict(path=name,classification=kind,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 return rows

def results():
 old=json.loads((OUT.parent/'ll0/regression-results.json').read_text());new=json.loads((OUT/'regression-results.json').read_text())
 assert len(old)==len(new)==24
 for a,b in zip(old,new):
  for key in ('module','exit_code','tests','failures','errors','skipped'):assert a.get(key)==b.get(key),(b['module'],key)
 details={}
 for module in ('linearno.test_legacy','linearno.test_static_integration'):
  extract=lambda s:[l for l in s.splitlines() if l.startswith(('FAIL:','AssertionError:'))]
  before=extract((OUT.parent/'ll0'/f'{module}.log').read_text());after=extract((OUT/f'{module}.log').read_text())
  assert before==after;details[module]=after
 summary=json.loads((OUT.parent/'ll0/regression-summary.json').read_text());summary.update(status='LL7 outcomes and historical failure text exactly equal approved LL0',historical_assertions=details)
 write('regression-summary.json',summary)
 for name in ('attnres-report.json','body-report.json','sr-parity.json','accounting.json','rb-report.json','lb-report.json','modes-report.json'):
  assert json.loads((OUT/name).read_text())==json.loads((OUT.parent/'ll6'/name).read_text()),name
 log=(OUT/'loop-tests.log').read_text();assert '\nOK (skipped=2)\n' in log
 match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log);assert match and int(match[1])==83
 history=(OUT/'history-industrial.log').read_text();assert '\nFAILED (failures=1)\n' in history
 assert [l for l in history.splitlines() if l.startswith('FAIL:')]==['FAIL: test_old_source_fingerprints_task_bodies_and_launchers (linearno.test_history_industrial.HistoryIndustrial.test_old_source_fingerprints_task_bodies_and_launchers)']
 historical=json.loads((OUT/'historical-fingerprint-review.json').read_text())
 assert historical['original_suite']==dict(methods=5,passed=4,failures=1,errors=0,seconds=688.606)
 for row in historical['hashes'].values():
  for key,value in row.items():
   assert value['current']==value['pre_LL7']
   assert value['recomputed_with_R8_git_base']==value['R8']
   if key=='source_sha256':assert value['current']==value['R8']
 assert len(historical['projected_sources'])==3 and all(r['exact_bytes'] for r in historical['projected_sources'].values())
 assert len(historical['launcher_checks'])==6 and all(r['exit_code']==0 for r in historical['launcher_checks'])
 assert len(historical['identical_helpers'])==8 and historical['car_objective_ast_equal']
 hm=re.search(r'Ran (\d+) tests in ([\d.]+)s',history)
 standard=json.loads((OUT/'native-matrix.json').read_text());industrial=json.loads((OUT/'industrial-matrix.json').read_text())
 assert standard['completed']==standard['expected']==36
 assert industrial['completed']==industrial['expected']==12
 for r in standard['rows']:
  for k in ('exact_weights','exact_optimizer_scheduler_rng','exact_batches','exact_eval'):assert r[k]
 for r in industrial['rows']:
  assert r['exact']
  for k,report in r['reports'].items():
   assert report['metadata_immutable']
   if 'eval' not in k:assert report['loop_forward_checks']>0
  if r['task']=='airfrans':assert r['reports']['full-ensemble']['distinct_members']
 replay=json.loads((OUT/'pre-checkpoint-replay.json').read_text());assert len(replay)==4 and all(r['exact_weights_optimizer_rng_output'] for r in replay)
 conflicts=json.loads((OUT/'saved-conflicts.json').read_text());assert len(conflicts)==12 and sum(r['checks'] for r in conflicts)==120 and all(r['passed'] for r in conflicts)
 projections={}
 for name in REPLACEMENTS:
  raw=(ROOT/name).read_text();old=(OUT/'before'/name).read_text();assert project(name,raw)==old
  projections[name]=dict(exact_bytes=True,sha256=hashlib.sha256(old.encode()).hexdigest())
 write('routing-projection.json',projections)
 from cdlno.linearno.air_entry import _provenance
 from cdlno.linearno.car_entry import task_provenance
 from cdlno.linearno_loop.provenance import provenance
 from cdlno.linearno_history.provenance import research_provenance
 previous=json.loads((OUT/'pre-edit.json').read_text());hashes={}
 for name,fn in [('air',_provenance),('car',task_provenance),('history_air',lambda:research_provenance(_provenance())),('history_car',lambda:research_provenance(task_provenance())),('loop',provenance)]:
  current=fn()
  for k in ('source_sha256','normalized_patch_sha256'):assert current[k]==previous[name][k],(name,k)
  hashes[name]={k:current[k] for k in ('source_sha256','normalized_patch_sha256')}
 write('preserved-provenance.json',hashes)
 syntax=[]
 for name in sorted(NEW|ALLOWED):
  if name.endswith('.py'):
   source=(ROOT/name).read_text();ast.parse(source,feature_version=(3,10));compile(source,name,'exec')
   assert all(l.rstrip()==l for l in source.splitlines());syntax.append(name)
 write('environment.json',dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('torch','numpy','torch-geometric','timm','einops')},device='CPU only',CUDA_VISIBLE_DEVICES='',real_data_gpu='NOT RUN'))
 write('delivery-review.json',dict(stage='LL7',status='PASS',tests=dict(loop_methods=83,pass_=81,skip=2,errors=0,failures=0,seconds=float(match[2]),
   saved_metadata_conflicts=120,industrial_cases=12,industrial_worker_processes=66,air_two_member_boundaries=12,
   standard_cases=36,standard_worker_processes=144,history_industrial_methods=int(hm[1]),history_industrial_seconds=float(hm[2]),
   history_industrial_pass=4,history_industrial_historical_failures=1,
   historical_fingerprint_diagnosis='Only git HEAD before-AST changed since R8; all pre-LL7 hashes preserved; remaining assertions separately passed; original test/golden unchanged',
   legacy=summary['total'],genuine_pre_edit_checkpoints=4),
   reviewed=dict(wrapper='Native stem/forward and temperature spellings unchanged; same LL6 seven numeric reports',
    science='Air native train, Car same inherited train function; unchanged objective/data/field/drag helpers; only optional checkpoint export at old save sites',
    serialization='Distinct schema/state_dict; metadata-first and120 actual-run conflict checks; strict shared-core/router states and optimizer signatures',
    ensemble='Two members independent params/router/core; within-member and between-member boundaries exact state/RNG/data/output',
    legacy='4 genuine pre-edit pure/history industrial resumes exact; LL6 Standard fingerprint preserved; full36 Standard cycles rerun; complete byte projections'),
   syntax_python310=syntax,not_run=['real training/data/VTK metrics','remote/GPU/AMP/compile','full-width industrial training','new loop launchers/monitor','LL8 and later']))

def freeze():
 initial=json.loads((OUT/'start-manifest.json').read_text());before={r['path']:r for r in initial['files']};now=inventory();diff=[]
 for name in sorted(ALLOWED):
  old=(OUT/'before'/name).read_text();assert hashlib.sha256(old.encode()).hexdigest()==before[name]['sha256']
  diff.extend(difflib.unified_diff(old.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before-LL7/'+name,tofile=name))
 for name in sorted(NEW):
  assert name not in before
  diff.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
 (OUT/'source.diff').write_text(''.join(diff));(OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
 now=inventory();changes=[dict(path=p,before=r,after=now.get(p)) for p,r in before.items() if p not in now or r!=now[p]]
 unexpected=[r for r in changes if r['path'] not in ALLOWED]
 added=[r for p,r in now.items() if p not in before]
 extras=[r for r in added if not (r['path'].startswith(AUDIT) or r['path'] in NEW)]
 ignored=[r for r in added if r['classification']=='ignored' and not (r['path'].startswith(AUDIT) and r['path'].endswith('.log'))]
 chk=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
 # Existing LL6 diff is the start truth; the seven LL7 deltas project byte-exactly back to it.
 entries=[f'{folder}/{name}.py' for folder in ('Airfoil-Design-AirfRANS','Car-Design-ShapeNetCar') for name in ('main','main_evaluation')]
 entries += [f'PDE-Solving-StandardBenchmark/exp_{n}.py' for n in ('airfoil','darcy','elas','pipe','ns','plas')]
 checks=dict(ten_entry_bytes_unchanged=all(before[p]==now[p] for p in entries),diff_check=chk.returncode==0,
  staged_unchanged=git('diff','--cached','--binary')==(OUT/'start-staged.diff').read_bytes(),head_unchanged=git('rev-parse','HEAD').decode().strip()==initial['head'])
 own={AUDIT+'end-freeze.json',AUDIT+'end-status.txt'}
 result=dict(repo=str(ROOT),head=initial['head'],tree=git('rev-parse','HEAD^{tree}').decode().strip(),branch=git('branch','--show-current').decode().strip(),remote=git('remote','-v').decode(),
  baseline_count=len(before),baseline_classifications=dict(Counter(r['classification'] for r in before.values())),unchanged=len(before)-len(changes),changed=changes,
  unexpected_changes=unexpected,unexpected_new=extras,unexpected_ignored=ignored,checks=checks,
  new_files=[r for r in added if r['path'] not in own],unhashed_self_records=sorted(own),allowed_ignored=AUDIT+'*.log; synthetic archives external',
  passed=not (unexpected or extras or ignored) and all(checks.values()))
 write('end-freeze.json',result)
 print(json.dumps({k:result[k] for k in ('passed','baseline_count','unchanged','unexpected_changes','unexpected_new','unexpected_ignored','checks')},indent=2))
 assert result['passed']
if __name__=='__main__':
 results()
 if '--results-only' not in sys.argv:freeze()
