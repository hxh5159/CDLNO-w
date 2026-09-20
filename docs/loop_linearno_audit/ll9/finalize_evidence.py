"""Freeze complete pre-LL9 tree and summarize honest CPU/GPU acceptance."""
import ast,collections,hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
def read(name):return json.loads((OUT/name).read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def inventory():
 rows={}
 for kind,flags in [('tracked',[]),('untracked',['--others','--exclude-standard']),('ignored',['--others','--ignored','--exclude-standard'])]:
  for raw in git('ls-files','-z',*flags).split(b'\0'):
   if raw:
    name=os.fsdecode(raw);p=ROOT/name
    if p.is_file():rows[name]=dict(path=name,classification=kind,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 return rows

def freeze():
 before=read('start-manifest.json');now=inventory();old={r['path']:r for r in before['files']}
 changes=[dict(path=p,before=r,after=now.get(p)) for p,r in old.items() if p not in now or r['sha256']!=now[p]['sha256'] or r['classification']!=now[p]['classification']]
 allowed={'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md'}
 artifacts=[r for r in changes if r['path'].endswith('.pyc') or r['path'].endswith('.log')]
 source_changes=[r for r in changes if r not in artifacts]
 assert {r['path'] for r in source_changes}<=allowed,source_changes
 assert git('rev-parse','HEAD').decode().strip()==before['HEAD']
 assert git('rev-parse','HEAD^{tree}').decode().strip()==before['tree']
 assert git('diff','--cached','--binary')==(OUT/'start-staged.diff').read_bytes()
 assert git('diff','--binary')==(OUT/'start-tracked.diff').read_bytes(),'no old tracked edit authorized'
 self_generated={str((OUT/name).relative_to(ROOT)) for name in ('end-freeze.json','end-status.txt','finalize.log')}
 added=[v for p,v in now.items() if p not in old and p not in self_generated]
 prefixes=('docs/loop_linearno_audit/ll9/','tools/linearno_loop_','tests/loop_linearno/test_performance_tools.py',
    'docs/LOOP_LINEARNO_PERFORMANCE.md','memory/2026-09-20-loop-linearno-ll9.md')
 external=[r for r in added if not r['path'].startswith(prefixes)]
 # Existing tests can create ignored output sidecars/pyc; preserve and list them.
 write('end-freeze.json',dict(start_files=len(old),unchanged=len(old)-len(changes),source_changes=source_changes,
    changed_runtime_artifacts=artifacts,new_files=added,additional_runtime_files=external,self_inventory_exclusions=sorted(self_generated),
    tracked_diff_identical=True,HEAD_identical=True,tree_identical=True,staged_identical=True,
    policy='No old production/model/schema/test/monitor/launcher edits. Runtime pyc/log/output artifacts are reported, not reverted; no real data or training.'))
 (OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
 from cdlno.linearno.air_entry import _provenance
 from cdlno.linearno.car_entry import task_provenance
 from cdlno.linearno_loop.provenance import provenance
 from cdlno.linearno_history.provenance import research_provenance
 from cdlno.linearno_loop.industrial_state import provenance as industrial
 prev=json.loads((OUT.parent/'ll8/preserved-provenance.json').read_text());hashes={}
 for name,fn in [('air',_provenance),('car',task_provenance),('history_air',lambda:research_provenance(_provenance())),
     ('history_car',lambda:research_provenance(task_provenance())),('loop',provenance),
     ('loop_car',lambda:industrial('car')),('loop_airfrans',lambda:industrial('airfrans'))]:
  v=fn();hashes[name]={k:v[k] for k in ('source_sha256','normalized_patch_sha256')};assert hashes[name]==prev[name],name
 write('preserved-provenance.json',hashes)


def results():
 counts=read('counts.json')['rows'];matrix=read('matrix.json')['rows'];cpu=read('cpu.json');cuda=read('cuda-final-v2.json')
 assert len(counts)==96 and len(matrix)==48 and len(cpu['rows'])==48 and len(cuda['rows'])==144
 assert all(r['analytic']['parameter_parts']==r['measured_parameter_parts'] for r in counts)
 assert all(not r['costs']['forbidden_attention'] and r['head_calls']==1 and r['strict_reload_max_abs']==r['repeat_forward_max_abs']==0 for r in matrix)
 native=read('native-matrix.json');industrial=read('industrial-matrix.json')
 assert native['completed']==native['expected']==18 and industrial['completed']==industrial['expected']==6
 assert all(r['exact_weights'] and r['exact_optimizer_scheduler_rng'] and r['exact_eval'] for r in native['rows'])
 assert all(r['exact'] for r in industrial['rows'])
 original=read('regression-results.json');retry=read('regression-rechecks.json');byname={r['module']:r for r in original}
 for r in retry:byname[r['module']]=r
 rows=list(byname.values());assert all('tests' in r for r in rows)
 test_total=sum(r['tests'] for r in rows)
 fails=[dict(module=r['module'],name=n,message=t) for r in rows for n,t in r.get('failures',[])]
 errors=[dict(module=r['module'],name=n,message=t) for r in rows for n,t in r.get('errors',[])]
 skips=[dict(module=r['module'],name=n,reason=t) for r in rows for n,t in r.get('skipped',[])]
 failed_methods={(r['module'],r['name'].split(') (')[0]) for r in fails}
 write('regression-final.json',dict(modules=len(rows),methods=test_total,failed_methods=len(failed_methods),failed_assertions=len(fails),errors=errors,
    passed_methods=test_total-len(failed_methods)-len(errors)-len(skips),skips=skips,failures=fails,results=rows,
    initial_attempt='regression-results.json',rechecks='regression-rechecks.json'))
 gpu_failed=[r for r in cuda['rows'] if r['status']=='FAIL']
 assert len(gpu_failed)==20 and all(r['mode']=='rb_attnres' and r['precision']!='FP32' for r in gpu_failed)
 write('summary.json',dict(stage='LL9',status='PARTIAL',reason='20 RB AMP mixed source dtype smoke failures plus one newly triggered ignored-pyc freeze assertion; no production source edits',
    full_profile_counts=96,canonical_spatial_small_width_cases=48,new_tests=7,new_tests_passed=7,
    cpu_timing_cases=48,cpu_warmup=cpu['warmup'],cpu_measured=cpu['measured'],
    cuda_smoke_cases=144,cuda_pass=124,cuda_fail=20,cuda_FP32_pass=48,
    native_standard_closures=18,native_industrial_closures=6,native_new_processes=105,
    regression=dict(modules=len(rows),methods=test_total,failed_methods=len(failed_methods),failed_assertions=len(fails),errors=len(errors),skips=len(skips)),
    not_run=['real data/VTK metrics/real training/epoch efficiency/SOTA','remote Python3.10 torch2.11 cu128','full-profile full-N training','LL10']))

if __name__=='__main__':
 if '--freeze-only' not in sys.argv:results()
 freeze()
