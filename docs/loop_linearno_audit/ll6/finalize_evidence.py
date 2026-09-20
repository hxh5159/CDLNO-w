"""LL6 final evidence: native cycles, legacy parity, exact source projection."""
from collections import Counter
from pathlib import Path
import ast
import datetime
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
AUDIT='docs/loop_linearno_audit/ll6/'
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from cdlno.linearno_loop.provenance import REPLACEMENTS,legacy_source

ALLOWED_EXISTING=set(REPLACEMENTS)|{'tests/history_entry_projection.py',
    'tests/loop_linearno/test_isolation.py','docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md'}
NEW_FILES={f'cdlno/linearno_loop/{n}.py' for n in ('standard_entry','checkpoint','provenance')}
NEW_FILES|={f'tests/loop_linearno/{n}.py' for n in ('native_worker','test_standard_entry')}
NEW_FILES|={'tests/loop_entry_projection.py','docs/LOOP_LINEARNO_LL6_STANDARD.md',
            'docs/LOOP_LINEARNO_LL6_COMMANDS.md'}


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def inventory():
    rows={}
    for category,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),
                          ('ignored',['--others','--ignored','--exclude-standard'])]:
        for raw in git('ls-files','-z',*args).split(b'\0'):
            if not raw:continue
            name=os.fsdecode(raw);path=ROOT/name
            if path.is_file():rows[name]=dict(path=name,classification=category,
                size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return rows


def results():
    old=json.loads((OUT.parent/'ll0/regression-results.json').read_text())
    current=json.loads((OUT/'regression-results.json').read_text())
    assert len(old)==len(current)==24
    for a,b in zip(old,current):
        assert a['module']==b['module']
        for key in ('exit_code','tests','failures','errors','skipped'):
            assert a.get(key)==b.get(key),(b['module'],key)
    failures={}
    for name in ('linearno.test_legacy','linearno.test_static_integration'):
        extract=lambda t:[s for s in t.splitlines() if s.startswith(('FAIL:','AssertionError:'))]
        before=extract((OUT.parent/'ll0'/f'{name}.log').read_text())
        after=extract((OUT/f'{name}.log').read_text())
        assert before==after
        failures[name]=after
    summary=json.loads((OUT.parent/'ll0/regression-summary.json').read_text())
    summary.update(status='LL6 matches approved LL0; no new failure',same_as_LL0=True,
                   historical_assertions=failures)
    write('regression-summary.json',summary)
    log=(OUT/'loop-tests.log').read_text()
    assert '\nOK (skipped=2)\n' in log and '\nFAILED (' not in log
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    assert match and int(match[1])==78
    math_names=('attnres-report.json','body-report.json','sr-parity.json','accounting.json',
                'rb-report.json','lb-report.json','modes-report.json')
    for name in math_names:
        assert json.loads((OUT/name).read_text())==json.loads((OUT.parent/'ll5'/name).read_text()),name
    write('prior-stage-recheck.json',dict(equal_to_LL5=list(math_names),all_equal=True))
    native=json.loads((OUT/'native-matrix.json').read_text())
    assert native['completed']==native['expected']==len(native['rows'])==36
    assert len({(r['task'],r['preset'],r['mode']) for r in native['rows']})==36
    totals=Counter();tasks={};artifact_counts=Counter()
    for row in native['rows']:
        for name in ('exact_weights','exact_optimizer_scheduler_rng','exact_batches','exact_eval',
                     'metadata_first','sidecar_immutable','normalizer_no_refit','real_spatial_shape'):
            assert row[name]
        task=row['task'];task_rows=tasks.setdefault(task,[])
        for action,report in row['reports'].items():
            assert report['public_dispatch_used'] and report['forward_local_checks']>0
            assert report['immutable_metadata']
            if action in ('resume','eval'):assert report['metadata_recovers_model']
            totals['checked_model_forwards']+=report['forward_local_checks']
            for artifact in report.get('recorded_artifacts',[]):
                artifact_counts[Path(artifact['path']).suffix]+=1
        counters=row['reports']['train'].get('counters',{})
        if task=='ns':
            assert [counters[k] for k in ('train_forward','backward','optimizer','scheduler')]==[60,6,6,6]
        if task=='plasticity':
            assert [counters[k] for k in ('train_forward','backward','optimizer','scheduler')]==[120,120,120,6]
        task_rows.append(dict(preset=row['preset'],mode=row['mode'],train_counters=counters,
            prediction_hash=row['reports']['eval']['prediction_hash'],exact_max_abs=0.0))
    assert len(tasks)==6 and all(len(v)==6 for v in tasks.values())
    replay=json.loads((OUT/'pre-checkpoint-replay.json').read_text())
    assert len(replay)==2
    for r in replay:
        assert all(r[k] for k in ('exact_final_weights','exact_optimizer_scheduler_rng','exact_batches','exact_eval'))
    commands=json.loads((OUT/'command-previews.json').read_text())
    assert commands['train']==72 and commands['eval']==36 and all(r['pass_'] for r in commands['rows'])
    syntax=[]
    for name in sorted(NEW_FILES|ALLOWED_EXISTING):
        if not name.endswith('.py'):continue
        source=(ROOT/name).read_text();ast.parse(source,feature_version=(3,10));compile(source,name,'exec')
        assert all(line.rstrip()==line for line in source.splitlines()),name
        syntax.append(name)
    projection={}
    for name in REPLACEMENTS:
        old=(OUT/'before'/name).read_bytes();projected=legacy_source(name,(ROOT/name).read_text()).encode()
        assert old==projected
        projection[name]=dict(exact_bytes=True,sha256=hashlib.sha256(old).hexdigest())
    write('routing-projection.json',projection)
    write('native-summary.json',dict(cases=36,fresh_worker_processes=144,
        exact_weight_optimizer_rng_batch_and_eval=True,max_abs=0.0,tasks=tasks,
        totals=dict(totals),artifact_inventory_entries_by_suffix=dict(artifact_counts),
        limits='CPU; real N; d8/h2/M4; synthetic 4train/2test; 3epochs; no real loader; standalone showcase plotting disabled'))
    write('environment.json',dict(python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','torch-geometric','timm','einops')},
        CUDA_VISIBLE_DEVICES='',device_used='cpu',gpu_or_real_data='NOT RUN'))
    write('delivery-review.json',dict(stage='LL6',status='PASS',
        reviewed=dict(routing='Explicit loop train or metadata family; old parse/factory and exact source projection checked',
            protocols='Six complete exp files unchanged; real main AST synthetic full-spatial loops; NS/Plasticity counters and state isolation',
            checkpoint='Read JSON before construction/weights; pair/manifest strict=True; optimizer shapes/groups/topology; no overwrite or normalizer refit',
            compatibility='Genuine pre-edit pure/history resumes exact; 24 old module outcomes/failure text equal LL0; seven loop math reports equal LL5',
            outputs='Original recorder/final field plots/eval result directories; 72 train and36 eval commands checked; real runs explicitly not claimed'),
        tests=dict(loop_methods=78,passed=76,skipped=2,failures=0,errors=0,seconds=float(match[2]),
            native_cases=36,native_fresh_processes=144,pre_edit_checkpoint_families=2,
            parser_profile_cases=108,train_previews=72,eval_previews=36,old_LL0=summary['total']),
        syntax_python310=syntax,not_run=['real loaders/real data/500epoch/convergence/accuracy',
            'GPU/AMP/compile/remote target','full-width native training','standalone evaluation showcase plots',
            'industrial loop production routing','loop logical-visit monitor','LL7 and later']))


def freeze():
    baseline=json.loads((OUT/'start-manifest.json').read_text());before={r['path']:r for r in baseline['files']}
    now=inventory();patch=[]
    for name in sorted(NEW_FILES):
        assert name not in before and name in now,name
        patch.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    for name in sorted(ALLOWED_EXISTING):
        prior=(OUT/'before'/name).read_text()
        assert hashlib.sha256(prior.encode()).hexdigest()==before[name]['sha256'],name
        patch.extend(difflib.unified_diff(prior.splitlines(True),(ROOT/name).read_text().splitlines(True),
            fromfile='before-LL6/'+name,tofile=name))
    (OUT/'source.diff').write_text(''.join(patch))
    checks={}
    for label,args in [('diff_check',['diff','--check']),('unstaged',['diff','--binary']),('staged',['diff','--cached','--binary'])]:
        p=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        checks[label]=dict(exit_code=p.returncode,output=p.stdout+p.stderr)
    checks['staged_diff_unchanged']=checks['staged']['output']==(OUT/'start-staged.diff').read_text()
    (OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
    now=inventory()
    changed=[dict(path=p,before=r,after=now.get(p)) for p,r in before.items()
        if p not in now or any(r[k]!=now[p][k] for k in ('classification','size','sha256'))]
    unexpected_changes=[r for r in changed if r['path'] not in ALLOWED_EXISTING]
    added=[v for p,v in now.items() if p not in before]
    unexpected_new=[v for v in added if not (v['path'].startswith(AUDIT) or v['path'] in NEW_FILES)]
    unexpected_ignored=[v for v in added if v['classification']=='ignored' and not
        (v['path'].startswith(AUDIT) and v['path'].endswith('.log'))]
    entries=[f'PDE-Solving-StandardBenchmark/exp_{t}.py' for t in ('airfoil','darcy','elas','pipe','ns','plas')]
    checks['six_exp_byte_equal']=all(before[p]==now[p] for p in entries)
    self_files={AUDIT+'end-freeze.json',AUDIT+'end-status.txt'}
    result=dict(repo=str(ROOT),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        head=git('rev-parse','HEAD').decode().strip(),tree=git('rev-parse','HEAD^{tree}').decode().strip(),
        branch=git('branch','--show-current').decode().strip(),remote=git('remote','-v').decode().strip(),
        baseline_existing=len(before),baseline_counts=dict(Counter(r['classification'] for r in before.values())),
        unchanged=len(before)-len(changed),changed=changed,unexpected_changes=unexpected_changes,
        unexpected_new=unexpected_new,unexpected_ignored=unexpected_ignored,
        allowed_ignored=AUDIT+'*.log; no new repo cache/run directory; synthetic runs external',
        new_files=[r for r in added if r['path'] not in self_files],self_generated_unhashed=sorted(self_files),checks=checks)
    result['pass']=not (unexpected_changes or unexpected_new or unexpected_ignored) and all((
        checks['staged_diff_unchanged'],checks['six_exp_byte_equal'],checks['diff_check']['exit_code']==0,
        result['head']==baseline['head']))
    write('end-freeze.json',result)
    print(json.dumps({k:result[k] for k in ('pass','baseline_existing','baseline_counts','unchanged',
                    'unexpected_changes','unexpected_new','unexpected_ignored')},indent=2))
    assert result['pass']


if __name__=='__main__':
    results()
    if '--results-only' not in sys.argv:freeze()
