"""LL5 regression comparison, numerical review and tracked/untracked/ignored freeze."""
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
AUDIT='docs/loop_linearno_audit/ll5/'
ALLOWED_EXISTING={'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md','tests/loop_linearno/test_sr_core.py',
                  'cdlno/linearno_loop/core.py','cdlno/linearno_loop/construction.py'}
NEW_FILES={f'tests/loop_linearno/{n}.py' for n in ('lb_oracle','lb_support','test_lb_core','test_modes')}
NEW_FILES.add('docs/LOOP_LINEARNO_CORE_REPORT.md')


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def inventory():
    result={}
    for category,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),
                          ('ignored',['--others','--ignored','--exclude-standard'])]:
        for raw in git('ls-files','-z',*args).split(b'\0'):
            if not raw:continue
            name=os.fsdecode(raw);path=ROOT/name
            if path.is_file():
                result[name]=dict(path=name,classification=category,size=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def results():
    old=json.loads((OUT.parent/'ll0/regression-results.json').read_text())
    current=json.loads((OUT/'regression-results.json').read_text())
    assert len(old)==len(current)==24
    for a,b in zip(old,current):
        assert a['module']==b['module']
        for key in ('exit_code','tests','failures','errors','skipped'):
            assert a.get(key)==b.get(key),(b['module'],key)
    failure_details={}
    for name in ('linearno.test_legacy','linearno.test_static_integration'):
        extract=lambda text:[s for s in text.splitlines() if s.startswith(('FAIL:','AssertionError:'))]
        before=extract((OUT.parent/'ll0'/f'{name}.log').read_text())
        after=extract((OUT/f'{name}.log').read_text())
        assert before==after,(name,before,after)
        failure_details[name]=after
    summary=json.loads((OUT.parent/'ll0/regression-summary.json').read_text())
    summary.update(status='LL5 rerun matches approved LL0 outcomes; no new failure',
                   same_as_LL0=True,historical_assertions=failure_details)
    write('regression-summary.json',summary)
    log=(OUT/'loop-tests.log').read_text()
    assert log.rstrip().endswith('OK (skipped=2)')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    assert match and int(match[1])==74
    numeric={}
    for label,name in [('attnres','attnres-report.json'),('body','body-report.json'),
                       ('sr','sr-parity.json'),('rb','rb-report.json'),('lb','lb-report.json')]:
        groups={}
        for row in json.loads((OUT/name).read_text())['rows']:
            key='/'.join((row.get('reference','independent mathematics' if label!='body' else 'native block'),
                          row.get('device','cpu'),row['dtype']))
            g=groups.setdefault(key,dict(cases=0,max_abs=0.,max_mean_abs=0.,max_relative=0.,max_mean_relative=0.))
            g['cases']+=1
            for e in row['errors'].values():
                for outkey,inkey in [('max_abs','max_abs'),('max_mean_abs','mean_abs'),
                                     ('max_relative','max_relative'),('max_mean_relative','mean_relative')]:
                    g[outkey]=max(g[outkey],e[inkey])
        numeric[label]=groups
    write('numeric-summary.json',numeric)
    syntax=[]
    for path in sorted([*(ROOT/'cdlno/linearno_loop').glob('*.py'),*(ROOT/'linearno_loop').glob('*.py'),
                        *(ROOT/'tests/loop_linearno').glob('*.py'),*OUT.glob('*.py')]):
        source=path.read_text();ast.parse(source,feature_version=(3,10));compile(source,str(path),'exec')
        assert all(line.rstrip()==line for line in source.splitlines()),str(path)
        syntax.append(str(path.relative_to(ROOT)))
    for name in ('sr-parity.json','accounting.json','rb-report.json'):
        assert json.loads((OUT.parent/'ll4'/name).read_text())==json.loads((OUT/name).read_text()),name
    lb=json.loads((OUT/'lb-report.json').read_text());modes=json.loads((OUT/'modes-report.json').read_text())
    assert len(lb['rows'])==36 and len(lb['adamw'])==24
    assert len(modes['counts'])==48 and len(modes['roundtrips'])==48 and len(modes['costs'])==6
    for row in lb['rows']:
        expected=dict(atol=1e-5,rtol=1e-4) if row['dtype']=='torch.float32' else dict(atol=1e-12,rtol=1e-10)
        assert row['propagated_weight_tolerance']==expected
    write('environment.json',dict(python=sys.version,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','torch-geometric','timm','einops')},
        CUDA_VISIBLE_DEVICES='',device_used='cpu',gpu_or_real_data='NOT RUN'))
    write('delivery-review.json',dict(stage='LL5',status='PASS',
        reviewed=dict(delta='Operator AND MLP scale1/R; actual Y-H object identity; independent FP64/FP32 full-core equations',
            timing='Only R-1 boundaries plus output; live anchor/Delta sources; zero-query mean; R1 output-only; no extra scale',
            modes='Single enum; exclusive router namespaces; 48 full-profile constructors match common weights and RNG exactly',
            serialization='48 small wrapper AdamW/strict reloads and 9 metadata-first fresh-process reloads; mode conflicts rejected before import/load',
            compatibility='SR numeric/accounting and RB numeric reports identical to LL4; old outcomes/failure details match LL0; complete content freeze'),
        tests=dict(loop_methods=74,passed=72,skipped=2,failures=0,errors=0,seconds=float(match[2]),
                   new_LL5_methods=10,lb_math_cases=36,lb_native_adamw_cases=24,full_profile_constructors=48,
                   small_wrapper_steps=48,fresh_process_reloads=9,measured_compute_cases=6,old_LL0=summary['total']),
        syntax_python310=syntax,
        numeric_limits='Only propagated FP32 LB source weights use atol1e-5/rtol1e-4; all other LL5 FP32 tensors 1e-6/1e-5; FP64 1e-12/1e-10; LL2/SR/RB unchanged',
        not_run=['production task parser/factory/launcher/monitor integration','production loop full resume archives',
                 'real data/training','GPU/AMP/compile','remote Python3.10 target','timing/accuracy','LL6 and later']))


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
            fromfile='LL4/'+name,tofile=name))
    (OUT/'source.diff').write_text(''.join(patch))
    checks={}
    for label,args in [('diff_check',['diff','--check']),('unstaged',['diff','--binary']),('staged',['diff','--cached','--binary'])]:
        p=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        checks[label]=dict(exit_code=p.returncode,output=p.stdout+p.stderr)
    checks['tracked_diff_unchanged']=checks['unstaged']['output']==(OUT/'start-tracked.diff').read_text()
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
    self_files={AUDIT+'end-freeze.json',AUDIT+'end-status.txt'}
    result=dict(repo=str(ROOT),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        head=git('rev-parse','HEAD').decode().strip(),tree=git('rev-parse','HEAD^{tree}').decode().strip(),
        branch=git('branch','--show-current').decode().strip(),remote=git('remote','-v').decode().strip(),
        baseline_existing=len(before),baseline_counts=dict(Counter(r['classification'] for r in before.values())),
        unchanged=len(before)-len(changed),changed=changed,unexpected_changes=unexpected_changes,
        unexpected_new=unexpected_new,unexpected_ignored=unexpected_ignored,
        allowed_ignored=AUDIT+'*.log; no new cache/run directory',
        new_files=[r for r in added if r['path'] not in self_files],self_generated_unhashed=sorted(self_files),checks=checks)
    result['pass']=not (unexpected_changes or unexpected_new or unexpected_ignored) and all((
        checks['tracked_diff_unchanged'],checks['staged_diff_unchanged'],checks['diff_check']['exit_code']==0,
        result['head']==baseline['head']))
    write('end-freeze.json',result)
    print(json.dumps({k:result[k] for k in ('pass','baseline_existing','baseline_counts','unchanged',
                    'unexpected_changes','unexpected_new','unexpected_ignored')},indent=2))
    assert result['pass']


if __name__=='__main__':
    results()
    if '--results-only' not in sys.argv:freeze()
