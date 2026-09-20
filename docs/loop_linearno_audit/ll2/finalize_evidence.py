"""LL2 evidence and full tracked/untracked/ignored freeze, no production imports."""
from collections import Counter
from pathlib import Path
import ast
import datetime
import difflib
import hashlib
import json
import os
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ALLOWED_EXISTING={'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md','tests/loop_linearno/test_artifacts.py'}
ALLOWED_NEW=('cdlno/linearno_loop/','docs/loop_linearno_audit/ll2/')
NEW_FILES={'tests/loop_linearno/point_attnres_oracle.py','tests/loop_linearno/test_point_attnres.py',
           'tests/loop_linearno/test_block_body.py','docs/LOOP_LINEARNO_PRIMITIVES.md'}


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


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


if __name__=='__main__':
    old=json.loads((OUT.parent/'ll0/regression-results.json').read_text())
    current=json.loads((OUT/'regression-results.json').read_text())
    assert len(old)==len(current)==24
    for a,b in zip(old,current):
        assert a['module']==b['module']
        for key in ('exit_code','tests','failures','errors','skipped'):
            assert a.get(key)==b.get(key),(b['module'],key)
    # Compare the actual failed assertions too, not only their test IDs.
    failure_details={}
    for name in ('linearno.test_legacy','linearno.test_static_integration'):
        extract=lambda text:[s for s in text.splitlines() if s.startswith(('FAIL:','AssertionError:'))]
        before=extract((OUT.parent/'ll0'/f'{name}.log').read_text())
        after=extract((OUT/f'{name}.log').read_text())
        assert before==after,(name,before,after)
        failure_details[name]=after
    summary=json.loads((OUT.parent/'ll0/regression-summary.json').read_text())
    summary.update(status='LL2 rerun matches approved LL0 outcomes; no new failure',
                   same_as_LL0=True,historical_assertions=failure_details)
    write('regression-summary.json',summary)

    loop_log=(OUT/'loop-tests.log').read_text()
    assert loop_log.rstrip().endswith('OK')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',loop_log)
    assert match and int(match[1])==41
    numeric={}
    for label,name in [('attnres','attnres-report.json'),('body','body-report.json')]:
        groups={}
        for row in json.loads((OUT/name).read_text())['rows']:
            k=row['device']+'/'+row['dtype']
            g=groups.setdefault(k,dict(cases=0,max_abs=0.,max_mean_abs=0.,max_relative=0.,max_mean_relative=0.))
            g['cases']+=1
            for e in row['errors'].values():
                for outkey,inkey in [('max_abs','max_abs'),('max_mean_abs','mean_abs'),
                                     ('max_relative','max_relative'),('max_mean_relative','mean_relative')]:
                    g[outkey]=max(g[outkey],e[inkey])
        numeric[label]=groups
    write('numeric-summary.json',numeric)
    syntax=[]
    paths=[*(ROOT/'cdlno/linearno_loop').glob('*.py'),
           *(ROOT/'tests/loop_linearno').glob('*.py'),*OUT.glob('*.py')]
    for path in sorted(paths):
        source=path.read_text();ast.parse(source,feature_version=(3,10));compile(source,str(path),'exec')
        assert all(line.rstrip()==line for line in source.splitlines()),str(path)
        syntax.append(str(path.relative_to(ROOT)))
    reviews=dict(oracle='Independent torch equations, scalar hand check, FP64 gradcheck; no production import',
        axes='S-only softmax, B/N/channel/source permutations and isolated perturbation; no matmul in AR trace',
        parameters='2H only, receiver independence, zero query/one scale; initialization consumes no RNG',
        body='60 same-weight cases, raw/native/scaled, identity unscaled, explicit terminal head, no state registration',
        compatibility='LL1 24 checks + exact LL0 test-outcome comparison + full file content/classification freeze')
    write('delivery-review.json',dict(stage='LL2',status='PASS',reviewed=reviews,
        tests=dict(loop_suite_count=41,loop_suite_passed=41,seconds=float(match[2]),
                   old_LL0=summary['total']),syntax_python310=syntax,
        known_test_correction='source permutation had an initial bitwise equality assertion; observed FP64 1.11e-16; corrected to the already declared FP64 tolerance, no math change',
        not_run=['full loop model','task integration','full checkpoint/resume for loop family',
                 'real data/training','remote environment','full CUDA AMP model support']))

    baseline=json.loads((OUT/'start-manifest.json').read_text());before={r['path']:r for r in baseline['files']}
    checks={}
    for label,args in [('diff_check',['diff','--check']),('unstaged',['diff','--binary']),('staged',['diff','--cached','--binary'])]:
        p=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        checks[label]=dict(exit_code=p.returncode,output=p.stdout+p.stderr)
    (OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
    now=inventory()
    changed=[dict(path=p,before=r,after=now.get(p)) for p,r in before.items()
        if p not in now or any(r[k]!=now[p][k] for k in ('classification','size','sha256'))]
    unexpected_changes=[r for r in changed if r['path'] not in ALLOWED_EXISTING]
    added=[v for p,v in now.items() if p not in before]
    # Explicit new-code diff; old code unchanged. The one prior test edit is
    # captured against its preserved LL1 source patch.
    patch=[]
    for row in added:
        name=row['path']
        if name.startswith(('cdlno/linearno_loop/','tests/loop_linearno/')) and name.endswith('.py'):
            patch.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    old_patch=(OUT.parent/'ll1/new-source.diff').read_text()
    start=old_patch.index('+++ tests/loop_linearno/test_artifacts.py\n')
    part=old_patch[start:].split('\n--- ',1)[0]
    prior=''.join(line[1:]+'\n' for line in part.splitlines()[1:] if line.startswith('+'))
    assert hashlib.sha256(prior.encode()).hexdigest()==before['tests/loop_linearno/test_artifacts.py']['sha256']
    patch.extend(difflib.unified_diff(prior.splitlines(True),(ROOT/'tests/loop_linearno/test_artifacts.py').read_text().splitlines(True),
        fromfile='LL1/tests/loop_linearno/test_artifacts.py',tofile='tests/loop_linearno/test_artifacts.py'))
    (OUT/'source.diff').write_text(''.join(patch))
    now=inventory();added=[v for p,v in now.items() if p not in before]
    unexpected_new=[v for v in added if not (v['path'].startswith(ALLOWED_NEW) or v['path'] in NEW_FILES)]
    unexpected_ignored=[v for v in added if v['classification']=='ignored' and not
        (v['path'].startswith('docs/loop_linearno_audit/ll2/') and v['path'].endswith('.log'))]
    self_files={'docs/loop_linearno_audit/ll2/end-freeze.json','docs/loop_linearno_audit/ll2/end-status.txt'}
    result=dict(repo=str(ROOT),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        head=git('rev-parse','HEAD').decode().strip(),tree=git('rev-parse','HEAD^{tree}').decode().strip(),
        baseline_existing=len(before),baseline_counts=dict(Counter(r['classification'] for r in before.values())),
        unchanged=len(before)-len(changed),changed=changed,unexpected_changes=unexpected_changes,
        unexpected_new=unexpected_new,unexpected_ignored=unexpected_ignored,
        allowed_ignored='docs/loop_linearno_audit/ll2/*.log; no new cache/run directory',
        new_files=[r for r in added if r['path'] not in self_files],self_generated_unhashed=sorted(self_files),checks=checks)
    result['pass']=not (unexpected_changes or unexpected_new or unexpected_ignored or checks['unstaged']['output'] or checks['staged']['output']) and checks['diff_check']['exit_code']==0 and result['head']==baseline['head']
    write('end-freeze.json',result)
    print(json.dumps({k:result[k] for k in ('pass','baseline_existing','baseline_counts','unchanged','unexpected_changes','unexpected_new','unexpected_ignored')},indent=2))
    assert result['pass']
