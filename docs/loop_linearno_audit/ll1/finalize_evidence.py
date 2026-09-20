"""LL1 all-file freeze and evidence; no production writes/imports."""
from collections import Counter
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def inventory():
    result={}
    for category,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),
                          ('ignored',['--others','--ignored','--exclude-standard'])]:
        for raw in git('ls-files','-z',*args).split(b'\0'):
            if not raw:continue
            name=os.fsdecode(raw);p=ROOT/name
            if p.is_file():
                result[name]=dict(path=name,classification=category,size=p.stat().st_size,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return result


def allowed(name):
    return (name.startswith(('linearno_loop/','tests/loop_linearno/','docs/loop_linearno_audit/ll1/'))
            or name in ('docs/LOOP_LINEARNO_CONFIGURATION.md','docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md'))


if __name__=='__main__':
    env=dict(python=platform.python_version(),executable=os.sys.executable,
        platform=platform.platform(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='CPU only; no GPU probing; no data; no dependency changes',packages={})
    for name in ('torch','numpy','torch-geometric','timm','einops'):
        try:env['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:env['packages'][name]='not installed'
    write('environment.json',env)
    baseline=json.loads((OUT/'start-manifest.json').read_text())
    before={v['path']:v for v in baseline['files']}
    checks={}
    for label,args in [('diff_check',['diff','--check']),('unstaged',['diff','--binary']),
                       ('staged',['diff','--cached','--binary'])]:
        proc=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        checks[label]=dict(exit_code=proc.returncode,output=proc.stdout+proc.stderr)
    (OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
    syntax=[]
    for path in sorted([*(ROOT/'linearno_loop').glob('*.py'),*(ROOT/'tests/loop_linearno').glob('*.py'),*OUT.glob('*.py')]):
        source=path.read_text();ast.parse(source,feature_version=(3,10));compile(source,str(path),'exec')
        syntax.append(str(path.relative_to(ROOT)))
    reviews={
        'old_behavior':'source/classification freeze + 33 unchanged legacy test methods',
        'sole_architecture_truth':'request reconstruction; PCRS/depth/rank conflict and tamper tests',
        'point_AR_not_old_history':'contract/count assertions and any A/K field presence rejection',
        'metadata_first':'strict flag, source/shape hashes, field diagnostics; no torch load/construct',
        'claim_limits':'312 config-only previews; real/model/backend restoration not run',
    }
    write('delivery-review.json',dict(scope='LL1',status='PASS',reviewed=reviews,
        python310_ast_and_compile_pass=syntax,
        tests=dict(new=dict(count=24,pass_count=24,seconds=3.490,log='schema-tests.log'),
                   legacy=dict(count=33,pass_count=33,seconds=23.008,log='legacy-regression.log')),
        no_real_data=True,no_loop_model=True,no_production_routing=True,
        old_LL0_historical_failures='2 methods / 4 old documentation freeze assertions; unchanged, not rerun here'))
    current=inventory()
    changed=[dict(path=p,before=v,after=current.get(p)) for p,v in before.items()
             if p not in current or any(v[k]!=current[p][k] for k in ('classification','size','sha256'))]
    unauthorized=[r for r in changed if r['path']!='docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md']
    added=[v for p,v in current.items() if p not in before]
    self_paths={'docs/loop_linearno_audit/ll1/end-freeze.json','docs/loop_linearno_audit/ll1/end-status.txt'}
    # Full LF diff for new executable schema/test sources; generated JSON has its
    # exact content hash in the manifest and is not repeated in a 4 MB patch.
    import difflib
    delta=[]
    for row in added:
        name=row['path']
        if name.startswith(('linearno_loop/','tests/loop_linearno/')) and name.endswith('.py'):
            delta.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),
                fromfile='/dev/null',tofile=name))
    (OUT/'new-source.diff').write_text(''.join(delta))
    current=inventory();added=[v for p,v in current.items() if p not in before]
    unexpected=[v for v in added if not allowed(v['path'])]
    bad_ignored=[v for v in added if v['classification']=='ignored' and not
                 (v['path'].startswith('docs/loop_linearno_audit/ll1/') and v['path'].endswith('.log'))]
    result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repo=str(ROOT),head=git('rev-parse','HEAD').decode().strip(),
        tree=git('rev-parse','HEAD^{tree}').decode().strip(),branch=git('branch','--show-current').decode().strip(),
        remote=git('remote','-v').decode().strip(),baseline_existing=len(before),
        baseline_counts=dict(Counter(v['classification'] for v in before.values())),
        existing_unchanged=len(before)-len(changed),existing_changed=changed,
        unauthorized_existing_changes=unauthorized,unexpected_new=unexpected,
        allowed_ignored_additions='docs/loop_linearno_audit/ll1/*.log only; no new cache/run dirs',
        unexpected_ignored=bad_ignored,new_files=[v for v in added if v['path'] not in self_paths],
        self_generated_unhashed=sorted(self_paths),checks=checks,
        tracked_diff_empty=not checks['unstaged']['output'] and not checks['staged']['output'])
    result['pass']=(not unauthorized and not unexpected and not bad_ignored
                    and result['tracked_diff_empty'] and checks['diff_check']['exit_code']==0
                    and result['head']==baseline['head'])
    write('end-freeze.json',result)
    print(json.dumps({k:result[k] for k in ('pass','baseline_existing','baseline_counts',
        'existing_unchanged','unauthorized_existing_changes','unexpected_new','unexpected_ignored','tracked_diff_empty')},indent=2))
    assert result['pass']
