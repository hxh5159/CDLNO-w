"""Recheck LL0 file content AND git classifications; no production mutations."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
base=json.loads((OUT/'start-manifest.json').read_text());before={r['path']:r for r in base['files']}
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
current={}
for category,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),('ignored',['--others','--ignored','--exclude-standard'])]:
 for raw in git('ls-files','-z',*args).split(b'\0'):
  if not raw:continue
  name=os.fsdecode(raw);p=ROOT/name
  if p.is_file():current[name]=dict(path=name,classification=category,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
changed=[dict(path=p,before=v,after=current.get(p)) for p,v in before.items() if p not in current or any(v[k]!=current[p][k] for k in ('classification','size','sha256'))]
def allowed(p):return p in ('docs/LOOP_LINEARNO_REFERENCE_AUDIT.md','docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md') or p.startswith('docs/loop_linearno_audit/ll0/')
new=[v for p,v in current.items() if p not in before]
checks={}
for label,args in [('diff_check',['diff','--check']),('unstaged',['diff','--binary']),('staged',['diff','--cached','--binary'])]:
 p=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True);checks[label]=dict(exit_code=p.returncode,output=p.stdout+p.stderr)
(OUT/'end-status.txt').write_bytes(git('status','--short','--branch','--untracked-files=all'))
self_files=('docs/loop_linearno_audit/ll0/end-freeze.json','docs/loop_linearno_audit/ll0/end-status.txt')
result=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=git('rev-parse','HEAD').decode().strip(),tree=git('rev-parse','HEAD^{tree}').decode().strip(),baseline_existing=len(before),baseline_counts=base['counts'],existing_changed_or_missing=changed,unexpected_new=[v for v in new if not allowed(v['path'])],new_files=[v for v in new if v['path'] not in self_files],self_generated_unhashed=list(self_files),checks=checks)
result['all_existing_unchanged']=not changed;result['only_allowed_additions']=not result['unexpected_new'];result['tracked_diff_empty']=not checks['unstaged']['output'] and not checks['staged']['output']
(OUT/'end-freeze.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('baseline_existing','baseline_counts','all_existing_unchanged','only_allowed_additions','tracked_diff_empty','existing_changed_or_missing','unexpected_new')},ensure_ascii=False,indent=2))
assert result['all_existing_unchanged'] and result['only_allowed_additions'] and result['tracked_diff_empty'] and checks['diff_check']['exit_code']==0
