"""Stage-local source snapshots and diffs; never edits production files."""
import argparse, difflib, hashlib, json, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def paths():
    items=set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0'))-{''}
    items.update(str(p.relative_to(ROOT)) for p in (ROOT/'docs/kcdno_audit').rglob('*') if p.is_file() and '__pycache__' not in str(p))
    return sorted(items)
def snapshot(stage):
    out=ROOT/'docs/kcdno_audit'/stage;out.mkdir(exist_ok=True)
    if (out/'before.json').exists():raise FileExistsError(out/'before.json')
    dest=Path(tempfile.mkdtemp(prefix=stage+'-before-',dir='/home/hwz/CDLNO-artifacts'))/'source';hashes={}
    for name in paths():
        p=ROOT/name
        if p.is_file():
            q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'before.json').write_text(json.dumps(dict(source_snapshot=str(dest),hashes=hashes,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')
    print(dest,len(hashes))
def diff(stage):
    out=ROOT/'docs/kcdno_audit'/stage;before=json.loads((out/'before.json').read_text());snap=Path(before['source_snapshot']);chunks=[];changes=[]
    for name in paths():
        if name.startswith('docs/kcdno_audit/') or name.startswith('PLAN_'):continue
        p=ROOT/name
        if not p.is_file():continue
        try:new=p.read_text();old=(snap/name).read_text() if (snap/name).exists() else ''
        except UnicodeDecodeError:continue
        if new==old:continue
        changes.append(name)
        chunks.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
    (out/'changes.patch').write_text(''.join(chunks));(out/'changed-files.json').write_text(json.dumps(changes,indent=2)+'\n')
    print(stage,len(changes),'reviewable files')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('snapshot','diff'));p.add_argument('stage');a=p.parse_args()
    {'snapshot':snapshot,'diff':diff}[a.action](a.stage)
