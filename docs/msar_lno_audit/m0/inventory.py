"""Read-only source inventory/snapshot for the retroactive M0 audit.

Writes only audit evidence and a unique external snapshot; never imports entries.
Inventory is not itself a semantic review: review notes live in the audit report.
"""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def command(*args):
    return subprocess.check_output(args, cwd=ROOT)


def main():
    if (OUT / 'inventory.json').exists():
        raise FileExistsError('M0 inventory already captured; preserve it and use a new stage audit directory')
    tracked = set(command('git', 'ls-files', '-z').decode().split('\0')) - {''}
    visible = set(command('rg', '--files', '--hidden', '-g', '!.git/**', '-g', '!**/__pycache__/**',
                          '-g', '!**/.pytest_cache/**').decode().splitlines())
    visible -= {p for p in visible if p.startswith('docs/msar_lno_audit/m0/')}
    paths = sorted(tracked | visible)
    snapshot = Path(tempfile.mkdtemp(prefix='msar-m0-after-m1-', dir='/home/hwz/CDLNO-artifacts'))
    records = []
    groups = defaultdict(list)
    source_extensions = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}
    for name in paths:
        path = ROOT/name
        if not path.is_file():
            records.append(dict(path=name, tracked=name in tracked, excluded='missing or non-file'))
            continue
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        row = dict(path=name, tracked=name in tracked, bytes=len(data), sha256=sha)
        try:
            content = data.decode('utf-8')
        except UnicodeDecodeError:
            content = None
        if content is None or '\0' in content:
            row['excluded'] = 'binary (figure/PDF/other); not executable source'
        else:
            row['lines'] = len(content.splitlines())
            row['text_read'] = True
            row['kind'] = ('source' if path.suffix in {'.py','.sh'} else
                           'documentation' if path.suffix == '.md' else
                           'configuration_or_evidence' if path.suffix in source_extensions else 'other_text')
            groups[sha].append(name)
            if path.suffix == '.py':
                tree = ast.parse(content, filename=name)
                row['symbols'] = [dict(name=n.name, line=n.lineno, end=n.end_lineno,
                                      kind=type(n).__name__) for n in ast.walk(tree)
                                  if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
                row['imports'] = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
            target = snapshot/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        records.append(row)
    result=dict(branch=command('git','branch','--show-current').decode().strip(),
                commit=command('git','rev-parse','HEAD').decode().strip(),snapshot=str(snapshot),
                chronology='M0 performed after M1; pre-M1 references are separately indexed, never backdated',
                rg_visible_count=len(visible),tracked_count=len(tracked),records=records,
                exact_text_duplicate_groups=[v for v in groups.values() if len(v)>1],
                exclusion_policy=['.git internals','__pycache__/.pytest_cache','binary figures/PDFs',
                                  'no traversal of external data/checkpoints/output; only listed source tree'],
                reading_boundary='Full text ingested; semantic notes and grouping decisions must be reviewed separately.')
    (OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'rg-files.txt').write_text('\n'.join(sorted(visible))+'\n')
    (OUT/'tracked-files.txt').write_text('\n'.join(sorted(tracked))+'\n')
    (OUT/'status-before.txt').write_bytes(command('git','status','--short','--untracked-files=all'))
    (OUT/'preexisting.diff').write_bytes(command('git','diff','--binary'))
    (OUT/'staged-before.diff').write_bytes(command('git','diff','--cached','--binary'))
    print(json.dumps(dict(snapshot=str(snapshot),files=len(records),tracked=len(tracked),
        texts=sum('text_read' in r for r in records),excluded=sum('excluded' in r for r in records),
        exact_duplicate_groups=len(result['exact_text_duplicate_groups'])),indent=2))


if __name__ == '__main__':
    main()
