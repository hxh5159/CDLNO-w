"""LL10 read-only compile/shell/diff and pinned-reference revalidation."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    inventory = subprocess.check_output(
        ['rg', '--files', '--hidden', '-g', '*.py', '-g', '*.sh', '-g', '!**/.git/**'],
        cwd=ROOT, text=True).splitlines()
    roots = ('cdlno/', 'linearno_loop/', 'linearno_history/', 'tests/', 'tools/',
             'monitor/', 'LINEARNO/', 'PDE-Solving-StandardBenchmark/',
             'Airfoil-Design-AirfRANS/', 'Car-Design-ShapeNetCar/', 'tran_evaluate/')
    py = sorted(p for p in inventory if p.endswith('.py') and
                (p.startswith(roots) or p.startswith('docs/loop_linearno_audit/ll10/')))
    shells = sorted(p for p in inventory if p.endswith('.sh'))
    results = []
    for name in py:
        then = time.monotonic()
        try:
            compile((ROOT/name).read_bytes(), name, 'exec')
            error = None
        except Exception as exc:
            error = repr(exc)
        results.append(dict(path=name, check='builtin compile (no import/pyc)',
                            seconds=time.monotonic()-then, error=error, sha256=sha(ROOT/name)))
    for name in shells:
        then = time.monotonic()
        proc = subprocess.run(['bash', '-n', name], cwd=ROOT, capture_output=True, text=True)
        results.append(dict(path=name, check='bash -n', seconds=time.monotonic()-then,
                            error=proc.stderr if proc.returncode else None, sha256=sha(ROOT/name)))
    proc = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    report = dict(python=sys.version, python_files=len(py), shell_files=len(shells),
                  commands=['rg --files --hidden -g *.py -g *.sh -g !**/.git/**',
                            'compile(bytes, path, exec)', 'bash -n FILE', 'git diff --check'],
                  files=results, diff_exit=proc.returncode, diff_output=proc.stdout+proc.stderr,
                  seconds=round(time.monotonic()-start, 3))
    report['status'] = 'PASS' if not proc.returncode and not any(r['error'] for r in results) else 'FAIL'
    (OUT/'static-checks.json').write_text(json.dumps(report, indent=2)+'\n')

    ledger=json.loads((OUT.parent/'ll0/reference-ledger.json').read_text())
    verified=[]
    def visit(node, group):
        if isinstance(node, dict):
            if 'path' in node and 'sha256' in node:
                path=Path(node['path'])
                if not path.is_absolute(): path=ROOT/path
                verified.append(dict(group=group, path=str(path), expected=node['sha256'],
                    actual=sha(path) if path.is_file() else None))
            for key,value in node.items():
                if isinstance(value,(dict,list)): visit(value,group)
        elif isinstance(node,list):
            for value in node: visit(value,group)
    for group,node in ledger.items():visit(node,group)
    references=dict(checked=len(verified), files=verified,
                    status='PASS' if all(r['actual']==r['expected'] for r in verified) else 'FAIL')
    (OUT/'reference-revalidation.json').write_text(json.dumps(references,indent=2)+'\n')

    symbols=[]
    mapping=['cdlno/linearno_loop/core.py','cdlno/linearno_loop/body.py',
             'cdlno/linearno_loop/attnres.py','cdlno/linearno_loop/construction.py',
             'PDE-Solving-StandardBenchmark/model/LinearNO.py',
             'cdlno/linearno/airfrans.py','cdlno/linearno/shapenet.py']
    mapping += [str(p.relative_to(ROOT)) for p in (ROOT/'cdlno/linearno_loop').glob('*.py')]
    mapping += [str(p.relative_to(ROOT)) for p in (ROOT/'linearno_loop').glob('*.py')]
    def walk(body,parents,path):
        for node in body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                name='.'.join([*parents,node.name])
                symbols.append(dict(path=path,symbol=name,line=node.lineno,end_line=node.end_lineno))
                walk(node.body,[*parents,node.name],path)
    for name in sorted(set(mapping)):walk(ast.parse((ROOT/name).read_text()).body,[],name)
    (OUT/'source-symbols.json').write_text(json.dumps(symbols,indent=2)+'\n')
    print(json.dumps(dict(static=report['status'],python_files=len(py),shell_files=len(shells),
                          seconds=report['seconds'],references=references['status'],reference_files=len(verified))))
    raise SystemExit(report['status']!='PASS' or references['status']!='PASS')


if __name__=='__main__':main()
