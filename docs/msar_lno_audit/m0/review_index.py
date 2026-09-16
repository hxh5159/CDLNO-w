"""Read every inventoried text file; preserve contracts without importing entries.

This is an audit index, not a claim that parsing proves scientific correctness.
The accompanying audit records the semantic review and executable evidence.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    inventory = json.loads((OUT / 'inventory.json').read_text())
    records, contracts, documents, tests, configs, shells = [], {}, {}, {}, {}, {}
    errors = []
    for row in inventory['records']:
        path = ROOT / row['path']
        if not row.get('text_read', False):
            continue
        text = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(text.encode()).hexdigest()
        record = dict(path=row['path'], sha256=digest, tracked=row['tracked'],
                      lines=len(text.splitlines()), kind=row['kind'])
        records.append(record)
        if path.suffix == '.py':
            tree = ast.parse(text)
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append(dict(name=node.name, line=node.lineno, end=node.end_lineno,
                                        ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest()))
            calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = ast.unparse(node.func)
                if any(word in fn for word in ('add_argument', 'save', 'load', 'backward', 'step',
                                               'loss', 'Loss', 'normaliz', 'Normaliz', 'model',
                                               'radius_graph', 'autocast', 'attention', 'mask')):
                    calls.append(dict(line=node.lineno, code=ast.unparse(node)))
            contracts[row['path']] = dict(symbols=symbols, calls=calls,
                                          module_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest())
            if row['path'].startswith('tests/'):
                tests[row['path']] = [dict(name=n.name, line=n.lineno,
                    checks=[ast.unparse(x) for x in ast.walk(n)
                            if isinstance(x, ast.Assert) or isinstance(x, ast.Call) and
                            any(w in ast.unparse(x.func) for w in ('assert', 'skip', 'subTest'))])
                    for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        elif path.suffix in ('.md', '.txt') and row['kind'] == 'documentation':
            # Preserve all prose, with headings and chronology. Generated logs are
            # kept separately by their original inventory hash rather than reprinted.
            documents[row['path']] = dict(headings=re.findall(r'^#{1,6} .+$', text, re.M),
                paragraphs=[p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()])
        if path.suffix in ('.json', '.yaml', '.toml') and ('configs/' in row['path'] or
                path.name in ('params.yaml', 'pyproject.toml')):
            configs[row['path']] = text
        if path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            shells[row['path']] = dict(returncode=result.returncode, error=result.stderr, source=text)
            if result.returncode:
                errors.append(row['path'])
    payloads = {'source-contracts.json': contracts, 'test-contracts.json': tests,
                'document-reading.json': documents, 'configs-shells.json': dict(configs=configs, shells=shells),
                'reading-index.json': dict(files=records, counts=dict(Counter(r['kind'] for r in records)),
                    python_files=len(contracts), test_methods=sum(map(len, tests.values())),
                    documents=len(documents), shell_scripts=len(shells), shell_syntax_failures=errors,
                    method='Full text read; every Python AST/function/call contract indexed; exact-hash duplicates only. '
                           'Semantic review is recorded separately, not inferred from parser success.')}
    for name, payload in payloads.items():
        with (OUT / name).open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps({k:v for k,v in payloads['reading-index.json'].items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
