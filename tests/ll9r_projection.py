"""Exact LL9R routing AST projection; never discard arbitrary conditionals."""
import ast
import copy
import json
from pathlib import Path


def strip_repair(tree):
    path = Path(__file__).resolve().parents[1]/'docs/loop_linearno_audit/ll9r/routing-replacements.json'
    changes = json.loads(path.read_text())
    pairs = []
    for rows in changes.values():
        for row in rows:
            import textwrap
            old = ast.parse(textwrap.dedent(row['before'])).body
            new = ast.parse(textwrap.dedent(row['after'])).body
            pairs.append((old, new))
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            for old, new in pairs:
                for i in range(len(node.body)-len(new)+1):
                    if [ast.dump(n) for n in node.body[i:i+len(new)]] == [ast.dump(n) for n in new]:
                        node.body[i:i+len(new)] = copy.deepcopy(old)
                        break
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
