"""Strip only the new epoch visualization calls, not any training logic."""
import ast
import copy


def strip_visualization(tree):
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in (
                    'record.visualize', 'cdlno_run.recorder.visualize'):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'dict':
                node.keywords = [k for k in node.keywords if k.arg != 'visualization_norm']
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name == 'main' and node.args.args and node.args.args[-1].arg == 'visualization_norm':
                assert isinstance(node.args.defaults[-1], ast.Constant) and node.args.defaults[-1].value is None
                node.args.args.pop(); node.args.defaults.pop()
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
