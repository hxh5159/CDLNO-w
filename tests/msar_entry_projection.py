"""Only remove the explicit new-family branches, retaining their legacy else.

The complete projected entries are separately compared to the pre-M5 snapshot.
Factory mapping is not stripped: it has its own exact branch assertion.
"""
import ast
import copy


def strip_msar(tree):
    from linearno_entry_projection import strip_linearno
    tree = strip_linearno(tree)
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'train' and node.args.args[-1].arg == 'msar_metrics':
                assert ast.unparse(node.args.defaults[-1]) == 'None'
                node.args.args.pop(); node.args.defaults.pop()
            return node if node.name == 'get_model' else self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == 'msar_training':
                assert ast.unparse(node.value) == "getattr(getattr(model, 'config', None), 'family', None) == 'msar_lno'"
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) in ("args.model == 'msar_lno'", "args.cfd_model == 'msar_lno'",
                                          "model == 'msar_lno'", 'msar_training'):
                result=[]
                for item in node.orelse:
                    value=self.visit(item)
                    if isinstance(value,list):result.extend(value)
                    elif value is not None:result.append(value)
                return result
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
