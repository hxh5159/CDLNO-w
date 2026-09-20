"""Remove only reviewed history dispatch when checking complete legacy ASTs."""
import ast
import copy


def strip_history(tree):
    from loop_entry_projection import strip_loop
    tree = strip_loop(tree)
    class Strip(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='DataLoader':
                node.keywords=[k for k in node.keywords if not (k.arg is None and isinstance(k.value,ast.IfExp) and ast.unparse(k.value.test)=="hasattr(linearno_run, 'loader_kwargs')")]
            return node
        def visit_ImportFrom(self,node):
            if node.module=='cdlno.linearno_history.standard_entry' and any(a.name=='intercept' for a in node.names):return None
            return node
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='history_args':
                assert ast.unparse(node.value)=='intercept_history(parser, task, tokens, selected.model)'
                return None
            return self.generic_visit(node)
        def visit_If(self,node):
            text=ast.unparse(node.test)
            if text in ("history_args is not None", "hasattr(args, '_linearno_history_config')"):
                return node.orelse
            return self.generic_visit(node)
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text in ("self.family in ('linearno', 'linearno_history')", "getattr(args, 'linearno_family', None) in ('linearno', 'linearno_history')"):
                return ast.Compare(left=node.left,ops=[ast.Eq()],comparators=[ast.Constant('linearno')])
            return self.generic_visit(node)
        def visit_Dict(self,node):
            pairs=[(k,v) for k,v in zip(node.keys,node.values) if not(isinstance(k,ast.Constant) and k.value=='linearno_history')]
            node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
