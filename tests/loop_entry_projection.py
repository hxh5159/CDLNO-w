"""Remove only LL6 dispatcher guards for older complete-entry AST assertions.

Production resume projection is independently byte-checked against pre-LL6
snapshots; this helper preserves the preexisting history/pure AST test layers.
"""
import ast
import copy


def strip_loop(tree):
    from ll9r_projection import strip_repair
    tree = strip_repair(tree)
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module in ('cdlno.linearno_loop.standard_entry','cdlno.linearno_loop.industrial_entry') and [(a.name,a.asname) for a in node.names]==[('intercept','intercept_loop')]:return None
            return node
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='loop_args':
                assert ast.unparse(node.value) in (
                    'intercept_loop(parser, task, tokens, selected.model)',
                    "intercept_loop(parser, tokens, task='airfrans', evaluation=evaluation, selected_model=selected.model)",
                    "intercept_loop(parser, tokens, task='car', evaluation=evaluation, selected_model=selected.cfd_model)")
                return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=="hasattr(linearno_run, 'export_final')":
                assert ast.unparse(node.body[0])=='linearno_run.export_final(model)'
                return node.orelse
            if ast.unparse(node.test) in ('loop_args is not None',"hasattr(args, '_linearno_loop_config')",
                    "getattr(args, 'linearno_family', None) == 'linearno_loop'","self.family == 'linearno_loop'"):
                assert not node.orelse
                return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            if ast.unparse(node)=="('linearno', 'linearno_history', 'linearno_loop')":
                node.elts=node.elts[:2]
            return self.generic_visit(node)
        def visit_Dict(self,node):
            pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value=='linearno_loop')]
            node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
