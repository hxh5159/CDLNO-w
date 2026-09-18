"""L4 exact projection of explicitly guarded additions to the pre-L4 source.

Used only by legacy source tests. New-branch math/protocol is tested separately;
no AST model/loss/data subtree is ignored in the remaining legacy program.
"""
import ast
import copy

TESTS = {"args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh')",
         "selected.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh')",
         "getattr(args, 'linearno_family', None) == 'linearno'", "self.family == 'linearno'"}
TESTS |= {"args.cfd_model == 'LinearNO'", "selected.cfd_model == 'LinearNO'", "linearno_run is not None"}


def strip_linearno(tree):
    from history_entry_projection import strip_history
    tree = strip_history(tree)
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.args.args and node.args.args[-1].arg == 'linearno_run' and node.name == 'main':
                assert isinstance(node.args.defaults[-1],ast.Constant) and node.args.defaults[-1].value is None
                node.args.args.pop(); node.args.defaults.pop()
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0]) == 'self.resuming':
                assert ast.unparse(node.value) == "self.family == 'linearno' and getattr(args, 'resume', False)"
                return None
            return self.generic_visit(node)
        def visit_Attribute(self,node):
            if ast.unparse(node) == 'self.resuming': return ast.Constant(False)
            return self.generic_visit(node)
        def visit_UnaryOp(self,node):
            relevant = 'self.resuming' in ast.unparse(node)
            node=self.generic_visit(node)
            if relevant and isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant):
                return ast.Constant(not node.operand.value)
            return node
        def visit_BoolOp(self,node):
            relevant = 'self.resuming' in ast.unparse(node)
            node=self.generic_visit(node)
            if not relevant: return node
            neutral=isinstance(node.op,ast.And)
            values=[]
            for value in node.values:
                if isinstance(value,ast.Constant):
                    if bool(value.value)!=neutral:return value
                else:values.append(value)
            return values[0] if len(values)==1 else ast.BoolOp(op=node.op,values=values)
        def visit_If(self,node):
            if ast.unparse(node.test) in TESTS:
                return self._body(node.orelse)
            relevant = 'self.resuming' in ast.unparse(node.test)
            node=self.generic_visit(node)
            if relevant and isinstance(node.test,ast.Constant):return node.body if node.test.value else node.orelse
            return node
        def _body(self,body):
            result=[]
            for item in body:
                value=self.visit(item)
                if isinstance(value,list):result.extend(value)
                elif value is not None:result.append(value)
            return result
        def visit_IfExp(self,node):
            relevant = 'self.resuming' in ast.unparse(node.test)
            node=self.generic_visit(node)
            return (node.body if node.test.value else node.orelse) if relevant and isinstance(node.test,ast.Constant) else node
        def visit_Dict(self,node):
            if any(isinstance(k,ast.Constant) and k.value=='linearno' for k in node.keys) and any(isinstance(k,ast.Constant) and k.value=='msar_lno' for k in node.keys):
                pairs=[(k,v) for k,v in zip(node.keys,node.values) if not(isinstance(k,ast.Constant) and k.value=='linearno')]
                node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
