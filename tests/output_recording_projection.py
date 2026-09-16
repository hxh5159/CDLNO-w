"""Remove only the approved observational additions for exact frozen AST checks."""
import ast


class WithoutRecording(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module in ('cdlno.experiment', 'kcdno_entry'):
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            function = ast.unparse(node.value.func)
            if function in ('start_experiment', 'finish_experiment') or function in {
                'cdlno_run.recorder.attach_model', 'cdlno_run.recorder.update_protocol',
                'cdlno_run.recorder.record_training_setup',
                'cdlno_run.recorder.record_epoch', 'cdlno_run.recorder.record_metrics',
                'cdlno_run.recorder.record_air_scores', 'cdlno_run.recorder.visualize'}:
                return None
        return self.generic_visit(node)

    def visit_With(self, node):
        if any(ast.unparse(item.optional_vars) == 'config_file'
               for item in node.items if item.optional_vars is not None):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if ast.unparse(node.targets[0]) == 'cdlno_run.recorder':
            return None
        if ast.unparse(node.targets[0]) in ('score_path', 'score_array_path'):
            if ast.unparse(node.value) in ('cdlno_run.recorder.result_dir', 'score_path'):
                return None
        return self.generic_visit(node)

    def visit_If(self, node):
        # Remove only the independent new-family selection; its own complete
        # projection to the pre-K4 entries is separately tested.
        if ast.unparse(node.test) in ("args.model == 'kcdno'", "args.model in ('kcdno', 'lrsa_matched')", "args.cfd_model == 'kcdno'", "args.cfd_model in ('kcdno', 'lrsa_matched')", "model == 'kcdno'", "model in ('kcdno', 'lrsa_matched')"):
            result = []
            for item in node.orelse:
                value = self.visit(item)
                if isinstance(value, list):
                    result.extend(value)
                elif value is not None:
                    result.append(value)
            return result
        if ast.unparse(node.test) == 'record is not None':
            return None
        node = self.generic_visit(node)
        return node if node.body else None

    def visit_Call(self, node):
        # Only remove the conditional observer kwargs from the two train calls.
        if ast.unparse(node.func) == 'train.main':
            node.keywords = [kw for kw in node.keywords if not (
                kw.arg is None and isinstance(kw.value, ast.IfExp)
                and ast.unparse(kw.value.test) == 'cdlno_run is not None'
                and isinstance(kw.value.body, ast.Call)
                and ast.unparse(kw.value.body.func) == 'dict'
                and {k.arg for k in kw.value.body.keywords} <= {'record', 'record_member', 'visualization_norm'})]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == 'main':
            while node.args.args and node.args.args[-1].arg in ('record', 'record_member', 'visualization_norm'):
                node.args.args.pop()
                node.args.defaults.pop()
        return self.generic_visit(node)


def strip_recording(tree):
    from msar_entry_projection import strip_msar
    return WithoutRecording().visit(strip_msar(tree))
