"""Exact later AirfRANS LinearNO nodes; retain the complete old program."""
import ast
import copy

ADDITIONS = (
    """if args.model == 'LinearNO':
    from cdlno.linearno.air_entry import run_cli
    run_cli(args)
    raise SystemExit(0)""",
    'linearno_start_epoch = 0',
    """if linearno_run is not None:
    linearno_start_epoch, linearno_history = linearno_run.prepare(
        model, optimizer, lr_scheduler, train_dataset, val_dataset, criterion, reg, val_iter, val_sample)
    if linearno_history:
        (train_loss_surf_list, train_loss_vol_list, loss_surf_var_list, loss_vol_var_list,
         val_surf_list, val_vol_list, val_surf_var_list, val_vol_var_list) = linearno_history['curves']
        val_loss = linearno_history['val_loss']
        if val_surf_list:
            val_surf = val_surf_list[-1]""",
    """if linearno_run is not None:
    linearno_run.complete_epoch(epoch + 1, model, dict(curves=[
        train_loss_surf_list, train_loss_vol_list, loss_surf_var_list, loss_vol_var_list,
        val_surf_list, val_vol_list, val_surf_var_list, val_vol_var_list],
        val_loss=val_loss if val_iter is not None else None))""",
)
EXACT = {ast.dump(ast.parse(source).body[0]) for source in ADDITIONS}


def strip_air(tree):
    class Strip(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, ast.stmt) and ast.dump(node) in EXACT:
                return None
            return super().visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == "range(linearno_start_epoch, hparams['nb_epochs'])":
                return ast.copy_location(ast.parse("range(hparams['nb_epochs'])", mode='eval').body, node)
            return node

        def visit_BoolOp(self, node):
            # Preserve radius_graph body and all unrelated conditions.
            if ast.dump(node) == ast.dump(ast.parse("name_mod != 'PointNet' and name_mod != 'MLP' and linearno_run is None", mode='eval').body):
                return ast.copy_location(ast.parse("name_mod != 'PointNet' and name_mod != 'MLP'", mode='eval').body, node)
            return self.generic_visit(node)
    return ast.fix_missing_locations(Strip().visit(copy.deepcopy(tree)))
