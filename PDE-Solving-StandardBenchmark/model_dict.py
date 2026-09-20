from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D, Transolver_Structured_Mesh_3D


def get_model(args):
    if hasattr(args, '_linearno_loop_config'):
        from cdlno.linearno_loop.standard_entry import model_module
        return model_module(args)
    if hasattr(args, '_linearno_history_config'):
        from cdlno.linearno_history.standard_entry import model_module
        return model_module(args)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        from model import LinearNO
        return LinearNO
    if args.model == 'msar_lno':
        if getattr(args, 'msar_task', None) in ('ns', 'plasticity'):
            from model import MSAR_Temporal
            return MSAR_Temporal
        if getattr(args, 'msar_task', None) in ('darcy', 'elasticity', 'airfoil', 'pipe'):
            from model import MSAR_Standard
            return MSAR_Standard
        from model import MSAR_LNO
        return MSAR_LNO
    if args.model in ('kcdno', 'lrsa_matched'):
        from model import KCDNO
        return KCDNO
    if args.model == 'CDLNO':
        task = getattr(args, 'cdlno_task', None)
        if task == 'elasticity':
            from model import CDLNO_Irregular_Mesh
            return CDLNO_Irregular_Mesh
        if task in ('darcy', 'airfoil', 'pipe'):
            from model import CDLNO_Structured_Mesh_2D
            return CDLNO_Structured_Mesh_2D
        if task in ('ns', 'plasticity'):
            from model import CDLNO_Temporal_Structured_Mesh_2D
            return CDLNO_Temporal_Structured_Mesh_2D
        raise ValueError('CDLNO is currently registered only for the six standard tasks')
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
        'Transolver_Structured_Mesh_2D': Transolver_Structured_Mesh_2D,
        'Transolver_Structured_Mesh_3D': Transolver_Structured_Mesh_3D,
    }
    return model_dict[args.model]
