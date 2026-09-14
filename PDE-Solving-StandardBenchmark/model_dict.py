from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D, Transolver_Structured_Mesh_3D


def get_model(args):
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
