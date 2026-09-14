"""Reject unsupported original Car drag-evaluation paths before dataset loading.

The frozen upstream cal_coefficient reads a fixed raw root and param0. This
check does not read data contents, rewrite paths, create links, or load models.
"""
import argparse
import ast
from pathlib import Path


def main():
    # Use only the real parser definitions, never import/execute the entry. This
    # also prevents argparse abbreviations (--fold, --nb_epochs=...) bypassing
    # the guard. Model-specific arguments are irrelevant to this path check.
    entry = Path(__file__).resolve().parents[1] / 'Car-Design-ShapeNetCar/main_evaluation.py'
    nodes = []
    for node in ast.parse(entry.read_text()).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'parser'):
            nodes.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and ast.unparse(node.value.func.value) == 'parser'
              and node.value.func.attr == 'add_argument'):
            nodes.append(node)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<car-eval-arguments-only>', 'exec'), scope)
    parser = scope['parser']
    parser.add_argument('--run_dir', type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.fold_id != 0:
        parser.error('Original drag evaluator hardcodes param0: full evaluation currently requires --fold_id 0. No metrics were changed.')
    original_root = Path('/data/PDE_data/mlcfd_data/training_data')
    if not (original_root / 'param0').is_dir():
        parser.error(f'Original drag evaluator requires {original_root}/param0. Make the same raw dataset accessible at that path before full evaluation; --data_dir alone does not redirect drag geometry.')
    if args.data_dir and Path(args.data_dir).resolve() != original_root.resolve():
        parser.error('--data_dir and the original hardcoded drag root must resolve to the same raw dataset; otherwise field/geometry evaluation can disagree.')
    for name in ('architecture.json', f'model_{args.nb_epochs}.pth'):
        if not (args.run_dir / name).is_file():
            parser.error(f'Missing existing checkpoint artifact: {args.run_dir / name}')
    print('Car evaluation path/fold check passed; model sidecar validation remains in CarRun.')


if __name__ == '__main__':
    main()
