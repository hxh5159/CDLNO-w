"""Read actual parser + saved config, then reuse the existing Car path guard."""
import argparse
import ast
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[2]
    project=root/'Car-Design-ShapeNetCar'
    nodes=[n for n in ast.parse((project/'main_evaluation.py').read_text()).body if
           (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser') or
           (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument')]
    scope={'argparse':argparse}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<parser-only>','exec'),scope)
    sys.path.insert(0,str(project))
    from models.cdlno_run import parse_args
    args=parse_args(scope['parser'],evaluation=True)
    subprocess.run([sys.executable,'-B',str(root/'tran_evaluate/check_car_eval.py'),
                    '--data_dir',args.data_dir,'--run_dir',str(args.kcdno_run_dir),
                    '--fold_id',str(args.fold_id),'--nb_epochs',str(args.nb_epochs)],check=True)


if __name__=='__main__':main()
