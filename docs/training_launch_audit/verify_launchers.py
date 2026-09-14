"""Data-free launch audit: shell dry-run plus only real argument-parser AST nodes."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
records = []
PARSER = r'''
import argparse, ast, json, sys
from pathlib import Path
entry, *tokens = sys.argv[1:]
nodes = []
for n in ast.parse(Path(entry).read_text()).body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'parser':
        nodes.append(n)
    elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and ast.unparse(n.value.func.value) == 'parser' and n.value.func.attr == 'add_argument':
        nodes.append(n)
scope = {'argparse': argparse}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<argument-definitions-only>', 'exec'), scope)
if entry == 'exp_ns.py':
    from cdlno_entry import parse_args
    args = parse_args(scope['parser'], 'ns', tokens)
else:
    from models.cdlno_run import parse_args
    args = parse_args(scope['parser'], evaluation=entry == 'main_evaluation.py', argv=tokens)
print(json.dumps(vars(args), default=str))
'''


def call(argv, cwd, env=None, success=True):
    result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=45)
    if success and result.returncode != 0:
        raise AssertionError(result.stderr + result.stdout)
    if not success:
        assert result.returncode != 0, result.stdout
    return result


def check_launch(task, mode, cwd, overrides):
    project = ROOT / ('Car-Design-ShapeNetCar' if task == 'car' else 'PDE-Solving-StandardBenchmark')
    env = {**os.environ, 'CDLNO_PYTHON': sys.executable, 'PYTHONPATH': str(ROOT)}
    result = call(['bash', str(ROOT/'tran_evaluate'/f'{task}.sh'), mode, *overrides, '--dry-run'], cwd, env)
    command = next(line.removeprefix('Command:') for line in result.stdout.splitlines() if line.startswith('Command:'))
    parts = shlex.split(command)
    assert parts[:2] == [sys.executable, '-u']
    expected_entry = ('main.py' if mode == 'train' else 'main_evaluation.py') if task == 'car' else 'exp_ns.py'
    assert parts[2] == expected_entry
    assert str(project) in result.stdout
    args = json.loads(call([sys.executable, '-B', '-c', PARSER, *parts[2:]], project, env).stdout)
    assert (args['n_hidden'], args['n_layers'], args['n_heads'], args['slice_num'], args['front_blocks']) == (256, 8, 8, 64, 2)
    assert args['mlp_ratio'] == args['latent_ffn_ratio'] == 2
    assert args['cdpa_mode'] == 'entry'
    if task == 'ns':
        assert args['max_grad_norm'] is None
        assert (args['epochs'], args['batch_size'], args['lr'], args['weight_decay'], args['downsample']) == (500, 2, .001, 1e-5, 1)
        assert args['eval'] == int(mode == 'eval')
        assert args['data_path'] == str(cwd/'data with spaces')
        assert args['cdlno_run_dir'] == str(cwd/'run with spaces')
    else:
        assert (args['nb_epochs'], args['fold_id'], args['weight'], args['r']) == (200, 0, .5, .2)
        if mode == 'train':
            assert (args['batch_size'], args['lr'], args['preprocessed'], args['val_iter']) == (1, .001, 1, 10)
        assert args['data_dir'] == str(cwd/'data with spaces')
        assert args['save_dir'] == str(cwd/'preprocessed with spaces')
        assert args['run_dir'] == str(cwd/'run with spaces')
    # Re-parse a real last-option override: does not construct or execute a model.
    chunk_flag = '--cdpa_source_chunk_size' if task == 'car' else '--cdpa-source-chunk-size'
    override = json.loads(call([sys.executable, '-B', '-c', PARSER, *parts[2:], chunk_flag+'=2', '--gpu', '3'], project, env).stdout)
    assert override['cdpa_source_chunk_size'] == 2 and str(override['gpu']) == '3'
    records.append(dict(task=task, mode=mode, command=parts, resolved_arguments=args,
                        override_chunk=2, override_gpu=3, status='passed'))


for script in sorted((ROOT/'tran_evaluate').glob('*.sh')):
    call(['bash', '-n', str(script)], ROOT)
ast.parse((ROOT/'tran_evaluate/check_car_eval.py').read_text(), feature_version=(3, 10))
with tempfile.TemporaryDirectory(prefix='cdlno launch audit ') as tmp:
    cwd = Path(tmp)
    for task in ('car', 'ns'):
        help_text = call(['bash', str(ROOT/'tran_evaluate'/f'{task}.sh'), 'help'], cwd).stdout
        assert '--dry-run' in help_text
        for mode in ('train', 'eval'):
            flags = (['--data_dir', 'data with spaces', '--save_dir=preprocessed with spaces', '--run_dir', 'run with spaces']
                     if task == 'car' else ['--data_path=data with spaces', '--cdlno-run-dir', 'run with spaces'])
            check_launch(task, mode, cwd, flags)
    assert list(cwd.iterdir()) == [], 'Dry-run must not create data or run directories'
    for flags in (['--data_path'], ['--data_path='], ['--data_path', '--dry-run']):
        call(['bash', str(ROOT/'tran_evaluate/ns.sh'), 'train', *flags, '--dry-run'], cwd, success=False)
    guard = call([sys.executable, '-B', str(ROOT/'tran_evaluate/check_car_eval.py'), '--fold_id', '1', '--run_dir', str(cwd/'missing')], cwd, success=False)
    assert 'hardcodes param0' in guard.stderr
    abbreviated = call([sys.executable, '-B', str(ROOT/'tran_evaluate/check_car_eval.py'), '--fold=1', '--run_dir', str(cwd/'missing')], cwd, success=False)
    assert 'hardcodes param0' in abbreviated.stderr
    missing = call([sys.executable, '-B', str(ROOT/'tran_evaluate/check_car_eval.py')], cwd, success=False)
    assert '--run_dir' in missing.stderr
    assert list(cwd.iterdir()) == []

payload = dict(status='passed', launches=records, shell_syntax='passed', help='passed',
               path_spaces_equals_and_relative='passed', dry_run_no_writes='passed',
               missing_path_rejected='passed', car_guard_fold_and_missing_run='passed',
               car_guard_with_real_dataset='not run', real_entry_execution='not run')
(EVIDENCE/'launcher-checks.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
print('PASS: four dry-run commands and actual AST-only parsers; overrides, path quoting, no writes, Car negative guards.')
