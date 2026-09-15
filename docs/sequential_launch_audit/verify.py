"""Validate sequential launch commands without data, task imports, or training."""
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ast.parse((ROOT/'docs/remote_launch_audit/verify_remote_launchers.py').read_text())
PARSER = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PARSER')
env = {k:v for k,v in os.environ.items() if not k.startswith('CDLNO_')}
env.update(CDLNO_PYTHON=sys.executable, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
launcher = ROOT/'tran_evaluate/train_eval.sh'
records = []


def run(command, cwd, expected=0, environment=None):
    result = subprocess.run(command, cwd=cwd, env=environment or env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == expected, (command, result.returncode, result.stdout, result.stderr)
    return result


with tempfile.TemporaryDirectory(prefix='cdlno sequential commands ') as tmp:
    cwd = Path(tmp)
    for task in ('darcy','elasticity','airfoil','pipe','ns','plasticity','car','airfrans'):
        for custom in (False, True):
            industrial = task in ('car','airfrans')
            extras = []
            if custom:
                extras = (['--run_dir','a run','--front_blocks','3','--cdpa_mode','off'] if industrial
                          else ['--cdlno-run-dir','a run','--front-blocks','3','--cdpa-mode','off'])
                if task == 'car':
                    extras += ['--train-args','--preprocessed','0','--lr','0.002']
                if task == 'airfrans':
                    extras += ['--train-args','--my_path','data dir/Dataset','--eval-args','--my_path','data dir']
            result = run(['bash',str(launcher),task,*extras,'--dry-run'], cwd)
            lines = result.stdout.splitlines()
            commands = [shlex.split(line[len('Command:'):]) for line in lines if line.startswith('Command:')]
            projects = [line[len('Working directory: '):] for line in lines if line.startswith('Working directory: ')]
            assert len(commands) == len(projects) == 2
            arguments = []
            for command, project in zip(commands, projects):
                assert command[:2] == [sys.executable,'-u']
                parsed = json.loads(run([sys.executable,'-B','-c',PARSER,task,*command[2:]],project).stdout)
                arguments.append(parsed)
            train, evaluation = arguments
            key = 'run_dir' if industrial else 'cdlno_run_dir'
            assert train[key] == evaluation[key]
            if custom:
                assert train[key] == str(cwd/'a run')
                for parsed in arguments:
                    assert parsed['front_blocks'] == 3 and parsed['cdpa_mode'] == 'off'
            else:
                preset_name = 'shapenet_car' if task == 'car' else task
                preset = json.loads((Path(projects[0])/'configs/CDLNO'/f'{preset_name}.json').read_text())
                for parsed in arguments:
                    assert all(parsed[k] == v for k,v in preset['model'].items())
                    if not industrial:
                        assert all(parsed[k] == v for k,v in preset['training'].items())
                    elif task == 'airfrans':
                        assert parsed['resolved_hparams'] == preset['initial_training']
            if not industrial:
                assert train['eval'] == 0 and evaluation['eval'] == 1
            elif task == 'car':
                assert (train['nb_epochs'],evaluation['nb_epochs']) == (200,200)
                assert 'preprocessed' in train and 'preprocessed' not in evaluation
                assert 'lr' not in evaluation
                if custom:
                    assert train['preprocessed'] == 0 and train['lr'] == .002
            else:
                assert Path(train['my_path']) == Path(evaluation['my_path'])/'Dataset'
            records.append(dict(task=task, custom=custom, commands=commands, arguments=arguments, status='passed'))
    assert list(cwd.iterdir()) == [], 'dry-runs created files'
    # Explicit tags persist across both subprocesses; default tags differ per pair.
    fixed = run(['bash',str(launcher),'ns','--dry-run'],cwd,environment={**env,'CDLNO_RUN_TAG':'fixed_tag'})
    assert fixed.stdout.count('/ns/fixed_tag') == 2
    auto = [run(['bash',str(launcher),'ns','--dry-run'],cwd).stdout.splitlines()[0] for _ in range(2)]
    assert auto[0] != auto[1]
    for tokens in (['unknown'], ['ns','--eval','1'], ['ns','--eval-args','--cdlno-run-dir','wrong'],
                   ['car','--train-args','--run_dir=wrong'], ['airfrans','--my_path','wrong']):
        result = run(['bash',str(launcher),*tokens,'--dry-run'],cwd,expected=2)
        assert 'Command:' not in result.stdout

    # Shell-only stand-ins exercise exit status/order; no Python/model/data stubs.
    harness = cwd/'shell harness'
    harness.mkdir()
    shutil.copy2(launcher,harness/'train_eval.sh')
    log = harness/'order.txt'
    (harness/'darcy.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|' "$1" "$CDLNO_RUN_TAG" >> "$ORDER_LOG"
printf ' %q' "$@" >> "$ORDER_LOG"
printf '\\n' >> "$ORDER_LOG"
if [[ "$1" == train ]]; then exit "${TRAIN_EXIT:-0}"; fi
exit "${EVAL_EXIT:-0}"
''')
    shell_env = {**env,'ORDER_LOG':str(log)}
    args = ['bash',str(harness/'train_eval.sh'),'darcy','--shared','with spaces',
            '--train-args','--train-option','one','--eval-args','--eval-option','two']
    run(args,cwd,environment=shell_env)
    first, second = log.read_text().splitlines()
    assert first.startswith('train|') and second.startswith('eval|')
    assert first.split('|')[1] == second.split('|')[1]
    assert '--train-option' in first and '--train-option' not in second
    assert '--eval-option' in second and '--eval-option' not in first
    assert 'with spaces' in shlex.split(first.split('|')[2])
    log.unlink()
    result = run(args,cwd,expected=17,environment={**shell_env,'TRAIN_EXIT':'17'})
    assert len(log.read_text().splitlines()) == 1
    assert 'evaluation was not started' in result.stderr
    log.unlink()
    run(args,cwd,expected=19,environment={**shell_env,'EVAL_EXIT':'19'})
    assert len(log.read_text().splitlines()) == 2

run(['bash','-n',str(launcher)],ROOT)
(OUT/'results.json').write_text(json.dumps(dict(
    passed=True, command_pairs=16, parsed_commands=32, records=records,
    shell_checks=['success train then eval','train exit17 prevents eval','eval exit19 propagated',
                  'stage-only args separated','spaces preserved','fixed/generated tag consistency',
                  '5 invalid invocations rejected','dry-run no files','bash -n'],
    not_run=['real data loading','training','evaluation','new model/PyG/GPU tests']),indent=2)+'\n')
print('PASS: 8 default + 8 override pairs, 32 real AST-parser commands; shell sequencing/failure/tag/quoting checks.')
