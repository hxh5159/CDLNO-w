"""Reproducible LAA1 checks only; no V3 tensor model or data loader."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMANDS = {
    'targeted': [sys.executable, '-B', '-m', 'unittest', '-v',
                 'loop_linearno_latent_adapter.test_laa1_config',
                 'loop_linearno_latent_adapter.test_laa1_costs',
                 'loop_linearno_latent_adapter.test_laa1_schema',
                 'loop_linearno_latent_adapter.test_laa1_isolation'],
    'legacy': [sys.executable, '-B', '-m', 'unittest', '-v',
               'linearno.test_profiles', 'linearno.test_schema',
               'linearno.test_history_schema', 'loop_linearno.test_config',
               'loop_linearno.test_schema', 'loop_linearno_ffn.test_lf1_config',
               'loop_linearno_ffn.test_lf1_schema'],
    'matrix': [sys.executable, '-B', '-m', 'linearno_loop.v3.matrix',
               '--output-dir', str(OUT / 'matrix')],
}


def main():
    name = sys.argv[1]
    command = COMMANDS[name]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join(map(str, [ROOT / 'tests', ROOT / 'tests/loop_linearno_ffn', ROOT])))
    label = sys.argv[2] if len(sys.argv) > 2 else 'verified'
    log = OUT / (name + '-' + label + '.log')
    start = time.perf_counter()
    with log.open('x') as stream:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    output = log.read_text()
    match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', output)
    result = dict(command=command, cwd=str(ROOT), exit_code=p.returncode,
                  wall_seconds=seconds, log=log.name,
                  status='PASS' if p.returncode == 0 else 'FAIL',
                  tests=int(match[1]) if match else None,
                  unittest_seconds=float(match[2]) if match else None,
                  environment=dict(python=sys.version, executable=sys.executable,
                                   platform=platform.platform(),
                                   packages={k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'torch-geometric')},
                                   overrides={k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    with (OUT / (name + '-' + label + '.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    raise SystemExit(p.returncode)


if __name__ == '__main__':
    main()
