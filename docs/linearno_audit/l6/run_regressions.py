"""Replay accepted L1–L5 checks; no real task/data entry imports or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
modules = [
    'linearno.test_profiles', 'linearno.test_schema', 'linearno.test_rng',
    'linearno.test_attention_parity', 'linearno.test_attention_structure',
    'linearno.test_standard_model', 'linearno.test_standard_structure',
    'linearno.test_static_integration', 'linearno.test_temporal_integration',
]
artifacts = Path(tempfile.mkdtemp(prefix='linearno-l6-regression-', dir=ROOT.parent/'CDLNO-artifacts'))
(OUT/'regression-root.txt').write_text(str(artifacts)+'\n')
results = []
for module in modules:
    env = dict(os.environ, PYTHONPATH=str(ROOT/'tests')+':'+str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    env.update(LINEARNO_L2_PARITY_REPORT=str(OUT/'l2-parity.json'),
               LINEARNO_L3_PARITY_REPORT=str(OUT/'l3-parity.json'),
               LINEARNO_L4_INTEGRATION_REPORT=str(OUT/'static-integration.json'),
               LINEARNO_L5_REPORT=str(OUT/'temporal-integration.json'))
    if module.endswith(('test_static_integration', 'test_temporal_integration')):
        env['CUDA_VISIBLE_DEVICES'] = ''
        folder = artifacts/module.rsplit('.', 1)[-1]
        folder.mkdir()
        env['LINEARNO_L4_ARTIFACT_ROOT'] = str(folder)
        env['LINEARNO_L5_ARTIFACT_ROOT'] = str(folder)
    log = OUT/(module.rsplit('.', 1)[-1]+'.txt')
    cmd = [sys.executable, '-B', '-m', 'unittest', module, '-v']
    started = time.perf_counter()
    with log.open('w') as stream:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(module=module, command=cmd, returncode=process.returncode,
               seconds=time.perf_counter()-started, log=log.name)
    results.append(row)
    (OUT/'regression-results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if process.returncode:
        print(log.read_text()[-6000:], flush=True)
        sys.exit(process.returncode)
