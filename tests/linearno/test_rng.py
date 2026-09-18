import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from cdlno.training_state import _same
from cdlno.linearno.schema import pack_state, unpack_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class RandomStreamRegression(unittest.TestCase):
    def test_fresh_process_continuation_and_real_visualization(self):
        for schedule in ('onecycle', 'cosine'):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory(prefix='linearno-l1-rng-') as temp:
                path = Path(temp)
                for action in ('continuous', 'instrumented', 'split', 'resume'):
                    folder = path/('split' if action == 'resume' else action)
                    proc = subprocess.run([sys.executable, '-B', str(HERE/'rng_worker.py'), action, str(folder), schedule],
                        cwd=ROOT/'PDE-Solving-StandardBenchmark', env=dict(os.environ, PYTHONPATH=str(ROOT),
                        PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'), text=True, capture_output=True, timeout=120)
                    self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
                ref = torch.load(path/'continuous/continuous.pt', weights_only=True)
                self.assertTrue(_same(ref, unpack_state(pack_state(ref))), 'real Transolver optimizer/RNG numerical state encoding')
                for name, file in (('instrumented','instrumented'), ('split','resume')):
                    got = torch.load(path/name/(file+'.pt'), weights_only=True)
                    self.assertTrue(_same(ref, got), f'{schedule}/{name}: all weights/states/order/RNG must match bitwise')
                figures = list((path/'instrumented').rglob('*.pdf'))
                self.assertGreaterEqual(len(figures), 4, 'must actually render, not a no-op callback')
