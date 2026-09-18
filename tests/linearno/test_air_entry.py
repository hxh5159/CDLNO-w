"""AirfRANS LinearNO native train/checkpoint/resume/eval acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'tests/linearno/air_entry_worker.py'


class AirEntryCycle(unittest.TestCase):
    def _worker(self, action, run_dir, artifact):
        report = artifact / f'{action}.json'
        result = subprocess.run([sys.executable, '-B', str(WORKER), action, str(run_dir), str(report)],
                                cwd=ROOT / 'Airfoil-Design-AirfRANS',
                                env=dict(__import__('os').environ, PYTHONPATH=str(ROOT),
                                         PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=''),
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        return json.loads(report.read_text())

    def test_native_two_member_ensemble_manifest_and_load_order(self):
        with tempfile.TemporaryDirectory(prefix='linearno-air-ensemble-') as temp:
            root = Path(temp)
            report = self._worker('ensemble', root / 'ensemble', root)
            self.assertEqual(report['nmodel'], 2)
            self.assertEqual(report['member_hashes'], report['loaded_hashes'])
            self.assertEqual(report['members'], [
                {'member_id': 'member_000', 'order': 0},
                {'member_id': 'member_001', 'order': 1},
            ])
            self.assertTrue(report['distinct_members'])
            for index in range(2):
                member = root / 'ensemble' / f'member_{index:03d}'
                self.assertTrue((member / 'checkpoints' / 'final.json').is_file())
                self.assertTrue((member / 'weights').is_dir())

    def test_native_train_resume_fresh_eval_and_strict_metadata(self):
        with tempfile.TemporaryDirectory(prefix='linearno-air-cycle-') as temp:
            root = Path(temp)
            continuous = root / 'continuous'
            split = root / 'split'
            train = self._worker('train', continuous, root)
            interrupted = self._worker('interrupt', split, root)
            resumed = self._worker('resume', split, root)
            evaluation = self._worker('eval', split, root)
            self.assertEqual(train['epoch'], 3)
            self.assertEqual(interrupted['epoch'], 1)
            self.assertEqual(resumed['epoch'], 3)
            self.assertEqual(train['state_hash'], resumed['state_hash'])
            self.assertEqual(train['resume_hash'], resumed['resume_hash'])
            self.assertEqual(train['batches'], interrupted['batches'] + resumed['batches'])
            self.assertEqual(train['prediction'], evaluation['prediction'])
            self.assertEqual(train['objective']['kind'], 'MSE')
            self.assertEqual(train['objective']['surface_weight'], 1)
            self.assertTrue(interrupted['pointer'] and evaluation['metadata_immutable'])
            self.assertEqual(evaluation['family'], 'linearno')
            self.assertEqual(evaluation['negative_count'], 10)
            self.assertEqual(evaluation['shape'], [13, 4])
            self.assertTrue((split / 'ensemble.json').is_file())
            ensemble = json.loads((split / 'ensemble.json').read_text())
            self.assertEqual(ensemble['members'][0]['format'], 'state_dict')
            self.assertTrue((split / ensemble['members'][0]['path']).is_file())

    def test_two_member_resume_before_and_during_later_member(self):
        with tempfile.TemporaryDirectory(prefix='linearno-air-resume-members-') as temp:
            root = Path(temp)
            complete = self._worker('ensemble', root/'complete', root)
            for action in ('ensemble_interrupt', 'ensemble_later_interrupt'):
                directory = root/action
                interrupted = self._worker(action, directory, root)
                resumed = self._worker('resume', directory, root)
                evaluated = self._worker('eval', directory, root)
                self.assertEqual(complete['member_hashes'], resumed['member_hashes'])
                self.assertEqual(complete['resume_hash'], resumed['resume_hash'])
                self.assertEqual(complete['batches'], interrupted['batches']+resumed['batches'])
                self.assertEqual(complete['prediction'], evaluated['prediction'])


if __name__ == '__main__':
    unittest.main()
