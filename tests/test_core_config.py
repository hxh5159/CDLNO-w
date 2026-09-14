"""Phase-4 configuration/loading boundaries; never import task entry scripts."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from cdlno import CDLNO, CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdlno.checkpoint import (
    SidecarMismatch, compare_architecture, save_sidecar,
    validate_sidecar,
)

ROOT = Path(__file__).resolve().parents[1]


def small_config(**changes):
    return replace(CDLNOArchitectureConfig(
        L=3, F=1, M=4, d_model=8, num_heads=2, output_dim=3,
    ), **changes)


def setUpModule():
    global previous_threads
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(previous_threads)


class CoreConfigChecks(unittest.TestCase):
    def test_illegal_architecture_and_core_policy_rejected(self):
        invalid = [
            dict(L=0, F=0), dict(F=-1), dict(F=3), dict(F=4),
            dict(M=0), dict(d_model=0), dict(num_heads=0), dict(output_dim=0),
            dict(d_model=9), dict(cdpa_mode='every'), dict(decoder_mode='coordinate'),
            dict(model_name='CDPA'), dict(history_rule='fused-history'),
            dict(structured=True), dict(grid_shape=(5, 7)),
            dict(structured=1), dict(structured=True, grid_shape=(0, 7)),
            dict(structured=True, grid_shape=(5, -7)),
            dict(structured=True, grid_shape=(True, 7)),
            dict(structured=True, grid_shape=(5, 7, 1)),
            dict(structured=True, grid_shape=(5.0, 7)),
            dict(structured=True, grid_shape=[5, 7]),
            dict(model_version='phase1'), dict(norm_type='layernorm'),
            dict(norm_type='unknown'), dict(attention_dropout=0.1),
            dict(attention_dropout=-0.1), dict(attention_dropout=float('nan')),
        ]
        for name in ('L', 'F', 'M', 'd_model', 'num_heads', 'output_dim'):
            invalid.extend({name: value} for value in (True, 2.0, '2'))
        for name in ('ffn_ratio', 'latent_ffn_ratio'):
            invalid.extend({name: value} for value in (
                0, -1, float('nan'), float('inf'), 0.3, True, '2',
            ))
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError, TypeError)):
                    CDLNO(small_config(**changes))
        with self.assertRaises(TypeError):
            CDLNO({})

    def test_invalid_input_and_chunk_rejected_before_computation(self):
        for structured in (False, True):
            cfg = small_config(structured=structured, grid_shape=(5, 7) if structured else None)
            for mode in ('off', 'entry', 'every_block'):
                model = CDLNO(replace(cfg, cdpa_mode=mode))
                n = 35 if structured else 11
                good = torch.randn(2, n, 8)
                bad_inputs = [None, [], torch.ones(2, n, 8, dtype=torch.int64),
                              torch.ones(n, 8), torch.ones(2, n, 7),
                              torch.ones(2, n, 8, 1)]
                if structured:
                    bad_inputs.append(torch.ones(2, 34, 8))
                with patch.object(model.front_blocks[0], 'forward', side_effect=AssertionError('front ran')), \
                        patch.object(model.bridge, 'forward', side_effect=AssertionError('bridge ran')):
                    for bad in bad_inputs:
                        with self.subTest(structured=structured, mode=mode, input_type=type(bad)):
                            with self.assertRaises((TypeError, ValueError)):
                                model(bad)
                    for chunk in (-1, True, 1.5, '1'):
                        with self.subTest(mode=mode, chunk=chunk):
                            with self.assertRaises((TypeError, ValueError)):
                                model(good, source_chunk_size=chunk)
                            with self.assertRaises((TypeError, ValueError)):
                                CDLNO(cfg, source_chunk_size=chunk)

    def test_derived_p_ratios_and_json_grid_roundtrip(self):
        cfg = small_config(structured=True, grid_shape=(5, 7), ffn_ratio=3, latent_ffn_ratio=1.5)
        payload = json.loads(json.dumps(cfg.to_dict()))
        self.assertEqual(payload['grid_shape'], [5, 7])
        self.assertNotIn('P', payload)
        self.assertEqual(CDLNOArchitectureConfig.from_dict(payload), cfg)
        self.assertEqual(CDLNOArchitectureConfig.from_dict(dict(payload, P=2)).P, 2)
        with self.assertRaises(ValueError):
            CDLNOArchitectureConfig.from_dict(dict(payload, P=6))
        for forbidden in ('P', 'rear_depth', 'latent_blocks'):
            with self.assertRaises(TypeError):
                CDLNOArchitectureConfig(**{forbidden: 6})
        # Frozen/slotted dataclass assignment raises TypeError on some Python
        # versions; both exceptions reject an independent P override.
        with self.assertRaises((AttributeError, TypeError)):
            cfg.P = 6
        model = CDLNO(cfg)
        front = model.front_blocks[0]
        self.assertEqual(front.latent_ffn_1.fc1.out_features, 24)
        self.assertEqual(front.latent_ffn_2.fc1.out_features, 24)
        self.assertEqual(front.point_ffn.fc1.out_features, 24)
        self.assertEqual(model.readout.point_ffn.fc1.out_features, 24)
        for block in model.latent_blocks:
            self.assertEqual(block.ffn.fc_in.out_features, 24)
            self.assertEqual(block.ffn.fc_out.in_features, 12)
        self.assertEqual(model(torch.randn(2, 35, 8)).shape, (2, 35, 3))

    def test_sidecar_runtime_allowed_architecture_rejected_bytes_preserved(self):
        cfg = small_config(structured=True, grid_shape=(5, 7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'architecture.json'
            save_sidecar(path, cfg, CDLNORuntimeConfig(device='cpu'))
            original = path.read_bytes()
            for chunk in (0, 1, 2, 99):
                runtime = CDLNORuntimeConfig(source_chunk_size=chunk, device='cuda',
                                             dtype='bfloat16', amp=True, sdpa_backend='math')
                self.assertEqual(validate_sidecar(path, cfg, runtime)['architecture'], cfg.to_dict())
                self.assertEqual(path.read_bytes(), original)
            for changes in (dict(L=4), dict(F=0), dict(M=5), dict(d_model=16),
                            dict(num_heads=4), dict(output_dim=4), dict(cdpa_mode='off'),
                            dict(cdpa_mode='every_block'), dict(grid_shape=(7, 5)),
                            dict(ffn_ratio=3), dict(latent_ffn_ratio=3),
                            dict(model_version='phase1'), dict(norm_type='layernorm')):
                requested = replace(cfg, **changes)
                self.assertEqual(compare_architecture(cfg, requested), sorted(changes))
                with self.assertRaises(SidecarMismatch):
                    validate_sidecar(path, requested)
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                validate_sidecar(path, replace(cfg, history_rule='future-state'))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                save_sidecar(path, cfg)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileNotFoundError):
                validate_sidecar(Path(directory) / 'missing.json', replace(cfg, F=-1))

    def test_strict_weights_and_existing_sidecar_in_fresh_process(self):
        script = '''
import sys
from pathlib import Path
import torch
from cdlno import CDLNO, CDLNOArchitectureConfig, CDLNORuntimeConfig
from cdlno.checkpoint import load_sidecar, validate_sidecar
torch.set_num_threads(1)
directory = Path(sys.argv[1])
path = directory / 'architecture.json'
original = path.read_bytes()
cfg = CDLNOArchitectureConfig.from_dict(load_sidecar(path)['architecture'])
validate_sidecar(path, cfg, CDLNORuntimeConfig(source_chunk_size=2, device='cpu'))
model = CDLNO(cfg, source_chunk_size=2).eval()
model.load_state_dict(torch.load(directory / 'weights.pt', weights_only=True), strict=True)
sample = torch.load(directory / 'sample.pt', weights_only=True)
with torch.no_grad():
    torch.testing.assert_close(model(sample['input']), sample['output'], atol=1e-5, rtol=3e-4)
assert path.read_bytes() == original
print('fresh-process core checkpoint passed')
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        for structured in (False, True):
            cfg = small_config(cdpa_mode='every_block', structured=structured,
                               grid_shape=(5, 7) if structured else None)
            model = CDLNO(cfg).eval()
            with self.assertRaises(RuntimeError):
                CDLNO(replace(cfg, M=5)).load_state_dict(model.state_dict(), strict=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                h = torch.randn(2, 35 if structured else 11, 8)
                save_sidecar(path / 'architecture.json', cfg, CDLNORuntimeConfig(device='cpu'))
                torch.save(model.state_dict(), path / 'weights.pt')
                with torch.no_grad():
                    torch.save({'input': h, 'output': model(h)}, path / 'sample.pt')
                result = subprocess.run([sys.executable, '-B', '-c', script, directory],
                                        cwd=directory, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('checkpoint passed', result.stdout)

    def test_three_task_cwd_fresh_imports_without_entry_or_install(self):
        script = '''
import sys
import cdlno
from cdlno import CDLNOArchitectureConfig
from cdlno.checkpoint import load_sidecar
assert 'torch' not in sys.modules
from cdlno import CDLNO
assert CDLNO.__module__ == 'cdlno.core'
assert not any(name.startswith(('exp_', 'dataset')) for name in sys.modules)
assert not any(name in sys.modules for name in ('main', 'main_evaluation', 'train', 'model_dict'))
print('lazy config and shared core import passed')
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        for name in ('PDE-Solving-StandardBenchmark', 'Car-Design-ShapeNetCar', 'Airfoil-Design-AirfRANS'):
            with self.subTest(cwd=name):
                result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT / name,
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('shared core import passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
