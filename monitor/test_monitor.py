"""Small, data-free tests for the optional LinearNO monitor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from monitor.kernels import kernel_cosine, pairwise_kernel_similarity
from monitor.runtime import install_runtime_monitor


class KernelTests(unittest.TestCase):
    def test_matches_explicit_point_kernel(self):
        torch.manual_seed(3)
        qi, ki = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
        qj, kj = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
        values = kernel_cosine(qi, ki, qj, kj, eps=1e-12)
        expected = []
        for batch in range(2):
            pi = qi[batch] @ ki[batch].transpose(0, 1)
            pj = qj[batch] @ kj[batch].transpose(0, 1)
            expected.append((pi * pj).sum() / (pi.square().sum().sqrt() * pj.square().sum().sqrt() + 1e-12))
        self.assertTrue(torch.allclose(values[:, 0], torch.stack(expected), atol=1e-6, rtol=1e-6))

    def test_batch_head_isolation_and_matrix_shape(self):
        torch.manual_seed(5)
        factors = [(torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 4)) for _ in range(3)]
        matrix = pairwise_kernel_similarity(factors)
        self.assertEqual(tuple(matrix.shape), (2, 3, 3, 3))
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertTrue(torch.allclose(matrix.diagonal(dim1=-2, dim2=-1), torch.ones(2, 3, 3), atol=1e-6))


class RuntimeTests(unittest.TestCase):
    def test_standard_model_hook_writes_snapshot(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PDE-Solving-StandardBenchmark"))
        from model.LinearNO import Model

        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(17)
            model = Model(space_dim=2, n_layers=3, n_hidden=16, n_head=4,
                          fun_dim=0, out_dim=1, linearno_variant="plain", linearno_rank=3).eval()
            reference = Model(space_dim=2, n_layers=3, n_hidden=16, n_head=4,
                              fun_dim=0, out_dim=1, linearno_variant="plain", linearno_rank=3)
            reference.load_state_dict(model.state_dict())
            inputs = torch.randn(2, 11, 2)
            with torch.no_grad():
                expected = reference(inputs, None)
            monitor = install_runtime_monitor({"output": temporary, "max_samples_per_snapshot": 2,
                                               "no_plots": True})
            with torch.no_grad():
                output = model(inputs, None)
            self.assertEqual(tuple(output.shape), (2, 11, 1))
            self.assertTrue(torch.equal(output, expected))
            summary = monitor.close()
            self.assertEqual(summary["snapshots"], 1)
            archive = Path(temporary) / "snapshots/validation_000001/similarity_values.npz"
            self.assertTrue(archive.is_file())
            with __import__("numpy").load(archive) as values:
                self.assertEqual(tuple(values["rho_mean"].shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
