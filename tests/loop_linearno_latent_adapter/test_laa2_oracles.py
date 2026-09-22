import unittest
import torch
from .primitive_oracles import adapter_indexed, adapter_einsum, adapter_merged_matrix, latent_expanded


class OracleTests(unittest.TestCase):
    def test_literal_rational_adapter_both_ends_three_independent_orders(self):
        for dtype in (torch.float64, torch.float32):
            x = torch.tensor([[[[1., 2.], [-1., 3.]]]], dtype=dtype)
            for a, b, expected in [
                ([[1., 0.], [0., 2.]], [[1., 0.], [0., 1.], [1., -1.]],
                 [[1.5, 6., -4.5], [-1.5, 9., -10.5]]),
                ([[2., 1.], [-1., 1.]], [[1., 2.], [-2., 1.], [0., -1.]],
                 [[9., -10.5, -1.5], [13.5, 3., -6.]])]:
                for oracle in (adapter_indexed, adapter_einsum, adapter_merged_matrix):
                    actual = oracle(x, torch.tensor(a, dtype=dtype), torch.tensor(b, dtype=dtype), 3.)
                    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype)[None, None], atol=0, rtol=0)

    def test_literal_latent_zero_mean_unit_variance_and_bias(self):
        # One token, H=2, channel vector [-1,1]. LN gamma compensates eps
        # so exact GELU(1) yields the independently tabulated scalar below.
        x = torch.tensor([[[[-1.]], [[1.]]]], dtype=torch.float64)
        p = {'norm.weight': torch.full((2,), (1+1e-5)**0.5, dtype=x.dtype),
             'norm.bias': torch.zeros(2, dtype=x.dtype),
             'linear1.weight': torch.tensor([[0., 1.]], dtype=x.dtype),
             'linear1.bias': torch.zeros(1, dtype=x.dtype),
             'linear2.weight': torch.tensor([[2.], [-1.]], dtype=x.dtype),
             'linear2.bias': torch.tensor([.5, -.25], dtype=x.dtype)}
        expected = torch.tensor([[[[1.1826894921370859]], [[-.09134474606854293]]]], dtype=x.dtype)
        torch.testing.assert_close(latent_expanded(x, p), expected, atol=1e-14, rtol=1e-14)
