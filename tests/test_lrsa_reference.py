"""Optional parity against an existing LRSA checkout; imports model files only.

CDLNO_LRSA_ROOT=/path/to/LRSA-Operator python -B -m unittest discover \
    -s tests -p test_lrsa_reference.py -v

No reference source is copied into CDLNO and no optional package is installed.
"""

import os
from pathlib import Path
import sys
import unittest

import torch

from cdlno.modules import LRSAFrontBlock


def setUpModule():
    root = os.environ.get("CDLNO_LRSA_ROOT")
    if root is None:
        raise unittest.SkipTest("set CDLNO_LRSA_ROOT to an audited LRSA checkout")
    source = Path(root) / "src"
    if not source.is_dir():
        raise ValueError(f"LRSA source directory missing: {source}")
    sys.path.insert(0, str(source))
    torch.set_num_threads(1)


def align_weights(ours, reference):
    """Map explicit v1.2 parameter names, checking every leaf was transferred."""
    pairs = []

    def parameter(a, b):
        with torch.no_grad():
            b.copy_(a)
        pairs.append((a, b))

    def norm(a, b):
        parameter(a.weight, b.nrm_scale)
        if getattr(a, "bias", None) is not None:
            parameter(a.bias, b.nrm_bias)

    def linear(a, b):
        parameter(a.weight, b.weight)
        if a.bias is not None:
            parameter(a.bias, b.bias)

    def attention(a, b):
        for name in ("to_q", "to_k", "to_v", "to_out"):
            if hasattr(a, name):
                linear(getattr(a, name), getattr(b, name))
        norm(a.q_norm, b.q_norm)
        norm(a.k_norm, b.k_norm)

    def plain(a, b):
        linear(a.fc1, b.up_proj)
        linear(a.fc2, b.down_proj)

    norm(ours.point_norm, reference.attn_norm)
    norm(ours.point_ffn_norm, reference.ffn_ln)
    parameter(ours.down.latent_queries, reference.attn.latents)
    attention(ours.down, reference.attn.down_project)
    norm(ours.latent_norm_1, reference.attn.channel_norm_1)
    plain(ours.latent_ffn_1, reference.attn.channel_mixing_1)
    norm(ours.latent_norm_sa, reference.attn.sm_norm)
    attention(ours.latent_sa.attn, reference.attn.latents_attention)
    norm(ours.latent_norm_2, reference.attn.channel_norm_2)
    plain(ours.latent_ffn_2, reference.attn.channel_mixing_2)
    norm(ours.up_latent_norm, reference.attn.up_project.out_latents_norm)
    attention(ours.up, reference.attn.up_project)
    if ours.structured:
        linear(ours.point_ffn.conv, reference.ffn.dwconv)
        norm(ours.point_ffn.norm, reference.ffn.norm)
        linear(ours.point_ffn.fc1, reference.ffn.pwconv)
        linear(ours.point_ffn.fc2, reference.ffn.pwconv2)
    else:
        plain(ours.point_ffn, reference.ffn)
    assert {id(p) for p in ours.parameters()} == {id(a) for a, _ in pairs}
    assert {id(p) for p in reference.parameters()} == {id(b) for _, b in pairs}
    return pairs


class LRSAReferenceChecks(unittest.TestCase):
    def compare_front(self, structured):
        from perceiverforpde.modeling.layers.attn import PerceiverAttentionConfig
        from perceiverforpde.modeling.layers.mlp import FeedForwardWithGatingConfig
        from perceiverforpde.modeling.perceiver import SinglePerceiverBlock
        from perceiverforpde.modeling.perceiver_structured import StructuredPerceiverBlock

        torch.manual_seed(82)
        ffn = FeedForwardWithGatingConfig(
            hidden_features=32, mlp_ratio=2, bias=True,
            disable_gate=True, act="gelu", dropout=0.0,
        )
        attn = PerceiverAttentionConfig(
            num_heads=4, dim_heads=4, num_latents=5, latents_dim=16,
            bias=False, ffn=ffn, enable_rope=False, qk_norm=True, norm_type="rmsnorm",
            attention_bias_down_out=True, attention_bias_up_out=True,
            attention_bias_interleaved_out=True,
        )
        cls = StructuredPerceiverBlock if structured else SinglePerceiverBlock
        reference = cls(16, 2, attn, ffn, "rmsnorm").double()
        ours = LRSAFrontBlock(16, 4, 5, structured=structured, grid_shape=(5, 7)).double()
        pairs = align_weights(ours, reference)
        inputs = torch.randn(2, 35, 16, dtype=torch.float64, requires_grad=True)
        ref_inputs = inputs.detach().clone().requires_grad_()
        positions = torch.randn(2, 35, 2, dtype=torch.float64)  # RoPE/mass disabled
        history = []
        hook = reference.attn.up_project.register_forward_pre_hook(
            lambda _module, args: history.append(args[1]))
        try:
            if structured:
                expected = reference(positions, ref_inputs, None, None, (5, 7))
            else:
                expected = reference(positions, ref_inputs, None, None)
        finally:
            hook.remove()
        actual, t = ours(inputs)
        self.assertEqual(len(history), 1)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(t, history[0], atol=1e-10, rtol=1e-8)
        v, vt = torch.randn_like(actual), torch.randn_like(t)
        ga = torch.autograd.grad((actual * v).sum() + (t * vt).sum(), (inputs, *(a for a, _ in pairs)))
        gb = torch.autograd.grad((expected * v).sum() + (history[0] * vt).sum(), (ref_inputs, *(b for _, b in pairs)))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-7)
        print(f"LRSA structured={structured}: y max error={(actual-expected).abs().max().item():.3g}; "
              f"T max error={(t-history[0]).abs().max().item():.3g}; "
              f"gradient max error={max((a-b).abs().max().item() for a,b in zip(ga,gb)):.3g}")

    def test_point_front_against_same_configuration(self):
        self.compare_front(False)

    def test_structured_front_against_same_configuration(self):
        self.compare_front(True)
