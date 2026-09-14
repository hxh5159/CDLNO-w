"""Untimed forward cost audit: exact dense matrix MACs + explicit non-MAC work.

SDPA MACs are obtained from live Q/K/V shapes, never profiler FLOP support.
Scalar/reduction/kernel inventories and temporary payloads are separate from
matrix MACs; allocation payload sums are NOT peak/live memory estimates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from cdlno.cdpa import CDPA
from cdlno.modules import ConvFFN, IPOTBridge, LRSAFrontBlock, LRSAFeatureReadout, PersistentLatentBlock, RMSNorm


def tensors(value):
    return [v for v in tree_flatten(value)[0] if isinstance(v, torch.Tensor)]


def storage_key(t):
    return (str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())


def category(path):
    if '.cdpa_at.' in path:
        return 'cdpa'
    if 'preprocess' in path or 'time_fc' in path:
        return 'stem_and_time'
    if 'latent_blocks' in path:
        return 'rear_geglu' if '.ffn.' in path else 'rear_attention'
    if 'latent_ffn_' in path:
        return 'front_two_latent_ffns'
    if 'point_ffn' in path:
        return 'point_ffn_dense_conv' if path.endswith('.conv') else 'point_ffn_linears'
    if 'output' in path or 'mlp2' in path:
        return 'output_head'
    if 'Attn' in path:
        return 'transolver_attention_projections'
    if '.mlp.' in path:
        return 'transolver_point_mlp'
    return 'front_bridge_readout_projections'


class Operations(TorchDispatchMode):
    """Observe actual tensor operators, including FP32 depth work and copies."""
    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.ops = defaultdict(lambda: dict(calls=0, output_elements=0, output_payload_bytes=0))
        self.copies = defaultdict(lambda: dict(calls=0, payload_bytes=0))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        ts = tensors(out)
        key = str(func)
        entry = self.ops[key]
        entry['calls'] += 1
        entry['output_elements'] += sum(t.numel() for t in ts)
        entry['output_payload_bytes'] += sum(t.numel() * t.element_size() for t in ts)
        if any(key.startswith('aten.' + op + '.') for op in ('stack', 'cat', 'clone', '_to_copy', 'repeat')):
            location = self.scope[-1] if self.scope else '<wrapper>'
            entry = self.copies[location + ':' + key]
            entry['calls'] += 1
            entry['payload_bytes'] += sum(t.numel() * t.element_size() for t in ts)
        return out


def audit(model, args, target, precision_context):
    """One synthetic training graph, outside all benchmark timing regions.

    Hooks keep only integer metadata. Saved tensors are returned unchanged to
    autograd; no detach, activation compression or checkpoint recomputation.
    """
    macs = Counter()
    bias_adds = Counter()
    counts = Counter(down_bridge=0, up_readout=0, latent_sa=0, structured_convffn=0,
                     cdpa_logical_sources=0, cdpa_locations=0, history_sdpa_calls=0, all_sdpa_calls=0)
    scope, sdpa_shapes, history_sizes, norm_work = [], [], [], []
    handles = []
    excluded = {storage_key(t) for t in [*model.parameters(), *model.buffers(), *tensors(args)]}
    saved, saved_logical_bytes = {}, 0
    point_feature_payloads, history_payloads, point_block_payloads = [], [], []

    def pack(t):
        nonlocal saved_logical_bytes
        key = storage_key(t)
        if key not in excluded:
            saved_logical_bytes += t.numel() * t.element_size()
            saved[key] = key[2]
        return t

    def pre(path, module, inputs):
        scope.append(path)
        if isinstance(module, (LRSAFrontBlock, IPOTBridge)):
            counts['down_bridge'] += 1
            point_block_payloads.append(inputs[0].numel() * inputs[0].element_size())
        if isinstance(module, (LRSAFrontBlock, LRSAFeatureReadout)):
            counts['up_readout'] += 1
        if isinstance(module, (LRSAFrontBlock, PersistentLatentBlock)):
            counts['latent_sa'] += 1
        if isinstance(module, ConvFFN):
            counts['structured_convffn'] += 1
        if isinstance(module, LRSAFeatureReadout):
            point_feature_payloads.append(inputs[0].numel() * inputs[0].element_size())
        if isinstance(module, CDPA):
            z, history = inputs
            s = len(history)
            counts['cdpa_logical_sources'] += s
            counts['cdpa_locations'] += bool(s)
            history_sizes.append(s)
            history_payloads.append(sum(t.numel() * t.element_size() for t in history))

    def post(path, module, inputs, out):
        if isinstance(module, nn.Linear):
            macs[category(path)] += out.numel() * module.in_features
            if module.bias is not None:
                bias_adds[category(path)] += out.numel()
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            macs[category(path)] += out.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            if module.bias is not None:
                bias_adds[category(path)] += out.numel()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            norm_work.append(dict(path=path, type=type(module).__name__, elements=inputs[0].numel(),
                                  width=inputs[0].shape[-1], dtype=str(inputs[0].dtype)))
        elif type(module).__name__.startswith('Physics_Attention_'):
            b, n, d = inputs[0].shape
            m = module.in_project_slice.out_features
            # Both original einsums plus manual QK and AV. Projections are
            # counted by their real Linear/Conv hooks, including per-head weights.
            macs['transolver_slice_deslice'] += 2 * b * n * m * d
            macs['transolver_latent_qk_av'] += 2 * b * m * m * d
            counts['transolver_physics_attention'] += 1
        scope.pop()

    original_sdpa = F.scaled_dot_product_attention
    def sdpa(q, k, v, *extra, **kwargs):
        path = scope[-1]
        value = math.prod(q.shape[:-2]) * q.shape[-2] * k.shape[-2] * (q.shape[-1] + v.shape[-1])
        is_history = '.cdpa_at.' in path
        macs['cdpa_qk_av' if is_history else 'non_cdpa_sdpa_qk_av'] += value
        counts['all_sdpa_calls'] += 1
        counts['history_sdpa_calls'] += is_history
        sdpa_shapes.append(dict(path=path, q=list(q.shape), k=list(k.shape), v=list(v.shape),
                                macs=value, q_dtype=str(q.dtype), k_dtype=str(k.dtype), v_dtype=str(v.dtype), softmax_logical_elements=math.prod(q.shape[:-1]) * k.shape[-2]))
        return original_sdpa(q, k, v, *extra, **kwargs)

    for path, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(lambda m, a, p=path: pre(p, m, a)))
        handles.append(module.register_forward_hook(lambda m, a, o, p=path: post(p, m, a, o)))
    dispatch = Operations(scope)
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), patch.object(F, 'scaled_dot_product_attention', sdpa):
            with dispatch, precision_context():
                output = model(*args)
            # Loss is outside the forward-only operation/MAC inventory.
            loss = (output.float() - target.float()).square().mean()
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        finite_output = bool(torch.isfinite(output).all())
    finally:
        for handle in handles:
            handle.remove()
    parameters = dict(registered=sum(p.numel() for p in model.parameters()),
                      requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      missing_grad_names=missing, nonfinite_grad_names=nonfinite)
    # Explicit source-fusion scalar work; this is NOT folded into matrix MACs.
    depth = []
    d = getattr(getattr(model, 'config', None), 'd_model', 0)
    m = getattr(getattr(model, 'config', None), 'M', 0)
    b = output.shape[0] if output.ndim == 3 else 1
    for s in history_sizes:
        rows, elems = b * m * (s + 1), b * m * (s + 1) * d
        depth.append(dict(sources=s, dtype='torch.float32', raw_stack_bytes=4 * elems,
                          normalized_keys_bytes=4 * elems, weighted_values_bytes=4 * elems,
                          score_and_weight_bytes=8 * rows,
                          # RMS square/mean/eps/rsqrt/scale, dot, RAW weighting
                          arithmetic_ops_excluding_rsqrt_exp_comparisons=7 * elems + b * m * d * s,
                          rsqrt_evaluations=rows, source_softmax_elements=rows, source_softmax_exp_evaluations=rows,
                          source_softmax_arithmetic_ops=3 * rows - b * m,
                          source_softmax_max_comparisons=rows - b * m,
                          source_softmax_axis=s + 1))
    result = dict(parameters=parameters, finite_output=finite_output, counts=dict(counts),
                  history_sizes=history_sizes, matrix_macs_by_component=dict(macs),
                  matrix_macs=sum(macs.values()), matrix_flops_2_per_mac=2 * sum(macs.values()),
                  bias_adds_by_component=dict(bias_adds), norm_work=norm_work,
                  sdpa=sdpa_shapes, depth_work_estimate=depth,
                  operations=dict(dispatch.ops), temporary_materializations=dict(dispatch.copies),
                  storage=dict(largest_point_block_input_payload_bytes=max(point_block_payloads, default=0),
                               h_f_payload_bytes=max(point_feature_payloads, default=0),
                               largest_history_list_payload_bytes=max(history_payloads, default=0),
                               current_latent_payload_bytes=b*m*d*4 if d else 0,
                               current_latent_payload_note='FP32 residual representation; actual mixed activation dtypes in norm/SDPA ledgers',
                               saved_activation_unique_backing_bytes=sum(saved.values()),
                               saved_activation_logical_bytes=saved_logical_bytes,
                               saved_storage_excludes='parameters, buffers, input backing storage; not allocator peak'))
    del output, loss
    model.zero_grad(set_to_none=True)
    return result
