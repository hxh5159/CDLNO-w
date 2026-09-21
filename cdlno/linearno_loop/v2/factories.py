"""Factories that construct only the requested v2 owner, never discarded blocks."""

import importlib

from torch import nn

from .attention import ContextLinearNOAttention


def standard_factories(*, hidden, heads, rank, variant, dropout, mlp_ratio, H, W, out_dim):
    native = importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO")

    def block_factory(*, last_layer):
        return native.LinearNOBlock(hidden=hidden, heads=heads, rank=rank, variant=variant,
            dropout=dropout, mlp_ratio=mlp_ratio, H=H, W=W, last_layer=last_layer, out_dim=out_dim)

    def operator_factory():
        kwargs = dict(heads=heads, dim_head=hidden // heads, rank=rank, variant=variant, dropout=dropout)
        if variant in ("conv", "conv_temp"):
            kwargs.update(H=H, W=W)
        return nn.LayerNorm(hidden), ContextLinearNOAttention(hidden, **kwargs)

    def point_ffn_factory():
        return nn.LayerNorm(hidden), native.PointwiseMLP(hidden, hidden * mlp_ratio, hidden)

    return block_factory, operator_factory, point_ffn_factory


def industrial_factories(kind, *, hidden, heads, rank, dropout, mlp_ratio, out_dim):
    if kind == "airfrans":
        from cdlno.linearno.airfrans import AirfRANSBlock as Block, PointMLP
        variant = "airfrans"
    elif kind == "car":
        from cdlno.linearno.shapenet import ShapeNetBlock as Block, PointMLP
        variant = "shapenet"
    else:
        raise ValueError("industrial kind must be airfrans or car")

    def block_factory(*, last_layer):
        return Block(hidden, heads, rank, mlp_ratio, dropout, out_dim, last_layer)

    def operator_factory():
        return nn.LayerNorm(hidden), ContextLinearNOAttention(hidden, heads=heads, dim_head=hidden // heads,
            rank=rank, variant=variant, dropout=dropout)

    def point_ffn_factory():
        return nn.LayerNorm(hidden), PointMLP(hidden, hidden * mlp_ratio, hidden)

    return block_factory, operator_factory, point_ffn_factory
