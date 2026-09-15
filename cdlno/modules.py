"""Phase-2 CDLNO building blocks.

This module contains only the reusable norm/attention/FFN pieces and the
non-CDPA parts of the planned architecture.  CDPA fusion and model assembly
are intentionally left for later phases.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import FrontLatentMode, _check_front_latent_mode


def _init_linear(module: nn.Linear, std: float = 0.02) -> None:
    nn.init.trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _hidden_width(dim: int, ratio: float) -> int:
    _positive_int("dim", dim)
    if isinstance(ratio, bool) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("FFN ratio must be finite and positive")
    width = dim * ratio
    if not float(width).is_integer():
        raise ValueError("dim * ratio must be an integer; hidden width is not rounded")
    return int(width)


def _check_tokens(x: Tensor, dim: int) -> None:
    if x.ndim != 3 or x.shape[-1] != dim or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"expected nonempty [B,N,{dim}], got {tuple(x.shape)}")


def _check_grid(grid_shape: Tuple[int, int], n: int) -> None:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be (height, width)")
    height, width = grid_shape
    _positive_int("height", height)
    _positive_int("width", width)
    if n != height * width:
        raise ValueError(f"N={n} does not match grid_shape={grid_shape}")


class RMSNorm(nn.Module):
    """RMSNorm with a scale and no bias, as required by v1.2."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        _positive_int("dim", dim)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate low-precision inputs in FP32; preserve FP64 references.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        variance = work.square().mean(dim=-1, keepdim=True)
        out = work * torch.rsqrt(variance + self.eps)
        return (out * self.weight.to(work.dtype)).to(x.dtype)


NormKind = Literal["rmsnorm", "layernorm"]


def make_norm(dim: int, kind: NormKind = "rmsnorm", eps: float = 1e-6) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind: {kind!r}")


def _head_view(x: Tensor, heads: int, head_dim: int) -> Tensor:
    b, n, d = x.shape
    if d != heads * head_dim:
        raise ValueError(f"expected last dimension {heads * head_dim}, got {d}")
    return x.view(b, n, heads, head_dim).transpose(1, 2)


class PlainFFN(nn.Module):
    """LRSA point/latent plain FFN: Linear(d,2d)-GELU-Linear(2d,d)."""

    def __init__(self, dim: int, ratio: float = 2.0) -> None:
        super().__init__()
        hidden = _hidden_width(dim, ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GEGLUFFN(nn.Module):
    """IPOT-style GEGLU with Linear(d,2*r*d) and Linear(r*d,d)."""

    def __init__(self, dim: int, ratio: float = 2.0) -> None:
        super().__init__()
        hidden = _hidden_width(dim, ratio)
        self.ratio = ratio
        self.fc_in = nn.Linear(dim, 2 * hidden, bias=True)
        self.fc_out = nn.Linear(hidden, dim, bias=True)
        _init_linear(self.fc_in)
        _init_linear(self.fc_out)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.fc_in(x).chunk(2, dim=-1)
        return self.fc_out(value * F.gelu(gate))


class ConvFFN(nn.Module):
    """LRSA structured point FFN with a regular, groups=1 3x3 Conv2d."""

    def __init__(self, dim: int, ratio: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        hidden = _hidden_width(dim, ratio)
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=True)
        # Conv2d intentionally keeps PyTorch's default Kaiming-uniform init.
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: Tensor, grid_shape: Tuple[int, int]) -> Tensor:
        _check_tokens(x, self.dim)
        b, n, d = x.shape
        _check_grid(grid_shape, n)
        height, width = grid_shape
        # [B,N,D] uses the existing row-major node order; no coordinate sort.
        image = x.reshape(b, height, width, d).permute(0, 3, 1, 2)
        image = self.conv(image)
        points = image.permute(0, 2, 3, 1).reshape(b, n, d)
        return self.fc2(F.gelu(self.fc1(self.norm(points))))


class _ProjectedAttention(nn.Module):
    """Multi-head attention using PyTorch SDPA and explicit projection policy."""

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        qk_norm: bool,
        bias_qkv: bool = False,
        bias_out: bool = True,
        dropout: float = 0.0,
        output_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("heads", heads)
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if not 0 <= dropout < 1 or not 0 <= output_dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.to_q = nn.Linear(dim, dim, bias=bias_qkv)
        self.to_k = nn.Linear(dim, dim, bias=bias_qkv)
        self.to_v = nn.Linear(dim, dim, bias=bias_qkv)
        self.to_out = nn.Linear(dim, dim, bias=bias_out)
        self.dropout = dropout
        self.output_dropout = output_dropout
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        for projection in (self.to_q, self.to_k, self.to_v, self.to_out):
            _init_linear(projection)

    def forward(self, q_input: Tensor, kv_input: Tensor) -> Tensor:
        _check_tokens(q_input, self.dim)
        _check_tokens(kv_input, self.dim)
        if q_input.shape[0] != kv_input.shape[0]:
            raise ValueError("query/context batch sizes must match (no broadcasting)")
        q = _head_view(self.to_q(q_input), self.heads, self.head_dim)
        k = _head_view(self.to_k(kv_input), self.heads, self.head_dim)
        v = _head_view(self.to_v(kv_input), self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.head_dim ** -0.5,
        )
        y = y.transpose(1, 2).contiguous().view(q_input.shape[0], q_input.shape[1], self.dim)
        return F.dropout(self.to_out(y), p=self.output_dropout, training=self.training)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, qk_norm: bool = True, dropout: float = 0.0, output_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = _ProjectedAttention(dim, heads, qk_norm=qk_norm, dropout=dropout, output_dropout=output_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.attn(x, x)


class _CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        qk_norm: bool,
        bias_qkv: bool = False,
        bias_out: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = _ProjectedAttention(
            dim, heads, qk_norm=qk_norm, bias_qkv=bias_qkv,
            bias_out=bias_out, dropout=dropout,
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        return self.attn(query, context)


class _DownAttention(nn.Module):
    """LRSA down projection: learned queries are direct Q, no Q residual."""

    def __init__(self, dim: int, heads: int, num_latents: int, dropout: float = 0.0) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("heads", heads)
        _positive_int("num_latents", num_latents)
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_latents = num_latents
        self.latent_queries = nn.Parameter(torch.empty(num_latents, heads, self.head_dim))
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        for projection in (self.to_k, self.to_v, self.to_out):
            _init_linear(projection)

        # Special query init comes last; parents must not recursively reset it.
        nn.init.orthogonal_(self.latent_queries.view(num_latents, dim))
        self.dropout = dropout

    def forward(self, points: Tensor) -> Tensor:
        _check_tokens(points, self.dim)
        b = points.shape[0]
        q = self.latent_queries.unsqueeze(0).expand(b, -1, -1, -1).permute(0, 2, 1, 3)
        k = _head_view(self.to_k(points), self.heads, self.head_dim)
        v = _head_view(self.to_v(points), self.heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.to(k.dtype)  # learned Q stays FP32 while K/V may be autocast
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.head_dim ** -0.5,
        )
        y = y.transpose(1, 2).contiguous().view(b, self.num_latents, self.dim)
        return F.dropout(self.to_out(y), p=self.dropout, training=self.training)


class _UpAttention(_ProjectedAttention):
    """LRSA up projection: point Q, latent K/V, independent from down."""

    pass


class LRSAFrontBlock(nn.Module):
    """LRSA point/latent block with a selectable latent processor.

    full: FFN1 -> SA -> FFN2, each with its own pre-norm and residual.
    no_sa: FFN1 -> FFN2; identity: T is the complete Down output itself.
    Returns (H_next, T), with live T before the retained Up-specific norm.
    Down, Up, point residuals and point FFN are identical in all modes.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        num_latents: int,
        *,
        structured: bool = False,
        grid_shape: Optional[Tuple[int, int]] = None,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        front_latent_mode: FrontLatentMode = "full",
    ) -> None:
        super().__init__()
        _check_front_latent_mode(front_latent_mode)
        self.front_latent_mode = front_latent_mode
        self.dim = dim
        self.num_latents = num_latents
        self.structured = structured
        self.grid_shape = grid_shape
        self.point_norm = RMSNorm(dim)
        self.down = _DownAttention(dim, heads, num_latents, dropout)
        # Keep full's names and construction/RNG order exactly as before.
        if front_latent_mode != "identity":
            self.latent_norm_1 = RMSNorm(dim)
            self.latent_ffn_1 = PlainFFN(dim, ffn_ratio)
        if front_latent_mode == "full":
            self.latent_norm_sa = RMSNorm(dim)
            self.latent_sa = _SelfAttention(dim, heads, qk_norm=True, output_dropout=dropout)
        if front_latent_mode != "identity":
            self.latent_norm_2 = RMSNorm(dim)
            self.latent_ffn_2 = PlainFFN(dim, ffn_ratio)
        self.up_latent_norm = RMSNorm(dim)
        self.up = _UpAttention(
            dim, heads, qk_norm=True,
            bias_qkv=False, bias_out=True, output_dropout=dropout,
        )
        self.point_ffn_norm = RMSNorm(dim)
        self.point_ffn = ConvFFN(dim, ffn_ratio) if structured else PlainFFN(dim, ffn_ratio)

    def forward(self, h: Tensor, *, grid_shape: Optional[Tuple[int, int]] = None) -> tuple[Tensor, Tensor]:
        # Trusted pre-A1 whole objects predate this attribute and are full.
        mode = getattr(self, "front_latent_mode", "full")
        _check_front_latent_mode(mode)
        _check_tokens(h, self.dim)
        shape = grid_shape if grid_shape is not None else self.grid_shape
        if self.structured:
            if shape is None:
                raise ValueError("structured LRSA block requires grid_shape")
            _check_grid(shape, h.shape[1])
        point_input = h
        h_norm = self.point_norm(h)
        z = self.down(h_norm)
        if mode == "identity":
            t = z
        else:
            z = z + self.latent_ffn_1(self.latent_norm_1(z))
            if mode == "full":
                z = z + self.latent_sa(self.latent_norm_sa(z))
            t = z + self.latent_ffn_2(self.latent_norm_2(z))
        delta = self.up(h_norm, self.up_latent_norm(t))
        u = point_input + delta
        if self.structured:
            point_update = self.point_ffn(self.point_ffn_norm(u), shape)
        else:
            point_update = self.point_ffn(self.point_ffn_norm(u))
        return u + point_update, t


class IPOTBridge(nn.Module):
    """IPOT-style learned-query bridge with a query residual and no encoder FFN."""

    def __init__(self, dim: int, heads: int, num_latents: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        _positive_int("dim", dim)
        _positive_int("heads", heads)
        _positive_int("num_latents", num_latents)
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.num_latents = num_latents
        self.latent_queries = nn.Parameter(torch.empty(num_latents, dim))
        self.query_norm = nn.LayerNorm(dim, eps=1e-6)
        self.context_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross = _CrossAttention(dim, heads, qk_norm=False, bias_qkv=False, bias_out=True, dropout=dropout)
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

    def forward(self, h: Tensor) -> Tensor:
        _check_tokens(h, self.dim)
        b = h.shape[0]
        q = self.latent_queries.unsqueeze(0).expand(b, -1, -1)
        return q + self.cross(self.query_norm(q), self.context_norm(h))


class PersistentLatentBlock(nn.Module):
    """One independently-instantiated IPOT-style latent processor block."""

    def __init__(self, dim: int, heads: int, *, geglu_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.norm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = _SelfAttention(dim, heads, qk_norm=False, dropout=dropout)
        self.norm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = GEGLUFFN(dim, geglu_ratio)

    def forward(self, z: Tensor) -> Tensor:
        _check_tokens(z, self.dim)
        normalized = self.norm_1(z)
        a = z + self.self_attention(normalized)
        return a + self.ffn(self.norm_2(a))


class LRSAFeatureReadout(nn.Module):
    """Final LRSA-style latent-to-point readout conditioned on ``H_F``."""

    def __init__(
        self,
        dim: int,
        heads: int,
        output_dim: int,
        *,
        structured: bool = False,
        grid_shape: Optional[Tuple[int, int]] = None,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int("output_dim", output_dim)
        self.dim = dim
        self.structured = structured
        self.grid_shape = grid_shape
        self.query_norm = RMSNorm(dim)
        self.latent_norm = RMSNorm(dim)
        self.up = _UpAttention(
            dim, heads, qk_norm=True,
            bias_qkv=False, bias_out=True, output_dropout=dropout,
        )
        self.point_ffn_norm = RMSNorm(dim)
        self.point_ffn = ConvFFN(dim, ffn_ratio) if structured else PlainFFN(dim, ffn_ratio)
        # Section 2.7 explicitly specifies LN_out at the task output head.
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        self.output = nn.Linear(dim, output_dim, bias=True)
        _init_linear(self.output)

    def forward_features(self, h_f: Tensor, z_p: Tensor, *, grid_shape: Optional[Tuple[int, int]] = None) -> Tensor:
        _check_tokens(h_f, self.dim)
        _check_tokens(z_p, self.dim)
        shape = grid_shape if grid_shape is not None else self.grid_shape
        if self.structured:
            if shape is None:
                raise ValueError("structured readout requires grid_shape")
            _check_grid(shape, h_f.shape[1])
        delta = self.up(self.query_norm(h_f), self.latent_norm(z_p))
        h_d = h_f + delta
        if self.structured:
            update = self.point_ffn(self.point_ffn_norm(h_d), shape)
        else:
            update = self.point_ffn(self.point_ffn_norm(h_d))
        return h_d + update

    def forward(self, h_f: Tensor, z_p: Tensor, *, grid_shape: Optional[Tuple[int, int]] = None) -> Tensor:
        return self.output(self.output_norm(self.forward_features(h_f, z_p, grid_shape=grid_shape)))


__all__ = [
    "RMSNorm", "make_norm", "PlainFFN", "GEGLUFFN", "ConvFFN",
    "LRSAFrontBlock", "IPOTBridge", "PersistentLatentBlock", "LRSAFeatureReadout",
]
