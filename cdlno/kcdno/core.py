"""KCDNO v1 feature core, assembled from the verified LRSA/K2 primitives.

Import explicitly from this module; configuration-only imports stay torch-free.
Task lifting, time/position conditioning and the output norm/head belong to the
task adapter. This core has exactly L point-to-latent-to-point blocks.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..modules import ConvFFN, PlainFFN, RMSNorm, _DownAttention, _UpAttention, _check_grid
from .config import KCDNOArchitectureConfig
from .history import KernelHistoryCache, KernelHistoryReader, KernelHistoryWriter


def _check_config(config: KCDNOArchitectureConfig) -> None:
    if not isinstance(config, KCDNOArchitectureConfig):
        raise TypeError("config must be KCDNOArchitectureConfig")
    config.validate()
    # K1 records explicit widths; this first core implements the approved 2d
    # branches only. Never silently ignore a resolved architecture field.
    for name in ("ffn1_hidden", "ffn2_hidden", "point_hidden"):
        if getattr(config, name) != 2 * config.d:
            raise ValueError(f"KCDNO v1 core requires {name}=2*d={2 * config.d}")


def _check_input(x: Tensor, config: KCDNOArchitectureConfig,
                 grid_shape: tuple[int, int] | None) -> None:
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise ValueError("X must be a floating-point [B,N,d] tensor")
    if x.ndim != 3 or x.shape[-1] != config.d or min(x.shape[:2]) < 1:
        raise ValueError(f"expected nonempty [B,N,{config.d}], got {tuple(x.shape)}")
    if config.point_module == "conv_ffn":
        if not isinstance(grid_shape, tuple):
            raise ValueError("conv_ffn requires explicit grid_shape=(height, width)")
        _check_grid(grid_shape, x.shape[1])
    elif grid_shape is not None:
        raise ValueError("point_ffn does not use grid_shape; pass None")


class KCDNOBlock(nn.Module):
    """One complete point-domain block; layer_index is zero-based.

    Internal return is (next points, optional source summary), consumed only
    by KCDNO. The public core returns a Tensor, never histories or attention.
    History must be the immutable snapshot of all preceding source summaries.
    There is no SA sublayer, bridge, persistent processor or task output head.
    """

    def __init__(self, config: KCDNOArchitectureConfig, layer_index: int) -> None:
        super().__init__()
        _check_config(config)
        if type(layer_index) is not int or not 0 <= layer_index < config.L:
            raise ValueError("layer_index must be an integer in [0, L)")
        self.config = config
        self.layer_index = layer_index
        d = config.d
        self.point_norm = RMSNorm(d, eps=config.norm_eps)
        self.down = _DownAttention(d, config.h, config.M)
        self.latent_norm_1 = RMSNorm(d, eps=config.norm_eps)
        self.latent_ffn_1 = PlainFFN(d, ratio=2.0)
        if config.history_mode == "all" and layer_index > 0:
            self.reader = KernelHistoryReader(d, config.kernel_rank)
        self.latent_norm_2 = RMSNorm(d, eps=config.norm_eps)
        self.latent_ffn_2 = PlainFFN(d, ratio=2.0)
        self.up_latent_norm = RMSNorm(d, eps=config.norm_eps)
        self.up = _UpAttention(d, config.h, qk_norm=True, bias_qkv=False, bias_out=True)
        self.point_ffn_norm = RMSNorm(d, eps=config.norm_eps)
        self.point_ffn = (ConvFFN(d, ratio=2.0, eps=config.norm_eps)
                          if config.point_module == "conv_ffn" else PlainFFN(d, ratio=2.0))
        if config.history_mode == "all" and layer_index < config.L - 1:
            self.writer = KernelHistoryWriter(d, config.kernel_rank)
        # Each primitive initializes itself once. No recursive reinitialization:
        # preserve orthogonal Down Q, native Conv, and K2 Xavier/w0/gamma0.1.

    def forward(self, x: Tensor, history: tuple[KernelHistoryCache, ...] = (), *,
                grid_shape: tuple[int, int] | None = None
                ) -> tuple[Tensor, KernelHistoryCache | None]:
        _check_input(x, self.config, grid_shape)
        if not isinstance(history, tuple):
            raise TypeError("history must be a tuple snapshot of previous summaries")
        expected = self.layer_index if self.config.history_mode == "all" else 0
        if len(history) != expected:
            raise ValueError(f"layer {self.layer_index} expects {expected} prior summaries, got {len(history)}")

        h = self.point_norm(x)
        s = self.down(h)
        u = s + self.latent_ffn_1(self.latent_norm_1(s))
        aligned = self.reader(u, history) if history else u
        t = aligned + self.latent_ffn_2(self.latent_norm_2(aligned))
        # Up itself only normalizes per-head projected Q/K. Its input T gets
        # precisely this one latent pre-norm; raw T remains the source value.
        v = x + self.up(h, self.up_latent_norm(t))
        points = self.point_ffn_norm(v)
        update = (self.point_ffn(points, grid_shape)
                  if self.config.point_module == "conv_ffn" else self.point_ffn(points))
        next_points = v + update
        cache = self.writer(t) if hasattr(self, "writer") else None
        return next_points, cache


class KCDNO(nn.Module):
    """L independent blocks: lifted X[B,N,d] -> features[B,N,d].

    Conv mode requires the original (H,W) on each call. Histories exist only
    as local variables within this forward, with live gradients. In particular
    an NS time step must call the core afresh, as any other independent input.
    """

    def __init__(self, config: KCDNOArchitectureConfig | None = None) -> None:
        super().__init__()
        self.config = KCDNOArchitectureConfig() if config is None else config
        _check_config(self.config)
        self.blocks = nn.ModuleList(KCDNOBlock(self.config, i) for i in range(self.config.L))

    def forward(self, x: Tensor, *, grid_shape: tuple[int, int] | None = None) -> Tensor:
        _check_input(x, self.config, grid_shape)
        history: list[KernelHistoryCache] = []
        for block in self.blocks:
            # Snapshot first; append the new raw-T summary only after this
            # block has consumed older sources and completed its point update.
            x, cache = block(x, tuple(history), grid_shape=grid_shape)
            if cache is not None:
                history.append(cache)
        return x


__all__ = ["KCDNO", "KCDNOBlock"]
