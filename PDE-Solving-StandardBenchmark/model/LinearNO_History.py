"""R2 Standard research wrapper; pure Model and all production entries stay unchanged.

Constructs the exact existing backbone first, then wraps its independent blocks
without any initialization/RNG draw. Direct internal use only until R5 routing.
"""
import torch
from .LinearNO import Model as PureModel, timestep_embedding
from cdlno.linearno_history.core import LinearNOHistoryCore


class Model(PureModel):
    """Internal R2 no-op wrapper. No A/K flags or production selection."""

    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, Time_Input=False, act='gelu', mlp_ratio=1, fun_dim=1,
                 out_dim=1, ref=8, unified_pos=False, H=85, W=85, *,
                 linearno_variant='plain', linearno_rank=4):
        super().__init__(space_dim=space_dim, n_layers=n_layers, n_hidden=n_hidden,
                         dropout=dropout, n_head=n_head, Time_Input=Time_Input,
                         act=act, mlp_ratio=mlp_ratio, fun_dim=fun_dim, out_dim=out_dim,
                         ref=ref, unified_pos=unified_pos, H=H, W=W,
                         linearno_variant=linearno_variant, linearno_rank=linearno_rank)
        self.blocks = LinearNOHistoryCore(self.blocks)

    def forward(self, x, fx, T=None, *, observe=None, latent_attnres=None, history_k=None):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or (not x.is_floating_point()):
            raise ValueError('x must be a floating tensor [B,N,space_dim]')
        B, N, channels = x.shape
        if B < 1 or N < 1 or channels != self.space_dim:
            raise ValueError(f'x requires B,N > 0 and space_dim={self.space_dim}; got {tuple(x.shape)}')
        if (self.unified_pos or self.linearno_variant in ('conv', 'conv_temp')) and N != self.H * self.W:
            raise ValueError(f'LinearNO conv/unified position requires N = H * W = {self.H} * {self.W}; got {N}')
        if fx is None:
            if self.fun_dim != 0:
                raise ValueError(f'fx=None requires fun_dim=0; configured fun_dim={self.fun_dim}')
        elif not isinstance(fx, torch.Tensor) or tuple(fx.shape) != (B, N, self.fun_dim) or fx.device != x.device or (fx.dtype != x.dtype):
            raise ValueError(f'fx must be [B,N,fun_dim]=[{B},{N},{self.fun_dim}] on x device/dtype')
        if T is not None:
            if not self.Time_Input:
                raise ValueError('T requires Time_Input=True')
            if not isinstance(T, torch.Tensor) or tuple(T.shape) != (B, 1) or T.device != x.device or (not T.is_floating_point()):
                raise ValueError('T must be a floating tensor [B,1] on x device')
        positions = self.pos.repeat(B, 1, 1, 1).reshape(B, N, self.ref * self.ref) if self.unified_pos else x
        if fx is None:
            features = self.preprocess(positions) + self.placeholder[None, None, :]
        else:
            features = self.preprocess(torch.cat((positions, fx), dim=-1))
        if T is not None:
            time = timestep_embedding(T, self.n_hidden).repeat(1, N, 1)
            features = features + self.time_fc(time.to(dtype=self.time_fc[0].weight.dtype))
        features = self.blocks(features, observe=observe, latent_attnres=latent_attnres, history_k=history_k)
        return features


class AttnResModel(Model):
    """Internal A-only model, not registered with any benchmark factory."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        from cdlno.linearno_history.attnres import LatentSummaryAttnRes
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), self.blocks[0].Attn.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)

    def forward(self, x, fx, T=None, *, observe=None):
        return super().forward(x, fx, T, observe=observe, latent_attnres=self.latent_attnres)


class HistoryKModel(Model):
    """Internal K-only model. Does not construct/import an A operator."""
    def __init__(self, *args, feature_seed, **kwargs):
        super().__init__(*args, **kwargs)
        from cdlno.linearno_history.history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, x, fx, T=None, *, observe=None):
        return super().forward(x, fx, T, observe=observe, history_k=self.history_k)


class JointHistoryModel(Model):
    """Internal A+K model; mechanisms own disjoint parameters."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        if float(attnres_history_dropout_p) == 0.0:
            raise ValueError('history dropout p=0 is only valid for A1K0')
        super().__init__(*args, **kwargs)
        from cdlno.linearno_history.attnres import LatentSummaryAttnRes
        from cdlno.linearno_history.history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), first.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, x, fx, T=None, *, observe=None):
        return super().forward(x, fx, T, observe=observe,
                               latent_attnres=self.latent_attnres, history_k=self.history_k)
