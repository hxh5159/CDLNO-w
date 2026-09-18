"""Industrial research wrappers: no-op, A-only, K-only, joint; no task entry imports."""
import torch
from cdlno.linearno.airfrans import AirfRANSLinearNO as PureAirfRANS, single_graph as air_single_graph
from cdlno.linearno.shapenet import ShapeNetLinearNO as PureShapeNet, single_graph as car_single_graph
from .core import LinearNOHistoryCore


class AirfRANSHistoryModel(PureAirfRANS):
    """Internal R2 no-op wrapper. No A/K flags or production selection."""

    def __init__(self, space_dim=7, n_layers=8, n_hidden=256, dropout=0.0,
                 n_head=8, act='gelu', mlp_ratio=2, fun_dim=0, out_dim=4,
                 linearno_rank=32, ref=8, unified_pos=True, linear=True):
        super().__init__(space_dim=space_dim, n_layers=n_layers, n_hidden=n_hidden,
                         dropout=dropout, n_head=n_head, act=act, mlp_ratio=mlp_ratio,
                         fun_dim=fun_dim, out_dim=out_dim, linearno_rank=linearno_rank,
                         ref=ref, unified_pos=unified_pos, linear=linear)
        self.blocks = LinearNOHistoryCore(self.blocks)

    def forward(self, data, *, observe=None, latent_attnres=None, history_k=None):
        x, pos = (getattr(data, 'x', None), getattr(data, 'pos', None))
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 7 or (x.shape[0] < 1):
            raise ValueError('LinearNO AirfRANS data.x must have shape [N,7], N>0')
        if not x.is_floating_point():
            raise ValueError('LinearNO AirfRANS data.x must be floating point')
        if not isinstance(pos, torch.Tensor) or pos.shape != (x.shape[0], 2):
            raise ValueError('LinearNO AirfRANS data.pos must have shape [N,2]; no implicit slicing')
        if pos.device != x.device or pos.dtype != x.dtype:
            raise ValueError('LinearNO AirfRANS data.x and data.pos must share device and floating dtype')
        air_single_graph(data, x.shape[0])
        features = x.unsqueeze(0)
        if self.unified_pos:
            features = torch.cat((features, self.get_grid(pos.unsqueeze(0))), dim=-1)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        hidden = self.blocks(hidden, observe=observe, latent_attnres=latent_attnres, history_k=history_k)
        return hidden[0]


class ShapeNetHistoryModel(PureShapeNet):
    """Internal R2 no-op wrapper. No A/K flags or production selection."""

    def __init__(self, space_dim=3, n_layers=8, n_hidden=256, dropout=0.0,
                 n_head=8, Time_Input=False, act='gelu', mlp_ratio=2, fun_dim=4,
                 out_dim=4, linearno_rank=32, ref=8, unified_pos=False,
                 H=85, W=85, isregular=False):
        super().__init__(space_dim=space_dim, n_layers=n_layers, n_hidden=n_hidden,
                         dropout=dropout, n_head=n_head, Time_Input=Time_Input,
                         act=act, mlp_ratio=mlp_ratio, fun_dim=fun_dim, out_dim=out_dim,
                         linearno_rank=linearno_rank, ref=ref, unified_pos=unified_pos,
                         H=H, W=W, isregular=isregular)
        self.blocks = LinearNOHistoryCore(self.blocks)

    def forward(self, data, *, observe=None, latent_attnres=None, history_k=None):
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError('ShapeNet LinearNO expects (cfd_data, geom)')
        cfd_data, _ = data
        x = getattr(cfd_data, 'x', None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 1 or (x.shape[1] != 7):
            raise ValueError('ShapeNet LinearNO requires data.x [N,7] with N>0')
        if not x.is_floating_point():
            raise ValueError('ShapeNet LinearNO data.x must be floating point')
        car_single_graph(cfd_data, x.shape[0])
        if self.unified_pos:
            if x.shape[0] != self.H * self.W:
                raise ValueError('ShapeNet unified_pos requires N=H*W')
            features = self.pos.reshape(1, self.H * self.W, self.ref * self.ref)
        else:
            features = x.unsqueeze(0)
        hidden = self.preprocess(features) + self.placeholder[None, None, :]
        hidden = self.blocks(hidden, observe=observe, latent_attnres=latent_attnres, history_k=history_k)
        return hidden[0]


class AirfRANSAttnResModel(AirfRANSHistoryModel):
    """Internal A-only model; Data contract is inherited unchanged."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        from .attnres import LatentSummaryAttnRes
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), self.blocks[0].Attn.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe, latent_attnres=self.latent_attnres)


class ShapeNetAttnResModel(ShapeNetHistoryModel):
    """Internal A-only model; tuple/Data contract is inherited unchanged."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        from .attnres import LatentSummaryAttnRes
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), self.blocks[0].Attn.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe, latent_attnres=self.latent_attnres)


class AirfRANSHistoryKModel(AirfRANSHistoryModel):
    """Internal K-only model; original Data contract is inherited unchanged."""
    def __init__(self, *args, feature_seed, **kwargs):
        super().__init__(*args, **kwargs)
        from .history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe, history_k=self.history_k)


class ShapeNetHistoryKModel(ShapeNetHistoryModel):
    """Internal K-only model; original tuple/Data contract is unchanged."""
    def __init__(self, *args, feature_seed, **kwargs):
        super().__init__(*args, **kwargs)
        from .history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe, history_k=self.history_k)


class AirfRANSJointHistoryModel(AirfRANSHistoryModel):
    """Internal A+K; preserve the original task input/output contract."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        if float(attnres_history_dropout_p) == 0.0:
            raise ValueError('history dropout p=0 is only valid for A1K0')
        super().__init__(*args, **kwargs)
        from .attnres import LatentSummaryAttnRes
        from .history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), first.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe,
                               latent_attnres=self.latent_attnres, history_k=self.history_k)


class ShapeNetJointHistoryModel(ShapeNetHistoryModel):
    """Internal A+K; preserve the original task input/output contract."""
    def __init__(self, *args, feature_seed, attnres_history_dropout_p=0.1, **kwargs):
        if float(attnres_history_dropout_p) == 0.0:
            raise ValueError('history dropout p=0 is only valid for A1K0')
        super().__init__(*args, **kwargs)
        from .attnres import LatentSummaryAttnRes
        from .history_k import HistoryConditionedK
        first = self.blocks[0].Attn
        self.latent_attnres = LatentSummaryAttnRes(
            len(self.blocks), first.dim_head, feature_seed=feature_seed,
            dropout_p=attnres_history_dropout_p)
        self.history_k = HistoryConditionedK(
            len(self.blocks), first.heads, first.dim_head, feature_seed=feature_seed)

    def forward(self, data, *, observe=None):
        return super().forward(data, observe=observe,
                               latent_attnres=self.latent_attnres, history_k=self.history_k)
