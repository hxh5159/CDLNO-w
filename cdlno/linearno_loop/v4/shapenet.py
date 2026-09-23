"""ShapeNet-Car v4 wrapper; M is independent of point count and d_h."""
import torch
from torch import nn
from .core import V4LoopCore
from .lifecycle import InitializedWrapper
from cdlno.linearno.shapenet import PointMLP, single_graph

class LoopedShapeNetModelV4(InitializedWrapper):
    def __init__(self, *, hidden_width=256, heads=8, actual_M=32, ffn_ratio=2, ref=8,
                 dropout=0., out_dim=4, space_dim=3, fun_dim=4, temperature_mode="base", public_seed=0, **_):
        super().__init__(); self.ref=ref; self.n_hidden=hidden_width
        if space_dim+fun_dim!=7 or out_dim!=4: raise ValueError("ShapeNet v4 requires 7 input and 4 output channels")
        self.preprocess=PointMLP(7,2*hidden_width,hidden_width)
        self.loop=V4LoopCore(hidden=hidden_width,heads=heads,rank=actual_M,variant="shapenet",dropout=dropout,
            H=1,W=1,out_dim=out_dim,ffn_ratio=ffn_ratio,temperature_mode=temperature_mode,public_seed=public_seed)
        self.final_norm=nn.LayerNorm(hidden_width);self.head=nn.Linear(hidden_width,out_dim)
        from cdlno.linearno.attention import initialize_release_weights
        self.apply(initialize_release_weights); self.placeholder=nn.Parameter(torch.rand(hidden_width)/hidden_width)
        self.loop.install_temperature_predictors()
        self._initialization_complete = True
        self.architecture="resmlp_dual_temp_v4"
    def forward(self,data):
        if not isinstance(data,(tuple,list)) or len(data)!=2: raise ValueError("ShapeNet v4 expects (data, geom)")
        cfd,_=data; x=getattr(cfd,'x',None)
        if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=7 or not x.is_floating_point(): raise ValueError("ShapeNet data.x must be [N,7]")
        single_graph(cfd,x.shape[0]); h=self.preprocess(x.unsqueeze(0))+self.placeholder[None,None,:]
        return self.head(self.final_norm(self.loop(h)))[0]
