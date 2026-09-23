"""AirfRANS v4 wrapper with the native x7/position-distance contract."""
import torch
from torch import nn
from .core import V4LoopCore
from .lifecycle import InitializedWrapper
from cdlno.linearno.airfrans import PointMLP, single_graph
import numpy as np

class LoopedAirfRANSModelV4(InitializedWrapper):
    def __init__(self, *, hidden_width=256, heads=8, actual_M=32, ffn_ratio=2, ref=8,
                 dropout=0., out_dim=4, temperature_mode="base", public_seed=0, **_):
        super().__init__(); self.ref=ref; self.n_hidden=hidden_width
        self.preprocess=PointMLP(7+ref*ref,2*hidden_width,hidden_width)
        self.loop=V4LoopCore(hidden=hidden_width,heads=heads,rank=actual_M,variant="airfrans",dropout=dropout,
            H=1,W=1,out_dim=out_dim,ffn_ratio=ffn_ratio,temperature_mode=temperature_mode,public_seed=public_seed)
        self.final_norm=nn.LayerNorm(hidden_width);self.head=nn.Linear(hidden_width,out_dim)
        self.apply(__import__('cdlno.linearno.attention',fromlist=['initialize_release_weights']).initialize_release_weights)
        self.placeholder=nn.Parameter(torch.rand(hidden_width)/hidden_width)
        self.reference=torch.tensor(np.stack(np.meshgrid(np.linspace(-2,4,ref),np.linspace(-1.5,1.5,ref),indexing='ij'),-1).reshape(ref*ref,2),dtype=torch.float32)
        self.loop.install_temperature_predictors()
        self._initialization_complete = True
        self.architecture="resmlp_dual_temp_v4"
    def forward(self,data):
        x,pos=getattr(data,'x',None),getattr(data,'pos',None)
        if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=7 or not x.is_floating_point(): raise ValueError("AirfRANS data.x must be [N,7]")
        if not isinstance(pos,torch.Tensor) or pos.shape!=(x.shape[0],2) or pos.dtype!=x.dtype or pos.device!=x.device: raise ValueError("AirfRANS data.pos mismatch")
        single_graph(data,x.shape[0]); ref=self.reference.to(device=pos.device,dtype=pos.dtype)
        dist=((pos[:,None,:]-ref[None,:,:])**2).sum(-1).sqrt(); h=self.preprocess(torch.cat((x,dist),-1).unsqueeze(0))+self.placeholder[None,None,:]
        return self.head(self.final_norm(self.loop(h)))[0]
