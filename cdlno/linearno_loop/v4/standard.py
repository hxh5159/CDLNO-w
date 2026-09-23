"""Standard six-task v4 wrapper, preserving native input/time semantics."""
import importlib
import torch
from torch import nn
from .core import V4LoopCore
from .lifecycle import InitializedWrapper

native=importlib.import_module("PDE-Solving-StandardBenchmark.model.LinearNO")

class LoopedStandardModelV4(InitializedWrapper):
    def __init__(self, *, space_dim, fun_dim, out_dim, time_input, ref, unified_pos,
                 hidden_width, grid_height, grid_width, actual_M, heads, variant,
                 dropout, activation, ffn_ratio, temperature_mode, public_seed=0, **_):
        super().__init__()
        if activation!="gelu": raise ValueError("v4 requires GELU")
        self.H,self.W,self.ref=grid_height,grid_width,ref; self.unified_pos=bool(unified_pos)
        self.fun_dim,self.out_dim,self.space_dim=fun_dim,out_dim,space_dim
        self.Time_Input=bool(time_input); self.n_hidden=hidden_width
        if self.unified_pos:
            self.register_buffer("pos",native._unified_positions(grid_height,grid_width,ref),persistent=False)
        channels=fun_dim+(ref*ref if self.unified_pos else space_dim)
        self.preprocess=native.PointwiseMLP(channels,hidden_width*2,hidden_width)
        if self.Time_Input:
            self.time_fc=nn.Sequential(nn.Linear(hidden_width,hidden_width),nn.SiLU(),nn.Linear(hidden_width,hidden_width))
        self.loop=V4LoopCore(hidden=hidden_width,heads=heads,rank=actual_M,variant=variant,
            dropout=dropout,H=grid_height,W=grid_width,out_dim=out_dim,ffn_ratio=ffn_ratio,
            temperature_mode=temperature_mode,public_seed=public_seed)
        self.final_norm=nn.LayerNorm(hidden_width)
        self.head=nn.Linear(hidden_width,out_dim)
        self.apply(native.initialize_release_weights)
        self.placeholder=nn.Parameter(torch.rand(hidden_width,dtype=torch.float32)/hidden_width)
        self.loop.install_temperature_predictors()
        self._initialization_complete = True
        self.architecture="resmlp_dual_temp_v4"

    def forward(self,x,fx,T=None):
        if not isinstance(x,torch.Tensor) or x.ndim!=3 or not x.is_floating_point(): raise ValueError("x must be floating [B,N,space_dim]")
        B,N,C=x.shape
        if C!=self.space_dim or B<1 or N<1: raise ValueError("x shape mismatch")
        if self.loop.blocks[0].Attn.variant in ("conv","conv_temp") and N!=self.H*self.W:
            raise ValueError("structured v4 input requires N=H*W")
        if fx is None:
            if self.fun_dim: raise ValueError("fx required")
            if self.unified_pos:
                if N==self.H*self.W:
                    positions=self.pos.repeat(B,1,1,1).reshape(B,N,self.ref*self.ref)
                elif self.space_dim==self.ref*self.ref:
                    positions=x
                else:
                    raise ValueError("unified position requires native grid N or precomputed ref features")
            else: positions=x
            h=self.preprocess(positions)+self.placeholder[None,None,:]
        else:
            if tuple(fx.shape)!=(B,N,self.fun_dim) or fx.device!=x.device or fx.dtype!=x.dtype: raise ValueError("fx shape/device/dtype mismatch")
            if self.unified_pos:
                if N==self.H*self.W: positions=self.pos.repeat(B,1,1,1).reshape(B,N,self.ref*self.ref)
                elif self.space_dim==self.ref*self.ref: positions=x
                else: raise ValueError("unified position requires native grid N or precomputed ref features")
            else: positions=x
            h=self.preprocess(torch.cat((positions,fx),dim=-1))
        if T is not None:
            if not self.Time_Input or tuple(T.shape)!=(B,1): raise ValueError("T requires Time_Input and [B,1]")
            time=native.timestep_embedding(T,self.n_hidden).repeat(1,N,1)
            h=h+self.time_fc(time.to(dtype=self.time_fc[0].weight.dtype))
        return self.head(self.final_norm(self.loop(h)))
