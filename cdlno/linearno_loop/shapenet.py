"""Synthetic ShapeNet loop wrapper; preserve native tuple/geometry semantics."""
import torch
from torch import nn

from cdlno.linearno.shapenet import ShapeNetLinearNO, ShapeNetBlock, PointMLP, initialize_release_weights
from .construction import LoopForwardView, validate_arguments
from .core import LinearNOLoopCore


class LoopedShapeNetModel(LoopForwardView,ShapeNetLinearNO):
    def __init__(self,*,space_dim,n_hidden,n_head,dropout,act,mlp_ratio,fun_dim,out_dim,
                 ref,unified_pos,prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks,
                 residual_mode,linearno_rank,Time_Input,H,W,isregular):
        validate_arguments('car',{k:v for k,v in locals().items() if k!='self'})
        nn.Module.__init__(self)
        self.H=H;self.W=W;self.ref=ref;self.unified_pos=unified_pos;self.Time_Input=Time_Input
        if unified_pos:self.register_buffer('pos',self.get_grid(),persistent=False)
        self.preprocess=PointMLP(ref*ref if unified_pos else space_dim+fun_dim,n_hidden*2,n_hidden)
        self.n_hidden=n_hidden;self.space_dim=space_dim
        if Time_Input:self.time_fc=nn.Sequential(nn.Linear(n_hidden,n_hidden),nn.SiLU(),nn.Linear(n_hidden,n_hidden))
        self.loop=LinearNOLoopCore(prefix_blocks=prefix_blocks,recurrent_core_blocks=recurrent_core_blocks,
            loop_repeats=loop_repeats,suffix_blocks=suffix_blocks,residual_mode=residual_mode,
            block_factory=lambda *,last_layer:ShapeNetBlock(n_hidden,n_head,linearno_rank,mlp_ratio,dropout,out_dim,last_layer))
        self.apply(initialize_release_weights)
        self.placeholder=nn.Parameter((1/n_hidden)*torch.rand(n_hidden,dtype=torch.float))
