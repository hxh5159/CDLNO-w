"""Synthetic AirfRANS loop wrapper; reuse native Data/position/forward code."""
import numpy as np
import torch
from torch import nn

from cdlno.linearno.airfrans import AirfRANSLinearNO, AirfRANSBlock, PointMLP, initialize_release_weights
from .construction import LoopForwardView, validate_arguments
from .core import LinearNOLoopCore


class LoopedAirfRANSModel(LoopForwardView,AirfRANSLinearNO):
    def __init__(self,*,space_dim,n_hidden,n_head,dropout,act,mlp_ratio,fun_dim,out_dim,
                 ref,unified_pos,prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks,
                 residual_mode,linearno_rank,linear):
        validate_arguments('airfrans',{k:v for k,v in locals().items() if k!='self'})
        nn.Module.__init__(self)
        self.__name__='LoopedLinearNO';self.ref=ref;self.unified_pos=unified_pos
        self.preprocess=PointMLP(space_dim+(ref*ref if unified_pos else 0),n_hidden*2,n_hidden)
        self.n_hidden=n_hidden;self.space_dim=space_dim
        self.loop=LinearNOLoopCore(prefix_blocks=prefix_blocks,recurrent_core_blocks=recurrent_core_blocks,
            loop_repeats=loop_repeats,suffix_blocks=suffix_blocks,residual_mode=residual_mode,
            block_factory=lambda *,last_layer:AirfRANSBlock(n_hidden,n_head,linearno_rank,mlp_ratio,dropout,out_dim,last_layer))
        self.apply(initialize_release_weights)
        self.placeholder=nn.Parameter((1/n_hidden)*torch.rand(n_hidden,dtype=torch.float))
        gx=torch.tensor(np.linspace(-2,4,ref),dtype=torch.float)
        gy=torch.tensor(np.linspace(-1.5,1.5,ref),dtype=torch.float)
        self.register_buffer('reference',torch.stack(torch.meshgrid(gx,gy,indexing='ij'),-1).reshape(ref*ref,2),persistent=False)
