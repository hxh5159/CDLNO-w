"""Synthetic Standard loop wrapper, preserving the native forward contract."""
import importlib

import torch
from torch import nn

from .construction import LoopForwardView, validate_arguments
from .core import LinearNOLoopCore

# Absolute namespace package import: no sys.path mutation or generic 'model'
# module collision across the three benchmark working directories.
native=importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO')


class LoopedStandardModel(LoopForwardView,native.Model):
    def __init__(self,*,space_dim,n_hidden,n_head,dropout,act,mlp_ratio,fun_dim,out_dim,
                 ref,unified_pos,prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks,
                 residual_mode,linearno_rank,Time_Input,H,W,linearno_variant):
        validate_arguments('standard',{k:v for k,v in locals().items() if k!='self'})
        nn.Module.__init__(self)
        self.H=H;self.W=W;self.ref=ref;self.unified_pos=unified_pos
        self.fun_dim=fun_dim;self.out_dim=out_dim;self.linearno_variant=linearno_variant;self.linearno_rank=linearno_rank
        if unified_pos:self.register_buffer('pos',native._unified_positions(H,W,ref),persistent=False)
        self.preprocess=native.PointwiseMLP(fun_dim+(ref*ref if unified_pos else space_dim),n_hidden*2,n_hidden)
        self.Time_Input=Time_Input;self.n_hidden=n_hidden;self.space_dim=space_dim
        if Time_Input:self.time_fc=nn.Sequential(nn.Linear(n_hidden,n_hidden),nn.SiLU(),nn.Linear(n_hidden,n_hidden))
        self.loop=LinearNOLoopCore(prefix_blocks=prefix_blocks,recurrent_core_blocks=recurrent_core_blocks,
            loop_repeats=loop_repeats,suffix_blocks=suffix_blocks,residual_mode=residual_mode,
            block_factory=lambda *,last_layer:native.LinearNOBlock(hidden=n_hidden,heads=n_head,rank=linearno_rank,
                variant=linearno_variant,dropout=dropout,mlp_ratio=mlp_ratio,H=H,W=W,last_layer=last_layer,out_dim=out_dim))
        self.apply(native.initialize_release_weights)
        self.placeholder=nn.Parameter(torch.rand(n_hidden,dtype=torch.float32)*(1/n_hidden))
