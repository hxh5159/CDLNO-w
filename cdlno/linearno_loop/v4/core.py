"""Task-independent v4 eight-block core."""
import math
import torch
from torch import nn
from cdlno.linearno.attention import initialize_release_weights
from .attention import V4LinearNOAttention
from .resmlp import ResidualMLP

SCHEDULE=("first","A","B","C","A","B","C","last")

class V4Block(nn.Module):
    def __init__(self, hidden, heads, rank, variant, dropout, H, W):
        super().__init__()
        self.ln_1=nn.LayerNorm(hidden)
        attention_kwargs=dict(heads=heads,dim_head=hidden//heads,rank=rank,variant=variant,dropout=dropout)
        if variant in ("conv","conv_temp"):
            attention_kwargs.update(H=H,W=W)
        self.Attn=V4LinearNOAttention(hidden,**attention_kwargs)
        self.ln_2=nn.LayerNorm(hidden)

class V4LoopCore(nn.Module):
    """Eight independent operators with five non-duplicated RMLP owners."""
    def __init__(self, *, hidden, heads, rank, variant, dropout, H, W, out_dim,
                 ffn_ratio, temperature_mode, public_seed):
        super().__init__()
        self.temperature_mode=temperature_mode; self.public_seed=public_seed
        self.prefix_blocks,self.recurrent_core_blocks,self.loop_repeats,self.suffix_blocks=1,3,2,1
        self.blocks=nn.ModuleList([V4Block(hidden,heads,rank,variant,dropout,H,W) for _ in range(8)])
        for block in self.blocks:block.Attn._planned_temperature_mode=temperature_mode
        self.rmlp=nn.ModuleDict({name: ResidualMLP(hidden,ffn_ratio,2 if name in ("first","last") else 3)
                                 for name in ("first","A","B","C","last")})
        self.schedule=SCHEDULE
        self.scale=1.0/math.sqrt(2.0)

    def install_temperature_predictors(self):
        for index,block in enumerate(self.blocks):
            block.Attn.install_temperature_predictors(mode=self.temperature_mode,
                public_seed=self.public_seed,logical_depth=index)

    def forward(self,x):
        for index,(block,owner) in enumerate(zip(self.blocks,self.schedule)):
            scale=1.0 if index in (0,7) else self.scale
            u=x + scale*block.Attn(block.ln_1(x),logical_depth=index)
            x=u + scale*self.rmlp[owner](block.ln_2(u))
        return x

    def parameter_ownership(self):
        return {name: sum(p.numel() for p in module.parameters()) for name,module in self.rmlp.items()}
