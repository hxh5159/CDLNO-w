"""Shared physical LinearNO blocks with the frozen SR 1/R execution rule."""
from torch import nn

from linearno_loop.contracts import integer
from .body import LinearNOBlockBody


def topology(P,C,R,S):
    for name,value,minimum in (('prefix_blocks',P,0),('recurrent_core_blocks',C,1),
                                ('loop_repeats',R,1),('suffix_blocks',S,1)):
        integer(value,name,minimum)
    return P+C+S,P+C*R+S


def require_sr(mode):
    if mode != 'sr_1_over_r' or type(mode) is not str:
        raise ValueError('LL3 implements sr_1_over_r only; RB/LB are not implemented')


class PhysicalBlock(nn.Module):
    """Register exactly one original block; expose actual visit hooks.

    An ephemeral LL2 view executes its original submodules. No parameter
    cloning, initialization, per-round module, or retained activation.
    """
    def __init__(self,block):
        super().__init__()
        LinearNOBlockBody(block)  # Validate without invoking or initializing.
        self.block=block

    def forward(self,x,*,scale=None,finalize=False):
        body=LinearNOBlockBody(self.block)
        x=body.native(x) if scale is None else body.scaled(x,scale)
        return body.finalize(x) if finalize else x


class LinearNOLoopCore(nn.Module):
    """Own P+C+S physical blocks; no stem or construction-time weight reset.

    block_factory(last_layer=bool) must create one fresh pure LinearNO block.
    The enclosing wrapper performs the sole native apply after its complete
    stem/time/core tree exists. Only suffix[-1] owns the native final head.
    """
    def __init__(self,*,prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks,
                 residual_mode,block_factory):
        super().__init__()
        require_sr(residual_mode)
        P,C,R,S=prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks
        topology(P,C,R,S)
        self.prefix_blocks=P;self.recurrent_core_blocks=C;self.loop_repeats=R;self.suffix_blocks=S
        self.residual_mode=residual_mode
        self.prefix=nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(P)])
        self.core=nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(C)])
        self.suffix=nn.ModuleList([PhysicalBlock(block_factory(last_layer=i==S-1)) for i in range(S)])
        self.validate_structure()

    @property
    def unique_depth(self):return self.prefix_blocks+self.recurrent_core_blocks+self.suffix_blocks

    @property
    def executed_depth(self):return self.prefix_blocks+self.recurrent_core_blocks*self.loop_repeats+self.suffix_blocks

    def validate_structure(self):
        """Scalar/tree checks only, including malicious reused factory modules."""
        require_sr(self.residual_mode)
        topology(self.prefix_blocks,self.recurrent_core_blocks,self.loop_repeats,self.suffix_blocks)
        for group,count in ((self.prefix,self.prefix_blocks),(self.core,self.recurrent_core_blocks),(self.suffix,self.suffix_blocks)):
            if len(group)!=count:raise ValueError('physical block count disagrees with topology')
        seen=set();signature=None
        for block in (*self.prefix,*self.core,*self.suffix):
            raw=block.block
            expected=block is self.suffix[-1]
            if raw.last_layer is not expected or (not expected and any(hasattr(raw,k) for k in ('ln_3','mlp2'))):
                raise ValueError('only final suffix may own ln_3/mlp2')
            attention=raw.Attn
            current=tuple(getattr(attention,k,None) for k in ('dim','heads','dim_head','rank','variant'))
            if None in current or signature is not None and current!=signature:
                raise ValueError('all physical blocks require matching LinearNO width/heads/rank/variant')
            signature=current
            for parameter in raw.parameters():
                if id(parameter) in seen:raise ValueError('physical prefix/core/suffix blocks must be distinct')
                seen.add(id(parameter))

    def forward(self,x):
        self.validate_structure()
        for block in self.prefix:x=block(x)
        for _ in range(self.loop_repeats):
            for block in self.core:x=block(x,scale=1/self.loop_repeats)
        for index,block in enumerate(self.suffix):x=block(x,finalize=index==self.suffix_blocks-1)
        return x
