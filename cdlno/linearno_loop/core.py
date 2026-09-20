"""Shared physical LinearNO blocks with three exclusive point residual modes."""
from torch import nn

from linearno_loop.contracts import integer, RESIDUAL_MODES
from .body import LinearNOBlockBody
from .attnres import PointDepthAttnRes


def topology(P,C,R,S):
    for name,value,minimum in (('prefix_blocks',P,0),('recurrent_core_blocks',C,1),
                                ('loop_repeats',R,1),('suffix_blocks',S,1)):
        integer(value,name,minimum)
    return P+C+S,P+C*R+S


def require_implemented(mode):
    if type(mode) is not str or mode not in RESIDUAL_MODES:
        raise ValueError(f'residual_mode must be one of {RESIDUAL_MODES}')


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
        require_implemented(residual_mode)
        P,C,R,S=prefix_blocks,recurrent_core_blocks,loop_repeats,suffix_blocks
        topology(P,C,R,S)
        self.prefix_blocks=P;self.recurrent_core_blocks=C;self.loop_repeats=R;self.suffix_blocks=S
        self.residual_mode=residual_mode
        self.prefix=nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(P)])
        self.core=nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(C)])
        self.suffix=nn.ModuleList([PhysicalBlock(block_factory(last_layer=i==S-1)) for i in range(S)])
        if residual_mode=='rb_attnres':
            hidden=self.core[0].block.Attn.dim
            # Zeros/ones consume no RNG; native whole-tree initialization does
            # not touch these parameters. Backbone/placeholder draws stay SR's.
            self.rb_receivers=nn.ModuleList([
                nn.ModuleList([PointDepthAttnRes(hidden) for _ in range(2*C)]) for _ in range(R)])
            self.rb_output=PointDepthAttnRes(hidden)
        elif residual_mode=='lb_attnres_1_over_r':
            hidden=self.core[0].block.Attn.dim
            self.lb_boundaries=nn.ModuleList([PointDepthAttnRes(hidden) for _ in range(R-1)])
            self.lb_output=PointDepthAttnRes(hidden)
        self.validate_structure()

    @property
    def unique_depth(self):return self.prefix_blocks+self.recurrent_core_blocks+self.suffix_blocks

    @property
    def executed_depth(self):return self.prefix_blocks+self.recurrent_core_blocks*self.loop_repeats+self.suffix_blocks

    def validate_structure(self):
        """Scalar/tree checks only, including malicious reused factory modules."""
        require_implemented(self.residual_mode)
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
        if self.residual_mode=='rb_attnres':
            if (not hasattr(self,'rb_receivers') or len(self.rb_receivers)!=self.loop_repeats or
                    any(len(row)!=2*self.recurrent_core_blocks for row in self.rb_receivers) or
                    not hasattr(self,'rb_output')):
                raise ValueError('RB requires 2*C*R sublayer receivers and one output receiver')
            for receiver in [*(v for row in self.rb_receivers for v in row),self.rb_output]:
                if not isinstance(receiver,PointDepthAttnRes) or receiver.hidden!=signature[0]:
                    raise ValueError('RB receivers must act on the point hidden dimension')
                for parameter in receiver.parameters():
                    if id(parameter) in seen:raise ValueError('RB receivers must be independent by round/sublayer/output')
                    seen.add(id(parameter))
        elif hasattr(self,'rb_receivers') or hasattr(self,'rb_output'):
            raise ValueError('SR must not register RB receiver parameters; neither may LB')
        if self.residual_mode=='lb_attnres_1_over_r':
            if (not hasattr(self,'lb_boundaries') or len(self.lb_boundaries)!=self.loop_repeats-1 or
                    not hasattr(self,'lb_output')):
                raise ValueError('LB requires R-1 boundary receivers and one output receiver')
            for receiver in [*self.lb_boundaries,self.lb_output]:
                if not isinstance(receiver,PointDepthAttnRes) or receiver.hidden!=signature[0]:
                    raise ValueError('LB receivers must act on the point hidden dimension')
                for parameter in receiver.parameters():
                    if id(parameter) in seen:raise ValueError('LB boundary/output receivers must be independent')
                    seen.add(id(parameter))
        elif hasattr(self,'lb_boundaries') or hasattr(self,'lb_output'):
            raise ValueError('only LB may register LB receiver parameters')

    @staticmethod
    def _rb_receive(receiver, sources, anchor):
        # The core-entry anchor owns RB's point-state dtype. Autocast raw
        # branches may differ; convert only this call's references, leaving
        # the authoritative raw sums untouched and their autograd edges live.
        canonical_dtype = anchor.dtype
        local_sources = tuple(source if source.dtype == canonical_dtype else
                              source.to(dtype=canonical_dtype) for source in sources)
        return receiver(local_sources)

    def _rb_forward(self,anchor):
        # Authoritative sources are completed *point* raw sums and anchor.
        # All lists/tensors are forward-local; tuple snapshots prevent mutation
        # of previously passed source lists. No ordinary residual or 1/R here.
        completed=[anchor]
        for receivers in self.rb_receivers:
            partial=None
            for index,physical in enumerate(self.core):
                body=LinearNOBlockBody(physical.block)
                for offset,branch in enumerate((body.operator,body.mlp)):
                    sources=tuple(completed) if partial is None else (*completed,partial)
                    h=self._rb_receive(receivers[2*index+offset],sources,anchor)
                    raw=branch(h)
                    partial=raw if partial is None else partial+raw
            completed.append(partial)
        return self._rb_receive(self.rb_output,tuple(completed),anchor)

    def _lb_forward(self,anchor):
        # Store true end-minus-entry updates, not a recomputed branch sum.
        # Sources and the current round entry are live forward-local tensors.
        deltas=[]
        x=anchor
        for r in range(self.loop_repeats):
            entry=x
            for block in self.core:x=block(x,scale=1/self.loop_repeats)
            deltas.append(x-entry)
            sources=(anchor,*deltas)
            x=self.lb_boundaries[r](sources) if r<self.loop_repeats-1 else self.lb_output(sources)
        return x

    def forward(self,x):
        self.validate_structure()
        for block in self.prefix:x=block(x)
        if self.residual_mode=='sr_1_over_r':
            for _ in range(self.loop_repeats):
                for block in self.core:x=block(x,scale=1/self.loop_repeats)
        elif self.residual_mode=='rb_attnres':
            x=self._rb_forward(x)
        else:
            x=self._lb_forward(x)
        for index,block in enumerate(self.suffix):x=block(x,finalize=index==self.suffix_blocks-1)
        return x
