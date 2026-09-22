"""V3 full-block sharing, with the frozen V1 point residual equations.

No stem, task input handling, training loop, file checkpoint or global RNG seed.
The owner constructs this core, completes its sole native tree initialization,
then calls install_features(). Only newly owned recurrent attention instances
change Python dispatch class; no module/parameter is cloned or reinitialized.
"""
from collections.abc import Mapping

import torch
from torch import nn

from linearno_loop.v3.config import validate_config
from linearno_loop.v3.contracts import TOPOLOGY_FIELDS, require_equal
from cdlno.linearno.attention import LinearNOAttention
from cdlno.linearno_loop.attnres import PointDepthAttnRes
from cdlno.linearno_loop.body import LinearNOBlockBody
from cdlno.linearno_loop.core import LinearNOLoopCore, PhysicalBlock
from .attention import V3LinearNOAttention


class VisitBody(LinearNOBlockBody):
    """Non-owning view: unchanged LN2/MLP and explicit current operator round."""
    __slots__ = ('round_index',)

    def __init__(self, block, round_index):
        super().__init__(block)
        self.round_index = round_index

    def operator(self, x):
        return self.block.Attn(self.block.ln_1(x), round_index=self.round_index)


class RecurrentPhysicalBlock(PhysicalBlock):
    def forward(self, x, *, round_index, scale):
        return VisitBody(self.block, round_index).scaled(x, scale)


class LinearNOLoopCoreV3(LinearNOLoopCore):
    """Own exactly P+C+S complete native blocks; core FFNs are also shared.

    Config is the validated LAA1 V3 document. block_factory(last_layer=bool)
    must return fresh native blocks matching its effective dimensions/variant.
    Prefix/suffix retain their original classes and native residual/head path.
    V1's properties, ownership checks and RB dtype helper are inherited intact.
    """

    def __init__(self, *, config, block_factory):
        checked = validate_config(config)  # Fail before calling a tensor factory.
        nn.Module.__init__(self)
        self._v3_config = checked
        self._features_ready = False
        spec = checked['loop_spec']
        for key in TOPOLOGY_FIELDS:
            setattr(self, key, spec[key])
        self.residual_mode = spec['residual_mode']
        P,C,R,S = (spec[k] for k in TOPOLOGY_FIELDS)
        self.prefix = nn.ModuleList([PhysicalBlock(block_factory(last_layer=False)) for _ in range(P)])
        self.core = nn.ModuleList([RecurrentPhysicalBlock(block_factory(last_layer=False)) for _ in range(C)])
        self.suffix = nn.ModuleList([PhysicalBlock(block_factory(last_layer=i==S-1)) for i in range(S)])
        H = spec['hidden_width']
        if self.residual_mode == 'rb_attnres':
            self.rb_receivers = nn.ModuleList([
                nn.ModuleList([PointDepthAttnRes(H) for _ in range(2*C)]) for _ in range(R)])
            self.rb_output = PointDepthAttnRes(H)
        elif self.residual_mode == 'lb_attnres_1_over_r':
            self.lb_boundaries = nn.ModuleList([PointDepthAttnRes(H) for _ in range(R-1)])
            self.lb_output = PointDepthAttnRes(H)
        # Validate complete fresh native tree before changing dispatch in core.
        self.validate_structure(require_features=False)
        for physical in self.core:
            if type(physical.block.Attn).forward is not LinearNOAttention.forward:
                raise ValueError('block_factory must supply fresh native attention')
            physical.block.Attn.__class__ = V3LinearNOAttention

    def install_features(self):
        """Install after the owner's release initialization; consumes no RNG.

        Per-position seed = (saved feature seed + physical index) modulo 2**63.
        Disabled features do not consume draws or register modules. There is no
        feature reinstallation or second whole-tree apply in this class.
        """
        if self._features_ready:
            raise RuntimeError('V3 core features already installed')
        self.validate_structure(require_features=False)
        spec = self._v3_config['loop_spec']; seeds = spec['initialization']
        for i,physical in enumerate(self.core):
            physical.block.Attn.install_features(latent_enabled=spec['latent_enabled'],
                latent_width=spec['latent_width'], adapter_mode=spec['adapter_mode'],
                adapter_rank=spec['adapter_rank'], adapter_alpha=spec['adapter_alpha'],
                latent_seed=(seeds['latent_seed']+i) % 2**63,
                adapter_seed=(seeds['adapter_seed']+i) % 2**63)
        self._features_ready = True
        self.validate_structure()

    def validate_structure(self, *, require_features=None):
        super().validate_structure()
        registered=list(self.named_parameters(remove_duplicate=False))
        if len(registered)!=len({id(p) for _,p in registered}):
            raise ValueError('each physical block/feature/router must have distinct parameter ownership')
        spec = self._v3_config['loop_spec']; model = self._v3_config['profile_spec']['values']['model']
        for key in (*TOPOLOGY_FIELDS, 'residual_mode'):
            require_equal(spec[key], getattr(self,key), 'core.'+key)
        require_features = self._features_ready if require_features is None else require_features
        if require_features and not self._features_ready:
            raise ValueError('install_features must follow common initialization before forward/load')
        for physical in (*self.prefix,*self.core,*self.suffix):
            block=physical.block; a=block.Attn; recurrent=physical in self.core
            for key,value in (('dim',spec['hidden_width']),('heads',spec['heads']),
                              ('dim_head',spec['head_dim']),('rank',spec['actual_M']),('variant',spec['variant'])):
                require_equal(value,getattr(a,key,None),'block.Attn.'+key)
            if not isinstance(a,LinearNOAttention):
                raise TypeError('V3 core requires native LinearNOAttention blocks')
            if type(a).forward not in (LinearNOAttention.forward,V3LinearNOAttention.forward):
                raise ValueError('custom attention forward cannot be promoted as a native block')
            if not recurrent and isinstance(a,V3LinearNOAttention):
                raise ValueError('prefix/suffix require native attention')
            if spec['variant'] in ('conv','conv_temp'):
                require_equal(spec['grid_height'],a.H,'block.grid_height')
                require_equal(spec['grid_width'],a.W,'block.grid_width')
            for norm in (block.ln_1,block.ln_2):
                if not isinstance(norm,nn.LayerNorm) or tuple(norm.normalized_shape)!=(spec['hidden_width'],) or norm.eps!=1e-5:
                    raise ValueError('native block LayerNorm contract mismatch')
            if (block.mlp.linear_pre[0].in_features!=spec['hidden_width'] or
                block.mlp.linear_pre[0].out_features!=spec['hidden_width']*spec['ffn_ratio'] or
                block.mlp.linear_post.out_features!=spec['hidden_width']):
                raise ValueError('point FFN dimensions disagree with V3 config')
            require_equal(float(model['dropout']),float(a.to_out[-1].p),'block.dropout')
            if block.last_layer and block.mlp2.out_features!=model['out_dim']:
                raise ValueError('suffix output head disagrees with config')
            for name,enabled in (('latent_processor',recurrent and spec['latent_enabled']),
                                 ('adapter',recurrent and spec['adapter_mode']!='none')):
                if require_features or not recurrent:
                    if (name in a._modules)!=enabled:
                        raise ValueError('feature registration mismatch: '+name)
            if recurrent and require_features:
                if not isinstance(a,V3LinearNOAttention) or not getattr(a,'_features_installed',False):
                    raise ValueError('core attention requires installed V3 round dispatch')
                if spec['latent_enabled']:
                    require_equal(spec['hidden_width'],a.latent_processor.hidden,'latent.hidden')
                    require_equal(spec['latent_width'],a.latent_processor.inner_width,'latent.width')
                if spec['adapter_mode']!='none':
                    for key,value in (('rank',spec['adapter_rank']),('alpha',spec['adapter_alpha']),
                                      ('latent_tokens',spec['actual_M']),('head_dim',spec['head_dim'])):
                        require_equal(value,getattr(a.adapter,key),'adapter.'+key)

    def _rb_forward(self, anchor):
        completed=[anchor]
        for round_index,receivers in enumerate(self.rb_receivers):
            partial=None
            for index,physical in enumerate(self.core):
                body=VisitBody(physical.block,round_index)
                for offset,branch in enumerate((body.operator,body.mlp)):
                    sources=tuple(completed) if partial is None else (*completed,partial)
                    h=self._rb_receive(receivers[2*index+offset],sources,anchor)
                    raw=branch(h)
                    partial=raw if partial is None else partial+raw
            completed.append(partial)
        return self._rb_receive(self.rb_output,tuple(completed),anchor)

    def _lb_forward(self, anchor):
        deltas=[];x=anchor
        for r in range(self.loop_repeats):
            entry=x
            for block in self.core:x=block(x,round_index=r,scale=1/self.loop_repeats)
            deltas.append(x-entry)
            sources=(anchor,*deltas)
            x=self.lb_boundaries[r](sources) if r<self.loop_repeats-1 else self.lb_output(sources)
        return x

    def forward(self, x):
        self.validate_structure(require_features=True)
        for block in self.prefix:x=block(x)
        if self.residual_mode=='sr_1_over_r':
            for r in range(self.loop_repeats):
                for block in self.core:x=block(x,round_index=r,scale=1/self.loop_repeats)
        elif self.residual_mode=='rb_attnres':x=self._rb_forward(x)
        else:x=self._lb_forward(x)
        for index,block in enumerate(self.suffix):x=block(x,finalize=index==self.suffix_blocks-1)
        return x

    def load_configured_state_dict(self, state_dict, *, saved_config, strict=True):
        """Core-only in-memory replay guard, not a task/file checkpoint backend.

        Check metadata first (including same-shape R/alpha/task conflicts), then
        all keys/shapes before applying any weight. File/archive loading remains
        the responsibility of the future V3 wrapper/checkpoint integration.
        """
        if strict is not True:
            raise ValueError('V3 core replay requires strict=True')
        checked=validate_config(saved_config)
        require_equal(self._v3_config,checked,'saved_config')
        self.validate_structure(require_features=True)
        if not isinstance(state_dict,Mapping) or any(type(k) is not str for k in state_dict):
            raise ValueError('state_dict must map exact string keys to tensors')
        expected=self.state_dict()
        require_equal(sorted(expected),sorted(state_dict),'state_dict.keys')
        for key,value in state_dict.items():
            if not isinstance(value,torch.Tensor) or value.shape!=expected[key].shape:
                raise ValueError('state_dict shape mismatch: '+key)
        return super().load_state_dict(state_dict,strict=True)
