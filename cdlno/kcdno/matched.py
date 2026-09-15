"""Trainable matched LRSA-full: the existing full block composed exactly L times."""
from torch import nn
from ..modules import LRSAFrontBlock, _check_tokens, _check_grid
from .matched_config import MatchedLRSAConfig


class MatchedLRSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        if type(config) is not MatchedLRSAConfig:raise TypeError('MatchedLRSAConfig required')
        config.validate(); self.config=config
        self.blocks=nn.ModuleList([
            LRSAFrontBlock(config.d,config.h,config.M,structured=config.point_module=='conv_ffn',
                           ffn_ratio=2.,dropout=0.,front_latent_mode='full') for _ in range(config.L)
        ])

    def forward(self,x,*,grid_shape=None):
        _check_tokens(x,self.config.d)
        for block in self.blocks:
            x,_=block(x,grid_shape=grid_shape)
        return x


def make_core(config):
    if config.family=='lrsa_matched':return MatchedLRSA(config)
    from .core import KCDNO
    return KCDNO(config)


def validate_core(core, config):
    """Validate trusted pickle layout before its strict weight/schema check."""
    from .core import KCDNO,KCDNOBlock
    expected=MatchedLRSA if config.family=='lrsa_matched' else KCDNO
    if type(core) is not expected or core.config!=config or len(core.blocks)!=config.L:
        raise ValueError('whole-model core configuration mismatch')
    for i,block in enumerate(core.blocks):
        if config.family=='lrsa_matched':
            if type(block) is not LRSAFrontBlock or block.front_latent_mode!='full':
                raise ValueError('matched LRSA requires full blocks')
        elif type(block) is not KCDNOBlock or block.layer_index!=i or block.config!=config:
            raise ValueError('KCDNO block configuration mismatch')
