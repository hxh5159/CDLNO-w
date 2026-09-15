"""Strict behavior/layout check inside already trusted industrial load boundaries."""
import torch
from .metadata import KCDNOMetadataMismatch

# Attributes affecting forward but absent from a state_dict. No learned tensor
# is compared with its initialization; these checks never modify the loaded model.
_BEHAVIOR = ('dim','heads','head_dim','num_latents','kernel_rank','eps','normalized_shape',
             'dropout','output_dropout','p','approximate','in_features','out_features',
             'in_channels','out_channels','kernel_size','stride','padding','dilation','groups',
             'padding_mode','structured','grid_shape','front_latent_mode')


def validate_whole_model(actual, expected):
    left,right=dict(actual.named_modules()),dict(expected.named_modules())
    if left.keys()!=right.keys():raise KCDNOMetadataMismatch('whole-model module paths differ')
    for path,wanted in right.items():
        saved=left[path]
        if type(saved) is not type(wanted):raise KCDNOMetadataMismatch('whole-model class mismatch: '+path)
        for name in _BEHAVIOR:
            if hasattr(wanted,name) and getattr(saved,name,None)!=getattr(wanted,name):
                raise KCDNOMetadataMismatch(f'whole-model behavior mismatch: {path}.{name}')
    # reference/pos are fixed, nonpersistent buffers; strict state_dict alone
    # cannot detect a stale or mutated reference grid in a pickle.
    a,b=dict(actual.named_buffers()),dict(expected.named_buffers())
    if a.keys()!=b.keys():raise KCDNOMetadataMismatch('whole-model buffer paths differ')
    for name,value in b.items():
        if a[name].shape!=value.shape or not torch.equal(a[name],value.to(a[name].device,a[name].dtype)):
            raise KCDNOMetadataMismatch('whole-model fixed buffer mismatch: '+name)
    expected.load_state_dict(actual.state_dict(),strict=True)
