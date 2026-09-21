"""Opt-in v2 loop observations; disabled mode installs no hooks.

Only detached scalar summaries survive a callback. No forward tensor, latent
context, history source, or autograd graph is retained across calls.
"""

import math

import torch


def _stats(value):
    value=value.detach().float()
    return dict(norm=float(value.norm()),rms=float(value.square().mean().sqrt()),
                nan=int(torch.isnan(value).sum()),inf=int(torch.isinf(value).sum()))


class V2LoopDiagnostics:
    def __init__(self,model,*,enabled=False):
        if type(enabled) is not bool:raise TypeError('enabled must be bool')
        self.model=model;self.enabled=enabled;self.handles=[];self.records=[]
        self._current=None;self._operator_counts=[];self._latent_counts=[]

    def __enter__(self):
        if not self.enabled:return self
        loop=self.model.loop
        if not hasattr(loop,'core_operators'):raise ValueError('v2 diagnostics require a v2 loop core')
        self._operator_counts=[0]*loop.recurrent_core_blocks
        self._latent_counts=[0]*loop.recurrent_core_blocks
        def start(module,args):
            self._operator_counts[:]=[0]*len(self._operator_counts)
            self._latent_counts[:]=[0]*len(self._latent_counts)
            self._current=dict(core_ffn_mode=loop.core_ffn_mode,residual_mode=loop.residual_mode,
                               operators=[],point_ffns=[],latent=[],completed=False)
        def finish(module,args,output):
            if self._current is not None:
                self._current['completed']=output is not None
                if output is not None:self._current['output']=_stats(output)
                self.records.append(self._current)
            self._current=None
        self.handles.extend([self.model.register_forward_pre_hook(start),
                             self.model.register_forward_hook(finish,always_call=True)])
        for position,operator in enumerate(loop.core_operators):
            def operator_visit(module,args,output,p=position):
                round_index=self._operator_counts[p];self._operator_counts[p]+=1
                if self._current is not None:
                    self._current['operators'].append(dict(position=p,round=round_index,
                                                           output=_stats(output)))
            self.handles.append(operator.register_forward_hook(operator_visit))
        for position,row in enumerate(loop.core_ffns):
            for round_index,point_ffn in enumerate(row):
                def point_visit(module,args,output,p=position,r=round_index):
                    if self._current is not None:
                        self._current['point_ffns'].append(dict(position=p,round=r,
                                                                output=_stats(output)))
                self.handles.append(point_ffn.register_forward_hook(point_visit))
        for position,latent in enumerate(getattr(loop,'latent_ffns',())):
            def latent_visit(module,args,output,p=position):
                round_index=self._latent_counts[p];self._latent_counts[p]+=1
                if self._current is not None:
                    update=output.detach()-args[0].detach()
                    self._current['latent'].append(dict(position=p,round=round_index,
                                                        context=_stats(args[0]),update=_stats(update)))
            self.handles.append(latent.register_forward_hook(latent_visit))
        return self

    def gradients(self):
        if not self.enabled:return None
        loop=self.model.loop;rows=[]
        groups=(('shared_operator',loop.core_operators),
                ('point_ffn',(module for row in loop.core_ffns for module in row)),
                ('latent',getattr(loop,'latent_ffns',())))
        for group,modules in groups:
            for index,module in enumerate(modules):
                gradients=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
                rows.append(dict(group=group,index=index,
                    gradient_norm=math.sqrt(sum(float(g.square().sum()) for g in gradients)),
                    nan=sum(int(torch.isnan(g).sum()) for g in gradients),
                    inf=sum(int(torch.isinf(g).sum()) for g in gradients),
                    missing=[name for name,p in module.named_parameters() if p.grad is None]))
        return rows

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[];self._current=None;self._operator_counts=[];self._latent_counts=[]
