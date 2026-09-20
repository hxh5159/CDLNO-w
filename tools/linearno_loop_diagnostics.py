"""Optional, external loop observations. No production model or old monitor edits.

Use ``with LoopDiagnostics(model, enabled=True) as observation`` explicitly.
Default disabled attaches no hooks, allocates no tensors and performs no CPU
synchronization. Enabled mode intentionally synchronizes for JSON summaries;
do NOT include it in performance measurements. Only detached observation
copies survive a sublayer, all are cleared on successful/exceptional forward.
No Q/K tensor, point history, or autograd graph is saved in returned records.
"""
import math
import torch


def stats(value):
    v=value.detach().float()
    return dict(norm=float(v.norm()),rms=float(v.square().mean().sqrt()),
                nan=int(torch.isnan(v).sum()),inf=int(torch.isinf(v).sum()))


class LoopDiagnostics:
    def __init__(self,model,*,enabled=False):
        if type(enabled) is not bool:raise TypeError('enabled must be bool')
        self.model=model;self.enabled=enabled;self.records=[];self.handles=[];self._reset()
    def _reset(self):
        self.pending={};self.previous_update=None;self.partial=None;self.visits=0;self.current=None
    def __enter__(self):
        if not self.enabled:return self
        if self.handles:raise RuntimeError('diagnostic context cannot be entered twice')
        if getattr(self.model,'family',None)!='linearno_loop':raise ValueError('requires linearno_loop model')
        loop=self.model.loop
        def start(m,a):
            self._reset();self.current=dict(index=len(self.records),residual_mode=loop.residual_mode,
                visits=[],routers=[],routing=[],round_deltas=[],completed=False)
        def finish(m,a,y):
            if self.current is not None:
                self.current['completed']=y is not None
                if y is not None:self.current['output']=stats(y)
                self.records.append(self.current)
            self._reset()
        self.handles.extend([self.model.register_forward_pre_hook(start),self.model.register_forward_hook(finish,always_call=True)])
        for index,physical in enumerate(loop.core):
            b=physical.block
            def entry(m,a,i=index):
                r=self.visits//loop.recurrent_core_blocks
                if i==0:self.partial=None
                self.pending=dict(round=r,physical_core_index=i,entry=a[0].detach(),branch_outputs=[])
            def raw(m,a,y,branch):
                if not self.pending:return
                self.pending['branch_outputs'].append(y.detach())
                if branch=='MLP':
                    item=self.pending;first,second=item['branch_outputs']
                    if loop.residual_mode=='rb_attnres':
                        update=first+second
                        self.partial=update if self.partial is None else self.partial+update
                        exit_state=self.partial
                        semantics='RB raw partial increment=u_operator+u_MLP; exit is round partial, not h+u'
                    else:
                        update=(first+second)/loop.loop_repeats
                        # ln_2 input is the actual state after operator residual.
                        exit_state=item['mlp_input']+second/loop.loop_repeats
                        semantics='SR/LB actual exit-entry; scaled branches'
                        update=exit_state-item['entry']
                    prev=self.previous_update
                    cosine=None if prev is None else float(torch.nn.functional.cosine_similarity(prev.flatten(),update.flatten(),dim=0))
                    row={k:item[k] for k in ('round','physical_core_index')}
                    row.update(logical_visit=self.visits,entry=stats(item['entry']),exit=stats(exit_state),
                        update=stats(update),adjacent_update_cosine=cosine,update_semantics=semantics,
                        operator_raw=stats(first),mlp_raw=stats(second))
                    self.current['visits'].append(row);self.previous_update=update;self.visits+=1;self.pending={}
            def mlp_entry(m,a):self.pending['mlp_input']=a[0].detach()
            self.handles.extend([b.ln_1.register_forward_pre_hook(entry),
                b.Attn.register_forward_hook(lambda m,a,y:raw(m,a,y,'operator')),
                b.ln_2.register_forward_pre_hook(mlp_entry),
                b.mlp.register_forward_hook(lambda m,a,y:raw(m,a,y,'MLP'))])
            for qk,projection in (('Q',b.Attn.to_q),('K',b.Attn.to_k)):
                def route(m,a,y,kind=qk,attention=b.Attn,i=index):
                    if self.current is None:return
                    logits=y.detach()
                    with torch.no_grad():
                        if attention.variant in ('temp','conv_temp'):
                            logits=logits/getattr(attention,'temperature_'+kind.lower()).detach().clamp(.01,1.)
                        elif attention.variant=='shapenet':
                            logits=logits/getattr(attention,'tempreature_'+kind.lower()).detach().clamp(.1,2.)
                        axis=-1 if kind=='Q' else -2;p=logits.float().softmax(axis)
                        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(axis)
                        row=dict(round=self.visits//loop.recurrent_core_blocks,physical_core_index=i,sublayer='operator',
                            kind=kind,axis=axis,shape=list(p.shape),mean_entropy=float(entropy.mean()),
                            max_sum_error=float((p.sum(axis)-1).abs().max()),**stats(p))
                    self.current['routing'].append(row)
                self.handles.append(projection.register_forward_hook(route))
        from cdlno.linearno_loop.attnres import PointDepthAttnRes
        for name,receiver in loop.named_modules():
            if not isinstance(receiver,PointDepthAttnRes):continue
            def router(m,a,y,name=name):
                sources=a[0]
                with torch.no_grad():
                    weights=m.source_weights(tuple(x.detach() for x in sources)).float()
                    e=-(weights*weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(0)
                    row=dict(receiver=name,sources=len(sources),source_weight_mean=weights.mean((1,2)).tolist(),
                        source_weight_min=weights.amin((1,2)).tolist(),source_weight_max=weights.amax((1,2)).tolist(),
                        mean_entropy=float(e.mean()),source_states=[stats(x) for x in sources],output=stats(y))
                    if name.startswith('rb_receivers.'):
                        _,r,s=name.split('.');row.update(round=int(r),physical_core_index=int(s)//2,sublayer='operator' if int(s)%2==0 else 'MLP')
                    else:row.update(round=len(sources)-2,physical_core_index=None,sublayer='output' if 'output' in name else 'boundary')
                    self.current['routers'].append(row)
                    if name=='lb_output':self.current['round_deltas']=[stats(x) for x in sources[1:]]
            self.handles.append(receiver.register_forward_hook(router))
        return self
    def gradients(self):
        """Call explicitly after backward; no parameter hook or gradient mutation."""
        if not self.enabled:return None
        result=[]
        for index,block in enumerate(self.model.loop.core):
            grads=[p.grad.detach().float() for p in block.parameters() if p.grad is not None]
            result.append(dict(physical_core_index=index,gradient_norm=math.sqrt(sum(float(g.square().sum()) for g in grads)),
                nan=sum(int(torch.isnan(g).sum()) for g in grads),inf=sum(int(torch.isinf(g).sum()) for g in grads),
                missing_gradients=[n for n,p in block.named_parameters() if p.grad is None]))
        return result
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[];self._reset()
