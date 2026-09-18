"""Opt-in bounded LinearNO A/K diagnostics; no model attributes or saved hooks.

A scoped process-local call interceptor consumes existing per-call observers.
Do not combine with torch.compile, threaded forwards or another monkey patch.
No diagnostics run unless installed explicitly. Only detached sampled Q/K and
scalar statistics survive a block; no forward graph is saved by this monitor.
"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import weakref
import torch
from torch import nn
from monitor.kernels import pairwise_kernel_similarity


def entropy(p, dim=-1):
    p = p.detach().float()
    return -(p * p.clamp_min(1e-30).log()).sum(dim).mean().item()


def stats(t):
    t = t.detach().float()
    return dict(shape=list(t.shape), mean=t.mean().item(), std=t.std(unbiased=False).item(),
                norm=t.norm().item(), finite=bool(torch.isfinite(t).all()))


class HistoryDiagnostics(contextlib.AbstractContextManager):
    def __init__(self, output, *, every=100, max_snapshots=8, max_points=256, plots=True):
        if min(every, max_snapshots, max_points) < 1:
            raise ValueError('diagnostic bounds must be positive')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.every, self.limit, self.points, self.plots = every, max_snapshots, max_points, plots
        self.calls = 0
        self.rows = []
        self.frames = []
        self.pending = None
        self.original_call = self.original_backward = None

    def __enter__(self):
        from cdlno.linearno.attention import LinearNOAttention
        from cdlno.linearno_history.core import LinearNOHistoryCore, attention_factors
        from cdlno.linearno_history.attnres import LatentSummaryAttnRes
        from cdlno.linearno_history.history_k import HistoryConditionedK
        if self.original_call is not None:
            raise RuntimeError('monitor already installed')
        original = self.original_call = nn.Module._call_impl
        backward = self.original_backward = torch.Tensor.backward
        observer = self

        def is_model(module):
            return (hasattr(module, 'blocks') and hasattr(module, 'preprocess') and
                    len(module.blocks) > 0 and isinstance(getattr(module.blocks[0], 'Attn', None), LinearNOAttention))

        def chained(existing, collect):
            def callback(trace):
                collect(trace)
                if existing is not None:
                    existing(trace)
            return callback

        def invoke(module, *args, **kwargs):
            if is_model(module):
                observer.calls += 1
                capture = ((observer.calls-1) % observer.every == 0 and len(observer.rows) < observer.limit)
                frame = dict(row=dict(call=observer.calls, model=type(module).__module__+'.'+type(module).__name__,
                    training=module.training, layers=[], A=[], K=[]), factors=[], indices={}, model=weakref.ref(module)) if capture else None
                observer.frames.append(frame)
                succeeded = False
                try:
                    result = original(module, *args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    observer.frames.pop()
                    if succeeded and frame is not None:
                        observer.finish(frame)
                        observer.pending = (frame['model'], frame['row']['call'])
            frame = observer.frames[-1] if observer.frames else None
            if frame is not None:
                if isinstance(module, LinearNOHistoryCore):
                    kwargs['observe'] = chained(kwargs.get('observe'), lambda t: observer.block(frame,t.index,t.factors))
                elif isinstance(module, LinearNOAttention):
                    # Pure attention has no observer. Recompute only deterministic
                    # factors, before to_out/dropout; never execute its RNG twice.
                    with torch.no_grad():
                        f = attention_factors(module, args[0], reconstruct=False)
                        observer.block(frame,len(frame['factors']),f)
                elif isinstance(module, LatentSummaryAttnRes):
                    index = args[0]
                    kwargs['observe'] = chained(kwargs.get('observe'), lambda t: observer.a(frame,module,index,t))
                elif isinstance(module, HistoryConditionedK):
                    index, _, base, _, _ = args
                    kwargs['observe'] = chained(kwargs.get('observe'), lambda t: observer.k(frame,index,base,t))
            return original(module, *args, **kwargs)

        def after_backward(tensor, *args, **kwargs):
            result = backward(tensor,*args,**kwargs)
            if observer.pending is not None:
                ref, call = observer.pending
                model = ref()
                if model is not None:
                    groups = {}
                    for name,p in model.named_parameters():
                        if p.grad is None:
                            continue
                        group = name.split('.')[0]
                        groups[group] = groups.get(group,0.) + p.grad.detach().float().square().sum().item()
                    observer.write('gradients.jsonl',dict(call=call, gradient_norms={k:v**.5 for k,v in groups.items()},
                        semantics='accumulated gradients after Tensor.backward; autograd.grad not intercepted'))
                observer.pending = None
            return result
        nn.Module._call_impl = invoke
        torch.Tensor.backward = after_backward
        return self

    def __exit__(self, *exc):
        nn.Module._call_impl = self.original_call
        torch.Tensor.backward = self.original_backward
        self.original_call = self.original_backward = None
        self.frames.clear()
        self.pending = None
        return False

    @torch.no_grad()
    def block(self,frame,index,f):
        n = f.Q.shape[-2]
        if n not in frame['indices']:
            frame['indices'][n] = torch.linspace(0,n-1,min(n,self.points),device=f.Q.device).long()
        idx = frame['indices'][n]
        # clone guarantees that sampled arrays never retain full input storage.
        pair = tuple(t.detach().index_select(-2,idx).float().cpu().clone() for t in (f.Q,f.K))
        frame['factors'].append(pair)
        frame['row']['layers'].append(dict(index=index, points=n, sampled_points=len(idx),
            raw=stats(f.C_raw), Q_entropy=entropy(f.Q), K_entropy=entropy(f.K,-2),
            Q_sum_max_error=(f.Q.sum(-1)-1).abs().max().item(), K_sum_max_error=(f.K.sum(-2)-1).abs().max().item()))

    @torch.no_grad()
    def a(self,frame,module,index,t):
        alpha=t.alpha.detach()
        frame['row']['A'].append(dict(receiver=index,gamma=module.receivers[str(index)].gamma.item(),
            real_weights=alpha.mean((0,1,2))[:-1].tolist(),null_weight=alpha[..., -1].mean().item(),
            source_entropy=entropy(alpha),drop_rate=t.drop_mask.float().mean().item(),
            mask_shape=list(t.drop_mask.shape),cross_shapes=[list(a.shape) for a in t.attention],
            label='current M by old source M cross-depth; not same-layer self-attention'))

    @torch.no_grad()
    def k(self,frame,index,base,t):
        increment=t.eta*t.delta
        frame['row']['K'].append(dict(receiver=index,eta=t.eta.flatten().tolist(),
            correction_ratio=(increment.norm()/base.detach().norm().clamp_min(1e-30)).item(),
            delta_mean_N_max=t.delta.mean(-2).abs().max().item(),
            history_attention_shape=list(t.attention.shape),correction_shape=list(t.delta.shape),
            label='M by S old raw token bank; N by M point-slot correction'))

    def write(self,name,row):
        with (self.output/name).open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')

    @torch.no_grad()
    def finish(self,frame):
        pairs=frame.pop('factors')
        p=pairwise_kernel_similarity(pairs).mean((0,1))
        def cosine(which):
            flat=torch.stack([pair[which].flatten(-2) for pair in pairs],-2)
            flat=flat/flat.norm(dim=-1,keepdim=True).clamp_min(1e-30)
            return (flat@flat.transpose(-1,-2)).mean((0,1))
        row=frame['row']
        row.update(P_similarity=p.tolist(),Q_similarity=cosine(0).tolist(),K_similarity=cosine(1).tolist(),
            kernel_scope='own block routing P=QK^T; excludes A branch Jacobian; same deterministic point subsample at each layer',
            sample_estimate=any(x['sampled_points']<x['points'] for x in row['layers']))
        self.rows.append(row)
        self.write('diagnostics.jsonl',row)
        if self.plots:
            # matplotlib has no training RNG purpose; preserve all three streams.
            from cdlno.training_state import capture_rng, restore_rng
            rng=capture_rng()
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(1,3,figsize=(12,3.5))
                for ax,key in zip(axes,('Q_similarity','K_similarity','P_similarity')):
                    im=ax.imshow(row[key],vmin=0,vmax=1,cmap='viridis')
                    ax.set_title(key);ax.set_xlabel('depth');ax.set_ylabel('depth')
                    fig.colorbar(im,ax=ax)
                fig.suptitle('LinearNO own-block routing; sampled' if row['sample_estimate'] else 'LinearNO own-block routing')
                fig.tight_layout();fig.savefig(self.output/f'call_{row["call"]:06d}.png',dpi=140);plt.close(fig)
            finally:
                restore_rng(rng)
