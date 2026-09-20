"""Independent structural hooks for native industrial synthetic invocations."""
from linearno_loop.config import resolve_config
from cdlno.linearno_loop.attnres import PointDepthAttnRes


def config(task,preset,mode,*,nmodel=1):
    extra={'training.epochs':3,'model.hidden':8,'model.heads':2,'model.dropout':.1,
           'runtime.seed':901 if task=='airfrans' else 19}
    if task=='airfrans':extra.update({'training.subsampling':11,'training.nmodel':nmodel})
    return resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode,linearno_rank=4),profile_overrides=extra)


def install_checks(model,checks):
    loop=model.loop;C=loop.recurrent_core_blocks;R=loop.loop_repeats
    mode=loop.residual_mode
    expected=([r+1+(j>0) for r in range(R) for j in range(2*C)]+[R+1]
              if mode=='rb_attnres' else list(range(2,R+2)) if mode=='lb_attnres_1_over_r' else [])
    current={};handles=[]
    def begin(*a):current.update(sources=[],attn=0,head=0)
    def router(mod,args):current['sources'].append(len(args[0]))
    def attention(*a):current['attn']+=1
    def head(*a):current['head']+=1
    def end(*a):
        assert current['sources']==expected
        assert current['attn']==loop.prefix_blocks+C*R+loop.suffix_blocks and current['head']==1
        checks.append(dict(sources=current['sources'],attention=current['attn'],head=current['head']))
    handles.append(model.register_forward_pre_hook(begin))
    for mod in model.modules():
        if isinstance(mod,PointDepthAttnRes):handles.append(mod.register_forward_pre_hook(router))
    for group in (loop.prefix,loop.core,loop.suffix):
        for block in group:handles.append(block.block.Attn.register_forward_hook(attention))
    handles.append(loop.suffix[-1].block.mlp2.register_forward_hook(head));handles.append(model.register_forward_hook(end))
    return handles
