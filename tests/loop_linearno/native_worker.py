"""LL6 real entry ASTs, full spatial shapes, CPU synthetic tensors only.

Reuse accepted L4/L5 workers verbatim for native losses/time loops/observer;
substitute only loop CLI flags and the real public Run dispatcher. No real
loader or replacement train/eval loop. Each invocation is a fresh process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from linearno_loop.config import resolve_config,run_directory_id


def fixture_config(task,preset,mode):
    return resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode,linearno_rank=4),
        profile_overrides={'training.epochs':3,'model.hidden':8,'model.heads':2,'training.batch_size':2,
                           'runtime.seed':17,'model.dropout':.1})


def run(task,action,directory,report,preset,mode):
    from linearno import temporal_worker,static_worker
    worker=temporal_worker if task in ('ns','plasticity') else static_worker
    from linearno_entry import StandardRun as dispatch_run
    from cdlno.linearno_loop.standard_entry import LoopStandardRun
    from cdlno.linearno_loop.attnres import PointDepthAttnRes
    from linearno_loop.contracts import PRESETS
    original_parse=worker.parse_args
    worker.StandardRun=LoopStandardRun
    def parse(parser,task,tokens):
        tokens=list(tokens)
        if action in ('train','interrupt'):
            i=tokens.index('--n-layers');del tokens[i:i+2]
            tokens += ['--linearno-loop','1','--linearno-loop-topology',preset,
                       '--linearno-loop-residual-mode',mode,'--dropout','.1']
        # Eval/resume deliberately omit --model: metadata must recover family
        # and the correct existing structured/irregular key before dispatch.
        elif '--model' in tokens:
            i=tokens.index('--model');del tokens[i:i+2]
        return original_parse(parser,task,tokens)
    worker.parse_args=parse
    native=worker.native_main;init=LoopStandardRun.__init__;checks=[]
    P,C,R,S=PRESETS[preset]
    expected_sources=([1 if r==0 and j==0 else r+1+(j>0) for r in range(R) for j in range(2*C)]+[R+1]
                      if mode=='rb_attnres' else list(range(2,R+2)) if mode=='lb_attnres_1_over_r' else [])
    def capture_init(self,*a,**kw):
        init(self,*a,**kw)
        active={}
        def begin(*args):active.update(sources=[],attn=0,head=0)
        def router(mod,args):active['sources'].append(len(args[0]))
        def attention(*args):active['attn']+=1
        def head(*args):active['head']+=1
        def end(*args):
            assert active['sources']==expected_sources,(active,expected_sources)
            assert active['attn']==P+C*R+S and active['head']==1
            checks.append(dict(sources=active['sources'],attention=active['attn'],head=active['head']))
        self.model.register_forward_pre_hook(begin)
        for mod in self.model.modules():
            if isinstance(mod,PointDepthAttnRes):mod.register_forward_pre_hook(router)
        for group in (self.model.loop.prefix,self.model.loop.core,self.model.loop.suffix):
            for b in group:b.block.Attn.register_forward_hook(attention)
        self.model.loop.suffix[-1].block.mlp2.register_forward_hook(head)
        self.model.register_forward_hook(end)
    LoopStandardRun.__init__=capture_init
    def native_main(task,values,scope):
        scope['LinearNORun']=dispatch_run
        return native(task,values,scope)
    worker.native_main=native_main
    worker.run(task,action,directory,report)
    rows=json.loads(report.read_text());rows.update(preset=preset,mode=mode,
        public_dispatch_used=True,metadata_recovers_model=action in ('resume','eval'),
        forward_local_checks=len(checks),expected_source_counts=expected_sources,
        each_forward_recomputed_attention=P+C*R+S)
    if action!='interrupt':
        artifact_files=[p for base in ('training_artifacts','visualizations','evaluations') for p in (directory/base).rglob('*') if p.is_file()]
        rows['recorded_artifacts']=[dict(path=str(p.relative_to(directory)),size=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in artifact_files]
    report.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('action');p.add_argument('directory',type=Path)
    p.add_argument('report',type=Path);p.add_argument('preset');p.add_argument('mode');a=p.parse_args()
    run(a.task,a.action,a.directory,a.report,a.preset,a.mode)
