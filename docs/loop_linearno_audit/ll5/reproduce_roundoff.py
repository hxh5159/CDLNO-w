"""Reproduce the recorded LL5 independent-oracle FP32 subtraction/RMS case."""
import json
from pathlib import Path

import torch
from cdlno.linearno_loop.construction import build_from_config
from loop_linearno.lb_oracle import lb_reference
from loop_linearno.lb_support import mode_config,excite,LBTrace
from loop_linearno.point_attnres_oracle import point_attnres
from loop_linearno.test_point_attnres import errors

rows=[]
for dtype in (torch.float32,torch.float64):
    core=build_from_config(mode_config('darcy','custom',variant='conv')).loop.to(dtype)
    excite(core)
    x=torch.randn(2,15,8,generator=torch.Generator().manual_seed(873),dtype=dtype)
    state={k:v.detach().clone() for k,v in core.named_parameters()}
    with LBTrace(core) as trace:y=core(x)
    ref,rt=lb_reference(x,state,P=0,C=2,R=3,S=1,variant='conv',heads=2,H=3,W=5)
    local=point_attnres(trace.routes[1]['sources'],core.lb_boundaries[1].query,core.lb_boundaries[1].norm_scale)[1]
    rows.append(dict(dtype=str(dtype),weights2=errors(trace.routes[1]['weights'],rt['rounds'][1]['weights']),
        same_sources_oracle_weights=errors(trace.routes[1]['weights'],local),
        delta_errors=[errors(t['sources'][-1],r['delta']) for t,r in zip(trace.routes,rt['rounds'])],
        delta_max=[t['sources'][-1].detach().abs().max().item() for t in trace.routes],final=errors(y,ref)))
expected=json.loads(Path(__file__).with_name('lb-roundoff.json').read_text())
assert rows==expected,'roundoff reproduction differs from saved evidence'
print('FP32/FP64 roundoff evidence reproduced exactly; same-source routing retains LL2 tolerance.')
