"""Independent matrix formulas versus actual ATen trace at canonical task N."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from linearno_loop.v4.config import resolve_config
from cdlno.linearno_loop.v4.construction import build_from_config
from tools.linearno_loop_accounting import analytic,analytic_pure_linearno,audit,measured_parameters
from tools.linearno_loop_support import inputs,configuration
from cdlno.linearno_loop.construction import build_from_config as loop_model

POINTS=dict(airfoil=11271,darcy=7225,elasticity=972,pipe=16641,ns=4096,plasticity=3131,airfrans=32000,car=32186)


def pure_model(config):
    task=config['task'];base=config['profile_spec']
    if task=='car':
        from cdlno.linearno.car_entry import constructor_kwargs
        from cdlno.linearno.shapenet import ShapeNetLinearNO
        return ShapeNetLinearNO(**constructor_kwargs(base))
    if task=='airfrans':
        from cdlno.linearno.air_entry import _constructor_kwargs
        from cdlno.linearno.airfrans import AirfRANSLinearNO
        return AirfRANSLinearNO(**_constructor_kwargs(base))
    from cdlno.linearno.standard_entry import constructor_kwargs
    return importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO').Model(**constructor_kwargs(base))


def main(path):
    torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=[]
    for task,N in POINTS.items():
        for mode in ('base','latent_k_point_q','point_k_point_q'):
            config=resolve_config(task,options=dict(architecture='resmlp_dual_temp_v4',temperature_mode=mode,seed=17))
            model=build_from_config(config).to(device);arg=inputs(config,canonical=True,batch=1,device=device)
            expected=analytic(config,B=1,N=N);actual=audit(model,arg,B=1,N=N)
            assert actual['parameter_parts']==expected['parameter_parts'],(task,mode)
            assert actual['matrix_macs']==expected['matrix_macs'],(task,mode,actual['matrix_macs'],expected['matrix_macs'])
            assert not actual['forbidden_attention']
            assert len(actual['contractions'])==16 and len(actual['softmax'])==16
            pure=analytic_pure_linearno(task,B=1,N=N)
            legacy=analytic(configuration(task,'p1_c3_r2_s1','sr_1_over_r',multiplier=1),B=1,N=N)
            rows.append(dict(task=task,mode=mode,analytic=expected,actual=actual,pure_linearno=pure,loop_v1_m_base=legacy,
                parameter_ratio=expected['parameters']/pure['parameters'],matrix_flops_ratio=expected['matrix_flops']/pure['matrix_flops']))
            print(task,mode,expected['parameters'],expected['matrix_flops']/1e9,'exact_trace',flush=True)
            del model,arg;gc.collect()
        base=config
        model=pure_model(base)
        assert sum(p.numel() for p in model.parameters())==pure['parameters'],task
        del model;gc.collect()
    path.write_text(json.dumps(dict(scope='canonical N B1 forward matrix accounting',device=device,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
