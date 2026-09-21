import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Batch, Data

from cdlno.linearno.checkpoint import strict_load
from cdlno.linearno_loop.industrial_state import construct
from cdlno.linearno_loop.v2 import checkpoint
from linearno_loop.contracts import PRESETS, RESIDUAL_MODES, seal
from linearno_loop.v2.config import resolve_config, run_directory_id
from linearno_loop.v2.contracts import CORE_FFN_MODES
from linearno_loop.v2.schema import write_metadata
from support import metadata

ROOT=Path(__file__).resolve().parents[2]


def small_config(task,preset,residual,mode,seed=23):
    overrides={"runtime.seed":seed,"model.hidden":8,"model.heads":2,
               "model.ffn_ratio":1,"model.ref":2}
    if task=="airfrans":overrides["training.nmodel"]=2
    return resolve_config(task,options={"topology_preset":preset,"residual_mode":residual,
        "core_ffn_mode":mode,"linearno_rank":4},profile_overrides=overrides)


def data(task,n=13):
    graph=Data(x=torch.randn(n,7),pos=torch.randn(n,2 if task=="airfrans" else 3),
        y=torch.randn(n,4),surf=torch.arange(n)<n//2)
    batch=Batch.from_data_list([graph])
    return (batch,) if task=="airfrans" else ((batch,torch.randn(5,3)),)


class LF6IndustrialEntryTests(unittest.TestCase):
    def test_real_pyg_two_tasks_full_mode_matrix_step_and_strict_state(self):
        for task in ("airfrans","car"):
            for preset in PRESETS:
                for residual in RESIDUAL_MODES:
                    for mode in CORE_FFN_MODES:
                        with self.subTest(task=task,preset=preset,residual=residual,mode=mode):
                            config=small_config(task,preset,residual,mode)
                            args=SimpleNamespace(_linearno_loop_config=config,linearno_task=task,seed=23)
                            model=construct(args);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
                            output=model(*data(task));self.assertEqual(tuple(output.shape),(13,4))
                            output.square().mean().backward();optimizer.step()
                            clone=construct(args);strict_load(clone,model.state_dict())
                            self.assertTrue(torch.equal(model(*data(task,11)).isfinite(),torch.ones(11,4,dtype=torch.bool)))

    def test_air_members_independent_car_rank_and_v2_pair(self):
        air=small_config("airfrans","p2_c2_r2_s2","rb_attnres","round_specific_latent")
        args=SimpleNamespace(_linearno_loop_config=air,linearno_task="airfrans",seed=23)
        first,second=construct(args,0),construct(args,1)
        self.assertFalse({id(p) for p in first.parameters()}&{id(p) for p in second.parameters()})
        self.assertFalse(torch.equal(first.preprocess.linear_pre[0].weight,second.preprocess.linear_pre[0].weight))
        for model in (first,second):
            model(*data("airfrans")).square().mean().backward()
        car=small_config("car","p2_c2_r2_s2","lb_attnres_1_over_r","round_specific_latent")
        car_model=construct(SimpleNamespace(_linearno_loop_config=car,linearno_task="car",seed=23))
        car_model(*data("car")).square().mean().backward()
        architecture=metadata(car);saved=copy.deepcopy(architecture)
        saved["resume_state"]["epoch"]=1;saved["resume_state"]["global_step"]=1;saved=seal(saved,"metadata_hash")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_metadata(root/"architecture.json",architecture)
            manifest=checkpoint.save_pair(root,car_model,saved)
            _,state=checkpoint.read_pair(manifest,expected=car)
            fresh=construct(SimpleNamespace(_linearno_loop_config=car,linearno_task="car",seed=23))
            strict_load(fresh,state)

    def test_real_industrial_parsers_v2_default_m1_nondefault_fold_and_conflicts(self):
        code=r'''
import sys
sys.path.insert(0,sys.argv[1]+'/tests')
task=sys.argv[2]
if task=='airfrans':
 from loop_linearno.air_worker import parser_for,parse_args
 key='--model';extra=[]
else:
 from loop_linearno.car_worker import parser_for,parse_args
 key='--cfd_model';extra=['--fold_id','3']
for preset in ('p1_c3_r2_s1','p2_c2_r2_s2'):
 for residual in ('sr_1_over_r','rb_attnres','lb_attnres_1_over_r'):
  for mode in ('round_specific','round_specific_latent'):
   flags=[key,'LinearNO','--linearno-loop','1','--linearno-loop-topology',preset,
    '--linearno-loop-residual-mode',residual,'--linearno-loop-core-ffn-mode',mode,*extra]
   args=parse_args(parser_for(False),argv=flags)
   assert args._linearno_loop_config['config_version']==2
   assert args.linearno_rank==args._linearno_loop_config['loop_spec']['base_rank']==32
   if task=='car':assert args.fold_id==3
print(task,'24 v2 parser configurations passed')
'''
        for task in ("airfrans","car"):
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT))
            result=subprocess.run([sys.executable,"-B","-c",code,str(ROOT),task],cwd=ROOT,
                env=environment,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=="__main__":unittest.main()
