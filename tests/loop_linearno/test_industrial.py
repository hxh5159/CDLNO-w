"""LL7 parser isolation and real PyG wrapper contracts; no dataset reads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch_geometric.data import Data,Batch
from linearno_loop.config import resolve_config
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
from cdlno.linearno_loop.construction import build_from_config
from cdlno.linearno_loop.checkpoint import strict_load
from loop_linearno.industrial_support import config

ROOT=Path(__file__).resolve().parents[2]


class IndustrialTests(unittest.TestCase):
    def test_car_nonzero_fold_drag_surface_boundary_without_vtk_claim(self):
        import numpy as np
        from types import SimpleNamespace
        from cdlno.linearno import car_entry as pure
        from cdlno.linearno.car_metrics import drag_pair
        cfg=config('car','p1_c3_r2_s1','sr_1_over_r')
        model=build_from_config(cfg).eval();graphs=[]
        for n in (11,17):
            graphs.append((Data(x=torch.randn(n,7),y=torch.randn(n,4)+2,
                surf=torch.arange(n)<n//2,pos=torch.randn(n,3)),torch.randn(5,3)))
        coef=(np.zeros(7),np.ones(7),np.arange(4),np.ones(4)*2)
        calls=[]
        def coefficient(path,pressure,velocity):
            calls.append((Path(path),pressure.copy(),velocity.copy()))
            self.assertEqual(pressure.shape,(len(velocity),1));self.assertEqual(velocity.shape[1],3)
            return float(np.abs(pressure).sum()+np.abs(velocity).sum()+1)
        def pair(data,out,normalizer,path):return drag_pair(data,out,normalizer,path,coefficient_fn=coefficient)
        with tempfile.TemporaryDirectory() as tmp:
            run=SimpleNamespace(recorder=None,directory=Path(tmp),config=cfg['profile_spec'],coef=coef,
                args=SimpleNamespace(device='cpu',data_dir=tmp,checkpoint='final'),
                data=dict(test_samples=['param3/case_a','param3/case_b']))
            with patch.object(pure,'drag_pair',side_effect=pair):result=pure.evaluate(run,model,graphs,force=True)
            self.assertEqual([p.relative_to(tmp).as_posix() for p,_,_ in calls],
                ['param3/case_a','param3/case_a','param3/case_b','param3/case_b'])
            self.assertEqual(len(result['force']['prediction']),2)
            self.assertTrue(np.isfinite(result['force']['relative_error']))

    def test_six_modes_real_pyg_contract_and_state_keys(self):
        for task in ('airfrans','car'):
            for preset in PRESETS:
                for mode in RESIDUAL_MODES:
                    with self.subTest(task=task,preset=preset,mode=mode):
                        model=build_from_config(config(task,preset,mode));model.eval()
                        self.assertNotIn('reference',model.state_dict());self.assertNotIn('pos',model.state_dict())
                        keys=model.state_dict()
                        temperature='temperature' if task=='airfrans' else 'tempreature_q'
                        self.assertEqual(sum(k.endswith('.'+temperature) for k in keys),model.loop.unique_depth)
                        if task=='car':self.assertEqual(sum(k.endswith('.tempreature_k') for k in keys),model.loop.unique_depth)
                        for n in (11,19):
                            data=Data(x=torch.randn(n,7),pos=torch.randn(n,2 if task=='airfrans' else 3))
                            batched=Batch.from_data_list([data])
                            invoke=lambda d:model(d if task=='airfrans' else (d,torch.randn(4,3)))
                            out=invoke(batched);self.assertEqual(out.shape,(n,4))
                            out.square().mean().backward()
                            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                            if task=='airfrans':
                                self.assertTrue(all(p.grad is None for k,p in model.named_parameters() if k.endswith('.temperature')))
                                sampled=batched.clone();sampled.x=sampled.x[:5];sampled.pos=sampled.pos[:5];sampled.batch=sampled.batch[:5]
                                self.assertEqual(invoke(sampled).shape,(5,4))
                            with self.assertRaises(ValueError):invoke(Batch.from_data_list([data,data]))
                            wrong=batched.clone();wrong.ptr=torch.tensor([0,n-1])
                            with self.assertRaises(ValueError):invoke(wrong)
                            wrong=batched.clone();wrong.x=wrong.x.unsqueeze(0)
                            with self.assertRaises(ValueError):invoke(wrong)
                            wrong=batched.clone();wrong.x=wrong.x.long()
                            with self.assertRaises(ValueError):invoke(wrong)
                            if task=='airfrans':
                                wrong=batched.clone();wrong.pos=wrong.pos.double()
                                with self.assertRaises(ValueError):invoke(wrong)
                                wrong=batched.clone();wrong.pos=wrong.pos.to('meta')
                                with self.assertRaises(ValueError):invoke(wrong)
                            wrong=batched.clone();wrong.x=wrong.x.double()
                            if task=='airfrans':wrong.pos=wrong.pos.double()
                            with self.assertRaises((ValueError,RuntimeError)):invoke(wrong)
                        state=copy.deepcopy(model.state_dict());fresh=build_from_config(config(task,preset,mode))
                        strict_load(fresh,state)
                        if mode!='sr_1_over_r':
                            del state[next(k for k in state if '.query' in k)]
                            with self.assertRaises(ValueError):strict_load(fresh,state)

    def test_custom_topology_rank_and_independent_members(self):
        from types import SimpleNamespace
        from cdlno.linearno_loop.industrial_state import construct
        for task in ('car','airfrans'):
            c=resolve_config(task,options=dict(topology_preset='custom',prefix_blocks=0,recurrent_core_blocks=2,
                loop_repeats=3,suffix_blocks=1,residual_mode='rb_attnres'),profile_overrides={'model.hidden':8,'model.heads':2})
            m=build_from_config(c);self.assertEqual((m.loop.unique_depth,m.loop.executed_depth),(3,7))
            args=SimpleNamespace(_linearno_loop_config=c,linearno_task=task,seed=0)
            a=construct(args,0);b=construct(args,1)
            self.assertFalse({id(p) for p in a.parameters()}&{id(p) for p in b.parameters()})
            self.assertFalse(torch.equal(a.preprocess.linear_pre[0].weight,b.preprocess.linear_pre[0].weight))
        with self.assertRaises(ValueError):resolve_config('car',options=dict(topology_preset='p1_c3_r2_s1',residual_mode='sr_1_over_r',linearno_rank=63))

    def test_native_parsers_defaults_fail_fast_and_old_paths(self):
        code=r'''
import sys,copy,torch
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/tests')
from linearno_loop.config import resolve_config
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES
from cdlno.linearno_loop import construction
if sys.argv[2]=='airfrans':
 from loop_linearno.air_worker import parser_for,parse_args
 task='airfrans';key='--model'
else:
 from loop_linearno.car_worker import parser_for,parse_args
 task='car';key='--cfd_model'
def parse(tokens):return parse_args(parser_for(False),argv=list(map(str,tokens)))
for preset in PRESETS:
 for mode in RESIDUAL_MODES:
  a=parse([key,'LinearNO','--linearno-loop',1,'--linearno-loop-topology',preset,'--linearno-loop-residual-mode',mode])
  assert a._linearno_loop_config==resolve_config(task,options=dict(topology_preset=preset,residual_mode=mode))
  assert a.linearno_rank==64
base=[key,'LinearNO','--linearno-loop',1,'--linearno-loop-topology','p1_c3_r2_s1','--linearno-loop-residual-mode','rb_attnres']
for extra in ([key,'Transolver'],['--linearno_latent_attnres',0],['--linearno_history_k_conditioning',0],
 ['--linearno-fair-run',1],['--linearno-rank',32,'--linearno-loop-rank-multiplier',1],['--linearno-loop-core-blocks',3],
 ['--linearno-loop-residual-mode','bad'],['--linearno-layers',8],['--batch_size',2],['--weight',2.]):
 rng=torch.get_rng_state().clone()
 with patch('torch.load',side_effect=AssertionError('early load')),patch.object(construction,'build_from_config',side_effect=AssertionError('early construct')):
  try:parse(base+extra)
  except SystemExit:pass
  else:raise AssertionError(extra)
 assert torch.equal(torch.get_rng_state(),rng)
for tokens in ([key,'LinearNO'],[key,'Transolver'],[key,'LinearNO','--linearno_latent_attnres',1]):
 with patch('cdlno.linearno_loop.industrial_entry.intercept',return_value=None):before=parse(tokens)
 after=parse(tokens)
 def stable(a):return {k:v for k,v in vars(a).items() if k not in ('linearno_run_dir',)}
 assert stable(before)==stable(after)
print(task,'6 resolved profiles,10 conflicts,3 old routes passed')
'''
        for task in ('airfrans','car'):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
            p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT),task],env=env,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_original_scientific_functions_reused_and_precise_freeze(self):
        from types import SimpleNamespace
        from cdlno.linearno import car_entry as car,air_entry as air
        from cdlno.linearno_history.car_entry import CarRun as HistoryCarRun
        from cdlno.linearno_loop import car_entry as lc,air_entry as la
        from cdlno.linearno_loop.ll7_projection import REPLACEMENTS,project
        self.assertIs(lc.CarRun.train,HistoryCarRun.train)
        self.assertIs(lc.CarRun.objective,car.CarRun.objective)
        self.assertIs(lc.evaluate,car.evaluate);self.assertIs(lc.load_data,car.load_data)
        self.assertIs(la._load_dataset,air._load_dataset);self.assertIs(la._data_spec,air._data_spec)
        args=SimpleNamespace(linearno_family='linearno_loop')
        for original,adapter in ((car,lc),(air,la)):
            with patch.object(adapter,'run_cli',return_value='loop_dispatch') as called:
                self.assertEqual(original.run_cli(args),'loop_dispatch');called.assert_called_once_with(args)
        for name in REPLACEMENTS:
            self.assertEqual(project(name,(ROOT/name).read_text()),(ROOT/'docs/loop_linearno_audit/ll7/before'/name).read_text())


if __name__=='__main__':unittest.main()
