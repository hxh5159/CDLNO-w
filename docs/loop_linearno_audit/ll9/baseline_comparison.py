"""Actual pure eight-block baseline counts; compare cost, never accuracy."""
import importlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
rows=json.loads((OUT/'counts.json').read_text())['rows'];base={}
for task in dict.fromkeys(r['task'] for r in rows):
 r=next(v for v in rows if v['task']==task and v['rank_multiplier']==1 and v['mode']=='sr_1_over_r')
 kw=dict(r['config']['model_spec']['constructor_kwargs'])
 for key in ('prefix_blocks','recurrent_core_blocks','loop_repeats','suffix_blocks','residual_mode'):kw.pop(key)
 kw['n_layers']=8
 module,cls={'airfrans':('cdlno.linearno.airfrans','AirfRANSLinearNO'),'car':('cdlno.linearno.shapenet','ShapeNetLinearNO')}.get(task,('PDE-Solving-StandardBenchmark.model.LinearNO','Model'))
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(17);model=getattr(importlib.import_module(module),cls)(**kw)
 actual=sum(p.numel() for p in model.parameters());cost=r['analytic'];parts=cost['parameter_parts']
 body=parts['shared_core']//r['config']['loop_spec']['recurrent_core_blocks']
 expected=parts['stem']+8*body+parts['head'];assert expected==actual
 base[task]=dict(actual_pure8_parameters=actual,base_rank=cost['M'],matrix_macs=cost['executed_matrix_macs'],
     scope='actual pure eight-block construction count; B1 canonical-N dense MAC same as E8 loop rank1, excludes router/scalars')
comparisons=[dict(task=r['task'],preset=r['preset'],mode=r['mode'],rank_multiplier=r['rank_multiplier'],M=r['analytic']['M'],
    parameters=r['analytic']['parameters'],parameter_ratio_vs_pure8_base=r['analytic']['parameters']/base[r['task']]['actual_pure8_parameters'],
    dense_MAC_ratio_vs_pure8_base=r['analytic']['executed_matrix_macs']/base[r['task']]['matrix_macs'],
    router_contraction_MAC_equivalents=r['analytic']['router_contraction_mac_equivalents'],
    accuracy_or_epoch_efficiency='NOT RUN') for r in rows]
(OUT/'baseline-comparisons.json').write_text(json.dumps(dict(baselines=base,rows=comparisons),indent=2))
print('8 actual pure counts exact; 96 cost comparisons only')
