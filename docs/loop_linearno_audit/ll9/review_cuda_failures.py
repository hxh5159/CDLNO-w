"""Record untouched-model AMP failure source, without a compatibility workaround."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from tools.linearno_loop_support import configuration,inputs,PRESETS
from cdlno.linearno_loop.construction import build_from_config
OUT=Path(__file__).resolve().parent
before={v['path']:v for v in json.loads((OUT/'start-manifest.json').read_text())['files']}
rows=[];torch.set_num_threads(1)
for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'):
 c=configuration(task,PRESETS[0],'rb_attnres',small=True);m=build_from_config(c).cuda();events=[];handles=[]
 for name,r in m.loop.named_modules():
  if name.startswith('rb_receivers.') and hasattr(r,'source_weights'):
   handles.append(r.register_forward_pre_hook(lambda mod,a,name=name:events.append(dict(receiver=name,source_dtypes=[str(t.dtype) for t in a[0]],source_shapes=[list(t.shape) for t in a[0]]))))
 try:
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):m(*inputs(c,device='cuda'))
  error=None
 except TypeError as e:error=str(e)
 finally:
  for h in handles:h.remove()
 rows.append(dict(task=task,error=error,first_receivers=events[:3]))
paths=['cdlno/linearno_loop/attnres.py','cdlno/linearno_loop/core.py','cdlno/linearno_loop/standard.py','cdlno/linearno_loop/airfrans.py','cdlno/linearno_loop/shapenet.py','PDE-Solving-StandardBenchmark/model/LinearNO.py','cdlno/linearno/airfrans.py','cdlno/linearno/shapenet.py']
frozen={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==before[p]['sha256'] for p in paths};assert all(frozen.values())
(OUT/'cuda-failure-review.json').write_text(json.dumps(dict(rows=rows,unchanged_from_LL9_start=frozen,
 interpretation='Existing FP32 placeholder promotion produces FP32 anchor; autocast branches yield FP16 partial. LL2 receiver forbids mixed source dtype. FP32 and homogeneous-source AMP pass; no casting or model workaround applied.',
 first_tool_bug='Initial GPU tool compared first parameter (unused placeholder for fx tasks). Corrected tool to require an update in any parameter with gradients; initial logs preserved.'),indent=2))
