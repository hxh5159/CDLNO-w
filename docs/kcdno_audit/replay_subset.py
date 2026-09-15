"""Replay the same K0 weights in the original task cwd, without data imports."""
import argparse,json,os,sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import make_regression_fixtures as f
p=argparse.ArgumentParser();p.add_argument('group',choices=('static','temporal','car','airfrans'));p.add_argument('--result',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[2];args.result=args.result.resolve()
project='standard' if args.group in ('static','temporal') else args.group
cwd=root/f.PROJECTS[project];sys.path.insert(0,str(root));sys.path.insert(0,str(cwd));os.chdir(cwd)
tasks={'static':('darcy','elasticity','airfoil','pipe'),'temporal':('ns','plasticity'),'car':('car',),'airfrans':('airfrans',)}[args.group]
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
index=json.loads((root/'docs/kcdno_audit/fixture_index.json').read_text());rows=[]
with sdpa_kernel(SDPBackend.MATH):
 for i,row in enumerate(f.cases(project)):
  if row['task'] in tasks:
   result=f.process(row,Path(index['artifact_root']),'replay',2026091500+i);rows.append(result);print(result,flush=True)
args.result.write_text(json.dumps(rows,indent=2)+'\n')
assert all(r['status']=='passed' for r in rows)
