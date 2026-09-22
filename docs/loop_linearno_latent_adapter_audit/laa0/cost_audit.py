"""LAA0 read-only analytic audit. No v3 production module is constructed."""
from pathlib import Path
import importlib,json,re,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from cdlno.linearno.profiles import resolve_config
TASKS=('airfoil','darcy','elasticity','pipe','plasticity','ns','airfrans','car')
PROMPT=ROOT/'PLAN_Looped_LinearNO/Looped_LinearNO_LatentFFN_BilateralAdapter_CostProfiles_Codex_Staged_Prompts.md'
TEXT=PROMPT.read_text()
# Frozen tabulated H/Dz are inputs. Expected ratios are extracted separately.
tables=[]
for line in TEXT.splitlines():
 cells=[x.strip() for x in line.strip('|').split('|')]
 if len(cells)==9 and re.fullmatch(r'\d+ → \d+',cells[0]) and all(re.fullmatch(r'\d+/\d+',x) for x in cells[1:]):
  tables.append((list(map(int,cells[0].split(' → '))),[tuple(map(int,x.split('/'))) for x in cells[1:]]))
assert len(tables)==8

def formula(m,task,H,L,E,C,Dz,B,N,latent=False,adapter=False):
 h=m['heads'];d=H//h;M=m['linearno_rank'];f=m['ffn_ratio'];o=m['out_dim']
 assert H%h==0
 k=9 if m['linearno_variant'] in ('conv','conv_temp') else 1
 q=2 if k==9 or task=='car' else 1
 tau=2*h if m['linearno_variant'] in ('temp','conv_temp','shapenet') else h if task=='airfrans' else 0
 # Sum individual tensor shapes, independently of the existing accounting helper.
 att=k*H*H+H+2*M*d+d*d+q*(H*H+H)+tau
 ffn=H*(f*H)+(f*H)+(f*H)*H+H
 body=att+4*H+ffn
 inch=(m['ref']**2 if m['unified_pos'] else m['space_dim'])+m['fun_dim']
 if task=='airfrans' and m['unified_pos']:inch+=m['space_dim']
 stem=inch*2*H+2*H+2*H*H+H+H
 tm=2*(H*H+H) if m['time_input'] else 0
 head=2*H+H*o+o
 latent_p=C*(2*H*Dz+Dz+3*H) if latent else 0
 adapter_p=C*2*4*(d+M) if adapter else 0
 parts={'stem_placeholder':stem,'time':tm,'body':L*body,'head':head,'latent':latent_p,'adapter':adapter_p}
 body_mac=B*N*(k*H*H+4*H*M+H*d+q*H*H+2*f*H*H)
 mac_parts={'stem':B*N*(2*inch*H+2*H*H),'time':2*B*N*H*H if m['time_input'] else 0,'body':E*body_mac,'head':B*N*H*o,'latent':4*B*C*M*H*Dz if latent else 0,'adapter':2*B*C*h*N*4*(d+M) if adapter else 0}
 return {'parameters':sum(parts.values()),'matrix_macs':sum(mac_parts.values()),'parameter_parts':parts,'matrix_mac_parts':mac_parts,'body_parameters':body,'body_mac':body_mac}

def pure_model(task,m):
 kw=dict(space_dim=m['space_dim'],n_layers=m['layers'],n_hidden=m['hidden'],dropout=m['dropout'],n_head=m['heads'],act=m['activation'],mlp_ratio=m['ffn_ratio'],fun_dim=m['fun_dim'],out_dim=m['out_dim'],linearno_rank=m['linearno_rank'],ref=m['ref'],unified_pos=m['unified_pos'])
 if task=='airfrans':
  from cdlno.linearno.airfrans import AirfRANSLinearNO
  return AirfRANSLinearNO(**kw)
 kw.update(Time_Input=m['time_input'],H=m['H'] or 85,W=m['W'] or 85)
 if task=='car':
  from cdlno.linearno.shapenet import ShapeNetLinearNO
  return ShapeNetLinearNO(**kw)
 return importlib.import_module('PDE-Solving-StandardBenchmark.model.LinearNO').Model(**kw,linearno_variant=m['linearno_variant'])

def main():
 import torch
 torch.set_num_threads(1);start=time.monotonic();anchors={};rows=[];differences=[]
 anchor_text=TEXT.split('## 3.2 ')[1].split('## 3.3 ')[0]
 base_lines=[l for l in anchor_text.splitlines() if re.match(r'^\| (Airfoil|Darcy|Elasticity|Pipe|Plasticity|Navier--Stokes|AirfRANS|ShapeNet-Car) \| \d+ \|',l)]
 assert len(base_lines)==8
 for task,line in zip(TASKS,base_lines):
  nums=[int(x.strip().replace(',','')) for x in line.strip('|').split('|')[1:]]
  H,M,f,N,B,expected_p,expected_mac=nums
  config=resolve_config(task);m=config['values']['model']
  assert (H,M,f)==(m['hidden'],m['linearno_rank'],m['ffn_ratio'])
  calc=formula(m,task,H,8,8,0,0,B,N);model=pure_model(task,m)
  actual=sum(p.numel() for p in model.parameters())
  row={'task':task,'model':m,'B':B,'N':N,'prompt_parameters':expected_p,'prompt_macs':expected_mac,'actual_parameters':actual,**calc}
  if not (actual==calc['parameters']==expected_p and calc['matrix_macs']==expected_mac):differences.append(row)
  anchors[task]=row
 for idx,((L,E),pairs) in enumerate(tables):
  profile='matched_v1' if idx<4 else 'efficient_v1';C=(E-4)//2
  for task,(H,Dz) in zip(TASKS,pairs):
   b=anchors[task];base=formula(b['model'],task,b['model']['hidden'],L,L,0,0,b['B'],b['N'])
   calc=formula(b['model'],task,H,L,E,C,Dz,b['B'],b['N'],True,True)
   rows.append({'task':task,'cost_profile':profile,'comparator_depth':L,'executed_depth':E,'unique_depth':4+C,'H':H,'Dz':Dz,'M':b['model']['linearno_rank'],'B':b['B'],'N':b['N'],**calc,'parameter_percent':100*calc['parameters']/base['parameters'],'mac_percent':100*calc['matrix_macs']/base['matrix_macs']})
 # Validate every percentage in the two D12 tables and every deeper interval.
 d12=[l for l in TEXT.splitlines() if l.startswith('| ') and '%' in l and '→' not in l and '任务' not in l]
 assert len(d12)==16
 for profile,subset in [('matched_v1',d12[:8]),('efficient_v1',d12[8:16])]:
  for task,line in zip(TASKS,subset):
   expected=[float(x.replace('%','').strip()) for x in line.strip('|').split('|')[1:]]
   r=next(x for x in rows if x['task']==task and x['cost_profile']==profile and x['comparator_depth']==8)
   actual=[r['parameter_percent'],r['mac_percent']] if profile=='matched_v1' else [r['parameter_percent'],100-r['parameter_percent'],r['mac_percent'],100-r['mac_percent']]
   if any(abs(a-e)>.00501 for a,e in zip(actual,expected)):differences.append({'line':line,'actual':actual})
 ranges=[]
 for profile in ('matched_v1','efficient_v1'):
  for L in (8,12,16,32):
   rr=[r for r in rows if r['cost_profile']==profile and r['comparator_depth']==L]
   values=[[r[k] if profile=='matched_v1' else 100-r[k] for r in rr] for k in ('parameter_percent','mac_percent')]
   ranges.append({'profile':profile,'L':L,'parameter_range':[min(values[0]),max(values[0])],'mac_range':[min(values[1]),max(values[1])]})
 range_lines=[l for l in TEXT.splitlines() if re.match(r'^\| \d+→\d+ \|',l)]
 assert len(range_lines)==7
 for r,line in zip([r for r in ranges if not (r['profile']=='efficient_v1' and r['L']==8)],range_lines):
  expected=list(map(float,re.findall(r'(\d+\.\d+)%',line)))
  actual=r['parameter_range']+r['mac_range']
  if any(abs(a-e)>.00501 for a,e in zip(actual,expected)):differences.append({'line':line,'actual':actual})
 result={'status':'PASS' if not differences else 'BLOCKED','scope':'independent formula; actual pure model parameters; no v3 tensor implementation','anchors':anchors,'rows':rows,'ranges':ranges,'differences':differences,'seconds':time.monotonic()-start}
 (OUT/'cost-recalculation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'rows':len(rows),'differences':differences,'D12':[{'task':r['task'],'profile':r['cost_profile'],'parameters':r['parameters'],'MAC':r['matrix_macs']} for r in rows if r['comparator_depth']==8]},indent=2))
if __name__=='__main__':main()
