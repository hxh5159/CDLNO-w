"""Capture existing pure/v1/v2 tensors; NOT a v3 implementation or oracle."""
from pathlib import Path
import copy,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import torch
from cost_audit import pure_model
from cdlno.linearno_loop.versioning import construct
from linearno_loop.config import resolve_config as v1
from linearno_loop.v2.config import resolve_config as v2
from tools.linearno_loop_support import inputs
from cdlno.linearno_loop.attnres import PointDepthAttnRes
OUT=Path(__file__).resolve().parent
TASKS=('ns','elasticity','plasticity','darcy','airfrans','car')
MODES=('sr_1_over_r','rb_attnres','lb_attnres_1_over_r')
TOPO=('p1_c3_r2_s1','p2_c2_r2_s2','custom')
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
 return x

def leaves(x,p='input'):
 if isinstance(x,torch.Tensor):
  if x.is_floating_point():x.requires_grad_(True);return {p:x}
 elif isinstance(x,(tuple,list)):
  return {k:v for i,t in enumerate(x) for k,v in leaves(t,p+'.'+str(i)).items()}
 elif hasattr(x,'to_dict'):return {k:v for n,t in x.to_dict().items() for k,v in leaves(t,p+'.'+n).items()}
 return {}

def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def run(family,task,preset,mode,device,precision,root):
 start=time.monotonic();dtype=torch.float64 if precision=='float64' else torch.float32
 options=dict(topology_preset=preset,residual_mode=mode,linearno_rank=4)
 if preset=='custom':options.update(prefix_blocks=0,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=1)
 overrides={'model.hidden':8,'model.heads':2,'model.ref':3,'runtime.seed':1701,'model.dropout':.1}
 if task not in ('airfrans','car'):overrides.update({'model.H':3,'model.W':5})
 if family.startswith('v2_'):options['core_ffn_mode']=family[3:]
 config=(v2 if family.startswith('v2_') else v1)(task,options=options,profile_overrides=overrides)
 def build():
  if family!='pure':return construct(config)
  m=copy.deepcopy(config['profile_spec']['values']['model']);m['layers']=config['loop_spec']['unique_depth'];m['linearno_rank']=4
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(1701);return pure_model(task,m)
 torch.manual_seed(1234);before_build=torch.get_rng_state().clone();model=build().to(device=device,dtype=dtype)
 assert torch.equal(before_build,torch.get_rng_state()),'constructor leaked CPU RNG'
 for module in model.modules():
  if isinstance(module,PointDepthAttnRes):
   with torch.no_grad():module.query.copy_(torch.linspace(-.13,.17,module.hidden,device=device,dtype=dtype))
 args=inputs(config,canonical=False,batch=2,device=device,dtype=dtype);leaf=leaves(args)
 state=cpu(model.state_dict());rng_before=cpu(torch.get_rng_state());gpu_before=cpu(torch.cuda.get_rng_state()) if device=='cuda' else None
 calls=[];handles=[]
 for name,module in model.named_modules():
  if name.endswith(('.to_q','.to_k','.to_v','.mlp2')):handles.append(module.register_forward_hook(lambda m,a,y,tag=name:calls.append(tag)))
 optim=torch.optim.AdamW(model.parameters(),lr=.001)
 amp_dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}.get(precision)
 model.train()
 with torch.autocast(device_type=device,dtype=amp_dtype,enabled=amp_dtype is not None):
  output=model(*args);loss=(output-.31).square().mean()
 loss.backward();grads={k:cpu(p.grad) for k,p in model.named_parameters()};igrads={k:cpu(p.grad) for k,p in leaf.items()}
 assert torch.isfinite(output).all() and all(v is None or torch.isfinite(v).all() for v in [*grads.values(),*igrads.values()])
 optim.step();after=cpu(model.state_dict());rng_after=cpu(torch.get_rng_state());gpu_after=cpu(torch.cuda.get_rng_state()) if device=='cuda' else None
 for h in handles:h.remove()
 clone=build().to(device=device,dtype=dtype);clone.load_state_dict(model.state_dict(),strict=True);model.eval();clone.eval()
 with torch.no_grad():reference=model(*args);replayed=clone(*args)
 assert torch.equal(reference,replayed),'strict reload differs'
 payload=dict(config=config,family=family,inputs=cpu(args),state_before=state,output=cpu(output),loss=cpu(loss),parameter_gradients=grads,input_gradients=igrads,step_state=after,optimizer=cpu(optim.state_dict()),cpu_rng_before=rng_before,cpu_rng_after=rng_after,cuda_rng_before=gpu_before,cuda_rng_after=gpu_after,call_schedule=calls,strict_eval=cpu(reference))
 name='__'.join((family,task,preset,mode,device,precision));path=root/(name+'.pt');torch.save(payload,path)
 return dict(name=name,family=family,task=task,preset=preset,mode=mode,device=device,precision=precision,artifact=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),state_keys={k:list(v.shape) for k,v in state.items()},state_hash=hashlib.sha256(''.join(k+thash(v) for k,v in state.items()).encode()).hexdigest(),output_hash=thash(output),gradient_hashes={k:thash(v) if v is not None else None for k,v in grads.items()},strict_reload_max_error=0.,status='PASS',seconds=time.monotonic()-start)

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True)
 root=OUT/'baseline-fixtures';root.mkdir(exist_ok=False);rows=[]
 plans=[('cpu','float64'),('cpu','float32')]
 if torch.cuda.is_available():plans += [('cuda','float32'),('cuda','float16'),('cuda','bfloat16')]
 for device,precision in plans:
  for family in ('pure','v1','v2_round_specific','v2_round_specific_latent'):
   for task in TASKS:
    for preset in (TOPO if device=='cpu' else ('p2_c2_r2_s2',)):
     for mode in (MODES if family!='pure' else ('sr_1_over_r',)):
      try:row=run(family,task,preset,mode,device,precision,root)
      except Exception as e:
       row=dict(family=family,task=task,preset=preset,mode=mode,device=device,precision=precision,status='FAIL',error=traceback.format_exc())
      rows.append(row);print({k:row[k] for k in ('family','task','preset','mode','device','precision','status')},flush=True)
      (OUT/'baseline-fixtures.json').write_text(json.dumps({'scope':'small synthetic, current pure/v1/v2 only','rows':rows,'pass':sum(r['status']=='PASS' for r in rows),'fail':sum(r['status']=='FAIL' for r in rows)},indent=2)+'\n')
if __name__=='__main__':main()
