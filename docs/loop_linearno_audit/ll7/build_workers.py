"""Adapt accepted native industrial test harnesses; keep real train/data seams."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
for task in ('air','car'):
 s=(ROOT/f'tests/linearno/history_{task}_worker.py').read_text()
 s=s.replace('from cdlno.linearno_history.'+task+'_entry import', 'from cdlno.linearno_loop.'+task+'_entry import')
 s=s.replace('from cdlno.linearno_history.industrial import inspect_checkpoint,read_pair','from cdlno.linearno_loop.checkpoint import inspect_checkpoint,read_pair')
 s=s.replace('from cdlno.linearno_history.industrial import inspect_checkpoint, read_pair','from cdlno.linearno_loop.checkpoint import inspect_checkpoint, read_pair')
 s=s.replace('from cdlno.linearno_history.factory import build_model','from cdlno.linearno_loop.construction import build_from_config as build_model')
 s=s.replace('from cdlno.linearno_history.checkpoint import config_from_metadata',"def config_from_metadata(metadata):return metadata['resolved_config']")
 s=s.replace('cdlno.linearno_history.'+task+'_entry.','cdlno.linearno_loop.'+task+'_entry.')
 s=s.replace('from cdlno.linearno_history.'+task+'_entry import','from cdlno.linearno_loop.'+task+'_entry import')
 s=s.replace(', _constructor_kwargs','').replace(',constructor_kwargs','')
 s=s.replace("'--linearno-layers','4',",'')
 s=s.replace("tokens += ['--linearno-fair-run','1','--linearno_latent_attnres',signature[1],'--linearno_history_k_conditioning',signature[3]]",
 "tokens += ['--linearno-loop','1','--linearno-loop-topology',PRESET,'--linearno-loop-residual-mode',MODE,'--linearno-dropout','.1']")
 # Existing resume/eval omit structure; also omit model selector to test metadata-first public guard.
 insert="    if action in ('resume','eval'):\n        del tokens[:2]\n"
 marker='    args = parse_args(' if task=='air' else '    args=parse_args('
 s=s.replace(marker,insert+marker,1)
 s=s.replace(',AirfRANSLinearNO)',')').replace(', AirfRANSLinearNO)',')').replace(',ShapeNetLinearNO)',')')
 if task=='air':
  s=s.replace('from cdlno.linearno_history.air_entry import check_structure','from linearno_loop.schema import validate_metadata as check_structure')
  s=s.replace('from cdlno.linearno_loop.air_entry import check_structure','from linearno_loop.schema import validate_metadata as check_structure')
  s=s.replace("whole=torch.load(member/'model',map_location='cpu',weights_only=False).eval()\n            with torch.no_grad():torch.testing.assert_close(whole(values[0]),predictions[i],rtol=0,atol=0)",
   "bare=torch.load(member/'model',map_location='cpu',weights_only=True)\n            for k in state:torch.testing.assert_close(bare[k],state[k],rtol=0,atol=0)")
  s=s.replace("family='linearno'","family='linearno_loop'")
 else:
  s=s.replace("'--seed','19','--cfd_mesh'","'--seed','19','--cfd_mesh','--fold_id','3'")
  s=s.replace("whole=torch.load(directory/f'model_{args.nb_epochs}.pth',map_location='cpu',weights_only=False).eval()\n        with torch.no_grad(): torch.testing.assert_close(whole(test_ds[0]),predictions[0],rtol=0,atol=0)",
    "bare=torch.load(directory/f'model_{args.nb_epochs}.pth',map_location='cpu',weights_only=True)\n        for k in state:torch.testing.assert_close(bare[k],state[k],rtol=0,atol=0)")
 # Hook each actual graph forward, including every sampled validation/visualization call.
 s=s.replace('    original_prepare = AirRun.prepare;', '    original_prepare = AirRun.prepare;')
 marker='        handles.append(model.register_forward_pre_hook('
 s=s.replace(marker,'        check_handles=install_checks(model,checks)\n        handles.extend(check_handles)\n'+marker,1)
 s=s.replace('    batches = []; predictions = []; states = []; handles = []','    batches = []; predictions = []; states = []; handles = []; checks=[]')
 s=s.replace('    batches=[];runs=[];predictions=[];handles=[]','    batches=[];runs=[];predictions=[];handles=[];checks=[]')
 s=s.replace('return dict(epoch=', 'return dict(loop_forward_checks=len(checks),preset=PRESET,mode=MODE,epoch=')
 # Use inherited synthetic sampling data but all construction/checkpoint paths are new loop production.
 s=s[:s.index("if __name__=='__main__':")]+'''if __name__=='__main__':
    action,directory,report,PRESET,MODE=sys.argv[1:]
    Path(report).write_text(json.dumps(run(action,Path(directory),'loop'),indent=2)+'\\n')
'''
 # Import parser-independent hooks without corrupting industrial project module resolution.
 s=s.replace('ROOT = Path(__file__).resolve().parents[2]',"ROOT = Path(__file__).resolve().parents[2]\nsys.path.insert(0,str(ROOT/'tests'))\nfrom loop_linearno.industrial_support import install_checks") if task=='air' else s.replace('ROOT=Path(__file__).resolve().parents[2];',"ROOT=Path(__file__).resolve().parents[2]\nsys.path.insert(0,str(ROOT/'tests'))\nfrom loop_linearno.industrial_support import install_checks\n")
 (ROOT/f'tests/loop_linearno/{task}_worker.py').write_text(s)
