"""Real Air parser -> run_cli -> native train/records, using in-memory PyG only.

Only dataset/VTK boundaries are patched. Native sampling, 20 validation repeats,
loss, Adam, OneCycle, checkpoint/ensemble and recorder paths execute unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from loop_linearno.industrial_support import install_checks
PROJECT = ROOT / 'Airfoil-Design-AirfRANS'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PROJECT))
from cdlno_entry import parse_args
from cdlno.linearno_loop.air_entry import AirRun, run_cli, _restore_coef_norm
from cdlno.linearno.airfrans import AirfRANSLinearNO
from cdlno.linearno_loop.checkpoint import inspect_checkpoint, read_pair
from cdlno.linearno.profiles import digest
from cdlno.experiment import session
from cdlno.linearno_loop.construction import build_from_config as build_model
def config_from_metadata(metadata):return metadata['resolved_config']


def parser_for(evaluation=False):
    tree = ast.parse((PROJECT / ('main_evaluation.py' if evaluation else 'main.py')).read_text())
    nodes = [n for n in tree.body if
        isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser' or
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'parser.add_argument']
    scope = dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native parser only>', 'exec'), scope)
    return scope['parser']


def _data():
    g = torch.Generator().manual_seed(901)
    result = []
    for i in range(10):
        n = 13 + i % 3
        result.append(Data(x=torch.randn(n,7,generator=g), pos=torch.randn(n,2,generator=g),
            y=torch.randn(n,4,generator=g)+1, surf=torch.arange(n)%2==0))
    return result


def run(action, directory, signature):
    torch.set_num_threads(1)
    evaluation = action == 'eval'
    tokens = ['--model', 'LinearNO', '--experiment-dir', str(directory), '--my_path', 'synthetic:in-memory']
    nmodel = 2 if action in ('ensemble','ensemble_interrupt','ensemble_later_interrupt') else 1
    if action in ('train','interrupt','ensemble','ensemble_interrupt','ensemble_later_interrupt'):
        tokens += ['--epochs','3','--linearno-hidden','8','--linearno-heads','2',
            '--linearno-rank','4','--seed','901',
            '--subsampling','11','--nmodel',str(nmodel),'--task','scarce']
    elif action == 'resume': tokens += ['--resume']
    if action not in ('resume','eval'):
        tokens += ['--linearno-loop','1','--linearno-loop-topology',PRESET,'--linearno-loop-residual-mode',MODE,'--linearno-dropout','.1']
    if action in ('resume','eval'):
        del tokens[:2]
    args = parse_args(parser_for(evaluation), argv=tokens, evaluation=evaluation)
    negative_count = 0
    if evaluation:
        for flags in (['--linearno-rank','99'], ['--linearno-profile','official_release'],
                      ['--seed','7'], ['--task','full'], ['--batch_size','2'],
                      ['--weight','0.5'], ['--linearno-hidden','12'], ['--checkpoint','absent']):
            try:
                parse_args(parser_for(True),argv=tokens+flags,evaluation=True)
            except SystemExit as error:
                assert error.code == 2; negative_count += 1
            else: raise AssertionError('negative parser accepted '+str(flags))
        from linearno_loop.schema import validate_metadata as check_structure
        import copy
        changed=copy.deepcopy(args._linearno_metadata)
        changed['model_spec']['constructor_kwargs']['n_hidden']=12
        try:check_structure(changed)
        except ValueError:negative_count += 1
        else:raise AssertionError('constructor/profile mismatch accepted')
        pointer=directory/'member_000/checkpoints/final.json'; content=pointer.read_bytes()
        try:
            corrupted=json.loads(content);corrupted['sha256']='0'*64;pointer.write_text(json.dumps(corrupted))
            try:inspect_checkpoint(directory/'member_000','final')
            except ValueError:negative_count += 1
            else:raise AssertionError('corrupt pointer accepted')
        finally:pointer.write_bytes(content)
    assert args.seed == 901 and args.task == 'scarce'
    config = args._linearno_config
    nmodel = args.nmodel
    values = _data()
    checksum = hashlib.sha256(b''.join(t.numpy().tobytes() for data in values for t in (data.x,data.y,data.pos,data.surf))).hexdigest()
    manifest = {'scarce_train':[f'synthetic:{i}' for i in range(10)],'full_test':['synthetic:0']}
    data_spec = dict(split='SYNTHETIC scarce last10% validation/full_test',task='airfrans',task_variant=args.task,
        checksums={'SYNTHETIC tensors':checksum}, sampling='native random.sample; PyG shuffle',
        train_count=9,validation_count=1,steps_per_epoch=9,scheduler_total_steps=30)
    coef = tuple(np.arange(n,dtype=np.float32) if i%2==0 else np.ones(n,dtype=np.float32)*2
        for i,n in enumerate((7,7,4,4)))
    before = {p.name:p.read_bytes() for p in (directory/'architecture.json',directory/'config.json') if p.exists()}
    def load(data_dir, args, *, train, coef_norm):
        assert session(args) is not None  # recorder reserved before loading
        if before:
            for a,b in zip(coef,_restore_coef_norm(args._linearno_metadata)): np.testing.assert_array_equal(a,b)
            for a,b in zip(coef,coef_norm): np.testing.assert_array_equal(a,b)
        else: assert coef_norm is None
        return (manifest,values[:9],values[9:],coef) if train else (manifest,coef)
    batches = []; predictions = []; states = []; handles = []; checks=[]
    original_prepare = AirRun.prepare; original_complete = AirRun.complete_epoch
    def prepare(self, model, *args):
        check_handles=install_checks(model,checks)
        handles.extend(check_handles)
        handles.append(model.register_forward_pre_hook(lambda m, inputs: batches.append(
            [self.current_member, hashlib.sha256(inputs[0].x.cpu().numpy().tobytes()).hexdigest()]) if m.training else None))
        return original_prepare(self, model, *args)
    class Interrupted(Exception): pass
    def complete(self, epoch, model, history):
        original_complete(self,epoch,model,history)
        if epoch == args.nb_epochs:
            for handle in handles: handle.remove()
        stop = (action=='interrupt' and epoch==1 or
                action=='ensemble_interrupt' and self.current_member==0 and epoch==3 or
                action=='ensemble_later_interrupt' and self.current_member==1 and epoch==1)
        if stop: raise Interrupted()
    def scores(device, models, hparams, coef_norm, *a, **kw):
        assert kw['criterion']=='MSE' and kw['s']=='full_test' and kw['n_test']==3
        assert len(models)==1 and len(models[0])==nmodel
        with torch.no_grad(): predictions.extend(m(values[0]).cpu() for m in models[0])
        return np.zeros(1),np.zeros(1),np.zeros(1)
    try:
        with patch('cdlno.linearno_loop.air_entry._data_root',return_value=Path('synthetic:in-memory')), \
             patch('cdlno.linearno_loop.air_entry._data_spec',return_value=data_spec), \
             patch('cdlno.linearno_loop.air_entry._load_dataset',side_effect=load), \
             patch.object(AirRun,'prepare',prepare), patch.object(AirRun,'complete_epoch',complete), \
             patch('utils.metrics.Results_test',side_effect=scores), \
             patch('cdlno.linearno_loop.air_entry._pressure_rL2_dataset',return_value={'SYNTHETIC':True}):
            run_cli(args)
    except Interrupted:
        session(args).finish(status='interrupted',error='synthetic committed boundary')
    for h in handles: h.remove()
    after = {p.name:p.read_bytes() for p in (directory/'architecture.json',directory/'config.json')}
    if before: assert before == after
    epochs=[]; resume_hash=[]
    for i in range(nmodel):
        member=directory/f'member_{i:03d}'
        if not (member/'checkpoints/latest.json').exists(): continue
        metadata,path=inspect_checkpoint(member,'latest')
        _,state=read_pair(path); states.append(state)
        epochs.append(metadata['resume_state']['epoch']); resume_hash.append(digest(metadata['resume_state']))
        if not evaluation:
            model=build_model(config_from_metadata(metadata));model.load_state_dict(state,strict=True);model.eval()
            with torch.no_grad():predictions.append(model(values[0]))
        elif metadata['resume_state']['epoch']==3:
            # This worker created the trusted object locally; no external pickle.
            bare=torch.load(member/'model',map_location='cpu',weights_only=True)
            for k in state:torch.testing.assert_close(bare[k],state[k],rtol=0,atol=0)
    hashes=[hashlib.sha256(b''.join(t.numpy().tobytes() for t in s.values())).hexdigest() for s in states]
    members=json.loads((directory/'ensemble.json').read_text())['members'] if (directory/'ensemble.json').exists() else []
    if not evaluation and action not in ('interrupt','ensemble_interrupt','ensemble_later_interrupt'):
        events=[json.loads(line) for p in (directory/'visualizations').rglob('events.jsonl') for line in p.read_text().splitlines()]
        assert events and all(e['status']=='completed' for e in events), events
    return dict(loop_forward_checks=len(checks),preset=PRESET,mode=MODE,epoch=epochs[0],epochs=epochs,state_hash=hashes[0],member_hashes=hashes,loaded_hashes=hashes,
        resume_hash=resume_hash,batches=batches,prediction=[p.detach().tolist() for p in predictions],nmodel=nmodel,
        members=[dict(member_id=m['member_id'],order=m['order']) for m in members],distinct_members=len(set(hashes))==len(hashes),
        metadata_immutable=not before or before==after,pointer=True,shape=list(predictions[0].shape),family='linearno_loop',
        objective=config['values']['objective'],seed=args.seed,task=args.task,negative_count=negative_count)

if __name__=='__main__':
    action,directory,report,PRESET,MODE=sys.argv[1:]
    Path(report).write_text(json.dumps(run(action,Path(directory),'loop'),indent=2)+'\n')
