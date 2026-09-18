"""Native Car parser/train/records/resume/eval, in-memory real PyG only."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT/'Car-Design-ShapeNetCar'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROJECT))
from models.cdlno_run import parse_args
from cdlno.linearno_history.car_entry import CarRun,run_cli,constructor_kwargs
from cdlno.linearno.profiles import digest
from cdlno.linearno.schema import read_metadata,unpack_state
from cdlno.linearno.shapenet import ShapeNetLinearNO
from cdlno.linearno_history.industrial import inspect_checkpoint,read_pair
from cdlno.experiment import session
from cdlno.linearno_history.factory import build_model
from cdlno.linearno_history.checkpoint import config_from_metadata


def parser_for(evaluation=False):
    filename='main_evaluation.py' if evaluation else 'main.py'
    tree=ast.parse((PROJECT/filename).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser' or
        isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument']
    scope=dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<native parser definitions only>','exec'),scope)
    return scope['parser']


def synthetic(args):
    from dataset.dataset import GraphDataset
    g=torch.Generator().manual_seed(981); graphs=[]
    # Last graph exercises the native Python random geometry subsampling.
    for n,ns in ((17,5),(23,9),(8205,8201)):
        x=torch.randn(n,7,generator=g);pos=torch.randn(n,3,generator=g)
        y=torch.randn(n,4,generator=g)+1
        surf=torch.arange(n)<ns
        graphs.append(Data(x=x,pos=pos,y=y,surf=surf,edge_index=torch.empty(2,0,dtype=torch.long)))
    coef=(np.arange(7,dtype=np.float32),np.ones(7,dtype=np.float32),np.arange(4,dtype=np.float32),np.ones(4,dtype=np.float32)*2)
    if hasattr(args,'_linearno_metadata'):
        from cdlno.linearno.car_entry import restore_coef_norm
        restored=restore_coef_norm(args._linearno_metadata)
        for a,b in zip(coef,restored):np.testing.assert_array_equal(a,b)
        coef=restored
    ds=GraphDataset(graphs,use_cfd_mesh=True)
    spec=dict(split='SYNTHETIC in-memory real PyG',sampling='native GraphDataset.get_shape; full graph',
        checksums={'SYNTHETIC':hashlib.sha256(b''.join(t.numpy().tobytes() for graph in graphs for t in (graph.x,graph.y,graph.pos,graph.surf))).hexdigest()},
        train_samples=['synthetic:0','synthetic:1','synthetic:2'],test_samples=['synthetic:0','synthetic:1'],
        ntrain=3,ntest=2,fold_id=args.fold_id,cfd_mesh=args.cfd_mesh,r=args.r,val_iter=args.val_iter,preprocessed=args.preprocessed)
    return spec,ds,GraphDataset(graphs[:2],use_cfd_mesh=True),coef


def run(action,directory,signature):
    torch.set_num_threads(1)
    tokens=['--cfd_model','LinearNO','--experiment-dir',str(directory),'--device','cpu']
    if action in ('train','interrupt'):
        tokens+=['--nb_epochs','3','--linearno-hidden','8',
            '--linearno-heads','2','--linearno-layers','4','--linearno-rank','4','--seed','19','--cfd_mesh']
    elif action=='resume':tokens+=['--resume']
    if action not in ('resume','eval'):
        tokens += ['--linearno-fair-run','1','--linearno_latent_attnres',signature[1],'--linearno_history_k_conditioning',signature[3]]
    args=parse_args(parser_for(action=='eval'),argv=tokens,evaluation=action=='eval')
    spec,train_ds,test_ds,coef=synthetic(args)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json') if p.is_file()}
    batches=[];runs=[];predictions=[];handles=[]
    original_prepare=CarRun.prepare;original_complete=CarRun.complete_epoch
    def prepare(self,model,*a):
        runs.append(self)
        handles.append(model.register_forward_pre_hook(lambda module,inputs: batches.append(hashlib.sha256(
            inputs[0][0].x.detach().cpu().numpy().tobytes()+inputs[0][1].detach().cpu().numpy().tobytes()).hexdigest()) if module.training else None))
        return original_prepare(self,model,*a)
    class Interrupted(Exception):pass
    def complete(self,*a):
        result=original_complete(self,*a)
        if a[0] == args.nb_epochs:
            for handle in handles: handle.remove()
        if action=='interrupt':raise Interrupted()
        return result
    from cdlno.linearno_history.car_entry import evaluate
    def eval_without_raw(run,model,dataset):
        with torch.no_grad():predictions.append(model(dataset[0]).cpu())
        return evaluate(run,model,dataset,force=False)
    try:
        with patch('cdlno.linearno_history.car_entry.inspect_data',return_value=spec),patch('cdlno.linearno_history.car_entry.load_data',return_value=(train_ds,test_ds,coef)),patch.object(CarRun,'prepare',prepare),patch.object(CarRun,'complete_epoch',complete),patch('cdlno.linearno_history.car_entry.evaluate',eval_without_raw):
            run_cli(args)
    except Interrupted:
        session(args).finish(status='interrupted',error='synthetic committed epoch boundary')
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json')}
    if before:assert before==after
    metadata,manifest=inspect_checkpoint(directory,'latest' if action=='interrupt' else 'final',ShapeNetLinearNO)
    saved,state=read_pair(manifest,ShapeNetLinearNO)
    if action=='eval':
        # This object was just produced locally by this test's native train;
        # never load an external or untrusted pickle here.
        whole=torch.load(directory/f'model_{args.nb_epochs}.pth',map_location='cpu',weights_only=False).eval()
        with torch.no_grad(): torch.testing.assert_close(whole(test_ds[0]),predictions[0],rtol=0,atol=0)
    if action!='eval':
        model=build_model(config_from_metadata(metadata));model.load_state_dict(state,strict=True);model.eval()
        with torch.no_grad():predictions.append(model(test_ds[0]))
    return dict(epoch=metadata['resume_state']['epoch'],global_step=metadata['resume_state']['global_step'],
        batches=batches,state_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in state.values())).hexdigest(),
        resume_hash=digest(saved['resume_state']),prediction=predictions[0].tolist(),metadata_immutable=not before or before==after,
        scheduler_total=unpack_state(saved['resume_state']['scheduler'])['total_steps'],
        scheduler_step=unpack_state(saved['resume_state']['scheduler'])['last_epoch'])

if __name__=='__main__':
    action,directory,report,signature=sys.argv[1:]
    Path(report).write_text(json.dumps(run(action,Path(directory),signature),indent=2)+'\n')
