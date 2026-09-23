"""Real Standard main ASTs, full spatial shapes, synthetic data, native losses.

Only the dataset-reading prefix and showcase plotting are replaced. All task
forward/loss/optimizer/time loops, normalizers, epoch records and checkpoints
execute as written. Device transfers may be adapted to CPU when CUDA is absent.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'PDE-Solving-StandardBenchmark')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from linearno import static_worker, temporal_worker
from cdlno_entry import parse_args
from model_dict import get_model
from linearno_entry import model_kwargs,StandardRun
from cdlno.linearno.standard_entry import normalizer,start,finish
from cdlno.linearno_loop.standard_entry import LoopStandardRun
from cdlno.experiment import session
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss
from tran_evaluate.linearno_loop.recording import observe,state_hash


def native_main(task,values,scope,device):
    source=temporal_worker if task in ('ns','plasticity') else static_worker
    path=source.PROJECT/f'exp_{source.SHORT[task]}.py'
    tree=ast.parse(path.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    normalized=task not in ('airfoil','ns')
    index=next(i for i,n in enumerate(main.body) if ('linearno_normalizer(' in ast.unparse(n) if normalized
        else isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='train_loader'))
    main.body=ast.parse('\n'.join(f'{name} = _synthetic_values[{name!r}]' for name in values)).body+main.body[index:]
    class Adapt(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if device=='cpu' and isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':return node.func.value
            return node
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='showcase':node.value=ast.Constant(0)
            return self.generic_visit(node)
    functions=[Adapt().visit(n) for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('count_parameters','random_collate_fn','central_diff','main')]
    scope['_synthetic_values']=values
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(path)+':synthetic', 'exec'),scope)
    return scope['main']


def run(task,mode,action,directory,report):
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=temporal_worker if task in ('ns','plasticity') else static_worker
    key='LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D'
    tokens=['--model',key,'--experiment-dir',str(directory)]
    if action in ('train','interrupt'):
        tokens += ['--linearno-loop','1','--linearno-loop-architecture','resmlp_dual_temp_v4',
                   '--linearno-loop-temperature-mode',mode,'--seed','17','--epochs','2','--batch-size','1']
    elif action=='resume':tokens += ['--resume']
    else:tokens += ['--eval','1']
    args=parse_args(source.parser_for(task),task,tokens)
    values,checksum=source.synthetic_values(task)
    values={k:(v[:1] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[0] in (2,4) else v) for k,v in values.items()}
    values.update(ntrain=1,ntest=1)
    data=args._linearno_config['values']['data']
    args._linearno_data=dict(split=data['split'],sampling=data['sampling'],checksums={'SYNTHETIC':checksum},
                            scope='SYNTHETIC full N, full H/M, 1train/1test')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json') if p.exists()}
    start(args,task)
    runs=[];counts=dict(train_forward=0,eval_forward=0,backward=0,optimizer=0)
    original_init,original_save=LoopStandardRun.__init__,LoopStandardRun.save
    backward=torch.Tensor.backward
    def init(self,*a,**kw):
        original_init(self,*a,**kw);runs.append(self);observe(args,self.model)
        def visit(model,inputs,kwargs):
            counts['train_forward' if model.training else 'eval_forward']+=1
        self.model.register_forward_pre_hook(visit,with_kwargs=True)
    def back(value,*a,**kw):
        assert torch.isfinite(value).all();counts['backward']+=1
        return backward(value,*a,**kw)
    class Interrupted(Exception):pass
    def save(self,model):
        result=original_save(self,model)
        if action=='interrupt':raise Interrupted()
        return result
    scope=dict(args=args,eval=bool(args.eval),epochs=args.epochs,save_name=args.save_name,torch=torch,np=np,F=F,
        rearrange=rearrange,plt=plt,os=os,get_model=get_model,TestLoss=TestLoss,UnitTransformer=UnitTransformer,
        linearno_model_kwargs=model_kwargs,LinearNORun=StandardRun,linearno_normalizer=normalizer)
    with patch.object(LoopStandardRun,'__init__',init),patch.object(LoopStandardRun,'save',save),patch.object(torch.Tensor,'backward',back):
        try:
            if action in ('resume','eval'):
                with patch.object(UnitTransformer,'__init__',side_effect=AssertionError('normalizer refit')):
                    native_main(task,values,scope,device)()
            else:native_main(task,values,scope,device)()
        except Interrupted:session(args).finish(status='interrupted',error='intentional synthetic epoch boundary')
        else:finish(args)
    run=runs[0]
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json')}
    if before:assert before==after
    epochs={'train':2,'interrupt':1,'resume':1,'eval':0}[action]
    expected=epochs*(20 if task=='plasticity' else 1)
    assert counts['backward']==expected,(counts,expected)
    if action!='eval':
        assert counts['train_forward']==epochs*(20 if task=='plasticity' else 10 if task=='ns' else 1),counts
    archive=directory/'checkpoints'/('latest.json' if action=='interrupt' else 'final.json')
    assert archive.exists()
    payload=dict(task=task,mode=mode,action=action,device=device,scope='SYNTHETIC original task loops',
        counts=counts,shape_facts=args._linearno_loop_config['model'],weights_hash=state_hash(run.model.state_dict()),
        completed_epoch=run.completed_epoch,metadata_unchanged=bool(before),
        visualization_files=[str(p.relative_to(directory)) for p in directory.rglob('*.png')],
        metrics_files=[str(p.relative_to(directory)) for p in directory.rglob('*results.json')])
    report.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task');parser.add_argument('mode');parser.add_argument('action')
    parser.add_argument('directory',type=Path);parser.add_argument('report',type=Path)
    a=parser.parse_args();run(a.task,a.mode,a.action,a.directory,a.report)
