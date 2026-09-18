"""Execute the actual four exp main ASTs with explicit synthetic tensors only.

The raw-file loading prefix is replaced by labeled in-memory tensors at real N.
CPU transfers replace entry .cuda() calls; standalone eval showcase plotting is
suppressed, while the real periodic recorder remains active. No exp imports,
fake dataset filenames, fake packages, or replacement training/eval loops.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'PDE-Solving-StandardBenchmark'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROJECT))
from cdlno_entry import parse_args
from model_dict import get_model
from utils.normalizer import UnitTransformer
from utils.testloss import TestLoss
from cdlno.linearno.standard_entry import (model_kwargs, StandardRun, normalizer, start, finish)
from cdlno.linearno.profiles import digest
from cdlno.experiment import session
from cdlno.linearno.checkpoint import inspect_checkpoint, read_pair

SHORT=dict(darcy='darcy',elasticity='elas',airfoil='airfoil',pipe='pipe')
SHAPES=dict(darcy=(85,85),elasticity=(972,1),airfoil=(221,51),pipe=(129,129))


def parser_for(task):
    tree=ast.parse((PROJECT/f'exp_{SHORT[task]}.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='parser' or
           isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='parser.add_argument']
    scope=dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real parser AST>','exec'),scope)
    return scope['parser']


def synthetic_values(task):
    H,W=SHAPES[task];N=H*W
    generator=torch.Generator().manual_seed(441)
    row=torch.arange(N)//W;col=torch.arange(N)%W
    coords=torch.stack((row/max(H-1,1),col/max(W-1,1)),dim=-1).float()
    x=coords.repeat(6,1,1).clone()
    # Point order remains row-major H,W; samples carry distinct geometry/field tags.
    x[:,:,0]+=torch.arange(6).reshape(-1,1)*.03
    y=1+torch.sin(x[:,:,0]*2)+x[:,:,1].square()+torch.arange(6).reshape(-1,1)*.02
    fx=torch.randn(6,N,generator=generator)+3
    result=dict(ntrain=4,ntest=2,s=H,s1=H,s2=W,dx=1/H)
    if task=='elasticity':result.update(train_s=y[:4],test_s=y[4:],train_xy=x[:4],test_xy=x[4:])
    else:result.update(x_train=fx[:4] if task=='darcy' else x[:4],x_test=fx[4:] if task=='darcy' else x[4:],y_train=y[:4],y_test=y[4:])
    checksum=hashlib.sha256(b''.join(v.numpy().tobytes() for v in (x,y,fx))).hexdigest()
    return result,checksum


def native_main(task, values, scope):
    tree=ast.parse((PROJECT/f'exp_{SHORT[task]}.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start_index=next(i for i,node in enumerate(main.body) if
        ('linearno_normalizer(' in ast.unparse(node) if task!='airfoil' else
         isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='train_loader'))
    prefix=ast.parse('\n'.join(f"{name} = _synthetic_values[{name!r}]" for name in values)).body
    main.body=prefix+main.body[start_index:]
    class CPU(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':return node.func.value
            return node
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='showcase':node.value=ast.Constant(0)
            return self.generic_visit(node)
    functions=[CPU().visit(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('count_parameters','central_diff','main')]
    module=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))
    scope['_synthetic_values']=values
    exec(compile(module,str(PROJECT/f'exp_{SHORT[task]}.py')+':synthetic-prefix/CPU','exec'),scope)
    return scope['main']


def run(task, action, directory, report):
    torch.set_num_threads(1)
    key='LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D'
    argv=['--model',key,'--experiment-dir',str(directory)]
    if action in ('train','interrupt'):
        argv+=['--epochs','3','--n-hidden','8','--n-heads','2','--n-layers','2',
               '--linearno-rank','4','--batch-size','2','--seed','17']
    elif action=='resume':argv+=['--resume']
    elif action=='eval':argv+=['--eval','1']
    else:raise ValueError(action)
    args=parse_args(parser_for(task),task,argv)
    values,checksum=synthetic_values(task)
    data=args._linearno_config['values']['data']
    args._linearno_data=dict(split=data['split'],sampling=data['sampling'],checksums={'SYNTHETIC_full_spatial_shape':checksum},
        scope='SYNTHETIC tensors, 4 train/2 test; no real loader acceptance')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json') if p.exists()}
    start(args,task)
    runs=[];batches=[]
    original_init=StandardRun.__init__;original_save=StandardRun.save
    def capture_init(self,*a,**kw):
        original_init(self,*a,**kw);runs.append(self)
        def record(model,inputs,kwargs):
            if model.training:
                tensors=[v for v in (*inputs,*kwargs.values()) if isinstance(v,torch.Tensor)]
                batches.append(hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in tensors)).hexdigest())
        self.model.register_forward_pre_hook(record,with_kwargs=True)
    class SyntheticInterruption(Exception):pass
    def capture_save(self,model):
        result=original_save(self,model)
        if action=='interrupt':raise SyntheticInterruption('synthetic interruption after committed epoch1')
        return result
    scope=dict(args=args,eval=bool(args.eval),epochs=args.epochs,save_name=args.save_name,torch=torch,np=np,F=F,
        rearrange=rearrange,plt=plt,os=__import__('os'),get_model=get_model,TestLoss=TestLoss,
        UnitTransformer=UnitTransformer,linearno_model_kwargs=model_kwargs,LinearNORun=StandardRun,
        linearno_normalizer=normalizer)
    try:
        with patch.object(StandardRun,'__init__',capture_init),patch.object(StandardRun,'save',capture_save):
            if action in ('resume','eval'):
                with patch.object(UnitTransformer,'__init__',side_effect=AssertionError('eval/resume must not refit')):
                    native_main(task,values,scope)()
            else:native_main(task,values,scope)()
    except SyntheticInterruption:
        session(args).finish(status='interrupted',error='intentional synthetic checkpoint boundary')
    else:finish(args)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'architecture.json',directory/'config.json')}
    if action in ('resume','eval'):assert before==after,(before,after)
    run=runs[0]
    x=values['train_xy'][:1] if task=='elasticity' else values['x_train'][:1]
    if task=='darcy':
        s=85; xx,yy=np.meshgrid(np.linspace(0,1,s),np.linspace(0,1,s))
        pos=torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float).unsqueeze(0)
        fx=args._linearno_normalizers['input'].encode(x).unsqueeze(-1)
    else:
        pos=args._linearno_normalizers['input'].encode(x) if task=='pipe' else x
        fx=None
    run.model.eval()
    with torch.no_grad():prediction=run.model(pos,fx)
    prediction_hash=hashlib.sha256(prediction.numpy().tobytes()).hexdigest()
    payload=dict(task=task,action=action,shape=list(prediction.shape),prediction_hash=prediction_hash,
        batches=batches,normalizer_refit=False if action in ('resume','eval') else None,
        immutable_metadata=True,completed_epoch=run.completed_epoch,source='real main AST',
        reductions=args._linearno_config['values']['objective']['reduction'])
    report.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=SHORT);p.add_argument('action');p.add_argument('directory',type=Path);p.add_argument('report',type=Path)
    options=p.parse_args();run(options.task,options.action,options.directory,options.report)
