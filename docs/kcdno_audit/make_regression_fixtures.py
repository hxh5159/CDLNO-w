#!/usr/bin/env python3
"""K0 only: capture/replay current models in independent task working directories.

No exp/main imports, datasets, optimizer steps, or production edits. Binary
artifacts live outside the repository. Capture refuses existing fixture paths;
replay ALWAYS loads the saved weights, inputs and expected outputs/gradients.
Only trusted locally created industrial model objects are unpickled.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {'standard': 'PDE-Solving-StandardBenchmark',
            'car': 'Car-Design-ShapeNetCar', 'airfrans': 'Airfoil-Design-AirfRANS'}
STEMS = dict(darcy='darcy', elasticity='elas', airfoil='airfoil', pipe='pipe', ns='ns', plasticity='plas')
MODES = ('full', 'no_sa', 'identity')


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + '\n')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def parser_only(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser') or
             (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and ast.unparse(n.value.func) == 'parser.add_argument')]
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + ':parser-only', 'exec'), scope)
    return scope['parser']


def parse_task(task, extras=(), evaluation=False, family='CDLNO'):
    cwd = Path.cwd()
    if task in STEMS:
        from cdlno_entry import parse_args
        return parse_args(parser_only(cwd / f'exp_{STEMS[task]}.py'), task,
                          ['--model', family, '--eval', str(int(evaluation)), *extras])
    entry = importlib.import_module('models.cdlno_run' if task == 'car' else 'cdlno_entry')
    path = cwd / ('main_evaluation.py' if evaluation else 'main.py')
    flag = '--cfd_model' if task == 'car' else '--model'
    return entry.parse_args(parser_only(path), evaluation=evaluation, argv=[flag, family, *extras])


def cases(project):
    tasks = tuple(STEMS) if project == 'standard' else (project,)
    rows = []
    for task in tasks:
        for family, mode in [('CDLNO', m) for m in MODES] + [('Transolver', None)]:
            kw = dict(n_hidden=8, n_layers=8, n_head=2, mlp_ratio=2, slice_num=4, dropout=0.)
            device = 'cpu' if family == 'CDLNO' else 'cuda:0'
            if task in STEMS:
                module = 'model.CDLNO_' if family == 'CDLNO' else 'model.Transolver_'
                if task == 'elasticity': module += 'Irregular_Mesh'
                elif task in ('ns', 'plasticity') and family == 'CDLNO': module += 'Temporal_Structured_Mesh_2D'
                else: module += 'Structured_Mesh_2D'
                kw.update(space_dim=2, fun_dim=10 if task == 'ns' else (1 if task in ('darcy','plasticity') else 0),
                          out_dim=4 if task == 'plasticity' else 1, Time_Input=task == 'plasticity',
                          ref=8, unified_pos=task in ('darcy','ns'))
                if task != 'elasticity':
                    kw.update(H=64 if task == 'ns' else (101 if task == 'plasticity' else 5),
                              W=64 if task == 'ns' else (31 if task == 'plasticity' else 7))
                classname = 'Model'
                if family == 'CDLNO': kw.update(task_name=task)
            else:
                module = 'models.' + family
                classname = 'Transolver' if task == 'airfrans' and family == 'Transolver' else 'Model'
                if family == 'Transolver': kw.update(space_dim=7, fun_dim=0, out_dim=4, ref=8, unified_pos=task=='airfrans')
            if family == 'CDLNO':
                kw.update(front_blocks=2, latent_ffn_ratio=2, cdpa_mode='entry',
                          cdpa_source_chunk_size=0, front_latent_mode=mode)
            rows.append(dict(name=f'{family.lower()}_{task}_{mode or "original"}', task=task,
                             family=family, module=module, classname=classname, kwargs=kw, device=device))
    if project == 'standard':
        for mode in MODES:
            for cdpa in ('off','entry','every_block'):
                rows.append(dict(name=f'core_{mode}_{cdpa}', task='core', family='CDLNO',
                                 module='cdlno.core', classname='CDLNO', device='cpu',
                                 kwargs=dict(L=8,F=2,M=4,d_model=8,num_heads=2,output_dim=2,
                                             front_latent_mode=mode,cdpa_mode=cdpa)))
    return rows


def build(row):
    cls = getattr(importlib.import_module(row['module']), row['classname'])
    if row['task'] == 'core':
        from cdlno.config import CDLNOArchitectureConfig
        model = cls(CDLNOArchitectureConfig(**row['kwargs']))
    else: model = cls(**row['kwargs'])
    return model.to(row['device']).eval()


def make_inputs(row):
    task, kw = row['task'], row['kwargs']
    if task == 'core': return [dict(h=torch.randn(2,11,8))]
    if task in STEMS:
        n = 972 if task == 'elasticity' else kw['H'] * kw['W']
        index = torch.linspace(-.7,1.4,n)
        x = torch.stack((index.sin()+index.square(), index.cos()-index),-1).repeat(2,1,1)
        return [dict(x=x, fx=torch.randn(2,n,kw['fun_dim']) if kw['fun_dim'] else None,
                     T=torch.tensor([[.15],[.8]]) if task=='plasticity' else None)]
    return [dict(x=torch.randn(n,7),pos=torch.randn(n,3 if task=='car' else 2),
                 y=torch.randn(n,4),surf=torch.arange(n)%2==0,
                 edge_index=torch.stack((torch.arange(n-1),torch.arange(1,n)))) for n in (11,19)]


def forward(model, inputs, task, device, grad=False):
    moved = {k: v.clone().to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    if grad:
        for k in ('h','x','fx','T'):
            if isinstance(moved.get(k),torch.Tensor): moved[k].requires_grad_(True)
    if task == 'core': output=model(moved['h'])
    elif task in STEMS: output=model(moved['x'],moved['fx'],moved['T'])
    else:
        from torch_geometric.data import Batch, Data
        graph=Batch.from_data_list([Data(**moved)])
        output=model((graph,Data(pos=torch.randn(3,3,device=device)))) if task=='car' else model(graph)
    return output, moved


def evaluate(model, data, row):
    outputs=[]
    with torch.no_grad():
        for item in data:
            y,_=forward(model,item,row['task'],row['device'])
            assert torch.isfinite(y).all()
            outputs.append(y.cpu())
    model.zero_grad(set_to_none=True)
    y,variables=forward(model,data[0],row['task'],row['device'],grad=True)
    # Diagnostic gradient only: this is NOT an original task loss/train step.
    y.square().mean().backward()
    parameters={k: None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
    inputs={k: None if v.grad is None else v.grad.detach().cpu().clone()
            for k,v in variables.items() if isinstance(v,torch.Tensor) and v.requires_grad}
    for v in [*parameters.values(),*inputs.values()]:
        assert v is None or torch.isfinite(v).all()
    model.zero_grad(set_to_none=True)
    return dict(outputs=outputs,parameter_gradients=parameters,input_gradients=inputs)


def schema(state):
    return {k: dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest())
            for k,v in state.items()}


def make_run(row, directory, model=None, evaluation=False, chunk=0):
    task,kw=row['task'],row['kwargs']
    run_flag='--cdlno-run-dir' if task in STEMS else '--run_dir'
    extras=[run_flag,str(directory),'--n-hidden',str(kw['n_hidden']), '--n-heads',str(kw['n_head']),
            '--n-layers',str(kw['n_layers']),'--slice_num',str(kw['slice_num']),
            '--front-blocks',str(kw['front_blocks']),'--cdpa-mode',kw['cdpa_mode'],
            '--cdpa-source-chunk-size',str(chunk)]
    if not evaluation: extras+=['--front-latent-mode',kw['front_latent_mode']]
    args=parse_task(task,extras,evaluation)
    assert args.front_latent_mode==kw['front_latent_mode']
    with contextlib.redirect_stdout(io.StringIO()):
        if task in STEMS:
            from cdlno_entry import StaticRun
            run=StaticRun(args,model)
        elif task=='car':
            from models.cdlno_run import CarRun
            run=CarRun(args,device=row['device'],model=model,evaluation=evaluation)
        else:
            import yaml
            from cdlno_entry import AirRun
            run=AirRun(args,yaml.safe_load(Path('params.yaml').read_text())['CDLNO'],
                       device=row['device'],evaluation=evaluation)
    return run,args,extras


def compare_trees(a,b,*,atol,rtol):
    if a is None or b is None: assert a is b;return
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare_trees(a[k],b[k],atol=atol,rtol=rtol)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_trees(x,y,atol=atol,rtol=rtol)
    else:torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def process(row, root, action, seed):
    path=root/row['name']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if row['device'].startswith('cuda') and not torch.cuda.is_available():
        return dict(name=row['name'],status='not_run',reason='unmodified baseline needs CUDA; no GPU')
    if row['task'] in ('car','airfrans'):
        from torch_geometric.data import Data, Batch  # real PyG, no stub
    model=build(row)
    source=Path(sys.modules[row['module']].__file__).resolve()
    if action=='capture':
        path.mkdir(parents=True,exist_ok=False)
        data=make_inputs(row)
        expected=evaluate(model,data,row)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(dict(state_dict=state,inputs=data,expected=expected),path/'fixture.pt')
        metadata=dict(**row,seed=seed,actual_class=type(model).__module__+'.'+type(model).__name__,
                      source=str(source),architecture=model.config.to_dict() if hasattr(model,'config') else None,
                      adapter=model.adapter_architecture() if hasattr(model,'adapter_architecture') else None,
                      state_dict=schema(state),parameters=sum(p.numel() for p in model.parameters()),
                      gradient_loss='diagnostic mean(output**2); no optimizer or task training',
                      outputs=[list(x.shape) for x in expected['outputs']],
                      tolerance=dict(atol=0.,rtol=0.),
                      cross_backend_tolerance=dict(atol=1e-5,rtol=3e-4))
        if row['family']=='CDLNO' and row['task']!='core':
            run,args,_=make_run(row,path/'run',model)
            metadata['run_arguments']=plain(vars(args))
            if row['task'] in STEMS:run.save(model)
            elif row['task']=='car':torch.save(model,run.checkpoint)
            else:
                torch.save([model],run.checkpoint)
                member=Path(run.member_dir(0));member.mkdir();torch.save(model,member/'model')
        elif row['task'] in ('car','airfrans'):
            # Preserve original whole-object/list formats for baseline as well.
            torch.save(model,path/'model_object.pt')
            if row['task']=='airfrans':torch.save([model],path/'model_list.pt')
        write_json(path/'metadata.json',metadata)
        return dict(name=row['name'],status='captured',device=row['device'],parameters=metadata['parameters'])
    stored=torch.load(path/'fixture.pt',map_location='cpu',weights_only=True)
    meta=json.loads((path/'metadata.json').read_text())
    assert schema(stored['state_dict'])==meta['state_dict']
    assert list(model.state_dict())==list(stored['state_dict'])
    model.load_state_dict(stored['state_dict'],strict=True)
    now=evaluate(model,stored['inputs'],row)
    compare_trees(now,stored['expected'],**meta['tolerance'])
    roundtrips=[];negative=[]
    if row['family']=='CDLNO' and row['task']!='core':
        before=(path/'run/architecture.json').read_bytes()
        for chunk in (0,1):
            run,_,extras=make_run(row,path/'run',model,True,chunk)
            if row['task'] in STEMS:run.load(model);loaded=[model]
            elif row['task']=='car':loaded=[run.load()]
            else:loaded=[*run.load(),run.load(member=0)]
            for obj in loaded:
                obj.eval()
                out=evaluate(obj,stored['inputs'],row)
                compare_trees(out,stored['expected'],**(meta['tolerance'] if chunk==0 else meta['cross_backend_tolerance']))
                roundtrips.append(dict(chunk=chunk,classname=type(obj).__module__+'.'+type(obj).__name__))
        from cdlno.checkpoint import SidecarMismatch
        for mode in MODES:
            if mode==row['kwargs']['front_latent_mode']:continue
            try:parse_task(row['task'],[*extras,'--front-latent-mode',mode],True)
            except SidecarMismatch:negative.append(mode)
            else:raise AssertionError('explicit mode mismatch accepted')
        assert (path/'run/architecture.json').read_bytes()==before
    elif row['task'] in ('car','airfrans'):
        obj=torch.load(path/'model_object.pt',weights_only=False,map_location=row['device']).eval()
        compare_trees(evaluate(obj,stored['inputs'],row),stored['expected'],**meta['tolerance'])
        roundtrips.append(dict(kind='trusted original whole object'))
        if row['task']=='airfrans':
            objs=torch.load(path/'model_list.pt',weights_only=False,map_location=row['device'])
            compare_trees(evaluate(objs[0].eval(),stored['inputs'],row),stored['expected'],**meta['tolerance'])
            roundtrips.append(dict(kind='trusted original list'))
    return dict(name=row['name'],status='passed',device=row['device'],tolerance=meta['tolerance'],
                max_output_abs=max((a-b).abs().max().item() for a,b in zip(now['outputs'],stored['expected']['outputs'])),
                input_and_parameter_gradients='compared including None',roundtrips=roundtrips,
                explicit_mode_conflicts_rejected=negative)


def worker(options):
    import cdlno
    assert Path(cdlno.__file__).resolve()==ROOT/'cdlno/__init__.py'
    assert Path.cwd()==ROOT/PROJECTS[options.project]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    rows=[]
    with sdpa_kernel(SDPBackend.MATH):
        for i,row in enumerate(cases(options.project)):
            try:result=process(row,options.artifacts,options.action,2026091500+i)
            except Exception as exc:
                result=dict(name=row['name'],status='failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(result)
            print(row['name'],result['status'],flush=True)
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda_build=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     sdpa='MATH forced',dtype='float32',threads=1,tf32=False,amp=False,compile=False,
                     cudnn_benchmark=False,cudnn_deterministic=True,cwd=str(Path.cwd()),package=str(cdlno.__file__))
    write_json(options.result,dict(environment=environment,action=options.action,rows=rows))
    return int(any(r['status']=='failed' for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('capture','replay'))
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--project',choices=tuple(PROJECTS))
    o=p.parse_args();o.artifacts=o.artifacts.resolve();o.result=o.result.resolve()
    if o.project:return worker(o)
    o.artifacts.mkdir(parents=True,exist_ok=True)
    results=[]
    for project,cwd in PROJECTS.items():
        target=o.result.with_name(o.result.stem+'-'+project+'.json')
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/cwd),PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),o.action,'--artifacts',str(o.artifacts),
             '--result',str(target),'--project',project]
        completed=subprocess.run(cmd,cwd=ROOT/cwd,env=env,text=True,capture_output=True)
        target.with_suffix('.log').write_text(completed.stdout+completed.stderr)
        results.append(dict(project=project,command=cmd,cwd=str(ROOT/cwd),returncode=completed.returncode,
                            result=str(target),summary=json.loads(target.read_text()) if target.exists() else completed.stderr))
        print(project,'exit',completed.returncode,flush=True)
    write_json(o.result,results)
    return int(any(x['returncode'] for x in results))


if __name__=='__main__':raise SystemExit(main())
