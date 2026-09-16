"""Independent real-model fixtures; never imports an exp/main or reads datasets.

Capture happens after M1, before any MSAR math/entry edit. It is not backdated.
Each project runs in its original cwd to preserve local model pickle imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PROJECTS = {'standard':'PDE-Solving-StandardBenchmark', 'car':'Car-Design-ShapeNetCar',
            'airfrans':'Airfoil-Design-AirfRANS'}


def worker(args):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from cdlno.kcdno.config import KCDNOArchitectureConfig
    from cdlno.kcdno.matched_config import MatchedLRSAConfig
    from types import SimpleNamespace
    torch.set_num_threads(1)
    if args.project == 'standard':
        from model_dict import get_model
        tasks = ('darcy','elasticity','airfoil','pipe','ns','plasticity')
    else:
        from torch_geometric.data import Data
        from models.KCDNO import Model
        tasks = (args.project,)

    def make(task, family, cfg):
        if args.project == 'standard':
            cls = get_model(SimpleNamespace(model=family)).Model
            shape = (64,64) if task=='ns' else (101,31) if task=='plasticity' else (5,7)
            kw = {} if task=='elasticity' else dict(H=shape[0],W=shape[1])
            return cls(config=cfg,task_name=task,**kw)
        return Model(config=cfg)

    def inputs(task):
        if args.project == 'standard':
            n = 4096 if task=='ns' else 3131 if task=='plasticity' else 11 if task=='elasticity' else 35
            b = 1 if task in ('ns','plasticity') else 2
            f = 10 if task=='ns' else 1 if task in ('plasticity','darcy') else 0
            return dict(x=torch.randn(b,n,2),fx=torch.randn(b,n,f) if f else None,
                        T=torch.tensor([[.37]]) if task=='plasticity' else None)
        n=13
        return dict(x=torch.randn(n,7),pos=torch.randn(n,2 if task=='airfrans' else 3),
                    batch=torch.zeros(n,dtype=torch.long),ptr=torch.tensor([0,n]),
                    y=torch.randn(n,4),surf=torch.arange(n)%2==0)

    def call(model, data):
        if args.project=='standard':
            return model(data['x'],data['fx'],T=data['T'])
        graph=Data(**data)
        return model((graph,Data(pos=data['pos']))) if args.project=='car' else model(graph)

    rows=[]
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for task in tasks:
            for mode in ('all','off','matched'):
                family='lrsa_matched' if mode=='matched' else 'kcdno'
                config_cls=MatchedLRSAConfig if mode=='matched' else KCDNOArchitectureConfig
                folder=args.artifacts/(task+'_'+mode)
                if args.action=='capture':
                    torch.manual_seed(260917)
                    values=dict(L=2,d=8,h=2,M=4,point_module='point_ffn' if task in ('elasticity','car','airfrans') else 'conv_ffn')
                    if mode!='matched':values.update(kernel_rank=3,history_mode=mode)
                    cfg=config_cls(**values)
                    model=make(task,family,cfg).eval()
                    data=inputs(task);out=call(model,data)
                    fixture=dict(config=cfg.to_dict(),adapter=model.adapter_architecture(),
                                 inputs=data,weights=model.state_dict(),output=out.clone(),
                                 family=family,task=task,class_path=type(model).__module__+'.'+type(model).__qualname__)
                    folder.mkdir(parents=True,exist_ok=False)
                    with (folder/'fixture.pt').open('xb') as stream:torch.save(fixture,stream)
                    if args.project!='standard':
                        with (folder/'whole.pt').open('xb') as stream:torch.save(model,stream)
                        if args.project=='airfrans':
                            with (folder/'list.pt').open('xb') as stream:torch.save([model],stream)
                else:
                    fixture=torch.load(folder/'fixture.pt',map_location='cpu',weights_only=True)
                    cfg=config_cls.from_dict(fixture['config'])
                    model=make(task,family,cfg).eval()
                    model.load_state_dict(fixture['weights'],strict=True)
                    assert model.adapter_architecture()==fixture['adapter']
                    torch.testing.assert_close(call(model,fixture['inputs']),fixture['output'],atol=0,rtol=0)
                    if args.project!='standard':
                        # Only our explicitly captured local trusted fixture objects.
                        restored=torch.load(folder/'whole.pt',map_location='cpu',weights_only=False).eval()
                        assert type(restored) is type(model)
                        restored.load_state_dict(fixture['weights'],strict=True)
                        torch.testing.assert_close(call(restored,fixture['inputs']),fixture['output'],atol=0,rtol=0)
                        if args.project=='airfrans':
                            restored_list=torch.load(folder/'list.pt',map_location='cpu',weights_only=False)
                            assert type(restored_list) is list and len(restored_list)==1
                            torch.testing.assert_close(call(restored_list[0].eval(),fixture['inputs']),fixture['output'],atol=0,rtol=0)
                rows.append(dict(name=folder.name,status='passed',action=args.action,fixture=str(folder/'fixture.pt'),
                    class_path=fixture['class_path'],config=fixture['config'],adapter=fixture['adapter'],
                    state_keys_shapes={k:list(v.shape) for k,v in fixture['weights'].items()},
                    output_shape=list(fixture['output'].shape),atol=0,rtol=0,dtype='float32',backend='math',device='cpu',
                    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.pt'))},
                    checkpoint_formats=['state_dict'] if args.project=='standard' else ['state_dict','whole_model']+(['model_list'] if args.project=='airfrans' else []),
                    evidence='same weights and fixed synthetic inputs; no task data/loss/optimizer/resume claim'))
    args.result.write_text(json.dumps(rows,indent=2)+'\n')
    print(args.project,args.action,len(rows),'passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('capture','replay'))
    parser.add_argument('--project',choices=PROJECTS)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args();args.artifacts=args.artifacts.resolve();args.result=args.result.resolve()
    if args.project:
        sys.path.insert(0,str(Path.cwd()))
        sys.path.insert(0,str(ROOT))
        worker(args)
    else:
        results=[]
        for name,cwd in PROJECTS.items():
            result=args.result.with_name(args.result.stem+'-'+name+'.json')
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),args.action,'--project',name,
                 '--artifacts',str(args.artifacts),'--result',str(result)]
            p=subprocess.run(cmd,cwd=ROOT/cwd,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=False)
            results.append(dict(project=name,cwd=str(ROOT/cwd),command=cmd,returncode=p.returncode,
                                rows=json.loads(result.read_text()) if result.exists() else []))
        args.result.write_text(json.dumps(results,indent=2)+'\n')
        if any(p['returncode'] for p in results):raise SystemExit(1)


if __name__=='__main__':main()
