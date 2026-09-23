"""Fresh-process native industrial synthetic PyG checkpoint/continuation worker."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch_geometric.data import Data
from tran_evaluate.linearno_loop.launch import native_parse
from tran_evaluate.linearno_loop.recording import observe,state_hash
from cdlno.linearno_loop.industrial_state import construct
from cdlno.experiment import start,finish,session


def run(task,mode,action,directory,report):
    torch.set_num_threads(1)
    project=ROOT/('Airfoil-Design-AirfRANS' if task=='airfrans' else 'Car-Design-ShapeNetCar')
    sys.path.insert(0,str(project))
    import train as native_train
    tokens=['--model' if task=='airfrans' else '--cfd_model','LinearNO','--experiment-dir',str(directory),'--device','cpu']
    if action in ('train','interrupt'):
        tokens += ['--linearno-loop','1','--linearno-loop-architecture','resmlp_dual_temp_v4',
                   '--linearno-loop-temperature-mode',mode,'--seed','17',
                   '--epochs' if task=='airfrans' else '--nb_epochs','2']
        if task=='car':tokens+=['--fold_id','3','--val_iter','1']
    elif action=='resume':tokens+=['--resume']
    args=native_parse(task,'eval' if action=='eval' else 'train',
                      'main_evaluation.py' if action=='eval' else 'main.py',tokens)
    cfg=args._linearno_loop_config
    generator=torch.Generator().manual_seed(934)
    graph=Data(x=torch.randn(9,7,generator=generator),pos=torch.randn(9,2 if task=='airfrans' else 3,generator=generator),
        y=torch.randn(9,4,generator=generator),surf=torch.tensor([True,False]*4+[True]))
    geom=Data(pos=torch.randn(5,3,generator=generator),x=torch.randn(5,3,generator=generator))
    coef=(np.zeros(7 if task=='car' else 4),np.ones(7 if task=='car' else 4),np.zeros(4),np.ones(4))
    start(args,task,evaluation=action=='eval')
    recorder=session(args)
    class Interrupted(Exception):pass
    counters=dict(backward=0)
    original_backward=torch.Tensor.backward
    def backward(value,*a,**kw):
        assert torch.isfinite(value).all();counters['backward']+=1
        return original_backward(value,*a,**kw)
    if task=='airfrans':
        from cdlno.linearno_loop import air_entry
        fake=dict(split='SYNTHETIC fixed',sampling='synthetic fixed',checksums={'manifest.json':'a'*64},task_variant='full',
          steps_per_epoch=1,scheduler_total_steps=4,manifest_hash='a'*64,train_count=1,validation_count=1,manifest_keys=[])
        with patch.object(air_entry,'_data_spec',return_value=fake):
            run=air_entry.AirRun(args,args._linearno_config,Path('/tmp/SYNTHETIC'),coef,recorder,manifest={},evaluation=action=='eval')
        hparams=dict(batch_size=1,nb_epochs=2,lr=args.lr,subsampling=9,r=.05,max_neighbors=64,debug=0)
        if action=='eval':
            models=run.load_models();model=models[0];prediction=model(graph)
            recorder.attach_model(model,member=0)
            from torch_geometric.loader import DataLoader
            with torch.no_grad():metric=native_train.test('cpu',model,DataLoader([graph],batch_size=1),criterion='MSE')
            recorder.record_metrics(dict(synthetic_native_test=[np.asarray(v).tolist() for v in metric]))
        else:
            models=[]
            original=run.complete_epoch
            def complete(epoch,model,history):
                original(epoch,model,history)
                if action=='interrupt':raise Interrupted()
            run.complete_epoch=complete
            try:
                with patch.object(torch.Tensor,'backward',backward):
                    for member in range(args.nmodel):
                        run.current_member=member;model=construct(args,member);observe(args,model,member)
                        recorder.attach_model(model,member=member)
                        native_train.main('cpu',[graph],[graph],model,hparams,str(run.member_dir),criterion='MSE_weighted',
                            reg=args.weight,val_iter=10,name_mod='LinearNO',val_sample=True,linearno_run=run)
                        models.append(model)
                run.finish_ensemble(models)
            except Interrupted:session(args).finish(status='interrupted',error='intentional synthetic boundary')
            prediction=model(graph)
            if len(models)>1:assert not ({id(p) for p in models[0].parameters()} & {id(p) for p in models[1].parameters()})
    else:
        from cdlno.linearno_loop import car_entry
        fake=dict(split='SYNTHETIC fixed',sampling='synthetic fixed',checksums={'raw':'b'*64},fold_id=3,
          train_samples=['a'],test_samples=['b'],cfd_mesh=False,r=.2,val_iter=1,preprocessed=1,ntrain=1,ntest=1)
        run=car_entry.CarRun(args,fake,coef,recorder)
        hparams=dict(lr=args.lr,batch_size=1,nb_epochs=2)
        if action=='eval':
            model=run.load();recorder.attach_model(model);prediction=model((graph,geom))
            with torch.no_grad():loss,pressure,velocity=run.objective(prediction,graph)
            recorder.record_metrics(dict(synthetic_total=loss.item(),pressure=pressure.item(),velocity=velocity.item()))
        else:
            model=construct(args);observe(args,model);recorder.attach_model(model)
            original=run.complete_epoch
            def complete(epoch,model,metrics):
                original(epoch,model,metrics)
                if action=='interrupt':raise Interrupted()
            run.complete_epoch=complete
            try:
                with patch.object(torch.Tensor,'backward',backward):
                    native_train.main('cpu',[(graph,geom)],[(graph,geom)],model,hparams,str(directory),reg=.5,
                        val_iter=1,coef_norm=coef,record=None,linearno_run=run)
            except Interrupted:session(args).finish(status='interrupted',error='intentional synthetic boundary')
            prediction=model((graph,geom))
    assert prediction.shape==(9,4) and torch.isfinite(prediction).all()
    if action!='interrupt':finish(args)
    report.write_text(json.dumps(dict(task=task,mode=mode,action=action,synthetic=True,real_pyg=True,
        counters=counters,parameters=sum(p.numel() for p in model.parameters()),weights_hash=state_hash(model.state_dict()),
        shape=list(prediction.shape),config_hash=cfg['config_hash'],nmodel=getattr(args,'nmodel',1)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('mode');p.add_argument('action')
    p.add_argument('directory',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();run(a.task,a.mode,a.action,a.directory,a.report)
