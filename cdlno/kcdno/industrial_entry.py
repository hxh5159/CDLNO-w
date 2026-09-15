"""Industrial new-family selection and LOCAL trusted whole-object loading."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from .config import KCDNORuntimeConfig
from .entry import ROOT, task_record
from .metadata import KCDNOMetadataMismatch, save_metadata
from .options import explicit_arguments

from .families import (architecture_from_dict, make_metadata, load_metadata,
                       resolve_training, resolve_evaluation, new_run_path)

CAR_CONTRACT = ('fold_id', 'nb_epochs', 'weight', 'cfd_mesh', 'r')


def resolve_car(parser, args, tokens, *, evaluation):
    explicit = explicit_arguments(parser, tokens)
    if args.kcdno_run_dir is None and args.run_dir is not None:
        args.kcdno_run_dir = args.run_dir
    elif args.run_dir is not None and args.kcdno_run_dir != args.run_dir:
        raise ValueError('conflicting --run_dir and --kcdno-run-dir')
    args.kcdno_task, args.kcdno_family, args.kcdno_evaluation = 'car', args.cfd_model, evaluation
    if evaluation:
        if args.kcdno_run_dir is None:
            parser.error('kcdno evaluation requires --kcdno-run-dir')
        saved = resolve_evaluation(args.kcdno_run_dir/'architecture.json', explicit, task='car').saved
        cfg = saved.architecture;args.profile = saved.profile
        task = task_record(args.kcdno_run_dir)
        if task['task'] != 'car' or set(task['protocol']) != set(CAR_CONTRACT):
            raise KCDNOMetadataMismatch('missing Car run contract')
        for name in CAR_CONTRACT:
            if name in explicit and explicit[name] != task['protocol'][name]:
                raise KCDNOMetadataMismatch('Car run contract mismatch: ' + name)
            setattr(args, name, task['protocol'][name])
    else:
        cfg = resolve_training('car', explicit)
        preset=json.loads((ROOT/'Car-Design-ShapeNetCar/configs/kcdno/shapenet_car.json').read_text())
        for name,value in preset['training'].items():
            if name not in explicit:setattr(args,name,value)
    if cfg.point_module != 'point_ffn' or not 0 <= args.fold_id <= 8 or args.nb_epochs < 1:
        raise ValueError('invalid Car point module/fold/epoch configuration')
    if not evaluation and args.batch_size != 1:
        raise ValueError('Car supports single-graph batch_size=1')
    args.batch_size = getattr(args, 'batch_size', 1)
    args.kcdno_architecture = cfg.to_dict()
    for name, value in dict(n_hidden=cfg.d,n_layers=cfg.L,n_heads=cfg.h,slice_num=cfg.M,
                            kernel_rank=getattr(cfg, 'kernel_rank', None),history_mode=getattr(cfg, 'history_mode', None),dropout=0.).items():
        setattr(args,name,value)
    if not evaluation and args.kcdno_run_dir is None:
        proposed=new_run_path(Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output')),'car',cfg)
        digest=hashlib.sha256(json.dumps(cfg.to_dict(),sort_keys=True).encode()).hexdigest()[:10]
        args.kcdno_run_dir=proposed.parent/args.profile/(proposed.name+'_'+digest)
    return args


def car_model_kwargs(args):
    return dict(config=architecture_from_dict(args.kcdno_architecture))


class CarRun:
    def __init__(self, args, *, device, model=None, evaluation=False):
        from ..experiment import session, reserve_directory, timestamp
        self.args,self.device,self.evaluation=args,device,evaluation
        self.directory=Path(args.kcdno_run_dir).resolve()
        self.architecture=architecture_from_dict(args.kcdno_architecture)
        self.contract={key:getattr(args,key) for key in CAR_CONTRACT}
        self.recorder=session(args)
        self.metadata=make_metadata(task='car',architecture=self.architecture,checkpoint_format='whole_model',
            profile=args.profile,runtime=KCDNORuntimeConfig(device=str(device),batch_size=1))
        if evaluation:
            self.validate()
        else:
            self._check_model(model)
            reserve_directory(args,self.directory)
            save_metadata(self.sidecar,self.metadata)
            self.adapter=model.adapter_architecture()
            with (self.directory/'task.json').open('x') as stream:
                json.dump(dict(schema_version=1,family=self.architecture.family,task='car',adapter=self.adapter,
                    protocol=self.contract,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),stream,indent=2)
        self.result_dir=str(self.directory/('eval_'+timestamp()))+'/'
        if self.recorder is not None:
            self.result_dir=self.recorder.result_dir
            if model is not None:self.recorder.attach_model(model,protocol=self.contract)
        print(self.architecture.family+' Car run:',self.directory,self.architecture.to_dict())

    @property
    def sidecar(self):return self.directory/'architecture.json'
    @property
    def checkpoint(self):return self.directory/f'model_{self.contract["nb_epochs"]}.pth'

    def validate(self):
        from models.KCDNO import Model
        from models.CDLNO import adapter_architecture
        saved=load_metadata(self.sidecar);record=task_record(self.directory)
        expected=adapter_architecture() | {'version':'kcdno-shapenet-car-v1','output_head':'layernorm-linear-v1'}
        if (saved.architecture!=self.architecture or saved.task!='car' or saved.checkpoint_format!='whole_model'
                or record['family']!=self.architecture.family or record['task']!='car' or record['adapter']!=expected or record['protocol']!=self.contract):
            raise KCDNOMetadataMismatch('Car model/task/run architecture mismatch')

    def _check_model(self,model):
        from models.KCDNO import Model
        from .matched import validate_core
        if type(model) is not Model or model.config != self.architecture:
            raise KCDNOMetadataMismatch('whole-model family/configuration mismatch')
        try: validate_core(model.core,self.architecture)
        except ValueError as e: raise KCDNOMetadataMismatch(str(e)) from e

    def load(self):
        import torch
        from models.KCDNO import Model
        if not self.evaluation:raise RuntimeError('load is for evaluation of a trusted run')
        self.validate()  # Always before deserializing the trusted project object.
        model=torch.load(self.checkpoint,map_location=self.device,weights_only=False)
        self._check_model(model)
        with torch.random.fork_rng(devices=[]):
            expected=Model(**car_model_kwargs(self.args))
        from .loading import validate_whole_model
        validate_whole_model(model,expected)
        if model.adapter_architecture()!=task_record(self.directory)['adapter']:
            raise KCDNOMetadataMismatch('whole-model wrapper contract mismatch')
        model=model.to(self.device)
        if self.recorder is not None:self.recorder.attach_model(model,protocol=self.contract)
        return model
