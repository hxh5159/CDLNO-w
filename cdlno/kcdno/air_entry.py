"""AirfRANS selection and local trusted model/list checkpoints; no dataset IO."""
import hashlib
import json
import os
from pathlib import Path

from .config import KCDNORuntimeConfig
from .entry import ROOT, task_record
from .metadata import KCDNOMetadataMismatch, save_metadata
from .options import explicit_arguments
from .airfrans import AirfRANSModel

from .families import (architecture_from_dict, make_metadata, load_metadata,
                       resolve_training, resolve_evaluation, new_run_path)


def resolve_air(parser, args, tokens, *, evaluation):
    import yaml
    explicit = explicit_arguments(parser, tokens)
    if args.kcdno_run_dir is None:
        args.kcdno_run_dir = args.run_dir
    elif args.run_dir is not None and args.run_dir != args.kcdno_run_dir:
        raise ValueError('conflicting --run_dir and --kcdno-run-dir')
    args.kcdno_task, args.kcdno_family, args.kcdno_evaluation = 'airfrans', args.model, evaluation
    if evaluation:
        if args.kcdno_run_dir is None:
            parser.error('kcdno eval requires --kcdno-run-dir')
        saved = resolve_evaluation(args.kcdno_run_dir/'architecture.json', explicit, task='airfrans').saved
        cfg, args.profile = saved.architecture, saved.profile
        record = task_record(args.kcdno_run_dir)
        if record['task'] != 'airfrans' or set(record['protocol']) != {'task','nmodel','weight','hparams'}:
            raise KCDNOMetadataMismatch('invalid AirfRANS run contract')
        contract = record['protocol']
        for name in ('task','nmodel','weight'):
            if name in explicit and explicit[name] != contract[name]:
                raise KCDNOMetadataMismatch('AirfRANS run contract mismatch: '+name)
            setattr(args, name, contract[name])
        hp = contract['hparams']
        for name in ('nb_epochs','batch_size','lr'):
            if name in explicit and explicit[name] != hp[name]:
                raise KCDNOMetadataMismatch('AirfRANS hparams mismatch: '+name)
    else:
        cfg = resolve_training('airfrans', explicit)
        hp = yaml.safe_load((ROOT/'Airfoil-Design-AirfRANS/params.yaml').read_text())['kcdno'].copy()
        for name in ('nb_epochs','batch_size','lr'):
            if name in explicit: hp[name] = explicit[name]
    if cfg.point_module != 'point_ffn' or args.task not in ('full','scarce','reynolds','aoa') or args.nmodel < 1:
        raise ValueError('invalid AirfRANS task/model configuration')
    if hp['batch_size'] != 1 or type(hp['nb_epochs']) is not int or hp['nb_epochs'] < 1:
        raise ValueError('AirfRANS requires batch_size=1 and positive nb_epochs')
    args.kcdno_hparams, args.kcdno_architecture = hp, cfg.to_dict()
    for name, value in dict(n_hidden=cfg.d,n_layers=cfg.L,n_heads=cfg.h,slice_num=cfg.M,
                            kernel_rank=getattr(cfg, 'kernel_rank', None),history_mode=getattr(cfg, 'history_mode', None),dropout=0.,
                            **{k: hp[k] for k in ('nb_epochs','batch_size','lr')}).items():
        setattr(args,name,value)
    if not evaluation and args.kcdno_run_dir is None:
        proposed = new_run_path(Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output')),'airfrans',cfg)
        digest = hashlib.sha256(json.dumps(cfg.to_dict(),sort_keys=True).encode()).hexdigest()[:10]
        args.kcdno_run_dir = proposed.parent/args.profile/(proposed.name+'_'+digest)
    return args


def model_kwargs(args):
    return dict(config=architecture_from_dict(args.kcdno_architecture))


def resolve_hparams(args, hparams):
    # Saved hparams win in eval; training already resolved explicit CLI over YAML.
    return dict(args.kcdno_hparams)


class AirRun:
    def __init__(self, args, hparams, *, device, evaluation=False):
        from ..experiment import session, reserve_directory, timestamp
        self.args, self.device, self.evaluation = args, device, evaluation
        self.directory = Path(args.kcdno_run_dir).resolve()
        self.architecture = architecture_from_dict(args.kcdno_architecture)
        self.kwargs, self.hparams = model_kwargs(args), resolve_hparams(args,hparams)
        self.contract = dict(task=args.task,nmodel=args.nmodel,weight=args.weight,hparams=self.hparams)
        self.recorder = session(args)
        self.metadata = make_metadata(task='airfrans',architecture=self.architecture,
            checkpoint_format='model_list',profile=args.profile,
            runtime=KCDNORuntimeConfig(device=str(device),batch_size=1))
        if evaluation:
            self.validate()
        else:
            from ..airfrans import adapter_architecture
            reserve_directory(args,self.directory)
            save_metadata(self.sidecar,self.metadata)
            adapter = adapter_architecture() | {'version':'kcdno-airfrans-v1','output_head':'layernorm-linear-v1'}
            with (self.directory/'task.json').open('x') as f:
                json.dump(dict(schema_version=1,family=self.architecture.family,task='airfrans',adapter=adapter,
                    protocol=self.contract,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),f,indent=2)
        self.result_dir = str(self.directory/('eval_'+timestamp()))
        if self.recorder is not None:self.result_dir = self.recorder.result_dir
        print(self.architecture.family+' AirfRANS run:',self.directory,self.architecture.to_dict())

    @property
    def sidecar(self):return self.directory/'architecture.json'
    @property
    def checkpoint(self):return self.directory/self.architecture.family
    def member_dir(self,index):
        if type(index) is not int or not 0 <= index < self.contract['nmodel']:
            raise ValueError('invalid model member index')
        return str(self.directory/f'member_{index:03d}')

    def validate(self):
        from ..airfrans import adapter_architecture
        saved, record = load_metadata(self.sidecar), task_record(self.directory)
        adapter = adapter_architecture() | {'version':'kcdno-airfrans-v1','output_head':'layernorm-linear-v1'}
        if (saved.architecture != self.architecture or saved.task != 'airfrans'
                or saved.checkpoint_format != 'model_list' or record['family'] != self.architecture.family or record['task'] != 'airfrans'
                or record['adapter'] != adapter or record['protocol'] != self.contract):
            raise KCDNOMetadataMismatch('AirfRANS model/task/run architecture mismatch')

    def load(self, member=None):
        import torch
        from .matched import validate_core
        if not self.evaluation:raise RuntimeError('load is for trusted evaluation runs')
        self.validate()
        path = self.checkpoint if member is None else Path(self.member_dir(member))/'model'
        result = torch.load(path,map_location=self.device,weights_only=False)
        models = result if member is None else [result]
        if type(models) is not list or len(models) != (self.contract['nmodel'] if member is None else 1):
            raise KCDNOMetadataMismatch('expected AirfRANS model list/member')
        for index,model in enumerate(models):
            if type(model) is not AirfRANSModel or model.config != self.architecture:
                raise KCDNOMetadataMismatch('AirfRANS whole-model family/configuration mismatch')
            try: validate_core(model.core,self.architecture)
            except ValueError as e: raise KCDNOMetadataMismatch(str(e)) from e
            with torch.random.fork_rng(devices=[]):expected = AirfRANSModel(**self.kwargs)
            from .loading import validate_whole_model
            validate_whole_model(model,expected)
            if model.adapter_architecture() != task_record(self.directory)['adapter']:
                raise KCDNOMetadataMismatch('AirfRANS adapter mismatch')
            models[index] = model.to(self.device)
            if self.recorder is not None:self.recorder.attach_model(models[index],member=index,protocol=self.contract)
        return models if member is None else models[0]
