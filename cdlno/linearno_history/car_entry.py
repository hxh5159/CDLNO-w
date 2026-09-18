"""Research industrial adapter. Native protocol copied at R8 with audited routing only."""
from __future__ import annotations
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from uuid import uuid4
import numpy as np
import torch
from cdlno.linearno.shapenet import ShapeNetLinearNO
from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES, resolve_config, require_resolved_objective, digest
from cdlno.linearno.schema import (make_metadata, read_metadata, write_metadata, normalizer_record,
                     restore_numerical_state, unpack_state)
from cdlno.linearno.checkpoint import (inspect_checkpoint, read_pair, strict_load, save_pair, sha256,
                         resume_state, restore_random_state)
from cdlno.linearno.car_metrics import fields, relative_l2, field_metrics, drag_pair
from cdlno.linearno.car_entry import constructor_kwargs, restore_coef_norm, inspect_data, load_data, task_provenance, evaluate
from .industrial import (finish_args, check_structure, read_metadata, write_metadata, make_metadata, inspect_checkpoint, read_pair, save_pair, construct, generators, verify_resume_source)
from .factory import build_model
from .checkpoint import config_from_metadata
from .provenance import research_provenance
ROOT = Path(__file__).resolve().parents[2]
CLASS_PATH = 'cdlno.linearno.shapenet.ShapeNetLinearNO'

def check_structure(metadata):
    from .checkpoint import validate_metadata
    validate_metadata(metadata)
    if metadata['profile_spec']['task'] != 'car':
        raise ValueError('Car checkpoint task mismatch')

def parse_args(parser, tokens, *, evaluation=False, history_features=None):
    parser.add_argument('--linearno-profile', choices=PROFILES, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-variant', choices=('shapenet',), default=argparse.SUPPRESS)
    for field in ('rank', 'hidden', 'heads', 'layers', 'ffn-ratio'):
        parser.add_argument('--linearno-' + field, type=int, default=argparse.SUPPRESS)
    parser.add_argument('--linearno-dropout', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--experiment-dir', '--linearno-run-dir', dest='linearno_run_dir', type=Path)
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--linearno-field-metric', choices=('physical_rL2', 'normalized_MSE'), default=argparse.SUPPRESS)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(tokens)
    probe = copy.deepcopy(parser); probe._defaults.clear()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    supplied = vars(probe.parse_args(tokens))
    mapping = dict(linearno_rank='model.linearno_rank', linearno_variant='model.linearno_variant',
        linearno_hidden='model.hidden', linearno_heads='model.heads', linearno_layers='model.layers',
        linearno_ffn_ratio='model.ffn_ratio', linearno_dropout='model.dropout', lr='training.lr',
        nb_epochs='training.epochs', batch_size='training.batch_size', preprocessed='training.preprocessed',
        seed='runtime.seed', save_name='runtime.save_name', weight='objective.surface_weight')
    explicit = {path: supplied[k] for k, path in mapping.items() if k in supplied}
    if 'linearno_field_metric' in supplied:
        explicit['evaluation.field_metric.primary'] = supplied['linearno_field_metric']
    try:
        forbidden = set(supplied) & {'n_hidden','n_layers','n_heads','slice_num','mlp_ratio','cdpa_mode',
            'front_blocks','front_latent_mode','history_mode','kernel_rank','profile','dropout','unified_pos'}
        if forbidden:
            raise ValueError('LinearNO requires its own model overrides; incompatible flags: ' + str(sorted(forbidden)))
        if args.cfd_model != 'LinearNO':
            raise ValueError('Car LinearNO requires --cfd_model LinearNO')
        if evaluation and args.resume:
            raise ValueError('evaluation and resume are separate actions')
        args.eval = int(evaluation)
        selector = args.checkpoint or ('latest' if args.resume else 'final')
        args.checkpoint = selector
        if evaluation or args.resume:
            if args.linearno_run_dir is None:
                raise ValueError('LinearNO eval/resume requires --experiment-dir; metadata cannot be inferred')
            args.linearno_run_dir = args.linearno_run_dir.resolve()
            root = read_metadata(args.linearno_run_dir/'architecture.json', constructor=ShapeNetLinearNO)
            check_structure(root)
            saved, manifest = inspect_checkpoint(args.linearno_run_dir, selector, ShapeNetLinearNO)
            check_structure(saved)
            for key in ('model_spec','profile_spec','data_spec','normalizer_spec'):
                if root[key] != saved[key]:
                    raise ValueError(f'root/checkpoint {key} mismatch')
            cfg = saved['profile_spec']
            if 'linearno_profile' in supplied and args.linearno_profile != cfg['profile']:
                raise ValueError('explicit profile conflicts with checkpoint')
            for path, value in explicit.items():
                expected = cfg['values']
                for part in path.split('.'):
                    expected = expected[part]
                if value != expected:
                    raise ValueError(f'explicit {path} conflicts with checkpoint')
            for field in ('fold_id','cfd_mesh','r','val_iter'):
                if field in supplied and supplied[field] != saved['data_spec'][field]:
                    raise ValueError(f'explicit {field} conflicts with checkpoint data protocol')
                setattr(args, field, saved['data_spec'][field])
            args._linearno_metadata, args._linearno_checkpoint = saved, manifest
        else:
            cfg = resolve_config('car', getattr(args,'linearno_profile',DEFAULT_PROFILE),
                                 explicit=explicit)
        require_resolved_objective(cfg)
        if not 0 <= args.fold_id <= 8 or cfg['values']['training']['batch_size'] != 1:
            raise ValueError('Car supports fold_id 0..8 and batch_size=1 only')
        if cfg['values']['training']['preprocessed'] not in (0,1):
            raise ValueError('preprocessed must be 0 or 1')
        if cfg['values']['model']['unified_pos']:
            raise ValueError('the Car task uses the released non-unified 7-feature wrapper')
        tr = cfg['values']['training']
        args.nb_epochs, args.lr, args.batch_size = tr['epochs'],tr['lr'],tr['batch_size']
        args.preprocessed, args.weight = tr['preprocessed'],cfg['values']['objective']['surface_weight']
        args.seed = cfg['values']['runtime']['seed']
        args.val_iter = getattr(args,'val_iter',10)
        if args.val_iter < 1:
            raise ValueError('val_iter must be positive')
        args.linearno_profile = cfg['profile']
        args.linearno_family, args.linearno_task = 'linearno','car'
        args._linearno_config = cfg
        args._linearno_model_spec = dict(class_path=CLASS_PATH,constructor_kwargs=constructor_kwargs(cfg))
        if args.linearno_run_dir is None:
            m = cfg['values']['model']
            leaf = args.save_name or uuid4().hex
            args.linearno_run_dir = Path(os.environ.get('CDLNO_RUNS_ROOT',ROOT/'output'))/'car'/'linearno'/cfg['profile']/f"shapenet_M{m['linearno_rank']}_eval{digest(cfg['values']['evaluation'])[:10]}"/f'seed{args.seed}_{leaf}'
        args.linearno_run_dir = args.linearno_run_dir.resolve()
        args.device = args.device or (f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        return finish_args(args, history_features or {}, tokens)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))

class CarRun:
    def __init__(self,args,data_spec,coef_norm,recorder=None):
        self.args,self.config,self.directory = args,args._linearno_config,Path(args.linearno_run_dir)
        self.data,self.coef,self.recorder = copy.deepcopy(data_spec),coef_norm,recorder
        self.completed_epoch = 0
        checksum = digest(self.data['checksums'])
        self.data['checksums']['normalizer_fit_dataset'] = checksum
        self.normalizers = dict(policy='saved_train_fit',records=dict(coef_norm=normalizer_record(
            dict(zip(('input_mean','input_std','output_mean','output_std'),coef_norm)),
            fit_split='train_samples only',data_checksum=checksum,
            algorithm='native all-point streaming mean/std; encode=(x-mean)/(std+1e-8); decode=x*std+mean')))
        tr = self.config['values']['training']
        self.steps = self.data['ntrain']//tr['batch_size']
        self.data['scheduler'] = dict(epochs=tr['epochs'],actual_steps_per_epoch=self.steps,
            total_steps=(self.steps+1)*tr['epochs'],final_div_factor=1000,drop_last=True)
        self.provenance = research_provenance(task_provenance()) if args.linearno_family == 'linearno_history' else task_provenance()
        self.generators = generators(args)
        if hasattr(args,'_linearno_metadata'):
            for key,actual in (('data_spec',self.data),('normalizer_spec',self.normalizers)):
                if args._linearno_metadata[key] != actual:
                    raise ValueError('Car saved '+key+' differs')

    def metadata(self,optimizer,scheduler,epoch):
        return make_metadata(profile_spec=self.config,constructor=ShapeNetLinearNO,
            model_spec=self.args._linearno_model_spec,data_spec=self.data,
            objective_spec=self.config['values']['objective'],evaluation_spec=self.config['values']['evaluation'],
            provenance_spec=self.provenance,normalizer_spec=self.normalizers,
            resume_state=resume_state(optimizer,scheduler,epoch,self.steps,self.args.nb_epochs,self.generators,
                dict(train='RandomSampler, drop_last=True, independent generator',test='SequentialSampler',epoch_boundary=True)),
            ensemble_manifest=[])

    def prepare(self,model,optimizer,scheduler):
        self.optimizer,self.scheduler = optimizer,scheduler
        if scheduler.total_steps != self.data['scheduler']['total_steps']:
            raise ValueError('Car OneCycle schedule differs from released total_steps')
        if self.recorder:
            self.recorder.record_training_setup(optimizer,scheduler)
        if self.args.resume:
            saved,state = read_pair(self.args._linearno_checkpoint,ShapeNetLinearNO)
            check_structure(saved)
            verify_resume_source(saved,self.provenance)
            strict_load(model,state)
            optimizer.load_state_dict(unpack_state(saved['resume_state']['optimizer']))
            scheduler.load_state_dict(unpack_state(saved['resume_state']['scheduler']))
            self.completed_epoch = saved['resume_state']['epoch']
            if (saved['resume_state']['global_step'] != self.completed_epoch*self.steps
                    or scheduler.last_epoch != self.completed_epoch*self.steps):
                raise ValueError('Car resume scheduler/global_step mismatch')
            restore_random_state(saved['resume_state'],self.generators)
        else:
            write_metadata(self.directory/'architecture.json',self.metadata(optimizer,scheduler,0),constructor=ShapeNetLinearNO)
        return self.completed_epoch

    def complete_epoch(self,epoch,model,metrics):
        if epoch != self.completed_epoch+1:
            raise ValueError('Car epoch checkpoint order mismatch')
        if self.recorder:
            self.recorder.record_epoch(epoch,metrics)
        save_pair(self.directory,model,self.metadata(self.optimizer,self.scheduler,epoch),ShapeNetLinearNO)
        self.completed_epoch = epoch

    def load(self):
        saved,state = read_pair(self.args._linearno_checkpoint,ShapeNetLinearNO)
        check_structure(saved)
        for key,expected in (('data_spec',self.data),('normalizer_spec',self.normalizers),('profile_spec',self.config)):
            if saved[key] != expected:
                raise ValueError('Car checkpoint '+key+' mismatch')
        model = build_model(config_from_metadata(saved))
        strict_load(model,state)
        return model.to(self.args.device).eval()

    def objective(self,out,data):
        spec = self.config['values']['objective']
        if spec['kind'] == 'MSE' and spec['space'] == 'normalized' and spec['volume_region'] == 'all points':
            pressure = (out[data.surf,3]-data.y[data.surf,3]).square().mean()
            velocity = (out[:,:3]-data.y[:,:3]).square().mean(0).mean()
        elif (spec['kind'] == 'relative_L2' and spec['space'] == 'physical'
              and spec['reduction'] == 'flatten region channels per sample then sample mean; no epsilon'):
            pred,truth = fields(out,data.y,data.surf,self.coef)
            velocity = relative_l2(pred[~data.surf,:3],truth[~data.surf,:3])
            pressure = relative_l2(pred[data.surf,3],truth[data.surf,3])
        else:
            raise ValueError('Car objective space/reduction is unresolved or unsupported')
        return velocity+spec['surface_weight']*pressure,pressure,velocity

    def train(self,device,train_dataset,val_dataset,model,hparams,path,reg,val_iter,coef_norm,record):
        """Called by native train.main only with an explicit LinearNO run."""
        from torch_geometric.loader import DataLoader
        optimizer = torch.optim.Adam(model.parameters(),lr=hparams['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=hparams['lr'],
            total_steps=(len(train_dataset)//hparams['batch_size']+1)*hparams['nb_epochs'],final_div_factor=1000.)
        first = self.prepare(model,optimizer,scheduler)
        for epoch in range(first,hparams['nb_epochs']):
            model.train(); rows=[]
            loader = DataLoader(train_dataset,batch_size=hparams['batch_size'],shuffle=True,drop_last=True,generator=self.generators['train'])
            for data,geom in loader:
                data,geom = data.to(device),geom.to(device)
                optimizer.zero_grad()
                loss,pressure,velocity = self.objective(model((data,geom)),data)
                loss.backward();optimizer.step();scheduler.step()
                rows.append([loss.item(),pressure.item(),velocity.item()])
            total,pressure,velocity = np.mean(rows,axis=0).tolist()
            metrics = dict(total=total,pressure=pressure,velocity=velocity,objective=self.config['values']['objective'])
            if val_iter is not None and (epoch == hparams['nb_epochs']-1 or epoch%val_iter == 0):
                model.eval(); values=[]
                with torch.no_grad():
                    for data,geom in DataLoader(val_dataset,batch_size=1,generator=self.generators['test']):
                        data,geom=data.to(device),geom.to(device)
                        values.append([v.item() for v in self.objective(model((data,geom)),data)])
                metrics['validation_components']=np.mean(values,axis=0).tolist()
            if record:
                record.visualize(model,epoch+1,hparams['nb_epochs'],dataset=val_dataset,coef_norm=coef_norm)
            print('LinearNO Car epoch',epoch+1,metrics)
            self.complete_epoch(epoch+1,model,metrics)
        # Preserve industrial local whole-object naming; metadata/state pairs are
        # authoritative for new-family eval/resume, never external pickle fallback.
        torch.save(model,Path(path)/f'model_{hparams["nb_epochs"]}.pth')
        return model

def run_cli(args):
    from cdlno.experiment import start,finish
    import train as native_train
    hparams=dict(lr=args.lr,batch_size=args.batch_size,nb_epochs=args.nb_epochs)
    recorder=start(args,'car',evaluation=bool(args.eval),hparams=hparams)
    data_spec=inspect_data(args)
    train_dataset,test_dataset,coef=load_data(args,data_spec)
    run=CarRun(args,data_spec,coef,recorder)
    model=run.load() if args.eval else construct(args).to(args.device)
    recorder.attach_model(model,hparams=hparams,protocol=data_spec)
    if args.eval:
        result=evaluate(run,model,test_dataset)
    else:
        result=native_train.main(args.device,train_dataset,test_dataset,model,hparams,str(run.directory),
            reg=args.weight,val_iter=args.val_iter,coef_norm=coef,record=recorder,linearno_run=run)
    finish(args)
    return result
