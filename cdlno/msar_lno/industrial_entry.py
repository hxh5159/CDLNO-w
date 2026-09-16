"""Industrial CLI and local trusted whole-object/list protocol; no data IO."""
import json
import os
from pathlib import Path

from .config import FAMILY, MSARArchitectureConfig, MSARTrainingConfig, MSARRuntimeConfig
from .metadata import MSARMetadata, MSARMetadataMismatch, load_metadata, save_metadata, resolve_evaluation, new_run_path
from .options import parser_for_family, explicit_arguments, resolve_training
from .industrial import CarModel, AirfRANSModel, adapter_architecture

ROOT = Path(__file__).resolve().parents[2]
CAR_CONTRACT = ('fold_id', 'nb_epochs', 'weight', 'cfd_mesh', 'r')
AIR_CONTRACT = ('task', 'nmodel', 'weight', 'hparams')
CLASSES = {'car': CarModel, 'airfrans': AirfRANSModel}


def task_record(directory, task):
    value = json.loads((Path(directory)/'task.json').read_text())
    required = {'schema_version','family','task','adapter','protocol','arguments'}
    if (type(value) is not dict or value.keys() != required or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['family'] != FAMILY or value['task'] != task
            or not all(type(value[k]) is dict for k in ('adapter','protocol','arguments'))
            or value['adapter'] != adapter_architecture(task)
            or set(value['protocol']) != set(CAR_CONTRACT if task == 'car' else AIR_CONTRACT)):
        raise MSARMetadataMismatch('invalid MSAR industrial task/wrapper/protocol sidecar')
    return value


def parse_args(parser, tokens, *, task, evaluation=False):
    import yaml
    parser = parser_for_family(parser, FAMILY)
    parser.add_argument('--msar-run-dir', type=Path)
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args(tokens)
    explicit = explicit_arguments(parser, tokens)
    if args.msar_run_dir is None:
        args.msar_run_dir = args.run_dir
    elif args.run_dir is not None and args.run_dir != args.msar_run_dir:
        raise ValueError('conflicting --run_dir and --msar-run-dir')
    args.msar_task, args.msar_family, args.msar_evaluation = task, FAMILY, evaluation
    project = 'Car-Design-ShapeNetCar' if task == 'car' else 'Airfoil-Design-AirfRANS'
    preset = json.loads((ROOT/project/'configs/msar_lno'/f'{task}.json').read_text())
    if preset['family'] != FAMILY or preset['task'] != task:
        raise ValueError('MSAR preset family/task mismatch')
    if evaluation:
        if args.msar_run_dir is None:
            parser.error('MSAR evaluation requires --msar-run-dir pointing to an existing run')
        resolution = resolve_evaluation(args.msar_run_dir/'architecture.json', explicit, task=task)
        saved = resolution.saved
        if saved.checkpoint_format != ('whole_model' if task == 'car' else 'model_list'):
            raise MSARMetadataMismatch('industrial checkpoint format mismatch')
        record = task_record(args.msar_run_dir, task)
        contract = record['protocol']
        for name in (CAR_CONTRACT if task == 'car' else ('task','nmodel','weight')):
            if name in explicit and explicit[name] != contract[name]:
                raise MSARMetadataMismatch('industrial run contract mismatch: '+name)
            setattr(args, name, contract[name])
        if task == 'airfrans':
            hp = contract['hparams']
            for name in ('nb_epochs','batch_size','lr'):
                if name in explicit and explicit[name] != hp.get(name):
                    raise MSARMetadataMismatch('AirfRANS hparams mismatch: '+name)
        args.profile, architecture, training = saved.profile, saved.architecture, saved.training
        args.msar_eval_training_overrides = resolution.requested_training.to_dict()
        args.msar_eval_training_differences = list(resolution.training_differences)
    else:
        resolved = resolve_training(task, dict(profile=preset['profile']) | preset['objective'] | explicit)
        args.profile, architecture, training = resolved.profile, resolved.architecture, resolved.training
        if task == 'car':
            for name,value in preset['training'].items():
                if name not in explicit:setattr(args,name,value)
        else:
            hp = yaml.safe_load((ROOT/project/'params.yaml').read_text())[FAMILY].copy()
            for name in ('nb_epochs','batch_size','lr'):
                if name in explicit:hp[name]=explicit[name]
        # Validate save_name even when a separate explicit run path is supplied.
        proposed = new_run_path(Path(os.environ.get('CDLNO_RUNS_ROOT', ROOT/'output')), resolved,
                                save_name=args.save_name)
        if args.msar_run_dir is None:args.msar_run_dir = proposed
    if task == 'car':
        if not 0 <= args.fold_id <= 8 or args.nb_epochs < 1:
            raise ValueError('Car requires fold_id in 0..8 and positive nb_epochs')
        if getattr(args,'batch_size',1) != 1:
            raise ValueError('MSAR Car supports single-graph batch_size=1 only')
        args.batch_size = 1
    else:
        if args.task not in ('full','scarce','reynolds','aoa') or args.nmodel < 1:
            raise ValueError('invalid AirfRANS task/nmodel')
        if hp.get('batch_size') != 1 or type(hp.get('nb_epochs')) is not int or hp['nb_epochs'] < 1:
            raise ValueError('MSAR AirfRANS requires batch_size=1 and positive nb_epochs')
        # All original sampling/graph fields are mandatory; never repair a sidecar.
        if set(hp) != {'batch_size','nb_epochs','lr','max_neighbors','subsampling','r'}:
            raise MSARMetadataMismatch('AirfRANS requires its complete saved sampling/training hparams')
        args.msar_hparams = hp
        for name in ('nb_epochs','batch_size','lr'):setattr(args,name,hp[name])
    args.msar_architecture, args.msar_training = architecture.to_dict(), training.to_dict()
    return args


def model_kwargs(args):
    if getattr(args,'msar_family',None) != FAMILY or args.msar_task not in CLASSES:
        raise ValueError('explicit MSAR industrial arguments required')
    return dict(config=MSARArchitectureConfig.from_dict(args.msar_architecture),
                training_config=MSARTrainingConfig.from_dict(args.msar_training))


def resolve_hparams(args, hparams):
    if getattr(args,'msar_task',None) != 'airfrans':
        raise ValueError('MSAR AirfRANS arguments required')
    return dict(args.msar_hparams)  # Saved complete values win in eval; no YAML overwrite.


class _IndustrialRun:
    def __init__(self, args, *, device, model=None, evaluation=False):
        from ..experiment import session, reserve_directory, timestamp
        self.args, self.device, self.evaluation = args, device, evaluation
        self.task = args.msar_task
        self.directory = Path(args.msar_run_dir).resolve()
        self.architecture = MSARArchitectureConfig.from_dict(args.msar_architecture)
        self.training_config = MSARTrainingConfig.from_dict(args.msar_training)
        self.adapter = adapter_architecture(self.task)
        self.contract = ({k:getattr(args,k) for k in CAR_CONTRACT} if self.task == 'car' else
                         dict(task=args.task, nmodel=args.nmodel, weight=args.weight, hparams=dict(args.msar_hparams)))
        if self.task == 'airfrans':self.hparams = self.contract['hparams']
        self.recorder = session(args)
        self.metadata = MSARMetadata(self.task, self.architecture,
            'whole_model' if self.task == 'car' else 'model_list', args.profile, training=self.training_config,
            runtime=MSARRuntimeConfig(device=str(device),batch_size=1))
        if evaluation:
            self.validate()  # Before construction of the expected model or unpickling.
        else:
            if model is not None:self._check_model(model)
            reserve_directory(args,self.directory)
            save_metadata(self.sidecar,self.metadata)
            with (self.directory/'task.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(schema_version=1,family=FAMILY,task=self.task,adapter=self.adapter,
                    protocol=self.contract,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),
                    stream,indent=2,allow_nan=False)
                stream.write('\n')
        self.result_dir = str(self.directory/('eval_'+timestamp()))+'/'
        if self.recorder is not None:
            self.result_dir = self.recorder.result_dir
            if model is not None:self.recorder.attach_model(model,protocol=self.contract)
        print('MSAR industrial run:',self.directory,self.architecture.to_dict(),self.adapter)
        print('MSAR coverage:',self.training_config.to_dict(),'effective:',self.training_config.effective_coverage_mode)

    @property
    def sidecar(self):return self.directory/'architecture.json'

    @property
    def checkpoint(self):
        return self.directory/(f'model_{self.contract["nb_epochs"]}.pth' if self.task == 'car' else FAMILY)

    def member_dir(self,index):
        if self.task != 'airfrans' or type(index) is not int or not 0 <= index < self.contract['nmodel']:
            raise ValueError('AirfRANS member index out of range')
        return str(self.directory/f'member_{index:03d}')

    def validate(self):
        saved, record = load_metadata(self.sidecar), task_record(self.directory,self.task)
        if (saved.task != self.task or saved.checkpoint_format != self.metadata.checkpoint_format
                or saved.architecture != self.architecture or saved.training != self.training_config
                or record['protocol'] != self.contract):
            raise MSARMetadataMismatch('MSAR industrial family/architecture/training/protocol mismatch')

    def _check_model(self,model):
        import torch
        if (type(model) is not CLASSES[self.task] or getattr(model,'config',None) != self.architecture
                or getattr(model,'training_config',None) != self.training_config
                or model.core.config != self.architecture or model.core.training_config != self.training_config
                or model.core.output_dim != 4 or model.adapter_architecture() != self.adapter):
            raise MSARMetadataMismatch('MSAR whole-model class/configuration/task mismatch')
        parameters = list(model.named_parameters(remove_duplicate=False))
        if len(parameters) != len({(p.device,p.untyped_storage().data_ptr()) for _,p in parameters}):
            raise MSARMetadataMismatch('unexpected shared MSAR parameter storage')
        dtype = next(model.parameters()).dtype
        if any(p.dtype != dtype for _,p in parameters):
            raise MSARMetadataMismatch('mixed parameter dtype in MSAR checkpoint')
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            expected = CLASSES[self.task](**model_kwargs(self.args)).to(dtype=dtype)
        from ..kcdno.loading import validate_whole_model
        try:validate_whole_model(model,expected)
        except (ValueError,RuntimeError) as error:raise MSARMetadataMismatch(str(error)) from error

    def load(self, *, member=None):
        import torch
        if not self.evaluation:
            raise RuntimeError('load requires evaluation of an existing trusted industrial run')
        self.validate()
        path = self.checkpoint if member is None else Path(self.member_dir(member))/'model'
        # Same explicit LOCAL trusted whole-object boundary as the original tasks.
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if self.task == 'airfrans' and member is None:
            if type(saved) is not list or len(saved) != self.contract['nmodel']:
                raise MSARMetadataMismatch('checkpoint must contain the complete AirfRANS model list')
            models = saved
        else:models = [saved]
        storages = set()
        for i,model in enumerate(models):
            self._check_model(model)
            for p in model.parameters():
                key=(p.device,p.untyped_storage().data_ptr())
                if key in storages:raise MSARMetadataMismatch('model list members must not share parameters')
                storages.add(key)
            model.to(self.device)
            if self.recorder is not None:
                self.recorder.attach_model(model,hparams=getattr(self,'hparams',None),
                    member=i if member is None else member,protocol=self.contract)
        return models if self.task == 'airfrans' and member is None else models[0]


class CarRun(_IndustrialRun):
    pass


class AirRun(_IndustrialRun):
    def __init__(self,args,hparams,*,device,evaluation=False):
        if hparams != resolve_hparams(args,hparams):
            raise MSARMetadataMismatch('AirfRANS hparams must match resolved/saved values')
        super().__init__(args,device=device,evaluation=evaluation)
