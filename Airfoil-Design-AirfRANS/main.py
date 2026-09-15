import argparse, yaml, json
import torch
import train
import utils.metrics as metrics
from dataset.dataset import Dataset
import os.path as osp
import numpy as np
from cdlno_entry import parse_args as parse_cdlno_args, model_kwargs as cdlno_model_kwargs, resolve_hparams, AirRun

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet',
                    type=str)
parser.add_argument('-n', '--nmodel', help='Number of trained models for standard deviation estimation (default: 1)',
                    default=1, type=int)
parser.add_argument('-w', '--weight', help='Weight in front of the surface loss (default: 1)', default=1, type=float)
parser.add_argument('-t', '--task',
                    help='Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)',
                    default='full', type=str)
parser.add_argument('-s', '--score',
                    help='If you want to compute the score of the models on the associated test set. (default: 0)',
                    default=0, type=int)
parser.add_argument('--my_path',
                    default='/data/path', type=str)
parser.add_argument('--save_path',
                    default='metrics', type=str)
args = parse_cdlno_args(parser)
if args.model == 'CDLNO':
    from cdlno.experiment import start as start_experiment, finish as finish_experiment
    with open('params.yaml', 'r') as config_file:
        recording_hparams = resolve_hparams(args, yaml.safe_load(config_file)['CDLNO'])
    start_experiment(args, 'airfrans', evaluation=False, hparams=recording_hparams)

with open(args.my_path + '/manifest.json', 'r') as f:
    manifest = json.load(f)

manifest_train = manifest[args.task + '_train']
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1 * len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]
print("start load data")
train_dataset, coef_norm = Dataset(train_dataset, norm=True, sample=None, my_path=args.my_path)
val_dataset = Dataset(val_dataset, sample=None, coef_norm=coef_norm, my_path=args.my_path)
print("load data finish")
# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f:  # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]

cdlno_run = None
if args.model == 'CDLNO':
    hparams = resolve_hparams(args, hparams)
    cdlno_run = AirRun(args, hparams, device=device)
    cdlno_run.recorder.update_protocol(dict(train_graphs=len(train_dataset), validation_graphs=len(val_dataset),
                                         val_iter=10, val_sample=True, criterion="MSE_weighted"))

from models.MLP import MLP

models = []
for i in range(args.nmodel):

    if args.model == 'Transolver':
        from models.Transolver import Transolver

        model = Transolver(n_hidden=256,
                           n_layers=8,
                           space_dim=7,
                           fun_dim=0,
                           n_head=8,
                           mlp_ratio=2,
                           out_dim=4,
                           slice_num=32,
                           unified_pos=1).cuda()
    elif args.model == 'CDLNO':
        from models.CDLNO import Model

        model = Model(**cdlno_model_kwargs(args)).to(device)
    else:
        encoder = MLP(hparams['encoder'], batch_norm=False)
        decoder = MLP(hparams['decoder'], batch_norm=False)
        if args.model == 'GraphSAGE':
            from models.GraphSAGE import GraphSAGE

            model = GraphSAGE(hparams, encoder, decoder)

        elif args.model == 'PointNet':
            from models.PointNet import PointNet

            model = PointNet(hparams, encoder, decoder)

        elif args.model == 'MLP':
            from models.NN import NN

            model = NN(hparams, encoder, decoder)

        elif args.model == 'GUNet':
            from models.GUNet import GUNet

            model = GUNet(hparams, encoder, decoder)

    log_path = cdlno_run.member_dir(i) if cdlno_run is not None else osp.join(args.save_path, args.task, args.model)
    if cdlno_run is not None:
        cdlno_run.recorder.attach_model(model, hparams=hparams, member=i, protocol=cdlno_run.contract)
    print('start training')
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path,
                       criterion='MSE_weighted', val_iter=10, reg=args.weight, name_mod=args.model, val_sample=True,
                       **(dict(record=cdlno_run.recorder, record_member=i) if cdlno_run is not None else {}))
    print('end training')
    models.append(model)
torch.save(models, cdlno_run.checkpoint if cdlno_run is not None else osp.join(args.save_path, args.task, args.model, args.model))

score_path = cdlno_run.result_dir if cdlno_run is not None else 'scores'
score_array_path = score_path if cdlno_run is not None else osp.join('scores', args.task)
if args.model == 'CDLNO':
    finish_experiment(args)

if bool(args.score):
    if args.model == 'CDLNO':
        cdlno_run.recorder = start_experiment(args, 'airfrans', evaluation=True, hparams=hparams)
        score_path = cdlno_run.recorder.result_dir
        score_array_path = score_path
    print('start score')
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(device, [models], [hparams], coef_norm, args.my_path, path_out=score_path, n_test=3,
                                 criterion='MSE', s=s)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
    np.save(osp.join(score_array_path, 'true_coefs'), coefs[0])
    np.save(osp.join(score_array_path, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join(score_array_path, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join(score_array_path, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join(score_array_path, 'surf_coefs_' + str(n)), file)
    np.save(osp.join(score_array_path, 'true_bls'), coefs[5])
    np.save(osp.join(score_array_path, 'bls'), coefs[6])
    print('end score')

    if args.model == 'CDLNO':
        cdlno_run.recorder.record_air_scores(score_path)
        finish_experiment(args)
