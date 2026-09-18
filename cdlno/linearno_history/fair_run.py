"""External fair-run manifest and independent DataLoader generators.

The pure checkpoint schema stays unchanged. Legacy commands do not opt in;
research models always opt in, and baseline controls use an explicit flag.
"""
import argparse
import hashlib
import json
from pathlib import Path

from linearno_history.schema import derive_fair_seeds
from .config import resolve_config

MANIFEST='linearno_run_manifest.json'


def take_fair(tokens):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--linearno-fair-run',choices=('0','1'),default=None)
    value,remaining=parser.parse_known_args(tokens)
    probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    probe.add_argument('--experiment-dir','--linearno-run-dir',dest='directory',type=Path)
    run,_=probe.parse_known_args(remaining)
    path=run.directory/MANIFEST if run.directory else None
    saved=json.loads(path.read_text()) if path is not None and path.exists() else None
    requested=None if value.linearno_fair_run is None else value.linearno_fair_run=='1'
    if saved is not None:
        if saved.get('fair_run_version')!=1:raise ValueError('unknown fair-run manifest version')
        if requested is False:raise ValueError('cannot disable the saved fair-run data generator protocol')
        requested=True
    return requested,remaining


def attach(args, enabled, *, directory_was_explicit=True):
    if not enabled:return args
    args._linearno_fair_run=True
    config=getattr(args,'_linearno_history_config',None) or resolve_config(args._linearno_config)
    args._linearno_fair_config=config
    if not directory_was_explicit:
        from cdlno.experiment import timestamp
        args.linearno_run_dir=args.linearno_run_dir.parent/(config['run_signature']+'__'+timestamp())
    if config['run_signature'] not in args.linearno_run_dir.name:
        raise ValueError('fair-run directory must contain '+config['run_signature'])
    return args


def record(args, model):
    config=args._linearno_fair_config
    structure=dict(fair_run_version=1,task=args.linearno_task,
        protocol_profile=config['profile_spec']['profile'],run_signature=config['run_signature'],
        initialization=config['initialization'],profile_config_hash=config['profile_spec']['config_hash'])
    path=args.linearno_run_dir/MANIFEST
    if args.eval or args.resume:
        saved=json.loads(path.read_text())
        if any(saved.get(k)!=v for k,v in structure.items()):raise ValueError('fair-run manifest/config mismatch')
        return
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(('latent_attnres.','history_k.')):continue
        x=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    structure['public_backbone_initial_sha256']=h.hexdigest()
    with path.open('x') as stream:json.dump(structure,stream,sort_keys=True,indent=2)


def prepare_loaders(args, train_loader, test_loader):
    import torch
    for name,loader in (('train',train_loader),('test',test_loader)):
        seed=derive_fair_seeds(args.seed,task=args.linearno_task,split=name)['dataloader_generator_seed']
        if loader.generator is not None:raise ValueError('fair run owns the explicit DataLoader generator')
        generator=torch.Generator().manual_seed(seed)
        loader.generator=generator
        if isinstance(loader.sampler,torch.utils.data.RandomSampler):loader.sampler.generator=generator
        if args.linearno_task == 'plasticity':
            if loader.num_workers != 0:
                raise ValueError('fair Plasticity resume requires the native num_workers=0 loader')
            loader.collate_fn = GeneratorCollate(loader.collate_fn, generator)


class GeneratorCollate:
    """Execute the unchanged per-sample time permutation on the data RNG.

    CPU RNG is restored after collation, so feature dropout cannot change the
    queries. The loader generator already belongs to the full resume archive.
    """
    def __init__(self, collate, generator):
        self.collate, self.generator = collate, generator

    def __call__(self, batch):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            result = self.collate(batch)
            self.generator.set_state(torch.get_rng_state())
        return result
