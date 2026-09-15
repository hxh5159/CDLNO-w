"""Dispatch only the two new independent families; legacy selectors stay intact."""
from dataclasses import fields
import json
from pathlib import Path
from datetime import datetime, timezone
from . import metadata as kernel
from .config import KCDNOArchitectureConfig
from .matched_config import MatchedLRSAConfig, MatchedMetadata
from .options import architecture_overrides
from .profiles import profile_values, PROFILE_NAMES

FAMILIES = ('kcdno','lrsa_matched')


def architecture_from_dict(data):
    family=data.get('family')
    if family=='kcdno':return KCDNOArchitectureConfig.from_dict(data)
    if family=='lrsa_matched':return MatchedLRSAConfig.from_dict(data)
    raise kernel.KCDNOMetadataMismatch('explicit new family required')


def selected_family(explicit):
    family=explicit.get('model',explicit.get('cfd_model','kcdno'))
    if family not in FAMILIES:raise ValueError('unknown new model family')
    return family


def overrides(explicit,family):
    if family=='kcdno':return architecture_overrides(explicit)
    if any(k in explicit for k in ('kernel_rank','history_mode')):
        raise ValueError('kernel_rank/history_mode are not applicable to lrsa_matched full')
    mapped=dict(explicit)
    for key in ('model','cfd_model'):
        if key in mapped:mapped[key]='kcdno'
    values=architecture_overrides(mapped)
    names={f.name for f in fields(MatchedLRSAConfig)}
    if any(k not in names for k in values):raise ValueError('inapplicable matched LRSA architecture option')
    return values


def resolve_training(task,explicit):
    family=selected_family(explicit)
    if family=='kcdno':
        from .options import resolve_training as resolve
        return resolve(task,explicit)
    base=profile_values(task,explicit.get('profile','kcdno_v1'))
    for k in ('kernel_rank','history_mode'):base.pop(k)
    return MatchedLRSAConfig(**(base | overrides(explicit,family)))


def make_metadata(**kwargs):
    cls=kernel.KCDNOMetadata if kwargs['architecture'].family=='kcdno' else MatchedMetadata
    return cls(**kwargs)


def load_metadata(path):
    payload=json.loads(Path(path).read_text())
    if payload.get('family')=='lrsa_matched':return MatchedMetadata.from_dict(payload)
    return kernel.KCDNOMetadata.from_dict(payload)  # missing family is never guessed


def resolve_evaluation(path,explicit,*,task):
    saved=load_metadata(path)
    family=selected_family(explicit)
    if saved.architecture.family!=family:raise kernel.KCDNOMetadataMismatch('checkpoint family mismatch')
    if family=='kcdno':return kernel.resolve_evaluation(path,explicit,task=task)
    if saved.task!=task:raise kernel.KCDNOMetadataMismatch('task mismatch')
    if explicit.get('profile',saved.profile) not in PROFILE_NAMES:raise ValueError('unknown profile')
    wanted=MatchedLRSAConfig.from_dict(saved.architecture.to_dict() | overrides(explicit,family))
    if wanted!=saved.architecture:raise kernel.KCDNOMetadataMismatch('matched LRSA architecture mismatch')
    return kernel.EvaluationResolution(saved,saved.runtime,(),explicit.get('profile'))


def new_run_path(output,task,config):
    if config.family=='kcdno':return kernel.new_run_path(output,task,config)
    config.validate()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return Path(output)/task/config.family/f'{stamp}_L{config.L}_d{config.d}_h{config.h}_M{config.M}_full'
