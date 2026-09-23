"""Actual parser/recorder contracts, without importing data-reading exp modules."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tran_evaluate.linearno_loop.launch import native_parse, ENTRIES
from linearno_loop.v4.config import resolve_config
from cdlno.linearno_loop.v4.construction import build_from_config
from cdlno.linearno_loop.v4.checkpoint import save_pair


def parse(task, action, extra=()):
    key = '--cfd_model' if task == 'car' else '--model'
    model = ('LinearNO' if task in ('car','airfrans') else
             'LinearNO_Irregular_Mesh' if task=='elasticity' else 'LinearNO_Structured_Mesh_2D')
    tokens = [key,model,*map(str,extra)]
    if action=='resume': tokens += ['--resume']
    if action=='eval' and task in ENTRIES: tokens += ['--eval','1']
    entry=ENTRIES.get(task) or ('main_evaluation.py' if action=='eval' else 'main.py')
    return native_parse(task,action,entry,tokens)


BASE=['--linearno-loop','1','--linearno-loop-architecture','resmlp_dual_temp_v4']


@pytest.mark.parametrize('task', ['airfrans','car'])
@pytest.mark.parametrize('extra', [['--batch_size','2'],['--weight','0.7']])
def test_industrial_rejects_invalid_protocol_overrides(task,extra):
    # Air batch-size is the adapter's hyphenated option.
    if task=='airfrans' and extra[0]=='--batch_size':extra=['--batch-size','2']
    with pytest.raises(SystemExit):parse(task,'train',[*BASE,*extra])


def test_launcher_observes_v4_real_ownership_and_first_forward(tmp_path):
    from tran_evaluate.linearno_loop.recording import observe
    cfg=resolve_config('elasticity', options=dict(architecture='resmlp_dual_temp_v4',temperature_mode='point_k_point_q'))
    model=build_from_config(cfg)
    args=SimpleNamespace(_linearno_loop_config=cfg,linearno_run_dir=tmp_path,
                         seed=cfg['seed'],linearno_task='elasticity',eval=0,resume=False)
    observe(args,model)
    model(torch.randn(1,9,2),None)
    row=json.loads((tmp_path/'loop_run_manifest.json').read_text())
    assert row['schema_version']==4
    assert row['members']['member_000']['observation']=='first_forward_verified'
    assert row['members']['member_000']['operator_instances']==8
    assert row['members']['member_000']['point_ffn_instances']==5
    assert row['members']['member_000']['actual_call_schedule']==row['expected_call_schedule']


@pytest.mark.parametrize('mode',['base','latent_k_point_q','point_k_point_q'])
def test_standard_factory_rejects_kwargs_drift(mode):
    from cdlno.linearno_loop.standard_entry import model_module,model_kwargs
    args=parse('elasticity','train',[*BASE,'--linearno-loop-temperature-mode',mode])
    kwargs=model_kwargs(args);kwargs['n_hidden']=64
    with pytest.raises(ValueError,match='constructor'):model_module(args).Model(**kwargs)
