"""Actual shell/planner tests for all V4 tasks and temperature modes."""
import json
from pathlib import Path
from unittest.mock import patch
import subprocess

import pytest

from tran_evaluate.linearno_loop_v4.launch import plan_v4,main
from .test_v4_entry_contracts import parse,BASE
from linearno_loop.v4.config import resolve_config
from cdlno.linearno_loop.v4.construction import build_from_config
from cdlno.linearno_loop.v4.checkpoint import save_pair

ROOT=Path(__file__).resolve().parents[2]
TASKS=('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car')


@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',('base','latent_k_point_q','point_k_point_q'))
def test_real_parser_unique_run_and_mode(task,mode,tmp_path):
    command=[task,'train','--temperature-mode',mode,'--seed','2','--gpu','0',
             '--data-root',str(tmp_path/'data with spaces'),'--output-root',str(tmp_path/'runs with spaces')]
    with patch('torch.load',side_effect=AssertionError('preview read tensors')):
        args,a=plan_v4(command);_,b=plan_v4(command)
    assert a['run']!=b['run']
    assert str(tmp_path/'runs with spaces') in a['run']
    assert not Path(a['run']).exists()
    c=a['config'];assert c['temperature_mode']==mode and c['seed']==2
    assert c['model']['ffn_ratio']==(2 if task in ('ns','airfrans','car') else 1)
    assert c['model']['hidden_width']==(256 if task in ('ns','airfrans','car') else 128)
    assert c['model']['actual_M']==(32 if task in ('ns','airfrans','car') else 64)
    assert c['rmlp_spec']['route']==['first','A','B','C','A','B','C','last']


def test_saved_config_does_not_receive_default_mode_or_seed(tmp_path):
    cfg=resolve_config('elasticity',options=dict(architecture='resmlp_dual_temp_v4',temperature_mode='point_k_point_q',seed=23))
    save_pair(tmp_path,build_from_config(cfg),cfg)
    for action in ('eval','resume'):
        _,value=plan_v4(['elasticity',action,'--experiment-dir',str(tmp_path)])
        assert value['config']==cfg
    with pytest.raises(SystemExit):
        plan_v4(['elasticity','eval','--experiment-dir',str(tmp_path),'--temperature-mode','latent_k_point_q'])


def test_then_eval_only_after_training_success():
    _,value=plan_v4(['elasticity','train','--then-eval'])
    with patch('tran_evaluate.linearno_loop_v4.launch.execute',return_value=13) as execute:
        assert main(['elasticity','train_eval'])==13
        assert execute.call_count==1
    with patch('tran_evaluate.linearno_loop_v4.launch.execute',return_value=0) as execute, \
         patch('tran_evaluate.linearno_loop_v4.launch.plan',return_value=value):
        assert main(['elasticity','train_eval'])==0
        assert execute.call_count==2
        assert execute.call_args_list[0].args[0]['run']==execute.call_args_list[1].args[0]['run']


def test_all_shell_syntax_and_actual_bash_preview():
    for path in sorted((ROOT/'tran_evaluate/linearno_loop_v4').glob('*.sh')):
        subprocess.run(['bash','-n',str(path)],check=True)
    process=subprocess.run(['bash',str(ROOT/'tran_evaluate/linearno_loop_v4/elasticity.sh'),
                            'train','--temperature-mode','point_k_point_q','--dry-run'],
                           text=True,capture_output=True,cwd='/tmp')
    assert process.returncode==0,process.stderr
    assert 'resmlp_dual_temp_v4' in process.stdout and 'DRY RUN' in process.stdout


def test_plasticity_default_is_mat_file_and_environment_override(monkeypatch,tmp_path):
    monkeypatch.setenv('CDLNO_PLASTICITY_FILE',str(tmp_path/'custom plasticity.mat'))
    _,value=plan_v4(['plasticity','train'])
    tokens=value['argv'];position=tokens.index('--data_path',tokens.index('--data_path')+1) if tokens.count('--data_path')>1 else tokens.index('--data_path')
    assert tokens[position+1]==str(tmp_path/'custom plasticity.mat')
    _,value=plan_v4(['plasticity','train','--data-root',str(tmp_path/'other root')])
    positions=[i for i,t in enumerate(value['argv']) if t=='--data_path']
    assert value['argv'][positions[-1]+1]==str(tmp_path/'other root'/'fno'/'plas_N987_T20.mat')
