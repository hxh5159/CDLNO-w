import copy
import math
import os
from pathlib import Path
import subprocess,sys,json
import tempfile
import torch
from types import SimpleNamespace
from unittest.mock import patch

from linearno_loop.v4.config import resolve_config, validate_config
from linearno_loop.v4.contracts import V4SchemaError
from cdlno.linearno_loop.v4.construction import build_from_config
from cdlno.linearno_loop.v4.resmlp import ResidualMLP
from cdlno.linearno_loop.v4.checkpoint import save_pair, load_pair
from tools.linearno_loop_accounting import analytic

def cfg(task='elasticity', mode='base', seed=5):
    return resolve_config(task, options={'architecture':'resmlp_dual_temp_v4','temperature_mode':mode,'seed':seed})

def test_contract_and_rmlp_linear_count():
    c=cfg(); validate_config(c)
    m=ResidualMLP(8,1,2)
    assert len([x for x in m.modules() if isinstance(x,torch.nn.Linear)])==4
    assert m.input_residual and m.output_residual
    m2=ResidualMLP(8,2,3)
    assert len([x for x in m2.modules() if isinstance(x,torch.nn.Linear)])==5
    assert not m2.input_residual and not m2.output_residual
    for options in ({'architecture':'operator_latent_adapter_v3'},
                    {'architecture':'resmlp_dual_temp_v4','residual_mode':'sr_1_over_r'},
                    {'architecture':'resmlp_dual_temp_v4','ffn_ratio':2}):
        try:resolve_config('darcy',options=options)
        except V4SchemaError:pass
        else:raise AssertionError('invalid v4 contract accepted')

def test_rmlp_independent_manual_oracle():
    torch.manual_seed(3);m=ResidualMLP(3,1,2).double();x=torch.randn(2,5,3,dtype=torch.float64)
    h=torch.nn.functional.gelu(torch.nn.functional.linear(x,m.fc1.weight,m.fc1.bias),approximate='tanh')+x
    for layer in m.hidden:
        h=h+torch.nn.functional.gelu(torch.nn.functional.linear(h,layer.weight,layer.bias),approximate='tanh')
    expected=torch.nn.functional.linear(h,m.fc2.weight,m.fc2.bias)+h
    assert torch.allclose(m(x),expected,atol=1e-12,rtol=1e-12)

def test_modes_zero_init_public_pair_and_gradients():
    models=[build_from_config(cfg(mode=mode,seed=11)) for mode in ('base','latent_k_point_q','point_k_point_q')]
    common=set(models[0].state_dict()) & set(models[1].state_dict()) & set(models[2].state_dict())
    for key in common:
        assert torch.equal(models[0].state_dict()[key],models[1].state_dict()[key])
        assert torch.equal(models[0].state_dict()[key],models[2].state_dict()[key])
    x=torch.randn(2,19,2)
    ys=[m(x,None) for m in models]
    assert torch.equal(ys[0],ys[1]) and torch.equal(ys[0],ys[2])
    loss=ys[2].square().mean();loss.backward()
    assert any(p.grad is not None for n,p in models[2].named_parameters() if 'q_temperature' in n)
    for key,value in models[1].loop.blocks[0].Attn.q_temperature.state_dict().items():
        assert torch.equal(value,models[2].loop.blocks[0].Attn.q_temperature.state_dict()[key])
    for model in models[1:]:
        q_ids={id(block.Attn.q_temperature) for block in model.loop.blocks}
        k_ids={id(block.Attn.k_temperature) for block in model.loop.blocks}
        assert len(q_ids)==len(k_ids)==8 and not (q_ids&k_ids)

def test_core_schedule_and_accounting():
    m=build_from_config(cfg(mode='point_k_point_q'))
    assert tuple(m.loop.schedule)==('first','A','B','C','A','B','C','last')
    assert len(m.loop.rmlp)==5 and len(m.loop.blocks)==8
    report=analytic(cfg(mode='point_k_point_q'),B=1,N=11)
    assert report['unique_depth']==8 and report['executed_depth']==8
    assert report['no_nxn_or_mxm_attention']
    ids=[id(p) for p in m.parameters()];assert len(ids)==len(set(ids))
    keys=list(m.state_dict());assert not any('A1' in k or 'A2' in k for k in keys)

def test_core_branch_scale_oracle():
    m=build_from_config(cfg(mode='base'))
    class Add(torch.nn.Module):
        def __init__(self,v):super().__init__();self.v=v
        def forward(self,x,**kw):return torch.ones_like(x)*self.v
    for i,b in enumerate(m.loop.blocks):b.ln_1=torch.nn.Identity();b.ln_2=torch.nn.Identity();b.Attn=Add(i+1)
    for i,name in enumerate(('first','A','B','C','last')):m.loop.rmlp[name]=Add(10+i)
    x=torch.zeros(1,2,128);expected=x
    route=(0,1,2,3,1,2,3,4)
    for i,r in enumerate(route):
        s=1 if i in (0,7) else 1/math.sqrt(2);expected=expected+s*(i+1)+s*(10+r)
    assert torch.allclose(m.loop(x),expected)

def test_strict_checkpoint_roundtrip(tmp_path):
    c=cfg(mode='latent_k_point_q'); m=build_from_config(c)
    save_pair(tmp_path,m,c)
    fresh=build_from_config(c); load_pair(tmp_path,fresh,c)
    for key,value in m.state_dict().items(): assert torch.equal(value,fresh.state_dict()[key])
    bad=copy.deepcopy(c);bad['temperature_mode']='base';bad['config_hash']=''
    try: load_pair(tmp_path,fresh,bad)
    except Exception: pass
    else: raise AssertionError('cross-mode checkpoint load accepted')

def test_checkpoint_fresh_process_strict_load(tmp_path):
    c=cfg('elasticity','point_k_point_q',seed=23);model=build_from_config(c)
    save_pair(tmp_path,model,c)
    code=("from cdlno.linearno_loop.v4.checkpoint import load_model;import sys;"
          "m,meta=load_model(sys.argv[1]);"
          "print(meta['architecture'],meta['config_hash'],len(m.state_dict()))")
    process=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,'PYTHONPATH':'.:cdlno'},text=True,capture_output=True)
    assert process.returncode==0,process.stdout+process.stderr
    assert 'resmlp_dual_temp_v4' in process.stdout and c['config_hash'] in process.stdout

def test_checkpoint_pointer_tamper_rejected_before_tensor_load(tmp_path):
    c=cfg('elasticity','latent_k_point_q');model=build_from_config(c);save_pair(tmp_path,model,c)
    pointer=tmp_path/'checkpoints/final.json';value=json.loads(pointer.read_text());value['sha256']='0'*64
    pointer.write_text(json.dumps(value))
    with patch('torch.load',side_effect=AssertionError('tensor load preceded pointer validation')) as loading:
        try:load_pair(tmp_path,build_from_config(c),c)
        except ValueError:pass
        else:raise AssertionError('tampered pointer accepted')
    loading.assert_not_called()

def test_industrial_shapes():
    c=cfg('airfrans','point_k_point_q');m=build_from_config(c)
    d=SimpleNamespace(x=torch.randn(13,7),pos=torch.randn(13,2),batch=torch.zeros(13,dtype=torch.long),ptr=torch.tensor([0,13]))
    assert m(d).shape==(13,4)
    c=cfg('car','latent_k_point_q');m=build_from_config(c)
    d=SimpleNamespace(x=torch.randn(13,7),batch=torch.zeros(13,dtype=torch.long),ptr=torch.tensor([0,13]))
    assert m((d,None)).shape==(13,4)

def test_task_ratios_and_predictor_shapes_and_rng():
    rng=torch.get_rng_state().clone();rows={t:cfg(t,'latent_k_point_q')['model']['ffn_ratio'] for t in ('darcy','ns','airfrans','car')}
    assert rows=={'darcy':1,'ns':2,'airfrans':2,'car':2};assert torch.equal(rng,torch.get_rng_state())
    m=build_from_config(cfg('elasticity','latent_k_point_q'));a=m.loop.blocks[0].Attn;s=torch.randn(2,8,7,16)
    assert a.q_temperature(s).shape==(2,8,7,1);assert a.k_temperature(s).shape==(2,8,1,64)
    p=build_from_config(cfg('elasticity','point_k_point_q')).loop.blocks[0].Attn
    assert p.k_temperature(s).shape==(2,8,7,1)
    assert torch.equal(a.q_temperature.fc1.weight,p.q_temperature.fc1.weight)

def test_dynamic_first_and_second_step_gradients():
    m=build_from_config(cfg(mode='point_k_point_q'));opt=torch.optim.AdamW(m.parameters(),lr=1e-3);x=torch.randn(1,9,2)
    m(x,None).square().mean().backward();a=m.loop.blocks[0].Attn
    assert a.q_temperature.fc2.weight.grad.abs().sum()>0
    assert torch.count_nonzero(a.q_temperature.fc1.weight.grad)==0
    opt.step();opt.zero_grad();m(x,None).square().mean().backward()
    assert a.q_temperature.fc1.weight.grad.abs().sum()>0

def test_config_module_import_does_not_import_torch():
    code="import sys;from linearno_loop.v4.config import resolve_config;resolve_config('darcy',options={'architecture':'resmlp_dual_temp_v4'});print('torch' in sys.modules)"
    out=subprocess.check_output([sys.executable,'-c',code],text=True,env={'PYTHONPATH':'.:cdlno'})
    # Base profile package currently imports torch-free modules only.
    assert out.strip()=='False'

def test_resealed_contract_mutations_are_rejected():
    from linearno_loop.v4.contracts import digest
    original=cfg('darcy','latent_k_point_q')
    for mutate in (
        lambda c:c['public_contract'].__setitem__('middle_scale','1/2'),
        lambda c:c['model'].__setitem__('actual_M',32),
        lambda c:c['model_spec']['constructor_kwargs'].__setitem__('ffn_ratio',2),
        lambda c:c['public_contract'].__setitem__('rmlp_schedule',['first']*8),
    ):
        broken=copy.deepcopy(original);mutate(broken)
        broken['config_hash']=digest({k:v for k,v in broken.items() if k!='config_hash'})
        try:validate_config(broken)
        except V4SchemaError:pass
        else:raise AssertionError('resealed mutation accepted')

def test_dynamic_attention_direct_call_cannot_bypass_installation():
    from cdlno.linearno_loop.v4.attention import V4LinearNOAttention
    torch.manual_seed(31)
    attention=V4LinearNOAttention(8,heads=2,dim_head=4,rank=3,variant='plain')
    attention._planned_temperature_mode='point_k_point_q'
    x=torch.randn(2,7,8)
    try:attention(x)
    except RuntimeError:pass
    else:raise AssertionError('dynamic attention silently used the base path before installation')
    attention.install_temperature_predictors(mode='point_k_point_q',public_seed=7,logical_depth=0)
    with torch.no_grad():
        attention.q_temperature.fc2.weight.fill_(.25)
        attention.k_temperature.fc2.weight.fill_(-.2)
    direct=attention(x)
    explicit=attention(x,logical_depth=0)
    assert torch.equal(direct,explicit)

def _standard_parser(task):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'tests'))
    sys.path.insert(0,str(root/'PDE-Solving-StandardBenchmark'))
    from linearno.static_worker import parser_for
    from cdlno_entry import parse_args
    return parser_for(task),parse_args

def _parse_standard(task,tokens):
    parser,parse_args=_standard_parser(task)
    return parse_args(parser,task,tokens)

def test_v4_cli_rejects_old_architecture_fields():
    base=['--model','LinearNO_Irregular_Mesh','--linearno-loop','1',
          '--linearno-loop-architecture','resmlp_dual_temp_v4']
    for extra in (
        ['--linearno-loop-residual-mode','sr_1_over_r'],
        ['--linearno-loop-latent','1'],
        ['--linearno-loop-topology','p2_c2_r2_s2'],
        ['--linearno-rank','32'],
    ):
        try:_parse_standard('elasticity',[*base,*extra])
        except SystemExit:pass
        else:raise AssertionError('legacy field was silently accepted by v4: '+str(extra))

def test_standard_v4_uses_versioned_pair_for_resume(tmp_path):
    from linearno_loop.v4.config import run_directory_id
    expected=cfg('elasticity','latent_k_point_q',seed=19)
    directory=tmp_path/run_directory_id(expected)
    args=_parse_standard('elasticity',['--model','LinearNO_Irregular_Mesh','--linearno-loop','1',
        '--linearno-loop-architecture','resmlp_dual_temp_v4','--linearno-loop-temperature-mode','latent_k_point_q',
        '--seed','19','--experiment-dir',str(directory)])
    args._linearno_data=dict(split='SYNTHETIC fixed',sampling='synthetic fixed',
        checksums={'synthetic':'a'*64},scope='SYNTHETIC')
    from cdlno.linearno.standard_entry import start,finish
    from cdlno.linearno_loop.standard_entry import LoopStandardRun
    from model_dict import get_model
    from linearno_entry import model_kwargs,StandardRun
    start(args,'elasticity')
    try:
        model=get_model(args).Model(**model_kwargs(args));run=StandardRun(args,model)
        assert isinstance(run,LoopStandardRun)
        values=torch.randn(2,9,2)
        train=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(values),batch_size=2,shuffle=True)
        test=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(values),batch_size=2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
        run.prepare(optimizer,scheduler,train,test)
        optimizer.zero_grad();model(values,None).square().mean().backward();optimizer.step();scheduler.step()
        run.complete_epoch(1);manifest=run.save(model)
    finally:
        finish(args)
    record=json.loads(manifest.read_text())
    assert record['format']=='resmlp-dual-temp-v4-pair-v1'
    assert (directory/'checkpoints/latest.json').is_file()
    assert not (directory/'checkpoints/latest.pt').exists()
    resume=_parse_standard('elasticity',['--resume','--experiment-dir',str(directory)])
    resume._linearno_data=copy.deepcopy(args._linearno_data)
    start(resume,'elasticity')
    try:
        restored=get_model(resume).Model(**model_kwargs(resume));run2=StandardRun(resume,restored)
        optimizer2=torch.optim.AdamW(restored.parameters(),lr=resume.lr,weight_decay=resume.weight_decay)
        scheduler2=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2,T_max=resume.epochs)
        train2=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(values),batch_size=2,shuffle=True)
        test2=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(values),batch_size=2)
        run2.prepare(optimizer2,scheduler2,train2,test2)
        assert run2.start_epoch==1
        for key,value in model.state_dict().items():assert torch.equal(value,restored.state_dict()[key])
    finally:
        finish(resume)

def test_checkpoint_conflict_rejected_before_tensor_load(tmp_path):
    c=cfg('elasticity','latent_k_point_q');model=build_from_config(c);save_pair(tmp_path,model,c)
    wrong=cfg('elasticity','point_k_point_q')
    with patch('torch.load',side_effect=AssertionError('tensor load happened before metadata reject')) as loading:
        try:load_pair(tmp_path,build_from_config(wrong),wrong)
        except (ValueError,V4SchemaError):pass
        else:raise AssertionError('cross-mode checkpoint accepted')
    loading.assert_not_called()

def test_car_v4_uses_the_reviewed_native_objective_contract():
    c=cfg('car','latent_k_point_q')
    objective=c['profile_spec']['values']['objective']
    assert c['profile_spec']['unresolved']==[]
    assert c['profile_spec']['integration_contract']=='car_transolver_mse_v1'
    assert objective['kind']=='MSE' and objective['space']=='normalized'
    assert objective['surface_weight']==.5

def test_v4_accounting_partition_matches_constructed_parameters():
    from tools.linearno_loop_accounting import measured_parameters
    for task in ('elasticity','plasticity','airfrans','car'):
        c=cfg(task,'latent_k_point_q');model=build_from_config(c);report=analytic(c,B=1,N=7)
        assert sum(report['parameter_parts'].values())==report['parameters']
        assert report['parameter_parts']==measured_parameters(model)

def test_industrial_v4_rejects_legacy_structure_fields():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'Car-Design-ShapeNetCar'))
    from models.cdlno_run import parse_args
    import argparse
    def parser():
        value=argparse.ArgumentParser()
        value.add_argument('--data_dir',default='/tmp');value.add_argument('--save_dir',default='/tmp')
        value.add_argument('--fold_id',default=0,type=int);value.add_argument('--gpu',default=0,type=int)
        value.add_argument('--val_iter',default=10,type=int);value.add_argument('--cfd_config_dir',default='x')
        value.add_argument('--cfd_model');value.add_argument('--cfd_mesh',action='store_true')
        value.add_argument('--r',default=.2,type=float);value.add_argument('--weight',default=.5,type=float)
        value.add_argument('--lr',default=.001,type=float);value.add_argument('--batch_size',default=1,type=int)
        value.add_argument('--nb_epochs',default=2,type=int);value.add_argument('--preprocessed',default=1,type=int)
        return value
    common=['--cfd_model','LinearNO','--linearno-loop','1','--linearno-loop-architecture','resmlp_dual_temp_v4']
    for field in (['--linearno-rank','64'],['--linearno-loop-topology','p2_c2_r2_s2'],
                  ['--linearno-loop-residual-mode','sr_1_over_r']):
        try:parse_args(parser(),argv=[*common,*field])
        except SystemExit:pass
        else:raise AssertionError('legacy V4 industrial field was ignored: '+str(field))
