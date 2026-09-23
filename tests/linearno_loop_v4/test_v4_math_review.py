"""Independent derivative, mutation, owner and full-wrapper initialization review."""
import copy
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from cdlno.linearno.attention import LinearNOAttention,initialize_release_weights
from cdlno.linearno_loop.v4.attention import V4LinearNOAttention
from cdlno.linearno_loop.v4.resmlp import ResidualMLP
from cdlno.linearno_loop.v4.core import V4LoopCore
from linearno_loop.v4.config import resolve_config
from cdlno.linearno_loop.v4.construction import build_from_config
from .test_v4_attention import _oracle


def manual_rmlp(module,x,skip=True,exact_gelu=False):
    act=lambda v:F.gelu(v,approximate='none' if exact_gelu else 'tanh')
    h=act(F.linear(x,module.fc1.weight,module.fc1.bias))
    if module.ratio==1 and skip:h=h+x
    for layer in module.hidden:
        update=act(F.linear(h,layer.weight,layer.bias));h=h+update if skip else update
    out=F.linear(h,module.fc2.weight,module.fc2.bias)
    return out+h if module.ratio==1 and skip else out


@pytest.mark.parametrize('ratio',[1,2])
@pytest.mark.parametrize('depth',[2,3])
def test_rmlp_vjp_and_mutation_sensitivity(ratio,depth):
    torch.manual_seed(148)
    model=ResidualMLP(3,ratio,depth).double()
    x=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
    y=model(x);expected=manual_rmlp(model,x)
    torch.testing.assert_close(y,expected,atol=1e-12,rtol=1e-12)
    params=(x,*model.parameters());v=torch.randn_like(y)
    a=torch.autograd.grad(y,params,v,retain_graph=True);b=torch.autograd.grad(expected,params,v)
    for actual,wanted in zip(a,b):torch.testing.assert_close(actual,wanted,atol=1e-12,rtol=1e-11)
    assert not torch.allclose(y,manual_rmlp(model,x,skip=False),atol=1e-10,rtol=1e-10)
    assert not torch.allclose(y,manual_rmlp(model,x,exact_gelu=True),atol=1e-10,rtol=1e-10)
    assert torch.autograd.gradcheck(model,(x.detach().requires_grad_(),),fast_mode=True)


def core_oracle(core,x):
    # Independent parameter algebra, including all LayerNorms and point FFNs.
    for index,owner in enumerate(('first','A','B','C','A','B','C','last')):
        block=core.blocks[index]
        norm=lambda value,layer:F.layer_norm(value,layer.normalized_shape,layer.weight,layer.bias,layer.eps)
        s=1. if index in (0,7) else 2.**-.5
        # The attention oracle is functional and never invokes tested forward.
        raw=_oracle(block.Attn,norm(x,block.ln_1))
        u=x+s*raw
        x=u+s*manual_rmlp(core.rmlp[owner],norm(u,block.ln_2))
    return x


def test_core_nonzero_functional_oracle_and_mutations():
    torch.manual_seed(89)
    core=V4LoopCore(hidden=8,heads=2,rank=3,variant='plain',dropout=0.,H=1,W=1,out_dim=1,
                   ffn_ratio=1,temperature_mode='point_k_point_q',public_seed=17).double()
    core.apply(initialize_release_weights);core.install_temperature_predictors();core.double()
    for block in core.blocks:
        with torch.no_grad():
            block.Attn.q_temperature.fc2.weight.fill_(.3);block.Attn.k_temperature.fc2.weight.fill_(-.2)
    x=torch.randn(2,7,8,dtype=torch.float64,requires_grad=True)
    actual=core(x);expected=core_oracle(core,x)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)
    v=torch.randn_like(actual);params=(x,*core.parameters())
    grads=torch.autograd.grad(actual,params,v,retain_graph=True)
    reference=torch.autograd.grad(expected,params,v)
    for a,b in zip(grads,reference):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    bad=copy.deepcopy(core);bad.scale=.5
    assert not torch.allclose(bad(x),expected,atol=1e-10,rtol=1e-10)
    bad=copy.deepcopy(core);bad.blocks[4]=bad.blocks[1]
    assert len({id(b.Attn) for b in bad.blocks})==7
    assert not torch.allclose(bad(x),expected,atol=1e-10,rtol=1e-10)
    bad=copy.deepcopy(core);bad.rmlp['A_copy']=copy.deepcopy(bad.rmlp['A'])
    bad.schedule=('first','A','B','C','A_copy','B','C','last')
    assert len(bad.rmlp)!=5
    # At identical values a copied owner has equal output; ownership assertion is essential.
    torch.testing.assert_close(bad(x),expected)


@pytest.mark.parametrize('variant',['plain','temp','conv','conv_temp','airfrans','shapenet'])
def test_base_exact_native_output_gradient_and_dropout_rng(variant):
    kw=dict(heads=2,dim_head=4,rank=3,variant=variant,dropout=.2)
    n=7
    if variant in ('conv','conv_temp'):kw.update(H=2,W=3);n=6
    torch.manual_seed(16);native=LinearNOAttention(8,**kw).double();native.apply(initialize_release_weights)
    new=V4LinearNOAttention(8,**kw).double();new.load_state_dict(native.state_dict(),strict=True)
    new.install_temperature_predictors(mode='base',public_seed=2,logical_depth=0)
    x=torch.randn(2,n,8,dtype=torch.float64,requires_grad=True)
    rng=torch.get_rng_state();a=native(x);after=torch.get_rng_state()
    torch.set_rng_state(rng);b=new(x);assert torch.equal(after,torch.get_rng_state())
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a.sum(),(x,*native.parameters()),allow_unused=True)
    gb=torch.autograd.grad(b.sum(),(x,*new.parameters()),allow_unused=True)
    for u,v in zip(ga,gb):assert u is None and v is None or torch.equal(u,v)


@pytest.mark.parametrize('task',['elasticity','airfrans','car'])
def test_wrapper_modes_common_tensors_and_one_head(task):
    models=[build_from_config(resolve_config(task,options=dict(architecture='resmlp_dual_temp_v4',temperature_mode=mode,seed=29)))
            for mode in ('base','latent_k_point_q','point_k_point_q')]
    public=models[0].state_dict()
    for model in models[1:]:
        for name,value in public.items():assert torch.equal(value,model.state_dict()[name])
    torch.manual_seed(4)
    if task=='elasticity':args=(torch.randn(1,9,2),None)
    else:
        from torch_geometric.data import Data
        graph=Data(x=torch.randn(9,7),pos=torch.randn(9,2 if task=='airfrans' else 3))
        args=(graph,) if task=='airfrans' else ((graph,None),)
    outputs=[]
    for model in models:
        counts=[];h=model.head.register_forward_hook(lambda *unused:counts.append(1))
        outputs.append(model(*args));h.remove();assert counts==[1]
    for y in outputs[1:]:torch.testing.assert_close(y,outputs[0],atol=0,rtol=0)
    for index in range(8):
        for key,value in models[1].loop.blocks[index].Attn.q_temperature.state_dict().items():
            assert torch.equal(value,models[2].loop.blocks[index].Attn.q_temperature.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('mode',['base','latent_k_point_q','point_k_point_q'])
def test_cuda_amp_finite_step_and_reload(dtype,mode):
    c=resolve_config('elasticity',options=dict(architecture='resmlp_dual_temp_v4',temperature_mode=mode,seed=29))
    model=build_from_config(c).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x=torch.randn(1,17,2,device='cuda')
    with torch.autocast('cuda',dtype=dtype,enabled=dtype!=torch.float32):loss=model(x,None).square().mean()
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();fresh=build_from_config(c).cuda();fresh.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(model(x,None),fresh(x,None),atol=0,rtol=0)

@pytest.mark.parametrize('mode',['latent_k_point_q','point_k_point_q'])
def test_both_predictors_learn_after_zero_initialized_first_step(mode):
    torch.manual_seed(21)
    model=V4LinearNOAttention(8,heads=2,dim_head=4,rank=3,variant='plain').double()
    model.apply(initialize_release_weights);model.install_temperature_predictors(mode=mode,public_seed=21,logical_depth=1);model.double()
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    x=torch.randn(2,9,8,dtype=torch.float64)
    model(x).square().sum().backward()
    for predictor in (model.q_temperature,model.k_temperature):
        assert predictor.fc2.weight.grad.abs().sum()>0
        assert torch.count_nonzero(predictor.fc1.weight.grad)==0
    optimizer.step();optimizer.zero_grad();model(x).square().sum().backward()
    for predictor in (model.q_temperature,model.k_temperature):assert predictor.fc1.weight.grad.abs().sum()>0


@pytest.mark.parametrize('variant',['plain','temp','conv','conv_temp','airfrans','shapenet'])
@pytest.mark.parametrize('mode',['latent_k_point_q','point_k_point_q'])
def test_nonzero_attention_vjp_and_softmax_mutation(variant,mode):
    kwargs=dict(heads=2,dim_head=4,rank=3,variant=variant,dropout=0.)
    if variant in ('conv','conv_temp'):kwargs.update(H=2,W=3)
    torch.manual_seed(235)
    module=V4LinearNOAttention(8,**kwargs).double();module.apply(initialize_release_weights)
    module.install_temperature_predictors(mode=mode,public_seed=12,logical_depth=0);module.double()
    with torch.no_grad():
        module.q_temperature.fc2.weight.fill_(.1);module.k_temperature.fc2.weight.fill_(-.2)
    x=torch.randn(2,6,8,dtype=torch.float64,requires_grad=True)
    actual=module(x);reference=_oracle(module,x);v=torch.randn_like(actual)
    a=torch.autograd.grad(actual,(x,*module.parameters()),v,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(reference,(x,*module.parameters()),v,allow_unused=True)
    for lhs,rhs in zip(a,b):
        if lhs is None:assert rhs is None
        else:torch.testing.assert_close(lhs,rhs,atol=1e-11,rtol=1e-9)
    if variant=='plain':
        assert torch.autograd.gradcheck(module,(x.detach().requires_grad_(),),fast_mode=True)
        original=torch.Tensor.softmax
        def wrong(tensor,dim,*args,**kw):return original(tensor,-2 if dim==-1 else -1,*args,**kw)
        with patch.object(torch.Tensor,'softmax',wrong):
            assert not torch.allclose(module(x),reference,atol=1e-10,rtol=1e-10)
        original_k=module.k_temperature.forward
        # Swap the singleton competitive dimension while leaving positive tau.
        def wrong_shape(s):return original_k(s).transpose(2,3)
        with patch.object(module.k_temperature,'forward',wrong_shape):
            with pytest.raises(RuntimeError):module(x)
        bad=copy.deepcopy(module)
        with torch.no_grad():bad.q_temperature.fc2.weight.zero_();bad.k_temperature.fc2.weight.zero_()
        assert not torch.allclose(module(x),bad(x),atol=1e-12,rtol=1e-12)


def test_industrial_member_parameter_and_predictor_isolation():
    from cdlno.linearno_loop.industrial_state import construct
    c=resolve_config('airfrans',options=dict(architecture='resmlp_dual_temp_v4',temperature_mode='latent_k_point_q',seed=29))
    args=SimpleNamespace(_linearno_loop_config=c,seed=29,linearno_task='airfrans')
    first,second=construct(args,0),construct(args,1)
    assert not ({id(p) for p in first.parameters()} & {id(p) for p in second.parameters()})
    assert not torch.equal(first.loop.blocks[0].Attn.q_temperature.fc1.weight,second.loop.blocks[0].Attn.q_temperature.fc1.weight)


def test_full_air_sampling_dependency_boundary():
    pytest.importorskip('torch_cluster',reason='full AirfRANS radius sampling requires local torch_cluster; no installation authorized')
