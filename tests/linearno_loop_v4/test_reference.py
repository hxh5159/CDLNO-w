"""Independent fixed-commit FLARE ResidualMLP reference comparison."""
import ast
from pathlib import Path
import pytest
import torch
from cdlno.linearno_loop.v4.resmlp import ResidualMLP


@pytest.mark.parametrize('ratio',[1,2])
@pytest.mark.parametrize('depth',[2,3])
def test_official_flare_reference_topology_values_gradients(ratio,depth):
    text=(Path(__file__).resolve().parents[2]/'docs/resmlp_dual_temp_v4/evidence/stage0/flare-4e053784.py.txt').read_text()
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ResidualMLP')
    scope=dict(nn=torch.nn,ACTIVATIONS={'gelu':torch.nn.GELU(approximate='tanh')})
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<pinned FLARE class only>','exec'),scope)
    official=scope['ResidualMLP'](3,3*ratio,3,num_layers=depth,input_residual=True,output_residual=True).double()
    adapted=ResidualMLP(3,ratio,depth).double()
    adapted.load_state_dict({name.replace('fcs.','hidden.'):v for name,v in official.state_dict().items()},strict=True)
    x=torch.linspace(-1.1,2.3,30,dtype=torch.float64).reshape(2,5,3).requires_grad_()
    a=official(x);b=adapted(x);torch.testing.assert_close(a,b,atol=0,rtol=0)
    ga=torch.autograd.grad(a.square().sum(),(x,*official.parameters()))
    gb=torch.autograd.grad(b.square().sum(),(x,*adapted.parameters()))
    for lhs,rhs in zip(ga,gb):torch.testing.assert_close(lhs,rhs,atol=0,rtol=0)
