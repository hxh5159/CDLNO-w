"""Finite synthetic CUDA observer check, no data, no model-performance claims."""
import copy,json,tempfile,sys,os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.standard import StandardModel
from cdlno.periodic_visualization import PeriodicFields
from cdlno.training_state import capture_rng,restore_rng


def main():
    if not torch.cuda.is_available():
        print(json.dumps({'status':'not_run','reason':'CUDA unavailable'}));return
    torch.set_num_threads(1);torch.manual_seed(12)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=StandardModel(config=KCDNOArchitectureConfig(L=2,d=8,h=2,M=3,point_module='conv_ffn'),task_name='darcy',H=5,W=7).cuda()
    data=torch.utils.data.TensorDataset(torch.rand(2,35,2),torch.rand(2,35),torch.rand(2,35))
    x,fx,y=[v.cuda() for v in data.tensors]
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    def step(model,optimizer):
        optimizer.zero_grad();loss=(model(x,fx[...,None])[...,0]-y).square().mean();loss.backward();optimizer.step()
    with sdpa_kernel(SDPBackend.MATH),tempfile.TemporaryDirectory() as tmp:
        step(m,opt);other=copy.deepcopy(m);oc=torch.optim.AdamW(other.parameters(),lr=.001);oc.load_state_dict(copy.deepcopy(opt.state_dict()))
        rng=capture_rng();before={k:v.clone() for k,v in m.state_dict().items()}
        with patch('cdlno.periodic_visualization.render_fields'):
            event=PeriodicFields(tmp,'darcy','KCDNO').after_epoch(m,50,500,dataset=data,grid_shape=(5,7))
        assert event['status']=='completed',event
        after_rng=capture_rng()
        # Tensor CUDA and CPU RNG evidence; Python/NumPy covered by CPU test.
        for k,v in rng.items():
            if torch.is_tensor(v):torch.testing.assert_close(v,after_rng[k],atol=0,rtol=0)
            elif isinstance(v,list) and all(torch.is_tensor(a) for a in v):
                for a,b in zip(v,after_rng[k]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
        step(m,opt);restore_rng(rng);step(other,oc)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],atol=0,rtol=0)
    print(json.dumps(dict(status='passed',torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
        device=torch.cuda.get_device_name(),dtype='float32',sdpa='math',tf32=False,compile=False,amp=False,deterministic=True,
        scope='synthetic 5x7 Darcy KCDNO observer + exact next-step weights; renderer mocked, CPU real rendering separately tested'),indent=2))


if __name__=='__main__':main()
