"""Reproduce the A1 finite GPU FP32 core check; no task inputs or training."""
import json
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from cdlno import CDLNO, CDLNOArchitectureConfig

result = dict(torch=str(torch.__version__), cuda_build=torch.version.cuda,
              cuda_available=torch.cuda.is_available(), checks=[])
if torch.cuda.is_available():
    result['gpu'] = torch.cuda.get_device_name(0)
    result['precision'] = 'FP32; math SDPA; TF32 off; no AMP/compile'
    old = torch.backends.cuda.matmul.allow_tf32
    old_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_num_threads(1)
        for mode in ('full', 'no_sa', 'identity'):
            torch.manual_seed(414)
            cfg = CDLNOArchitectureConfig(L=8, F=2, M=4, d_model=8, num_heads=2, output_dim=3,
                front_latent_mode=mode, cdpa_mode='every_block', structured=True, grid_shape=(5, 7))
            cpu, gpu = CDLNO(cfg), CDLNO(cfg).cuda()
            gpu.load_state_dict(cpu.state_dict(), strict=True)
            x = torch.randn(2, 35, 8, requires_grad=True)
            xg = x.detach().cuda().requires_grad_()
            cot = torch.randn(2, 35, 3)
            with sdpa_kernel(SDPBackend.MATH):
                y, yg = cpu(x), gpu(xg)
                (y * cot).sum().backward()
                (yg * cot.cuda()).sum().backward()
            torch.testing.assert_close(y, yg.cpu(), atol=1e-5, rtol=3e-4)
            maxgrad = 0.
            for (name, p), (ng, pg) in zip(cpu.named_parameters(), gpu.named_parameters()):
                assert name == ng and pg.grad is not None and torch.isfinite(pg.grad).all(), name
                torch.testing.assert_close(p.grad, pg.grad.cpu(), atol=1e-5, rtol=3e-4)
                maxgrad = max(maxgrad, (p.grad - pg.grad.cpu()).abs().max().item())
            torch.testing.assert_close(x.grad, xg.grad.cpu(), atol=1e-5, rtol=3e-4)
            result['checks'].append(dict(mode=mode, status='passed',
                max_output_error=(y - yg.cpu()).abs().max().item(), max_parameter_gradient_error=maxgrad))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
        torch.backends.cudnn.allow_tf32 = old_c
else:
    result['not_run'] = 'GPU unavailable'
Path(__file__).with_name('gpu.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
