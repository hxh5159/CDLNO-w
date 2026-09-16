"""Capture/replay current M3 cores outside the repo; never task/data entries.

These are post-M3 synthetic references, not historical or trained checkpoints.
Capture refuses existing files. Replay loads the identical weights and inputs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from cdlno.msar_lno.config import MSARArchitectureConfig, MSARTrainingConfig, input_layout
from cdlno.msar_lno.core import MSARLNO
from cdlno.msar_lno.profiles import resolve_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('capture', 'replay'))
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(916303)
    results = []
    for profile, batch, n in (('light', 2, 35), ('full', 1, 7)):
        path = args.artifacts / f'{profile}.pt'
        if args.action == 'capture':
            if path.exists():
                raise FileExistsError(path)
            config = resolve_profile(profile)
            model = MSARLNO(config, output_dim=4)
            x = torch.randn(batch, n, config.d)
        else:
            data = torch.load(path, weights_only=True, map_location='cpu')
            config = MSARArchitectureConfig.from_dict(data['config'])
            model = MSARLNO(config, output_dim=data['output_dim'])
            model.load_state_dict(data['state_dict'], strict=True)
            x = data['input']
        trace, handles = [], []
        def record(name):
            def hook(module, inputs, output):
                trace.append(dict(module=name, inputs=[list(a.shape) for a in inputs], output=list(output.shape)))
            return hook
        for group in ('downs', 'encoders', 'decoders', 'ups', 'fusions'):
            for index, module in enumerate(getattr(model, group)):
                handles.append(module.register_forward_hook(record(f'{group}.{index}')))
        for name in ('final_up', 'output_norm', 'output'):
            handles.append(getattr(model, name).register_forward_hook(record(name)))
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            prediction = model.eval()(x)
            for handle in handles:
                handle.remove()
            aux = model.train()(x, return_aux=True)
        if args.action == 'capture':
            data = dict(config=config.to_dict(), training_config=MSARTrainingConfig().to_dict(),
                output_dim=model.output_dim, state_dict=model.state_dict(), input=x,
                prediction=prediction, training_prediction=aux.prediction,
                raw_coverage=aux.coverage_per_level)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                torch.save(data, stream)
        else:
            for a, b in ((prediction, data['prediction']), (aux.prediction, data['training_prediction']),
                         (aux.coverage_per_level, data['raw_coverage'])):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        groups = {name:sum(p.numel() for p in getattr(model, name).parameters()) for name in
                  ('downs', 'encoders', 'decoders', 'ups', 'fusions', 'final_up', 'output_norm', 'output')}
        counts = Counter(type(m).__name__ for m in model.modules())
        results.append(dict(profile=profile, action=args.action, status='passed', config=config.to_dict(),
            input_layout=input_layout(config, n), input_shape=list(x.shape), output_shape=list(prediction.shape),
            output_dim=4, parameters=sum(p.numel() for p in model.parameters()), parameter_groups=groups,
            module_counts=dict(counts), shape_trace=trace, state_keys_shapes={k:list(v.shape) for k,v in model.state_dict().items()},
            fixture=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
            dtype='float32', device='cpu', sdpa='MATH', atol=0, rtol=0,
            evidence='post-M3 current synthetic lifted inputs; no task/data/training checkpoint claim'))
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('profile','action','status','parameters','parameter_groups')} for r in results]))


if __name__ == '__main__':
    main()
