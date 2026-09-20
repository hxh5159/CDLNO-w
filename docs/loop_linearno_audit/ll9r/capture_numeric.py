"""Pre/post repair capture, observations only; no task data or model monkeypatch."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
if os.environ.get('LL9R_FROZEN_SOURCE'):
    sys.path.insert(0, os.environ['LL9R_FROZEN_SOURCE'])
import torch
from cdlno.linearno_loop.construction import build_from_config
from cdlno.linearno_loop.attnres import PointDepthAttnRes
from tools.linearno_loop_support import TASKS, PRESETS, MODES, configuration, inputs


def tensor_record(x):
    raw = x.detach().cpu().contiguous()
    return dict(dtype=str(x.dtype), shape=list(x.shape), device=str(x.device),
                sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())


def leaves(args):
    result = {}
    def walk(x, name):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            x.requires_grad_(True); result[name] = x
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x): walk(v, name + '.' + str(i))
        elif hasattr(x, 'to_dict'):
            for k, v in x.to_dict().items(): walk(v, name + '.' + k)
    walk(args, 'input')
    return result


def capture(config, device, precision):
    dtype = {'FP32': None, 'FP16': torch.float16, 'BF16': torch.bfloat16}[precision]
    model = build_from_config(config).to(device).train()
    # Exercise active RMS-scale gradients, identically before/after.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PointDepthAttnRes):
                module.query.copy_(torch.linspace(-.13, .17, module.hidden, device=device))
    args = inputs(config, device=device); tensors = leaves(args)
    events = []; handles = []
    def observation(name, x):
        events.append(dict(name=name, tensors=[tensor_record(v) for v in x]))
    handles.append(model.preprocess.register_forward_hook(lambda m, a, y: observation('preprocess', [y])))
    observation('placeholder', [model.placeholder])
    handles.append(model.loop.register_forward_pre_hook(lambda m, a: observation('loop_input', a)))
    for name, module in model.loop.named_modules():
        if isinstance(module, PointDepthAttnRes):
            handles.append(module.register_forward_pre_hook(lambda m, a, n=name: observation(n + '.sources', a[0])))
            handles.append(module.register_forward_hook(lambda m, a, y, n=name: observation(n + '.output', [y])))
        if name.startswith('core.') and name.endswith(('.Attn', '.mlp')):
            handles.append(module.register_forward_hook(lambda m, a, y, n=name: observation(n + '.raw', [y])))
    # Trace local references without assigning to them. No observers survive this function.
    def trace(frame, event, arg):
        if frame.f_code.co_name not in ('_rb_forward', '_lb_forward'): return None
        if event == 'line':
            values = frame.f_locals
            keys = ('anchor', 'partial', 'raw') if frame.f_code.co_name == '_rb_forward' else ('anchor', 'entry', 'x')
            row = {k: tensor_record(values[k]) for k in keys if isinstance(values.get(k), torch.Tensor)}
            if 'completed' in values: row['completed'] = [tensor_record(v) for v in values['completed']]
            if 'deltas' in values: row['deltas'] = [tensor_record(v) for v in values['deltas']]
            events.append(dict(local=frame.f_code.co_name, line=frame.f_lineno, values=row))
        return trace
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    input_state = {k: v.detach().cpu().clone() for k, v in tensors.items()}
    torch.manual_seed(4001)
    rng_before = torch.get_rng_state().clone()
    gpu_before = torch.cuda.get_rng_state().cpu() if device == 'cuda' else None
    try:
        sys.settrace(trace)
        with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
            out = model(*args); loss = (out.float() - .31).square().mean()
        sys.settrace(None)
        loss.backward()
        assert torch.isfinite(out).all() and torch.isfinite(loss)
        gradients = {k: None if p.grad is None else p.grad.detach().cpu().clone() for k, p in model.named_parameters()}
        assert all(v is None or torch.isfinite(v).all() for v in gradients.values())
        result = dict(config=config, initial=initial, inputs=input_state, output=out.detach().cpu(),
                      loss=loss.detach().cpu(), input_gradients={k: None if v.grad is None else v.grad.detach().cpu().clone() for k, v in tensors.items()},
                      gradients=gradients, count=sum(p.numel() for p in model.parameters()),
                      rng_before=rng_before, rng_after=torch.get_rng_state().clone(),
                      cuda_rng_before=gpu_before, cuda_rng_after=torch.cuda.get_rng_state().cpu() if device == 'cuda' else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.step()
        result['step_state'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return result, dict(status='PASS', events=events)
    except (TypeError, RuntimeError) as error:
        return None, dict(status='FAIL', error=str(error), events=events)
    finally:
        sys.settrace(None)
        for handle in handles: handle.remove()


def compare(before, after, path='root'):
    if isinstance(before, torch.Tensor):
        assert before.dtype == after.dtype and torch.equal(before, after), path
    elif isinstance(before, dict):
        assert before.keys() == after.keys(), path
        for key in before: compare(before[key], after[key], path + '.' + key)
    else: assert before == after, path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('before', 'after'))
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args(); out = Path(__file__).resolve().parent
    snapshot = Path(json.loads((out/'start-manifest.json').read_text())['snapshot'])
    label = '-deterministic' if args.deterministic else ''
    store = snapshot/('numeric'+label); store.mkdir(exist_ok=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    rows = []
    for device in ('cpu', 'cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            rows.append(dict(device=device, status='NOT RUN')); continue
        precisions = ('FP32',) if device == 'cpu' else ('FP32', 'FP16', 'BF16')
        for task in TASKS:
            for preset in PRESETS:
                for mode in MODES:
                    for precision in precisions:
                        if precision == 'BF16' and not torch.cuda.is_bf16_supported():
                            rows.append(dict(device=device, precision=precision, status='NOT RUN')); continue
                        name = '__'.join((device, task, preset, mode, precision))
                        config = configuration(task, preset, mode, small=True)
                        value, audit = capture(config, device, precision)
                        row = dict(name=name, **audit)
                        if value is not None:
                            path = store/(name+'.pt')
                            if args.action == 'before':
                                assert not path.exists(); torch.save(value, path)
                            elif path.exists():
                                compare(torch.load(path, weights_only=True), value)
                                row['pre_post_exact'] = True
                            else: row['previously_failed'] = True
                        rows.append(row)
                    print(device, task, preset, mode, rows[-1]['status'], flush=True)
    (out/(args.action+'-numeric'+label+'.json')).write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__': main()
