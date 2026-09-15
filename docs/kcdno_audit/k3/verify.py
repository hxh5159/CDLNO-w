"""Reproducible K3 evidence only; no task/data imports or production hooks."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def core_evidence():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from cdlno.kcdno.config import KCDNOArchitectureConfig
    from cdlno.kcdno.core import KCDNO
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_kcdno_core import count_execution
    torch.set_num_threads(1)
    torch.manual_seed(3103)
    rows = []
    for point in ('point_ffn', 'conv_ffn'):
        for mode in ('all', 'off'):
            cfg = KCDNOArchitectureConfig(point_module=point, history_mode=mode)
            model = KCDNO(cfg).eval()
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), count_execution(model) as counts:
                output = model(torch.randn(2, 35, cfg.d), grid_shape=(5, 7) if point == 'conv_ffn' else None)
            parameters = sum(p.numel() for p in model.parameters())
            block_parameters = [{name: sum(p.numel() for p in part.parameters())
                                 for name, part in b.named_children()} for b in model.blocks]
            d, h, M, r, L = cfg.d, cfg.h, cfg.M, cfg.kernel_rank, cfg.L
            public = (19*d*d + 16*d if point == 'point_ffn' else 28*d*d + 17*d) + 4*(d//h) + M*d
            expected = L*public + ((L-1)*(2*d*r + 4*d + 1) if mode == 'all' else 0)
            assert parameters == expected
            assert all(p.requires_grad for p in model.parameters())
            assert counts['down'] == counts['up'] == L
            assert counts['logical_reads'] == (L*(L-1)//2 if mode == 'all' else 0)
            rows.append(dict(config=cfg.to_dict(), parameters=parameters, trainable_parameters=parameters,
                             independent_parameter_formula=expected, per_block=block_parameters,
                             actual_calls=dict(counts), output_shape=list(output.shape)))
    write('core-evidence.json', dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device='cpu', dtype='float32', sdpa='MATH', threads=1, amp=False, compile=False,
        scope='feature core only, no task lift/output head; parameter counts are not latency/MAC estimates', rows=rows))
    print([(r['config']['point_module'], r['config']['history_mode'], r['parameters']) for r in rows])


def old_core_replay():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sys.path.insert(0, str(ROOT / 'docs/kcdno_audit'))
    import make_regression_fixtures as fixtures
    index = json.loads((ROOT / 'docs/kcdno_audit/fixture_index.json').read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = []
    with sdpa_kernel(SDPBackend.MATH):
        for i, row in enumerate(fixtures.cases('standard')):
            if row['task'] != 'core':
                continue
            result = fixtures.process(row, Path(index['artifact_root']), 'replay', 2026091500+i)
            rows.append(result)
            print(row['name'], result['status'])
    assert len(rows) == 9 and all(r['status'] == 'passed' for r in rows)
    write('old-core-replay.json', dict(artifact_root=index['artifact_root'], same_weights=True,
          device='cpu', dtype='float32', sdpa='MATH', atol=0, rtol=0,
          scope='K0 nine CDLNO front-mode x CDPA-mode core fixtures; saved output/input+parameter gradient replay', rows=rows))


def freeze():
    before = json.loads((OUT / 'before.json').read_text())
    changed, missing = [], []
    for name, digest in before['hashes'].items():
        p = ROOT / name
        if not p.is_file():
            missing.append(name)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            changed.append(name)
    allowed = ['docs/KCDNO_IMPLEMENTATION_STATUS.md', 'memory/current-state.md']
    unexpected = sorted(set(changed) - set(allowed))
    checks = {}
    for path in (ROOT / 'cdlno/kcdno/core.py', ROOT / 'tests/test_kcdno_core.py', Path(__file__)):
        ast.parse(path.read_text(), feature_version=(3, 10))
        checks[str(path.relative_to(ROOT))] = 'Python 3.10 syntax parsed; not remote runtime acceptance'
    result = dict(baseline_files=len(before['hashes']), unchanged=len(before['hashes'])-len(changed)-len(missing),
                  changed=changed, missing=missing, unexpected_changes=unexpected,
                  python_syntax=checks, source_snapshot=before['source_snapshot'])
    write('freeze.json', result)
    print(json.dumps(result, ensure_ascii=False))
    assert not unexpected and not missing


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('core', 'old-core', 'freeze'))
    args = p.parse_args()
    {'core': core_evidence, 'old-core': old_core_replay, 'freeze': freeze}[args.action]()
