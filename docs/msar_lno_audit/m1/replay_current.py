"""M1-only same-weight reference for existing KCDNO/matched cores, not MSAR math."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from cdlno.kcdno.config import KCDNOArchitectureConfig
from cdlno.kcdno.matched_config import MatchedLRSAConfig
from cdlno.kcdno.matched import make_core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('capture', 'replay'))
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    with sdpa_kernel(SDPBackend.MATH):
        for family in ('kcdno_all', 'kcdno_off', 'lrsa_matched'):
            for point in ('point_ffn', 'conv_ffn'):
                path = args.artifacts / (family + '_' + point + '.pt')
                cls = MatchedLRSAConfig if family == 'lrsa_matched' else KCDNOArchitectureConfig
                if args.mode == 'capture':
                    torch.manual_seed(260916)
                    kw = dict(L=2, d=8, h=2, M=4, point_module=point)
                    if family != 'lrsa_matched':kw.update(kernel_rank=3, history_mode=family.split('_')[-1])
                    config = cls(**kw)
                    model = make_core(config).eval()
                    x = torch.randn(2, 35, 8)
                    y = model(x, grid_shape=(5, 7) if point == 'conv_ffn' else None)
                    data = dict(config=config.to_dict(), state=model.state_dict(), x=x, y=y.detach())
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:torch.save(data, stream)
                else:
                    data = torch.load(path, weights_only=True, map_location='cpu')
                    model = make_core(cls.from_dict(data['config'])).eval()
                    model.load_state_dict(data['state'], strict=True)
                    with torch.no_grad():y = model(data['x'], grid_shape=(5, 7) if point == 'conv_ffn' else None)
                    torch.testing.assert_close(y, data['y'], atol=0, rtol=0)
                results.append(dict(name=path.stem, status='passed', mode=args.mode,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    keys_shapes={k:list(v.shape) for k,v in data['state'].items()},
                                    config=data['config'], atol=0, rtol=0, dtype='float32',
                                    device='cpu', backend='math', fixture=str(path)))
    args.result.write_text(json.dumps(results, indent=2) + '\n')
    print(f'{args.mode}: {len(results)}/{len(results)} passed')


if __name__ == '__main__':main()
