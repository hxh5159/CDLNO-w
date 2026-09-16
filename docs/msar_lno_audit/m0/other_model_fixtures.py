"""Small current references for other real legacy registry models, no task import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def worker(args):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(260918)
    rows = []
    if args.project == 'air':
        import yaml
        from torch_geometric.data import Data
        from models.MLP import MLP
        from models.NN import NN
        from models.PointNet import PointNet
        from models.GraphSAGE import GraphSAGE
        configurations = yaml.safe_load(Path('params.yaml').read_text())
        classes = {'MLP': NN, 'PointNet': PointNet, 'GraphSAGE': GraphSAGE}

        def make(name, config):
            return classes[name](config, MLP(config['encoder'], batch_norm=False),
                                 MLP(config['decoder'], batch_norm=False)).eval()

        def inputs():
            n = 13
            return dict(x=torch.randn(n, 7), pos=torch.randn(n, 2), y=torch.randn(n, 4),
                        surf=torch.arange(n) % 2 == 0, batch=torch.zeros(n, dtype=torch.long),
                        edge_index=torch.stack((torch.arange(n), torch.arange(n).roll(1))))

        def call(model, data):
            return model(Data(**data))
    else:
        from model_dict import get_model
        from types import SimpleNamespace
        classes = {'Transolver_Structured_Mesh_3D': get_model(
            SimpleNamespace(model='Transolver_Structured_Mesh_3D')).Model}
        configurations = {'Transolver_Structured_Mesh_3D': dict(space_dim=3, fun_dim=1,
            out_dim=1, H=3, W=5, D=2, n_hidden=8, n_head=2, slice_num=4, n_layers=2,
            unified_pos=False, dropout=0)}

        def make(name, config):
            return classes[name](**config).eval()

        def inputs():
            return dict(x=torch.randn(2, 30, 3), fx=torch.randn(2, 30, 1))

        def call(model, data):
            return model(**data)

    with torch.no_grad():
        for name in classes:
            folder = args.artifacts / name
            if args.action == 'capture':
                model = make(name, configurations[name])
                data = inputs()
                fixture = dict(config=configurations[name], inputs=data, output=call(model, data),
                    weights=model.state_dict(), class_path=type(model).__module__ + '.' + type(model).__qualname__)
                folder.mkdir(parents=True, exist_ok=False)
                with (folder / 'fixture.pt').open('xb') as stream:
                    torch.save(fixture, stream)
                if args.project == 'air':
                    with (folder / 'list.pt').open('xb') as stream:
                        torch.save([model], stream)
            else:
                fixture = torch.load(folder / 'fixture.pt', weights_only=True, map_location='cpu')
                model = make(name, fixture['config'])
                model.load_state_dict(fixture['weights'], strict=True)
                torch.testing.assert_close(call(model, fixture['inputs']), fixture['output'], atol=0, rtol=0)
                if args.project == 'air':
                    saved = torch.load(folder / 'list.pt', weights_only=False, map_location='cpu')
                    assert type(saved) is list and len(saved) == 1 and type(saved[0]) is type(model)
                    torch.testing.assert_close(call(saved[0].eval(), fixture['inputs']), fixture['output'], atol=0, rtol=0)
            rows.append(dict(name=name, status='passed', config=fixture['config'],
                class_path=fixture['class_path'], fixture=str(folder / 'fixture.pt'),
                keys_shapes={k:list(v.shape) for k,v in fixture['weights'].items()},
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.pt')},
                output_shape=list(fixture['output'].shape), device='cpu', dtype='float32', atol=0, rtol=0,
                scope='post-M1 current synthetic reference; not historic training weights; no real graph construction'))
    if args.project == 'air':
        rows.append(dict(name='GUNet', status='not_run',
                         reason='real torch_cluster absent; multiscale radius/nearest graph path unavailable'))
    args.result.write_text(json.dumps(rows, indent=2) + '\n')
    print(args.project, args.action, [(r['name'],r['status']) for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('capture', 'replay'))
    p.add_argument('--project', choices=('air', 'standard'))
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    a = p.parse_args()
    a.artifacts, a.result = a.artifacts.resolve(), a.result.resolve()
    if a.project:
        sys.path[:0] = [str(Path.cwd()), str(ROOT)]
        worker(a)
    else:
        rows = []
        for project, directory in [('air', 'Airfoil-Design-AirfRANS'), ('standard', 'PDE-Solving-StandardBenchmark')]:
            output = a.result.with_name(a.result.stem + '-' + project + '.json')
            cmd = [sys.executable, '-B', str(Path(__file__).resolve()), a.action, '--project', project,
                   '--artifacts', str(a.artifacts), '--result', str(output)]
            result = subprocess.run(cmd, cwd=ROOT / directory, env=dict(os.environ, PYTHONPATH=str(ROOT)))
            rows.append(dict(project=project, cwd=str(ROOT / directory), command=cmd,
                             returncode=result.returncode, rows=json.loads(output.read_text()) if output.exists() else []))
        a.result.write_text(json.dumps(rows, indent=2) + '\n')
        if any(r['returncode'] for r in rows):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
