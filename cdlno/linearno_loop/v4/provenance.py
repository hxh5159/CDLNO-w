"""Portable hashes of the actual v4 construction, task and checkpoint chain."""
import hashlib
from pathlib import Path
import subprocess

from linearno_loop.v4.config import validate_config
from linearno_loop.v4.contracts import digest


def provenance(config, *, task=None):
    config = validate_config(config)
    root = Path(__file__).resolve().parents[3]
    paths = set()
    for folder in ('linearno_loop/v4', 'cdlno/linearno_loop/v4',
                   'cdlno/linearno_loop', 'cdlno/linearno', 'cdlno/linearno_history'):
        paths.update((root / folder).glob('*.py'))
    for relative in ('linearno_loop/versioning.py', 'cdlno/training_state.py',
                     'cdlno/experiment.py', 'cdlno/visualization.py',
                     'tran_evaluate/linearno_loop/entry.py',
                     'tran_evaluate/linearno_loop/recording.py'):
        paths.add(root / relative)
    project = {'car': 'Car-Design-ShapeNetCar', 'airfrans': 'Airfoil-Design-AirfRANS'}.get(
        config['task'], 'PDE-Solving-StandardBenchmark')
    paths.update((root / project).glob('*.py'))
    for folder in ('model', 'models', 'utils', 'dataset'):
        paths.update((root / project / folder).glob('*.py'))
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(paths)}
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = '0' * 40
    return dict(code_version='resmlp_dual_temp_v4', task=config['task'],
                target_sha=head, source_sha256=digest(sources), sources=sources)
