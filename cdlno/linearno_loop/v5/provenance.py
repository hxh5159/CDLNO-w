"""V5 provenance for metadata records."""
import hashlib
from pathlib import Path


def provenance(config, task=None):
    root = Path(__file__).resolve().parents[3]
    paths = [*sorted((root / "linearno_loop" / "v5").glob("*.py")),
             *sorted((root / "cdlno" / "linearno_loop" / "v5").glob("*.py"))]
    paths += [
        root / "linearno_loop" / "versioning.py",
        root / "cdlno" / "linearno_loop" / "versioning.py",
        root / "cdlno" / "linearno_loop" / "standard_entry.py",
        root / "cdlno" / "linearno_loop" / "industrial_entry.py",
        root / "cdlno" / "linearno_loop" / "air_entry.py",
        root / "cdlno" / "linearno_loop" / "car_entry.py",
        root / "cdlno" / "linearno" / "standard_entry.py",
        root / "tran_evaluate" / "linearno_loop" / "recording.py",
        root / "tran_evaluate" / "linearno_loop_v5" / "launch.py",
        root / "PDE-Solving-StandardBenchmark" / "cdlno_entry.py",
        root / "PDE-Solving-StandardBenchmark" / "linearno_entry.py",
        root / "PDE-Solving-StandardBenchmark" / "model_dict.py",
        *[root / "PDE-Solving-StandardBenchmark" / f"exp_{name}.py"
          for name in ("airfoil", "darcy", "elas", "pipe", "ns", "plas")],
        root / "Airfoil-Design-AirfRANS" / "main.py",
        root / "Airfoil-Design-AirfRANS" / "train.py",
        root / "Car-Design-ShapeNetCar" / "main.py",
        root / "Car-Design-ShapeNetCar" / "train.py",
    ]
    paths = sorted(set(paths))
    value = hashlib.sha256()
    for path in paths:
        value.update(str(path.relative_to(root)).encode()); value.update(path.read_bytes())
    try:
        import subprocess
        target = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        target = "unknown"
    return dict(target_sha=target, source_sha256=value.hexdigest(),
                code_version="partial_share_feature_gate_v5", task=task or config["task"])
