#!/usr/bin/env python3
"""Data-free dependency checks for the Transolver/CDPA research environment.

This script does not install packages, download datasets, run training entry
points, or modify repository files. It creates tiny temporary synthetic files
only. A successful run establishes dependency checks, not training correctness.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--repo", type=Path, help="Optional Transolver repository root")
    parser.add_argument("--json-path", type=Path, help="Optional machine-readable report")
    parser.add_argument(
        "--check-amp", action="store_true",
        help="Optional CUDA BF16 SDPA check; does not enable AMP in training",
    )
    return parser.parse_args()


class SkipCheck(Exception):
    pass


def main():
    args = arguments()
    # Only this process and its child import probes use these settings.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    results = []
    modules = {}
    report = {
        "purpose": "Dependency and primitive checks only; no real dataset or training validation",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "requested_device": args.device,
        "target": {"torch": "2.11.x", "torch_cuda_runtime": "12.8"},
        "versions": {},
        "checks": results,
        "gpu_verified": False,
    }

    def emit(name, status, detail):
        results.append({"name": name, "status": status, "detail": str(detail)})
        print(f"[{status}] {name}: {detail}", flush=True)

    def run(name, function):
        try:
            detail = function()
            emit(name, "PASS", detail if detail is not None else "completed")
            return True
        except SkipCheck as error:
            emit(name, "SKIP", error)
        except Exception as error:
            emit(name, "FAIL", f"{type(error).__name__}: {error}")
        return False

    def require(*names):
        missing = [name for name in names if name not in modules]
        if missing:
            raise SkipCheck("Unavailable dependency: " + ", ".join(missing))
        return [modules[name] for name in names]

    imports = {
        "torch": "torch", "torchvision": "torchvision",
        "torch_geometric": "torch-geometric", "pyg_lib": "pyg-lib",
        "numpy": "numpy", "scipy": "scipy", "einops": "einops",
        "timm": "timm", "matplotlib": "matplotlib", "tqdm": "tqdm",
        "sklearn": "scikit-learn", "yaml": "PyYAML", "seaborn": "seaborn",
        "pandas": "pandas", "pyvista": "pyvista", "vtk": "vtk",
    }
    for module_name, distribution in imports.items():
        def check_import(name=module_name, dist=distribution):
            module = importlib.import_module(name)
            modules[name] = module
            try:
                version = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                version = str(getattr(module, "__version__", "unknown"))
            report["versions"][dist] = version
            return version
        run("import." + module_name, check_import)

    def pip_check():
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "check"], capture_output=True,
            text=True, timeout=60,
        )
        if proc.returncode:
            raise RuntimeError((proc.stdout + proc.stderr).strip())
        return proc.stdout.strip()
    run("pip.check", pip_check)

    def target_build():
        (torch,) = require("torch")
        version = torch.__version__
        runtime = torch.version.cuda
        report["torch_cuda_runtime"] = runtime
        if version.split("+")[0].split(".")[:2] != ["2", "11"]:
            raise RuntimeError(f"Expected torch 2.11.x; found {version}")
        if runtime != "12.8":
            raise RuntimeError(f"Expected torch CUDA runtime 12.8; found {runtime}")
        return f"torch={version}, torch.version.cuda={runtime}; not inferred from nvcc/nvidia-smi"
    run("torch.target_build", target_build)

    cuda_ready = False
    if args.device == "cuda":
        def cuda_check():
            (torch,) = require("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
            probe = torch.ones(4, device="cuda")
            assert probe.sum().item() == 4
            torch.cuda.synchronize()
            report["cuda_device"] = torch.cuda.get_device_name()
            report["cuda_capability"] = list(torch.cuda.get_device_capability())
            return report["cuda_device"]
        cuda_ready = run("cuda.available", cuda_check)
    else:
        emit("cuda.validation", "SKIP", "CPU mode explicitly selected; GPU operation is NOT verified")

    def sdpa(device):
        (torch,) = require("torch")
        q = torch.randn(2, 2, 7, 8, device=device, requires_grad=True)
        k = torch.randn(2, 2, 5, 8, device=device, requires_grad=True)
        v = torch.randn(2, 2, 5, 8, device=device, requires_grad=True)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False,
        )
        assert out.shape == q.shape and torch.isfinite(out).all().item()
        out.square().mean().backward()
        for tensor in (q, k, v):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
            assert tensor.grad.abs().sum().item() > 0
        if device == "cuda":
            torch.cuda.synchronize()
        return "FP32 noncausal cross-attention forward/backward finite; backend selection automatic"

    run("sdpa.cpu", lambda: sdpa("cpu"))
    gpu_sdpa_ok = run("sdpa.cuda", lambda: sdpa("cuda")) if cuda_ready else False

    def radius_check(device):
        torch, _ = require("torch", "torch_geometric")
        from torch_geometric.nn import radius_graph
        # Identical coordinates in different graphs expose accidental cross-batch edges.
        x = torch.tensor([[0., 0.], [.1, 0.], [0., 0.], [.1, 0.]], device=device)
        batch = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        edge = radius_graph(x, r=.2, batch=batch, loop=True, max_num_neighbors=8)
        assert edge.dtype == torch.long and edge.shape == (2, 8)
        assert edge.device == x.device
        assert torch.equal(batch[edge[0]], batch[edge[1]])
        # No neighbor truncation here; compare edge sets, not implementation ordering.
        actual = set(map(tuple, edge.cpu().T.tolist()))
        expected = {(i, j) for i in range(4) for j in range(4) if i // 2 == j // 2}
        assert actual == expected
        if device == "cuda":
            torch.cuda.synchronize()
        return "radius_graph executes with loop=True; expected edges and batch isolation verified"

    run("pyg.radius_graph.cpu", lambda: radius_check("cpu"))
    gpu_radius_ok = run("pyg.radius_graph.cuda", lambda: radius_check("cuda")) if cuda_ready else False

    def pyg_batch():
        torch, _ = require("torch", "torch_geometric")
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.utils import k_hop_subgraph, subgraph
        data = Data(
            x=torch.randn(3, 7), pos=torch.randn(3, 3), y=torch.randn(3, 4),
            surf=torch.tensor([True, False, True]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        batch = next(iter(DataLoader([data, data.clone()], batch_size=2)))
        assert batch.x.shape == (6, 7) and batch.y.shape == (6, 4)
        assert batch.surf.dtype == torch.bool
        assert torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]])
        restored = batch.to_data_list()
        assert len(restored) == 2 and torch.equal(restored[1].surf, data.surf)
        ids, edges, _, _ = k_hop_subgraph(0, 1, data.edge_index, relabel_nodes=True)
        assert ids.numel() == 2 and edges.shape[0] == 2
        sub_edges, _ = subgraph(torch.tensor([0, 1]), data.edge_index, relabel_nodes=True)
        assert sub_edges.shape == (2, 2)
        # Car loader collates a pair: (CFD Data, geometry point tensor).
        pair = next(iter(DataLoader([(data, torch.randn(2, 3)),
                                     (data.clone(), torch.randn(2, 3))], batch_size=2)))
        assert pair[0].x.shape == (6, 7) and pair[1].shape == (2, 2, 3)
        return "Data, masks, pair batching, DataLoader, k_hop_subgraph and subgraph checked"
    run("pyg.data_and_loader", pyg_batch)

    def timm_init():
        torch, _ = require("torch", "timm")
        from timm.models.layers import trunc_normal_
        tensor = torch.empty(16, 16)
        trunc_normal_(tensor, std=.02)
        assert torch.isfinite(tensor).all().item()
        return "Original repository timm.models.layers import remains functional"
    run("timm.legacy_initializer", timm_init)

    def scipy_io():
        np, _ = require("numpy", "scipy")
        from scipy.io import loadmat, savemat
        from scipy.spatial import ConvexHull
        from scipy.stats import spearmanr
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        stream = io.BytesIO()
        savemat(stream, {"coeff": arr, "sol": arr + 1})
        stream.seek(0)
        restored = loadmat(stream)
        np.testing.assert_array_equal(restored["coeff"], arr)
        stream = io.BytesIO()
        np.save(stream, arr)
        stream.seek(0)
        np.testing.assert_array_equal(np.load(stream, allow_pickle=False), arr)
        hull = ConvexHull(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]))
        assert np.isclose(hull.volume, 1.)
        assert np.isclose(spearmanr([1, 2, 3], [2, 4, 6]).statistic, 1.)
        if "torch" in modules:
            converted = modules["torch"].from_numpy(arr)
            assert tuple(converted.shape) == arr.shape
        return "Synthetic MAT/NPY roundtrips, ConvexHull and Spearman; no benchmark files used"
    run("scipy_numpy.io_and_metrics", scipy_io)

    def sklearn_check():
        require("sklearn")
        from sklearn.neighbors import NearestNeighbors
        distances, indices = NearestNeighbors(n_neighbors=1).fit([[0., 0.], [1., 0.]]).kneighbors([[.1, 0.]])
        assert indices[0, 0] == 0 and abs(distances[0, 0] - .1) < 1e-6
        return "NearestNeighbors native extension executes"
    run("sklearn.neighbors", sklearn_check)

    def mesh_io():
        pv, vtk, np = require("pyvista", "vtk", "numpy")
        from vtk.util.numpy_support import vtk_to_numpy
        plane = pv.Plane(i_resolution=2, j_resolution=2)
        grid = plane.cast_to_unstructured_grid()
        grid.point_data["U"] = np.asarray(grid.points).copy()
        grid.point_data["p"] = np.arange(grid.n_points, dtype=float)
        grid.point_data["nut"] = np.ones(grid.n_points)
        sized = grid.compute_cell_sizes(length=False, volume=False)
        assert np.all(sized.cell_data["Area"] > 0)
        # Exercise the exact alias still used in the original AirfRANS evaluator.
        converted = grid.ptc(pass_point_data=True)
        assert "p" in converted.cell_data and "p" in converted.point_data
        derived = grid.compute_derivative(scalars="U", gradient="pred_grad")
        assert np.isfinite(derived.point_data["pred_grad"]).all()
        sampled = grid.sample_over_line((-.25, 0., 0.), (.25, 0., 0.), resolution=8)
        assert sampled.n_points == 9 and "U" in sampled.point_data
        with tempfile.TemporaryDirectory(prefix="cdpa_mesh_check_") as tmp:
            folder = Path(tmp)
            grid.save(folder / "internal.vtu")
            plane.save(folder / "aerofoil.vtp")
            grid.save(folder / "legacy.vtk")
            assert pv.read(folder / "internal.vtu").n_points == grid.n_points
            assert pv.read(folder / "aerofoil.vtp").n_cells == plane.n_cells
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(str(folder / "legacy.vtk"))
            reader.Update()
            restored = reader.GetOutput()
            assert restored.GetNumberOfPoints() == grid.n_points
            assert vtk_to_numpy(restored.GetPoints().GetData()).shape == (grid.n_points, 3)
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(restored)
            surface.Update()
            assert surface.GetOutput().GetNumberOfCells() > 0
            p2c = vtk.vtkPointDataToCellData()
            p2c.SetInputData(restored)
            p2c.PassPointDataOn()
            p2c.Update()
            assert p2c.GetOutput().GetCellData().GetArray("p") is not None
        return "Synthetic VTU/VTP/legacy VTK, ptc, cell area, gradient and line sampling; no window"
    run("pyvista_vtk.io_and_filters", mesh_io)

    def checkpoints():
        (torch,) = require("torch")
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
        model.eval()
        x = torch.randn(4, 3)
        expected = model(x).detach()
        with tempfile.TemporaryDirectory(prefix="cdpa_checkpoint_check_") as tmp:
            folder = Path(tmp)
            state_path = folder / "state.pt"
            object_path = folder / "model.pt"
            list_path = folder / "models.pt"
            torch.save(model.state_dict(), state_path)
            other = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2))
            other.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
            torch.testing.assert_close(other(x), expected)
            # These objects were constructed and saved just above by this process.
            # Never apply this setting blindly to externally obtained checkpoints.
            torch.save(model, object_path)
            loaded = torch.load(object_path, map_location="cpu", weights_only=False)
            torch.testing.assert_close(loaded(x), expected)
            torch.save([model], list_path)
            loaded_list = torch.load(list_path, map_location="cpu", weights_only=False)
            torch.testing.assert_close(loaded_list[0](x), expected)
        return "state_dict weights_only=True; trusted temporary Module/list weights_only=False"
    run("torch.checkpoint_roundtrip", checkpoints)

    if args.check_amp:
        def amp_check():
            (torch,) = require("torch")
            if not cuda_ready or not torch.cuda.is_bf16_supported():
                raise SkipCheck("Optional CUDA BF16 probe unavailable; FP32 checks remain authoritative")
            q = torch.randn(1, 2, 8, 16, device="cuda", requires_grad=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=0.)
            out.float().square().mean().backward()
            assert torch.isfinite(out).all().item() and torch.isfinite(q.grad).all().item()
            return "Optional BF16 SDPA only; training remains FP32 unless separately configured"
        run("optional.bf16_sdpa", amp_check)
    else:
        emit("optional.bf16_sdpa", "SKIP", "Not requested; no AMP training configuration is implied")

    if args.repo is not None:
        repo = args.repo.expanduser().resolve()
        probes = {
            "PDE-Solving-StandardBenchmark": ["model_dict", "utils.normalizer", "utils.testloss"],
            "Car-Design-ShapeNetCar": ["models.Transolver", "dataset.dataset", "dataset.load_dataset", "train", "utils.drag_coefficient"],
            "Airfoil-Design-AirfRANS": ["models.Transolver", "dataset.dataset", "train", "utils.metrics"],
        }
        for subdir, names in probes.items():
            def import_repo(folder=subdir, imports=names):
                cwd = repo / folder
                if not cwd.is_dir():
                    raise FileNotFoundError(cwd)
                # Explicit allowlist: no exp_*.py, main.py or main_evaluation.py.
                code = "import importlib\n" + "\n".join(
                    f"importlib.import_module({name!r})" for name in imports
                ) + "\nprint('safe module imports completed')\n"
                proc = subprocess.run(
                    [sys.executable, "-B", "-c", code], cwd=str(cwd),
                    env=os.environ.copy(), capture_output=True, text=True, timeout=60,
                )
                if proc.returncode:
                    raise RuntimeError((proc.stdout + proc.stderr).strip()[-6000:])
                return ", ".join(imports) + "; entry points not executed"
            run("repo.imports." + subdir, import_repo)
    else:
        emit("repo.imports", "SKIP", "--repo not supplied; original repository imports not verified")

    report["gpu_verified"] = bool(cuda_ready and gpu_sdpa_ok and gpu_radius_ok)
    failures = sum(item["status"] == "FAIL" for item in results)
    report["failure_count"] = failures
    report["status"] = "FAIL" if failures else ("PASS_GPU_NOT_VERIFIED" if args.device == "cpu" else "PASS")
    print("\nResult:", report["status"], flush=True)
    print("This result does NOT verify real data loading, complete metrics, training convergence, or model speed.", flush=True)
    if args.json_path is not None:
        output = args.json_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("JSON report:", output, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
