"""LAA5 wrapper/checkpoint fixtures. No dataset reads or production entry imports."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import hashlib
import random

import numpy as np
import torch

from cdlno.linearno.profiles import resolve_config as base_resolve
from cdlno.linearno.schema import normalizer_record
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.contracts import ARCHITECTURE_SELECTOR


TASK_VARIANTS = (
    ("ns", "plain"),
    ("elasticity", "temp"),
    ("plasticity", "conv"),
    ("airfoil", "conv_temp"),
    ("airfrans", "airfrans"),
    ("car", "shapenet"),
)
MODES = ("sr_1_over_r", "rb_attnres", "lb_attnres_1_over_r")
ABLATIONS = ((False, False), (False, True), (True, False), (True, True))
WRAPPER_ROWS = []
CHECKPOINT_ROWS = []


def _contract(task):
    if task in ("ns", "plasticity"):
        return {"contract": "standard_temporal_l5"}
    if task not in ("airfrans", "car"):
        return {"contract": "standard_static_l4"}
    return {}


def config(task="elasticity", *, cost="custom", mode=MODES[0], latent=True,
           adapter=True, seed=17, formal_width=False):
    """Resolve a valid saved profile with a 2x3 synthetic structured grid.

    formal_width retains the exact matched/efficient D12 H/Dz/M/head table. Only
    the saved grid dimensions and runtime seed are synthetic test facts.
    """
    explicit = {"model.H": 2, "model.W": 3, "runtime.seed": seed}
    base = base_resolve(task, "paper_table8_on_release_model", explicit=explicit,
                        **_contract(task))
    options = dict(architecture=ARCHITECTURE_SELECTOR, cost_profile=cost,
                   topology_preset="d12", residual_mode=mode,
                   latent_enabled=latent,
                   adapter_mode=("bilateral_qk_lowrank_second_visit" if adapter else "none"))
    if cost == "custom":
        options.update(hidden_width=8, latent_width=11, heads=2, actual_M=5,
                       adapter_rank=2, adapter_alpha=3.0)
    elif not formal_width:
        raise ValueError("tabulated profiles always use their formal widths")
    with patch("linearno_loop.v3.config._profile", return_value=base):
        return resolve_config(task, "paper_table8_on_release_model", options=options)


def construct(c, *, member_seed=None, dtype=torch.float32):
    from cdlno.linearno_loop.v3.construction import build_from_config
    return build_from_config(c, initialization_seed=member_seed).to(dtype=dtype)


def example(c, *, batch=2, points=6, dtype=torch.float32):
    task = c["loop_spec"]["task"]
    model = c["profile_spec"]["values"]["model"]
    if task == "airfrans":
        from torch_geometric.data import Data
        return (Data(x=torch.randn(points, 7, dtype=dtype),
                     pos=torch.randn(points, 2, dtype=dtype),
                     batch=torch.zeros(points, dtype=torch.long),
                     ptr=torch.tensor([0, points])),)
    if task == "car":
        from torch_geometric.data import Data
        data = Data(x=torch.randn(points, 7, dtype=dtype),
                    pos=torch.randn(points, 3, dtype=dtype),
                    batch=torch.zeros(points, dtype=torch.long),
                    ptr=torch.tensor([0, points]))
        return ((data, torch.randn(4, 3, dtype=dtype)),)
    x = torch.randn(batch, points, model["space_dim"], dtype=dtype)
    fx = (None if model["fun_dim"] == 0 else
          torch.randn(batch, points, model["fun_dim"], dtype=dtype))
    T = torch.rand(batch, 1, dtype=dtype) if model["time_input"] else None
    return (x, fx, T)


def invoke(model, c, args):
    task = c["loop_spec"]["task"]
    if task in ("airfrans", "car"):
        return model(*args)
    return model(*args)


def public_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if ".latent_processor." not in key and ".adapter." not in key and
            not key.startswith("loop.rb_") and not key.startswith("loop.lb_")}


def finite_step(model, c):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = invoke(model, c, example(c, batch=1))
    loss = output.float().square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if not torch.isfinite(output).all():
        raise AssertionError("nonfinite wrapper output")
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
        raise AssertionError("nonfinite wrapper gradient")
    return output.detach(), optimizer


def provenance():
    from linearno_loop.schema import REFERENCE_PINS
    return dict(target_sha="a" * 40, base_commit="a" * 40, dirty=True,
        **REFERENCE_PINS, paper_version="2511.06294v3",
        residual_scaling_version="2606.18524v1", source_sha256="b" * 64,
        normalized_patch_sha256="c" * 64, code_version="LAA5-synthetic",
        config_schema_version=3, metadata_schema_version=3,
        command=["python", "-B", "LAA5 synthetic"], environment={"scope": "synthetic"})


def checkpoint_metadata(c, model, optimizer, scheduler, *, epoch=1, total=2,
                        generators=None, scaler=None, ensemble=None):
    from cdlno.linearno_loop.v3.checkpoint import measure_parameters, resume_state
    from linearno_loop.v3.schema import make_metadata
    checksum = hashlib.sha256(b"LAA5 synthetic data").hexdigest()
    normalizers = dict(policy="saved_train_fit", records={"input": normalizer_record(
        {"mean": torch.tensor([0.]), "std": torch.tensor([1.])},
        fit_split="synthetic train", data_checksum=checksum, algorithm="fixed fixture")})
    state = resume_state(optimizer, scheduler, epoch=epoch, steps_per_epoch=1,
                         total_epochs=total, generators=generators or {},
                         sampler={"scope": "synthetic"}, scaler=scaler)
    return make_metadata(c,
        data_spec=dict(protocol=c["profile_spec"]["values"]["data"],
                       split="synthetic fixed", sampling="synthetic fixed",
                       checksums={"synthetic": checksum}, scope="synthetic", runtime={}),
        provenance_spec=provenance(), normalizer_spec=normalizers,
        resume_state=state, ensemble_manifest=[] if ensemble is None else ensemble,
        parameter_measurement=measure_parameters(model, c))


@contextmanager
def seeded(seed):
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(torch_state)

