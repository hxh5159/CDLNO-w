"""Small valid v2 metadata fixture; no model construction or dataset access."""

import hashlib
import random

from cdlno.linearno.schema import numerical_state, pack_state
from linearno_loop.v2.config import resolve_config
from linearno_loop.v2.schema import make_metadata


def config(task="darcy", mode="round_specific", **options):
    return resolve_config(task, options={"topology_preset": "p1_c3_r2_s1",
        "residual_mode": "sr_1_over_r", "core_ffn_mode": mode, **options})


def metadata(checked=None):
    import numpy as np
    import torch
    checked = config() if checked is None else checked
    checksum = hashlib.sha256(b"v2 synthetic metadata fixture").hexdigest()
    generator = torch.Generator().manual_seed(11).get_state()
    provenance = dict(
        target_sha="8" * 40, base_commit="8" * 40, dirty=True,
        transolver_sha="75e0f67643806a81cd1d3f6adc88dd8c02416fe7",
        linearno_sha="3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269",
        paper_version="2511.06294v3",
        paper_sha256="637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd",
        attnres_sha="85e22310fe5ee860b4a023de312d791de8a5a5e6",
        kimi_k3_sha="3cb39dfd32e51c3328e2e4b4af21341247d06c43",
        residual_scaling_version="2606.18524v1", source_sha256="1" * 64,
        normalized_patch_sha256="2" * 64, code_version="LF1-v2-contract",
        config_schema_version=2, metadata_schema_version=2,
        command=["python", "-B", "-m", "unittest"], environment={"scope": "synthetic"},
    )
    rng = dict(python=random.Random(11).getstate(), numpy=np.random.RandomState(11).get_state(),
               torch_cpu=generator, torch_cuda=[])
    return make_metadata(checked,
        data_spec=dict(protocol=checked["profile_spec"]["values"]["data"], scope="synthetic",
            split="fixed train/test", sampling="fixed", checksums={"synthetic": checksum}, runtime={}),
        provenance_spec=provenance,
        normalizer_spec=dict(policy="saved_train_fit", records={"input": dict(
            algorithm="fixture", fit_split="train", data_checksum=checksum,
            states={"mean": numerical_state(torch.tensor([0.])), "std": numerical_state(torch.tensor([1.]))})}),
        resume_state=dict(checkpoint_role="epoch", selection_split=None, selection_metric=None,
            epoch=0, global_step=0, optimizer=pack_state({"state": {}, "param_groups": []}),
            scheduler=pack_state({"last_epoch": 0}), rng=pack_state(rng),
            dataloader_generators=pack_state({"train": {"device": "cpu", "state": generator}}),
            sampler_state=pack_state({"scope": "fixture"})), ensemble_manifest=[])

