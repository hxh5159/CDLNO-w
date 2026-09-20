"""Synthetic metadata fixture: real encodings, no model/data/optimizer creation."""
import hashlib
from pathlib import Path
import random
import subprocess

from linearno_loop.config import resolve_config
from linearno_loop.contracts import digest, seal
from linearno_loop.schema import make_metadata

ROOT = Path(__file__).resolve().parents[2]


def config(task='darcy', **options):
    return resolve_config(task, options={'topology_preset':'p1_c3_r2_s1',
        'residual_mode':'rb_attnres', **options})


def metadata(c=None):
    # Only fixture generation imports tensor libraries; validator/import tests do not.
    import numpy as np
    import torch
    from cdlno.linearno.schema import pack_state, numerical_state
    c = config() if c is None else c
    checksum=hashlib.sha256(b'LL1 metadata-only synthetic input, no dataset').hexdigest()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(p.relative_to(ROOT)):p.read_text().replace('\r\n','\n')
             for p in sorted((ROOT/'linearno_loop').glob('*.py'))}
    source_hash=digest({k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()})
    # All package files are additions against the LL1 start manifest. This is a
    # path-sorted LF-normalized patch encoding, without host paths/timestamps.
    patch_hash=digest({'format':'path-before-after-lf-v1',
        'files':{k:{'before':None,'after':v} for k,v in sources.items()}})
    prov=dict(target_sha=head,base_commit=head,dirty=True,
        transolver_sha='75e0f67643806a81cd1d3f6adc88dd8c02416fe7',
        linearno_sha='3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269',
        paper_version='2511.06294v3',paper_sha256='637e2953c0e223df7ffe2be92dd3825fccf55232cd73e2c955935a3944a470fd',
        attnres_sha='85e22310fe5ee860b4a023de312d791de8a5a5e6',
        kimi_k3_sha='3cb39dfd32e51c3328e2e4b4af21341247d06c43',residual_scaling_version='2606.18524v1',
        source_sha256=source_hash, normalized_patch_sha256=patch_hash,
        code_version='LL1-contract-only',config_schema_version=1,metadata_schema_version=1,
        command=['python','-B','-m','unittest','discover','-s','tests/loop_linearno'],
        environment={'scope':'synthetic metadata only','python_backend':'fixture generation; no model'})
    rng=dict(python=random.Random(3).getstate(),numpy=np.random.RandomState(3).get_state(),
        torch_cpu=torch.Generator().manual_seed(3).get_state(),torch_cuda=[])
    return make_metadata(c,
        data_spec=dict(protocol=c['profile_spec']['values']['data'],scope='synthetic',
            split='synthetic train/test, no real data',sampling='fixed fixture',
            checksums={'synthetic':checksum},runtime={}),provenance_spec=prov,
        normalizer_spec=dict(policy='saved_train_fit',records={'input':dict(algorithm='fixture mean/std',
            fit_split='train',data_checksum=checksum,states={'mean':numerical_state(torch.tensor([1.,2.])),
                'std':numerical_state(np.array([3.,4.],dtype='>f8'))})}),
        resume_state=dict(checkpoint_role='epoch',selection_split=None,selection_metric=None,epoch=0,global_step=0,
            optimizer=pack_state({'state':{},'param_groups':[]}),scheduler=pack_state({'last_epoch':0}),
            rng=pack_state(rng),dataloader_generators=pack_state({'train':{'device':'cpu','state':rng['torch_cpu']}}),
            sampler_state=pack_state({'scope':'synthetic fixture'})),ensemble_manifest=[])


def rehash(value):
    return seal(value,'metadata_hash' if 'metadata_hash' in value else 'config_hash')
