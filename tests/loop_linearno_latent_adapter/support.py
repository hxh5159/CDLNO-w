"""Stdlib-only schema fixtures. Encoded RNG bytes are structural test data.

These fixtures are NOT backend-restorable training checkpoints; LAA1 validates
JSON structure only. No actual torch model/optimizer/RNG tensor is constructed.
"""
import base64
import random
from linearno_loop.v3.contracts import seal, ARCHITECTURE_SELECTOR
from linearno_loop.v3.config import resolve_config
from linearno_loop.v3.schema import make_metadata, REFERENCE_PINS


def config(task='darcy',**options):
    return resolve_config(task,options={'architecture':ARCHITECTURE_SELECTOR,**options})


def packed(value):
    if isinstance(value,Encoded):return {'type':'numeric','value':value.spec}
    if isinstance(value,dict):return {'type':'dict','value':[[packed(k),packed(v)] for k,v in value.items()]}
    if type(value) in (tuple,list):return {'type':type(value).__name__,'value':[packed(v) for v in value]}
    return {'type':'scalar','value':value}


class Encoded:
    def __init__(self,backend,dtype,shape,size):
        self.spec=seal(dict(backend=backend,dtype=dtype,shape=shape,encoding='base64',
                             data=base64.b64encode(bytes(size)).decode()),'sha256')


def metadata(c=None):
    c=config() if c is None else c
    provenance=dict(target_sha='1'*40,base_commit='1'*40,dirty=True,**REFERENCE_PINS,
        paper_version='2511.06294v3',residual_scaling_version='2606.18524v1',source_sha256='2'*64,
        normalized_patch_sha256='3'*64,code_version='LAA1-schema-fixture',config_schema_version=3,
        metadata_schema_version=3,command=['schema-fixture-only'],environment={'scope':'structural test only'})
    g=Encoded('torch','uint8',[8],8)
    rng=dict(python=random.Random(17).getstate(),numpy=('MT19937',Encoded('numpy','<u4',[624],2496),0,0,0.0),torch_cpu=g,torch_cuda=[])
    return make_metadata(c,
        data_spec=dict(protocol=c['profile_spec']['values']['data'],split='synthetic fixed split',sampling='synthetic fixed order',
                       scope='synthetic',checksums={'synthetic':'4'*64},runtime={}),
        provenance_spec=provenance,normalizer_spec=dict(policy='none',records={}),
        resume_state=dict(checkpoint_role='epoch',selection_split=None,selection_metric=None,epoch=0,global_step=0,
            optimizer=packed({'state':{},'param_groups':[]}),scheduler=packed({'last_epoch':0}),scaler=packed(None),
            rng=packed(rng),dataloader_generators=packed({'train':{'device':'cpu','state':g}}),sampler_state=packed({})),
        ensemble_manifest=[],parameter_measurement=dict(status='pending_model_construction',total=None,trainable=None,groups=None))
