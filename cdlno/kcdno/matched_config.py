"""Independent, fixed-full LRSA structural-control contract (not paper reproduction)."""
from dataclasses import dataclass, fields
from .config import _Record, _positive_int, KCDNOArchitectureConfig, KCDNORuntimeConfig
from .metadata import KCDNOMetadataMismatch
from .profiles import TASKS, PROFILE_NAMES


@dataclass(frozen=True, slots=True)
class MatchedLRSAConfig(_Record):
    family: str = 'lrsa_matched'
    architecture_version: str = 'lrsa-matched-full-v1'
    L: int = 8
    d: int = 128
    h: int = 8
    M: int = 64
    ffn1_hidden: int | None = None
    ffn2_hidden: int | None = None
    point_hidden: int | None = None
    latent_activation: str = 'gelu'
    point_activation: str = 'gelu'
    point_module: str = 'point_ffn'
    norm: str = 'rmsnorm'
    qk_norm: str = 'per_head_rmsnorm'
    output_norm: str = 'layernorm'
    norm_eps: float = 1e-6
    attention_dropout: float = 0.
    latent_processor: str = 'full'

    def __post_init__(self):
        _positive_int('d',self.d)
        for name in ('ffn1_hidden','ffn2_hidden','point_hidden'):
            if getattr(self,name) is None:object.__setattr__(self,name,2*self.d)
        self.validate()

    def validate(self):
        if self.family != 'lrsa_matched' or self.architecture_version != 'lrsa-matched-full-v1' or self.latent_processor != 'full':
            raise ValueError('matched LRSA is the independent fixed full architecture')
        common = {f.name:getattr(self,f.name) for f in fields(self) if f.name not in ('family','architecture_version','latent_processor')}
        KCDNOArchitectureConfig(**common).validate()
        if any(getattr(self,k) != 2*self.d for k in ('ffn1_hidden','ffn2_hidden','point_hidden')):
            raise ValueError('matched LRSA uses the same hidden2d FFNs')
        return self

    @classmethod
    def from_dict(cls, data):
        if isinstance(data,dict):
            for key in ('ffn1_hidden','ffn2_hidden','point_hidden'):
                if key in data:_positive_int(key,data[key])
        return super(MatchedLRSAConfig,cls).from_dict(data)


@dataclass(frozen=True, slots=True)
class MatchedInitialization(_Record):
    initialization_version: str = 'lrsa-matched-init-v1'
    public_init: str = 'cdlno-front-v1'
    seed: int | None = None
    def validate(self):
        if not all(isinstance(getattr(self,k),str) and getattr(self,k) for k in ('initialization_version','public_init')):
            raise ValueError('initialization provenance strings required')
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):raise ValueError('invalid seed')
        return self


@dataclass(frozen=True, slots=True)
class MatchedMetadata:
    task: str
    architecture: MatchedLRSAConfig
    checkpoint_format: str
    profile: str = 'kcdno_v1'
    runtime: KCDNORuntimeConfig = KCDNORuntimeConfig()
    initialization: MatchedInitialization = MatchedInitialization()

    def validate(self):
        if self.task not in TASKS or self.profile not in PROFILE_NAMES:raise KCDNOMetadataMismatch('unknown task/profile')
        formats = ('whole_model',) if self.task=='car' else ('model_list','whole_model') if self.task=='airfrans' else ('state_dict',)
        if self.checkpoint_format not in formats:raise KCDNOMetadataMismatch('wrong checkpoint format')
        for value,kind in ((self.architecture,MatchedLRSAConfig),(self.runtime,KCDNORuntimeConfig),(self.initialization,MatchedInitialization)):
            if type(value) is not kind:raise KCDNOMetadataMismatch('wrong metadata record type')
            value.validate()
        return self

    def to_dict(self):
        self.validate()
        return dict(schema_version=1,family='lrsa_matched',task=self.task,profile=self.profile,
            checkpoint_format=self.checkpoint_format,architecture=self.architecture.to_dict(),
            initialization=self.initialization.to_dict(),runtime=self.runtime.to_dict())

    @classmethod
    def from_dict(cls,p):
        keys={'schema_version','family','task','profile','checkpoint_format','architecture','initialization','runtime'}
        if type(p) is not dict or p.keys()!=keys or type(p['schema_version']) is not int or p['schema_version']!=1 or p['family']!='lrsa_matched':
            raise KCDNOMetadataMismatch('invalid matched LRSA metadata')
        return cls(p['task'],MatchedLRSAConfig.from_dict(p['architecture']),p['checkpoint_format'],p['profile'],
            KCDNORuntimeConfig.from_dict(p['runtime']),MatchedInitialization.from_dict(p['initialization'])).validate()
