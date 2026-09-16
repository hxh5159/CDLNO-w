"""Four-scale profiles: identical across tasks, no old F/P/M/task defaults."""
from dataclasses import dataclass
from .config import MSARArchitectureConfig, MSARTrainingConfig

PROFILE_NAMES = ('light', 'full')
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')


def profile_values(profile: str = 'light') -> dict:
    if profile not in PROFILE_NAMES:raise ValueError(f'unknown MSAR-LNO profile: {profile!r}')
    return dict(num_latents=[512,256,128,64] if profile == 'light' else [1024,512,256,128],
                d=96 if profile == 'light' else 192, heads=[4,4,8,8],
                encoder_depths=[3,1,1,1], decoder_depths=[3,1,1,1],
                latent_ffn_ratio=2, activation='gelu')


def resolve_profile(profile: str = 'light', **overrides) -> MSARArchitectureConfig:
    return MSARArchitectureConfig(**(profile_values(profile) | overrides))


@dataclass(frozen=True, slots=True)
class ResolvedMSARConfig:
    task: str
    profile: str
    architecture: MSARArchitectureConfig
    training: MSARTrainingConfig

    def __post_init__(self):
        if self.task not in TASKS:raise ValueError(f'unknown MSAR-LNO task: {self.task!r}')
        if self.profile not in PROFILE_NAMES:raise ValueError(f'unknown MSAR-LNO profile: {self.profile!r}')
        if type(self.architecture) is not MSARArchitectureConfig:raise ValueError('MSARArchitectureConfig required')
        if type(self.training) is not MSARTrainingConfig:raise ValueError('MSARTrainingConfig required')
        self.architecture.validate(); self.training.validate()

    def to_dict(self) -> dict:
        defaults = resolve_profile(self.profile).to_dict()
        return dict(task=self.task, profile=self.profile, architecture=self.architecture.to_dict(),
                    training=self.training.to_dict(), effective_coverage_mode=self.training.effective_coverage_mode,
                    profile_overrides=[k for k,v in self.architecture.to_dict().items() if v != defaults[k]])
