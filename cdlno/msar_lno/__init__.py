"""MSAR-LNO configuration foundation. No core/model placeholder exists in M1."""
from .config import FAMILY, MODEL_NAME, MSARArchitectureConfig, MSARTrainingConfig, MSARRuntimeConfig, input_layout
from .profiles import PROFILE_NAMES, TASKS, ResolvedMSARConfig, resolve_profile
from .registry import family_for_model_key, checkpoint_family

__all__ = ['FAMILY', 'MODEL_NAME', 'MSARArchitectureConfig', 'MSARTrainingConfig', 'MSARRuntimeConfig',
           'PROFILE_NAMES', 'TASKS', 'ResolvedMSARConfig', 'resolve_profile', 'input_layout',
           'family_for_model_key', 'checkpoint_family']
