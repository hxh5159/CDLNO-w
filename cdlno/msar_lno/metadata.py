"""MSAR JSON sidecars only; no weights, pickle loading, model construction or resume."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from .config import FAMILY, MSARArchitectureConfig, MSARTrainingConfig, MSARRuntimeConfig
from .options import architecture_overrides, training_overrides
from .profiles import PROFILE_NAMES, TASKS, ResolvedMSARConfig
from .registry import checkpoint_family

SCHEMA_VERSION = 1


class MSARMetadataMismatch(ValueError):
    """Saved structure/family/task cannot supply the requested new model."""


def compare_architecture(expected: MSARArchitectureConfig, actual: MSARArchitectureConfig) -> list[str]:
    left, right = expected.to_dict(), actual.to_dict()
    return sorted(k for k in left if left[k] != right[k])


@dataclass(frozen=True, slots=True)
class MSARMetadata:
    task: str
    architecture: MSARArchitectureConfig
    checkpoint_format: str
    profile: str = 'light'
    training: MSARTrainingConfig = field(default_factory=MSARTrainingConfig)
    runtime: MSARRuntimeConfig = field(default_factory=MSARRuntimeConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MSARMetadata:
        if self.task not in TASKS:raise MSARMetadataMismatch(f'unknown task: {self.task!r}')
        if self.profile not in PROFILE_NAMES:raise MSARMetadataMismatch(f'unknown profile: {self.profile!r}')
        formats = ('whole_model',) if self.task == 'car' else (
            ('model_list', 'whole_model') if self.task == 'airfrans' else ('state_dict',))
        if self.checkpoint_format not in formats:
            raise MSARMetadataMismatch(f'{self.task} retains checkpoint format {formats}')
        for name, cls in (('architecture', MSARArchitectureConfig), ('training', MSARTrainingConfig),
                          ('runtime', MSARRuntimeConfig)):
            value = getattr(self, name)
            if type(value) is not cls:raise MSARMetadataMismatch(f'{name} must be {cls.__name__}')
            value.validate()
        return self

    def to_dict(self) -> dict:
        self.validate()
        return dict(schema_version=SCHEMA_VERSION, family=FAMILY, task=self.task, profile=self.profile,
                    checkpoint_format=self.checkpoint_format, architecture=self.architecture.to_dict(),
                    training=self.training.to_dict(), runtime=self.runtime.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> MSARMetadata:
        if checkpoint_family(data) != FAMILY:
            raise MSARMetadataMismatch('expected explicit family=msar_lno; missing family follows legacy routing')
        required = {'schema_version', 'family', 'task', 'profile', 'checkpoint_format',
                    'architecture', 'training', 'runtime'}
        if data.keys() != required:
            raise MSARMetadataMismatch(f'metadata fields: missing={sorted(required-data.keys())}, unknown={sorted(data.keys()-required)}')
        if type(data['schema_version']) is not int or data['schema_version'] != SCHEMA_VERSION:
            raise MSARMetadataMismatch('unsupported MSAR metadata schema_version')
        return cls(task=data['task'], profile=data['profile'], checkpoint_format=data['checkpoint_format'],
                   architecture=MSARArchitectureConfig.from_dict(data['architecture']),
                   training=MSARTrainingConfig.from_dict(data['training']),
                   runtime=MSARRuntimeConfig.from_dict(data['runtime']))


def save_metadata(path: str | Path, metadata: MSARMetadata) -> Path:
    text = json.dumps(metadata.to_dict(), sort_keys=True, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:stream.write(text)
    return path


def load_metadata(path: str | Path) -> MSARMetadata:
    return MSARMetadata.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True, slots=True)
class EvaluationResolution:
    saved: MSARMetadata
    requested_training: MSARTrainingConfig
    training_differences: tuple[str, ...]
    runtime: MSARRuntimeConfig
    runtime_differences: tuple[str, ...]
    requested_profile: str | None


def resolve_evaluation(path: str | Path, explicit: dict, *, task: str,
                       runtime: MSARRuntimeConfig | None = None) -> EvaluationResolution:
    """Load first; rebuild SAVED resolved structure, then verify explicit overrides.

    Profile is a provenance label, not replacement weights/shape. Valid coverage
    overrides are recorded as requests but never overwrite saved training settings.
    This pure helper computes no auxiliary loss and cannot initialize model weights.
    """
    saved = load_metadata(path)
    if task != saved.task:raise MSARMetadataMismatch(f'task mismatch: saved={saved.task}, requested={task}')
    requested_profile = explicit.get('profile')
    if requested_profile is not None and requested_profile not in PROFILE_NAMES:
        raise MSARMetadataMismatch(f'unknown requested profile: {requested_profile!r}')
    wanted = MSARArchitectureConfig.from_dict(saved.architecture.to_dict() | architecture_overrides(explicit))
    differences = compare_architecture(saved.architecture, wanted)
    if differences:
        raise MSARMetadataMismatch('MSAR architecture mismatch: ' + '; '.join(
            f'{k}: saved={getattr(saved.architecture,k)!r}, requested={getattr(wanted,k)!r}' for k in differences))
    training = MSARTrainingConfig.from_dict(saved.training.to_dict() | training_overrides(explicit))
    actual_runtime = saved.runtime if runtime is None else runtime
    if type(actual_runtime) is not MSARRuntimeConfig:raise MSARMetadataMismatch('MSARRuntimeConfig required')
    actual_runtime.validate()
    return EvaluationResolution(saved, training,
        tuple(k for k,v in saved.training.to_dict().items() if training.to_dict()[k] != v),
        actual_runtime, tuple(k for k,v in saved.runtime.to_dict().items() if actual_runtime.to_dict()[k] != v),
        requested_profile)


def new_run_path(output_root: str | Path, resolved: ResolvedMSARConfig, *, save_name: str | None = None) -> Path:
    """Pure path proposal: output/task/msar_lno/profile/coverage_floor|off/unique_name.

    Explicit save_name is preserved as the leaf, still under the isolated model
    hierarchy. This helper never changes existing entry output behavior.
    """
    resolved.__post_init__()
    if save_name is None:
        save_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:12]
    if (type(save_name) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]+', save_name)
            or save_name in ('.', '..')):
        raise ValueError('save_name must be a filename stem, not a path')
    return (Path(output_root) / resolved.task / FAMILY / resolved.profile /
            ('coverage_' + resolved.training.effective_coverage_mode) / save_name)


def reserve_run_directory(path: str | Path) -> Path:
    """Same exclusive reservation rule as existing experiments; never resume/overwrite."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
