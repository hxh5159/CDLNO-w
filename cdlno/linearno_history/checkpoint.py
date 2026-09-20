"""Internal metadata-first strict research checkpoints; baseline delegates unchanged.

Only native state dictionaries (weights_only=True), never external pickle. Model,
protocol, numeric normalizers and all epoch-boundary resume state stay separate.
No benchmark is wired to this module in R5.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import torch

from cdlno.linearno import checkpoint as legacy
from cdlno.linearno import schema as base_schema
from cdlno.linearno.profiles import digest
from cdlno.training_state import _atomic, _same, _model_state_check
from linearno_history.schema import (
    ARCHITECTURE_EXTENSION, HistorySchemaError, validate_strict_load_policy,
    validate_legacy_metadata,
)
from .config import (resolve_config, validate_config, innovation_spec, baseline_model_spec,
                     structural_differences)
from .factory import build_model, class_for, validate_constructor

FORMAT = 'linearno-history-epoch-pair-v1'
EXTRAS = {'architecture_extension', 'feature_signature', 'innovation_spec', 'resolved_config'}


def source_hash():
    """Canonical source inventory, independent of cwd and absolute root path."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/'cdlno/linearno_history').glob('*.py'))
    paths += sorted((root/'linearno_history').glob('*.py'))
    paths += [root/'PDE-Solving-StandardBenchmark/model/LinearNO_History.py']
    from cdlno.linearno_loop.provenance import legacy_source
    import hashlib
    return digest({str(p.relative_to(root)): hashlib.sha256(legacy_source(str(p.relative_to(root)),p.read_text()).encode()).hexdigest() for p in paths})


def rehash(metadata):
    metadata['metadata_hash'] = digest({k: v for k, v in metadata.items() if k != 'metadata_hash'})
    return metadata


def baseline_view(metadata):
    """Validate unchanged protocol/numeric-state schema without weakening it.

    This ephemeral validation view is never saved or exposed as a baseline
    checkpoint; the actual research model/spec/family are validated separately.
    """
    view = {k: deepcopy(v) for k, v in metadata.items() if k not in EXTRAS}
    view['family'] = 'linearno'
    view['model_spec'] = baseline_model_spec(view['profile_spec'])
    return rehash(view)


def make_metadata(config, **sections):
    c = validate_config(config)
    pure_spec = baseline_model_spec(c['profile_spec'])
    pure = class_for(pure_spec['class_path'])
    # These axes come directly from the existing profile, without reinterpreting
    # industrial MSE/metrics or any data, optimizer or scheduler protocol.
    sections = deepcopy(sections)
    sections.setdefault('objective_spec', c['profile_spec']['values']['objective'])
    sections.setdefault('evaluation_spec', c['profile_spec']['values']['evaluation'])
    if c['family'] == 'linearno_history':
        sections['provenance_spec']['research_source_sha256'] = source_hash()
    base = base_schema.make_metadata(profile_spec=c['profile_spec'], model_spec=pure_spec,
                                     constructor=pure, **sections)
    if c['family'] == 'linearno':
        return base
    base.update(family='linearno_history', architecture_extension=ARCHITECTURE_EXTENSION,
                feature_signature=c['features']['feature_signature'], innovation_spec=innovation_spec(c),
                model_spec=c['model_spec'], resolved_config=c)
    return validate_metadata(rehash(base))


def config_from_metadata(metadata):
    if metadata.get('family') == 'linearno_history':
        return validate_config(metadata['resolved_config'])
    validate_legacy_metadata(metadata)
    return resolve_config(metadata['profile_spec'])


def validate_metadata(metadata, *, expected=None, resume=False):
    if not isinstance(metadata, dict):
        raise HistorySchemaError('checkpoint metadata must be an object')
    family = metadata.get('family')
    if family == 'linearno':
        validate_legacy_metadata(metadata)
        if EXTRAS & metadata.keys():
            raise HistorySchemaError('A0K0 must keep the original metadata schema')
        c = config_from_metadata(metadata)
        pure = validate_constructor(c)
        base_schema.validate_metadata(metadata, constructor=pure)
        if metadata['model_spec'] != c['model_spec']:
            raise HistorySchemaError('baseline model_spec/profile structural mismatch')
    elif family == 'linearno_history':
        required = {'family', 'schema_version', 'config_version', 'config_hash', 'metadata_hash',
                    *base_schema.SECTIONS, *EXTRAS}
        if set(metadata) != required:
            raise HistorySchemaError(f'research metadata missing={sorted(required-set(metadata))}, '
                                     f'unknown={sorted(set(metadata)-required)}')
        if metadata['metadata_hash'] != digest({k: v for k, v in metadata.items() if k != 'metadata_hash'}):
            raise HistorySchemaError('research metadata checksum mismatch')
        c = config_from_metadata(metadata)
        if c['family'] != family:
            raise HistorySchemaError('research metadata requires an enabled innovation')
        validate_constructor(c)
        expected_parts = dict(model_spec=c['model_spec'], resolved_config=c,
            profile_spec=c['profile_spec'], innovation_spec=innovation_spec(c),
            feature_signature=c['features']['feature_signature'], architecture_extension=ARCHITECTURE_EXTENSION)
        for key, value in expected_parts.items():
            if metadata[key] != value:
                raise HistorySchemaError(f'research metadata structural mismatch at {key}')
        base_schema._hash(metadata['provenance_spec'].get('research_source_sha256'), 'research source hash')
        view = baseline_view(metadata)
        base_schema.validate_metadata(view, constructor=class_for(view['model_spec']['class_path']))
    else:
        raise HistorySchemaError('unknown checkpoint family; never infer research from weights')
    if expected is not None:
        target = validate_config(expected)
        differences = structural_differences(c, target)
        if differences:
            raise HistorySchemaError('structural checkpoint mismatch before weight load:\n'+'\n'.join(differences))
        if resume:
            for key in ('training', 'objective', 'evaluation'):
                if c['profile_spec']['values'][key] != target['profile_spec']['values'][key]:
                    raise HistorySchemaError(f'resume {key} protocol mismatch')
    return deepcopy(metadata)


def runtime_changes(metadata, expected):
    if expected is None:
        return {}
    saved = config_from_metadata(metadata)
    # Structural differences are already rejected. Protocol changes never alter
    # the saved metadata used for eval; caller sees them explicitly.
    changes = {}
    for key in ('training', 'runtime', 'objective', 'evaluation'):
        a, b = saved['profile_spec']['values'][key], expected['profile_spec']['values'][key]
        if a != b:
            changes[key] = dict(saved=a, requested=b)
    return changes


def read_json(path):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise HistorySchemaError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)


def inspect_checkpoint(directory, selector='final', *, expected=None, resume=False):
    """Read/validate JSON and checksums before constructing or reading tensors."""
    directory = Path(directory).resolve()
    if selector in ('latest', 'final'):
        pointer = read_json(directory/'checkpoints'/f'{selector}.json')
        name = pointer['manifest']
        if Path(name).name != name or legacy.sha256(directory/'checkpoints'/name) != pointer['sha256']:
            raise HistorySchemaError('checkpoint pointer checksum/path mismatch')
    elif re.fullmatch(r'epoch_[0-9]{4,}', selector or ''):
        name = selector + '.json'
    else:
        raise HistorySchemaError('checkpoint selector requires final, latest or epoch_XXXX')
    manifest_path = directory/'checkpoints'/name
    manifest = read_json(manifest_path)
    if manifest.get('format') not in (FORMAT, 'linearno-epoch-pair-v1'):
        raise HistorySchemaError('unknown checkpoint family/format')
    for key, subdir, suffix in (('checkpoint', 'checkpoints', '.pt'), ('weights', 'weights', '.pt'),
                                ('metadata', 'checkpoints', '.metadata.json')):
        path = (directory/manifest[key]['path']).resolve()
        if path.parent != directory/subdir or path.name != manifest_path.stem + suffix:
            raise HistorySchemaError('checkpoint pair path mismatch')
        if legacy.sha256(path) != manifest[key]['sha256']:
            raise HistorySchemaError(f'checkpoint {key} checksum mismatch')
    metadata = validate_metadata(read_json(directory/manifest['metadata']['path']), expected=expected, resume=resume)
    required_format = FORMAT if metadata['family'] == 'linearno_history' else 'linearno-epoch-pair-v1'
    if manifest['format'] != required_format:
        raise HistorySchemaError('manifest/metadata family mismatch')
    if metadata['resume_state']['epoch'] != manifest['epoch']:
        raise HistorySchemaError('manifest/metadata epoch mismatch')
    if selector == 'final' and metadata['resume_state']['checkpoint_role'] != 'final':
        raise HistorySchemaError('final selector requires a final checkpoint')
    if metadata['family'] == 'linearno':
        # The old native loader is authoritative for all A0K0 checkpoints.
        return legacy.inspect_checkpoint(directory, selector, class_for(metadata['model_spec']['class_path']))
    return metadata, manifest_path


def _read_pair(path, metadata):
    if metadata['family'] == 'linearno':
        return legacy.read_pair(path, class_for(metadata['model_spec']['class_path']))[1]
    manifest, directory = read_json(path), path.parent.parent
    payload = torch.load(directory/manifest['checkpoint']['path'], map_location='cpu', weights_only=True)
    state = torch.load(directory/manifest['weights']['path'], map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'model'} or payload['metadata'] != metadata:
        raise HistorySchemaError('checkpoint payload/metadata mismatch')
    if not _same(payload['model'], state):
        raise HistorySchemaError('checkpoint/weights pair mismatch')
    return state


def save_checkpoint(directory, model, metadata):
    metadata = validate_metadata(metadata)
    c = config_from_metadata(metadata)
    if type(model) is not validate_constructor(c):
        raise HistorySchemaError('model class and checkpoint metadata disagree')
    if c['family'] == 'linearno':
        return legacy.save_pair(directory, model, metadata, type(model))
    if getattr(model, '_history_model_spec', None) != c['model_spec']:
        raise HistorySchemaError('research model constructor and metadata disagree')
    # Verify all parameter keys/shapes before committing a metadata/weight pair.
    with torch.random.fork_rng(devices=[]):
        template = build_model(c)
    _model_state_check(model.state_dict(), template)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ckdir, wdir = directory/'checkpoints', directory/'weights'
    ckdir.mkdir(exist_ok=True); wdir.mkdir(exist_ok=True)
    epoch = metadata['resume_state']['epoch']
    if epoch < 1:
        raise HistorySchemaError('only completed epochs can be committed')
    name = f'epoch_{epoch:04d}'
    manifest_path = ckdir/(name+'.json')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    import fcntl
    with (directory/'.checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            old, path = inspect_checkpoint(directory, name, expected=c)
            if metadata != old or not _same(state, _read_pair(path, old)):
                raise HistorySchemaError('refusing to overwrite committed epoch')
            return manifest_path
        committed = [int(p.stem.split('_')[1]) for p in ckdir.glob('epoch_*.json')
                     if '.metadata.' not in p.name]
        if committed and epoch <= max(committed):
            raise HistorySchemaError('refusing to roll back committed progress')
        paths = dict(checkpoint=ckdir/(name+'.pt'), weights=wdir/(name+'.pt'), metadata=ckdir/(name+'.metadata.json'))
        for path in paths.values():
            if path.exists():
                raise HistorySchemaError(f'preserve uncommitted file for inspection: {path}')
        _atomic(paths['checkpoint'], dict(model=state, metadata=metadata))
        _atomic(paths['weights'], state)
        _atomic(paths['metadata'], metadata, json_file=True)
        record = dict(format=FORMAT, epoch=epoch,
                      **{k: dict(path=str(p.relative_to(directory)), sha256=legacy.sha256(p)) for k, p in paths.items()})
        _atomic(manifest_path, record, json_file=True)
        pointer = dict(manifest=manifest_path.name, sha256=legacy.sha256(manifest_path))
        _atomic(ckdir/'latest.json', pointer, json_file=True)
        if metadata['resume_state']['checkpoint_role'] == 'final':
            _atomic(ckdir/'final.json', pointer, json_file=True)
        _atomic(directory/'model.pt', state)
    return manifest_path


def load_checkpoint(directory, selector='final', *, expected=None, strict=True):
    validate_strict_load_policy(strict)
    metadata, path = inspect_checkpoint(directory, selector, expected=expected)
    # Constructor allowed only after complete metadata validation/comparison.
    model = build_model(config_from_metadata(metadata))
    state = _read_pair(path, metadata)
    legacy.strict_load(model, state)  # checks every key/shape before strict=True
    model.eval()
    return model, metadata, runtime_changes(metadata, expected)


def resume_checkpoint(directory, *, optimizer_factory, scheduler_factory,
                      generators, samplers=None, expected=None, strict=True):
    validate_strict_load_policy(strict)
    metadata, _ = inspect_checkpoint(directory, 'latest', expected=expected, resume=True)
    state = metadata['resume_state']
    samplers = {} if samplers is None else samplers
    saved_samplers = base_schema.unpack_state(state['sampler_state'])
    saved_generators = base_schema.unpack_state(state['dataloader_generators'])
    if set(saved_samplers) != set(samplers) or set(saved_generators) != set(generators):
        raise HistorySchemaError('resume requires exactly the recorded sampler/generator names')
    model, metadata, changes = load_checkpoint(directory, 'latest', expected=expected, strict=True)
    optimizer = optimizer_factory(model.parameters())
    scheduler = scheduler_factory(optimizer)
    optimizer.load_state_dict(base_schema.unpack_state(state['optimizer']))
    scheduler.load_state_dict(base_schema.unpack_state(state['scheduler']))
    for name, sampler in samplers.items():
        sampler.load_state_dict(saved_samplers[name])
    model.train()
    # Restore last: construction, checkpoint IO and callbacks cannot consume the
    # next training draws (notably A's sample-by-history dropout).
    legacy.restore_random_state(state, generators)
    return model, optimizer, scheduler, metadata, changes
