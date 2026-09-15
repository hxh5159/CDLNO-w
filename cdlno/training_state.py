"""Epoch-boundary training archives; independent of model math and task loaders.

Only plain containers/tensors are serialized. Each archive has a per-epoch
commit manifest; latest/final pointers are published after BOTH checkpoint and
weights are durable. A bare old state_dict is never treated as resumable.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch

from .checkpoint import compare_architecture, load_sidecar


VERSION = 1
TASKS = ('darcy', 'elasticity', 'airfoil', 'pipe', 'ns', 'plasticity', 'car', 'airfrans')


class ResumeError(ValueError):
    """Incomplete archive or a mismatch in the requested continuation."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ResumeError(f'{name} must be an integer >= {minimum}')


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _cpu(value):
    """Owned snapshots: no live parameter, optimizer, or NumPy storage aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return _cpu(value.item())
    if isinstance(value, dict):
        out = OrderedDict() if isinstance(value, OrderedDict) else {}
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ResumeError('state mapping keys must be str/int')
            out[key] = _cpu(item)
        if hasattr(value, '_metadata'):
            out._metadata = _cpu(value._metadata)
        return out
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ResumeError('nonfinite scalar in training state')
        return value
    raise ResumeError(f'unsupported state type: {type(value).__name__}')


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def ordered_fingerprint(items):
    """Order-sensitive digest of ALREADY loaded identifiers/arrays (no data IO)."""
    digest = hashlib.sha256()
    def visit(value):
        value = _cpu(value)
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ResumeError('fingerprint requires dense tensors')
            header = json.dumps([str(value.dtype), list(value.shape)]).encode()
            data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(b'tensor' + str(len(header)).encode() + b':' + header)
            digest.update(str(len(data)).encode() + b':' + data)
        elif isinstance(value, dict):
            digest.update(b'dict{')
            for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(value[key])
            digest.update(b'}')
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + b'[')
            for item in value:
                visit(item)
            digest.update(b']')
        else:
            encoded = json.dumps(value, allow_nan=False).encode()
            digest.update(type(value).__name__.encode() + str(len(encoded)).encode() + b':' + encoded)
    for item in items:
        visit(item)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingProtocol:
    """Explicit task facts, supplied by future entry adapters, never inferred.

    Counts are per completed epoch. This handles Plasticity's 20 updates per
    batch without changing its scheduler rate. training/data/adapter must be
    JSON mappings; normalizer tensors live separately in the archive.
    """
    task: str
    total_epochs: int
    updates_per_epoch: int
    scheduler_steps_per_epoch: int
    training: dict
    data: dict
    adapter: dict

    def __post_init__(self):
        if self.task not in TASKS:
            raise ResumeError(f'unsupported task: {self.task}')
        for name in ('total_epochs', 'updates_per_epoch', 'scheduler_steps_per_epoch'):
            _integer(getattr(self, name), name, 1)
        for name in ('training', 'data', 'adapter'):
            if not isinstance(getattr(self, name), dict):
                raise ResumeError(f'{name} must be an explicit mapping')
            object.__setattr__(self, name, _json_copy(getattr(self, name)))
        if not {'optimizer', 'scheduler', 'loss', 'batch_size'} <= self.training.keys():
            raise ResumeError('training requires optimizer/scheduler/loss/batch_size')
        _integer(self.training['batch_size'], 'batch_size', 1)
        if not {'split', 'ordered_fingerprint'} <= self.data.keys():
            raise ResumeError('data requires split and ordered_fingerprint')
        fingerprint = self.data['ordered_fingerprint']
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ResumeError('data.ordered_fingerprint must be a SHA256 digest')

    def to_dict(self):
        return _json_copy(dict(version=VERSION, task=self.task, total_epochs=self.total_epochs,
            updates_per_epoch=self.updates_per_epoch, scheduler_steps_per_epoch=self.scheduler_steps_per_epoch,
            training=self.training, data=self.data, adapter=self.adapter))


def capture_rng(generators: Mapping[str, torch.Generator] | None = None):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(algorithm=numpy_state[0],
        keys=torch.tensor(numpy_state[1].astype(np.int64)), position=int(numpy_state[2]),
        has_gauss=int(numpy_state[3]), cached_gaussian=float(numpy_state[4])),
        cpu=torch.get_rng_state().clone(),
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        generators={k: dict(device=str(g.device), state=g.get_state().clone()) for k, g in (generators or {}).items()})


def _validate_rng(state, generators):
    if not isinstance(state, dict) or set(state) != {'python', 'numpy', 'cpu', 'cuda', 'generators'}:
        raise ResumeError('missing/incompatible RNG state')
    try:
        random.Random().setstate(state['python'])
        n = state['numpy']
        np.random.RandomState().set_state((n['algorithm'], n['keys'].numpy().astype(np.uint32),
                                           n['position'], n['has_gauss'], n['cached_gaussian']))
        torch.Generator().set_state(state['cpu'])
        expected = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(state['cuda']) != expected:
            raise ResumeError('CUDA RNG device count differs; exact continuation requires the same visible devices')
        for i, value in enumerate(state['cuda']):
            torch.Generator(device=f'cuda:{i}').set_state(value)
        if state['generators'].keys() != generators.keys():
            raise ResumeError('DataLoader/Sampler generator names differ')
        for name, g in generators.items():
            saved = state['generators'][name]
            if str(g.device) != saved['device']:
                raise ResumeError(f'generator device differs: {name}')
            torch.Generator(device=g.device).set_state(saved['state'])
    except ResumeError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as e:
        raise ResumeError(f'invalid RNG state: {e}') from e


def restore_rng(state, generators: Mapping[str, torch.Generator] | None = None):
    generators = dict(generators or {})
    _validate_rng(state, generators)
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['algorithm'], n['keys'].numpy().astype(np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    for name, generator in generators.items():
        generator.set_state(state['generators'][name]['state'])


@contextmanager
def isolated_evaluation(model, *, generators=None):
    """Restore RNG, each module's train/eval flag and buffers, also on exceptions.

    Callbacks must only infer/render, never update parameters or call optimizer.
    The context does not modify gradients, parameters or model forward methods.
    """
    rng = capture_rng(generators)
    flags = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in flags:
            module.training = training
        restore_rng(rng, generators)


def _qualified(obj):
    cls = type(obj)
    return cls.__module__ + '.' + cls.__qualname__


def _optimizer_signature(model, optimizer):
    if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
        raise ResumeError('epoch continuation currently supports original Adam/AdamW only')
    names = {id(p): name for name, p in model.named_parameters()}
    seen, groups = set(), []
    for group in optimizer.param_groups:
        keys = []
        for p in group['params']:
            if id(p) not in names or id(p) in seen:
                raise ResumeError('optimizer has foreign or repeated parameters')
            seen.add(id(p)); keys.append(names[id(p)])
        groups.append(dict(names=keys, options=_cpu({k: v for k, v in group.items()
                      if k not in ('params', 'lr', 'betas', 'momentum')})))
    return dict(type=_qualified(optimizer), defaults=_cpu(optimizer.defaults), groups=groups)


def _scheduler_signature(scheduler):
    if type(scheduler) not in (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise ResumeError('epoch continuation supports original OneCycleLR/CosineAnnealingLR only')
    mutable = {'last_epoch', '_step_count', '_last_lr', '_get_lr_called_within_step', '_is_initial'}
    return dict(type=_qualified(scheduler), plan=_cpu({k: v for k, v in scheduler.state_dict().items() if k not in mutable}))


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, value, *, json_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            if json_file:
                f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
            else:
                torch.save(value, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_state_check(saved, model):
    expected = model.state_dict()
    if not isinstance(saved, dict) or saved.keys() != expected.keys():
        raise ResumeError('model state_dict keys differ (strict loading required)')
    for name, value in expected.items():
        other = saved[name]
        if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype:
            raise ResumeError(f'model shape/dtype differs: {name}')
        if other.is_floating_point() and not torch.isfinite(other).all():
            raise ResumeError(f'nonfinite model weight: {name}')


def _environment(model):
    packages = {}
    for name in ('numpy', 'matplotlib', 'torch-geometric'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version.split()[0], torch=str(torch.__version__), cuda=torch.version.cuda,
        packages=packages,
        devices=sorted({str(p.device) for p in model.parameters()}),
        dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        sdpa_enabled=dict(math=torch.backends.cuda.math_sdp_enabled(), flash=torch.backends.cuda.flash_sdp_enabled(),
                          memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(), cudnn=torch.backends.cuda.cudnn_sdp_enabled()),
        source=_source_provenance(model))


def _source_provenance(model):
    # Hash the actual shared code and wrapper modules, including dirty edits.
    # Task adapters additionally supply their loader/entry provenance in extra.
    root = Path(__file__).resolve().parents[1]
    paths = set(Path(__file__).resolve().parent.glob('*.py'))
    for module in model.modules():
        name = type(module).__module__
        filename = getattr(sys.modules.get(name), '__file__', None)
        if filename and not name.startswith('torch.') and Path(filename).is_file():
            paths.add(Path(filename).resolve())
    hashes = {str(p.relative_to(root) if p.is_relative_to(root) else p): _file_sha(p) for p in sorted(paths)}
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5, check=False)
        head = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        head = None
    return dict(git_head=head, shared_and_model_source_sha256=hashes)


class TrainingArchive:
    """An existing run/sidecar, with immutable epoch pairs and explicit restore.

    No directory is created by construction/read/restore. Future task adapters
    reserve fresh runs separately. One writer per run is required; a local
    exclusive lock rejects concurrent writes rather than overwriting archives.
    """
    def __init__(self, directory, protocol: TrainingProtocol, *, sidecar=None):
        self.directory = Path(directory).resolve()
        self.protocol = protocol
        self.sidecar = Path(sidecar).resolve() if sidecar else self.directory / 'architecture.json'
        load_sidecar(self.sidecar)

    def _check_architecture(self, model):
        saved = load_sidecar(self.sidecar)
        if compare_architecture(model.config, saved['architecture']):
            raise ResumeError('model architecture differs from existing sidecar')
        if saved.get('metadata', {}).get('wrapper_architecture', {}) != self.protocol.adapter:
            raise ResumeError('wrapper architecture differs from protocol/sidecar')
        if hasattr(model, 'adapter_architecture') and model.adapter_architecture() != self.protocol.adapter:
            raise ResumeError('actual model adapter differs from protocol')
        if hasattr(model, 'core'):
            from .checkpoint import validate_model_front_mode
            validate_model_front_mode(model, model.config)
        return saved

    def save(self, model, optimizer, scheduler, *, completed_epochs, global_step,
             scheduler_steps, normalizers, generators=None, scaler=None, history=None, extra=None):
        """Snapshot AFTER all original work in an epoch; no gradients are saved."""
        self._check_architecture(model)
        _integer(completed_epochs, 'completed_epochs', 1)
        if completed_epochs > self.protocol.total_epochs:
            raise ResumeError('completed_epochs exceeds original total_epochs')
        _integer(global_step, 'global_step', 1); _integer(scheduler_steps, 'scheduler_steps', 1)
        if global_step != completed_epochs * self.protocol.updates_per_epoch:
            raise ResumeError('global_step differs from completed epochs/protocol')
        if scheduler_steps != completed_epochs * self.protocol.scheduler_steps_per_epoch:
            raise ResumeError('scheduler_steps differs from completed epochs/protocol')
        if scheduler.optimizer is not optimizer or scheduler.last_epoch != scheduler_steps:
            raise ResumeError('actual scheduler progress/optimizer differs')
        if not isinstance(normalizers, dict):
            raise ResumeError('normalizers must be explicit state mappings, including {} if none')
        state = _cpu(model.state_dict())
        _model_state_check(state, model)
        payload = dict(format='cdlno-training', version=VERSION, protocol=self.protocol.to_dict(),
            architecture=model.config.to_dict(), sidecar_sha256=_file_sha(self.sidecar),
            model=state, optimizer=_cpu(optimizer.state_dict()), optimizer_signature=_optimizer_signature(model, optimizer),
            scheduler=_cpu(scheduler.state_dict()), scheduler_signature=_scheduler_signature(scheduler),
            progress=dict(completed_epochs=completed_epochs, next_epoch=completed_epochs, global_step=global_step,
                          scheduler_steps=scheduler_steps), normalizers=_cpu(normalizers), rng=capture_rng(generators),
            scaler=None if scaler is None else dict(type=_qualified(scaler), state=_cpu(scaler.state_dict())),
            history=_cpu(history if history is not None else {}), extra=_cpu(extra if extra is not None else {}),
            environment=_environment(model))
        self._validate(payload, model, optimizer, scheduler, normalizers, generators, scaler)
        ckdir = self.directory / 'checkpoints'; wdir = self.directory / 'weights'
        ckdir.mkdir(exist_ok=True); wdir.mkdir(exist_ok=True)
        lock = ckdir / '.writer.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as e:
            raise ResumeError('checkpoint writer lock exists; inspect interrupted/concurrent writer') from e
        os.close(fd)
        try:
            return self._publish(payload, completed_epochs)
        finally:
            lock.unlink()

    def _publish(self, payload, epoch):
        ckdir = self.directory / 'checkpoints'
        cp = ckdir / f'epoch_{epoch:04d}.pt'
        wp = self.directory / 'weights' / cp.name
        manifest = cp.with_suffix('.json')
        latest = ckdir / 'latest.json'
        protocol_file = self.directory / 'training_protocol.json'
        if protocol_file.exists():
            if json.loads(protocol_file.read_text()) != self.protocol.to_dict():
                raise ResumeError('existing training_protocol.json differs')
        else:
            _atomic(protocol_file, self.protocol.to_dict(), json_file=True)
        if self._newest_committed_epoch() > epoch:
            raise ResumeError('refusing rollback over a later checkpoint')
        if manifest.exists():
            old = self.read(manifest)
            if not _same(old, payload):
                raise ResumeError('epoch already committed with different state')
        else:
            if cp.exists() or wp.exists():
                raise ResumeError('uncommitted epoch files exist; inspect incomplete archive')
            try:
                _atomic(cp, payload)
                _atomic(wp, payload['model'])
                record = dict(format='cdlno-epoch-pair', version=VERSION, completed_epochs=epoch,
                    checkpoint=dict(path=str(cp.relative_to(self.directory)), sha256=_file_sha(cp)),
                    weights=dict(path=str(wp.relative_to(self.directory)), sha256=_file_sha(wp)))
                _atomic(manifest, record, json_file=True)  # the pair's commit point
            except BaseException:
                # Do not remove a committed pair if fsync/index publication fails.
                if not manifest.exists():
                    cp.unlink(missing_ok=True); wp.unlink(missing_ok=True)
                raise
        pointer = dict(format='cdlno-archive-pointer', version=VERSION,
                       manifest=manifest.name, sha256=_file_sha(manifest))
        _atomic(latest, pointer, json_file=True)
        if epoch == self.protocol.total_epochs:
            _atomic(ckdir / 'final.json', pointer, json_file=True)
        return cp

    def _newest_committed_epoch(self):
        # The pair manifest commits BEFORE latest.json. A crash in between
        # must not make an older epoch eligible to overwrite this continuation.
        ckdir = self.directory / 'checkpoints'
        epochs = []
        for path in ckdir.glob('epoch_*.json'):
            try:
                epochs.append((int(path.stem.removeprefix('epoch_')), path))
            except ValueError as e:
                raise ResumeError(f'invalid epoch manifest filename: {path.name}') from e
        newest = self._manifest(max(epochs)[1])['completed_epochs'] if epochs else 0
        latest = ckdir / 'latest.json'
        if latest.exists():
            newest = max(newest, self._manifest(latest)['completed_epochs'])
        return newest

    def _manifest(self, path):
        try:
            return self._read_manifest(path)
        except ResumeError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError, OSError) as e:
            raise ResumeError(f'invalid/incomplete archive manifest: {e}') from e

    def _read_manifest(self, path):
        path = Path(path).resolve()
        if path.suffix == '.pt':
            if path.parent not in (self.directory / 'checkpoints', self.directory / 'weights'):
                raise ResumeError('checkpoint/weights must belong to this run')
            # An independent weights path resolves to its verified training pair.
            path = self.directory / 'checkpoints' / path.with_suffix('.json').name
        if not path.is_relative_to(self.directory / 'checkpoints'):
            raise ResumeError('archive must belong to this run')
        if not path.exists():
            raise ResumeError('no committed training archive; bare weights/legacy objects cannot resume')
        value = json.loads(path.read_text())
        if value.get('format') == 'cdlno-archive-pointer':
            if value.get('version') != VERSION:
                raise ResumeError('unsupported archive pointer version')
            target = path.parent / value['manifest']
            if target.resolve().parent != path.parent or _file_sha(target) != value['sha256']:
                raise ResumeError('archive pointer checksum/path mismatch')
            path = target; value = json.loads(path.read_text())
        if value.get('format') != 'cdlno-epoch-pair' or value.get('version') != VERSION:
            raise ResumeError('unsupported/incomplete epoch manifest')
        _integer(value['completed_epochs'], 'manifest completed_epochs', 1)
        if path.name != f"epoch_{value['completed_epochs']:04d}.json":
            raise ResumeError('manifest filename/epoch mismatch')
        for key in ('checkpoint', 'weights'):
            target = (self.directory / value[key]['path']).resolve()
            expected_dir = self.directory / ('checkpoints' if key == 'checkpoint' else 'weights')
            if target.parent != expected_dir or target.name != path.with_suffix('.pt').name:
                raise ResumeError('archive pair path mismatch')
            if not target.exists() or _file_sha(target) != value[key]['sha256']:
                raise ResumeError(f'{key} file missing or checksum mismatch')
        return value

    def read(self, path):
        """Safe CPU read, no model/RNG mutation, no sidecar write."""
        manifest = self._manifest(path)
        payload = torch.load(self.directory / manifest['checkpoint']['path'], map_location='cpu', weights_only=True)
        required = {'format', 'version', 'protocol', 'architecture', 'sidecar_sha256', 'model', 'optimizer',
                    'optimizer_signature', 'scheduler', 'scheduler_signature', 'progress', 'normalizers',
                    'rng', 'scaler', 'history', 'extra', 'environment'}
        if not isinstance(payload, dict) or set(payload) != required or payload['format'] != 'cdlno-training' or payload['version'] != VERSION:
            raise ResumeError('incomplete/unsupported training checkpoint; weights alone cannot resume')
        if not isinstance(payload['progress'], dict) or payload['progress'].get('completed_epochs') != manifest['completed_epochs']:
            raise ResumeError('manifest epoch disagrees with checkpoint')
        weights = torch.load(self.directory / manifest['weights']['path'], map_location='cpu', weights_only=True)
        if not _same(weights, payload['model']):
            raise ResumeError('independent weights differ from checkpoint')
        return payload

    def _validate(self, saved, model, optimizer, scheduler, normalizers, generators, scaler):
        try:
            self._validate_state(saved, model, optimizer, scheduler, normalizers, generators, scaler)
        except ResumeError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as e:
            raise ResumeError(f'invalid/incomplete training state: {e}') from e

    def _validate_state(self, saved, model, optimizer, scheduler, normalizers, generators, scaler):
        self._check_architecture(model)
        if saved['sidecar_sha256'] != _file_sha(self.sidecar):
            raise ResumeError('sidecar bytes differ from saved training run')
        if saved['protocol'] != self.protocol.to_dict():
            raise ResumeError('training/data/adapter protocol mismatch; original total epochs must be retained')
        if compare_architecture(model.config, saved['architecture']):
            raise ResumeError('checkpoint architecture mismatch')
        _model_state_check(saved['model'], model)
        if not _same(saved['normalizers'], _cpu(normalizers)):
            raise ResumeError('normalizer state mismatch')
        if not _same(saved['optimizer_signature'], _optimizer_signature(model, optimizer)):
            raise ResumeError('optimizer type/defaults/parameter group order mismatch')
        if not _same(saved['scheduler_signature'], _scheduler_signature(scheduler)) or scheduler.optimizer is not optimizer:
            raise ResumeError('scheduler type/plan mismatch')
        progress = saved['progress']
        if set(progress) != {'completed_epochs', 'next_epoch', 'global_step', 'scheduler_steps'}:
            raise ResumeError('incomplete progress counters')
        ep = progress['completed_epochs']
        _integer(ep, 'completed_epochs', 1)
        if ep > self.protocol.total_epochs or progress != dict(completed_epochs=ep, next_epoch=ep,
                global_step=ep*self.protocol.updates_per_epoch, scheduler_steps=ep*self.protocol.scheduler_steps_per_epoch):
            raise ResumeError('progress counters disagree with training protocol')
        ss = saved['scheduler']
        if ss.get('last_epoch') != progress['scheduler_steps'] or not {'_step_count', '_last_lr'} <= ss.keys():
            raise ResumeError('scheduler state missing or progress mismatch')
        mutable = {'last_epoch', '_step_count', '_last_lr', '_get_lr_called_within_step', '_is_initial'}
        if not _same({k: v for k, v in ss.items() if k not in mutable}, saved['scheduler_signature']['plan']):
            raise ResumeError('scheduler state differs from saved schedule plan')
        if ss['_step_count'] != progress['scheduler_steps'] + 1:
            raise ResumeError('scheduler step count mismatch')
        if not isinstance(ss['_last_lr'], (list, tuple)) or len(ss['_last_lr']) != len(optimizer.param_groups):
            raise ResumeError('scheduler learning rate group count mismatch')
        optim = saved['optimizer']
        if set(optim) != {'state', 'param_groups'} or len(optim['param_groups']) != len(optimizer.param_groups):
            raise ResumeError('incomplete optimizer state')
        if not optim['state']:
            raise ResumeError('optimizer has no accumulated state')
        known = set()
        for index, (old, group) in enumerate(zip(optim['param_groups'], optimizer.param_groups)):
            static = {k: v for k, v in old.items() if k not in ('params', 'lr', 'betas', 'momentum')}
            if not _same(static, saved['optimizer_signature']['groups'][index]['options']):
                raise ResumeError('optimizer group options differ from saved signature')
            if old.get('lr') != ss['_last_lr'][index]:
                raise ResumeError('optimizer learning rate differs from scheduler state')
            if not isinstance(old['lr'], (int, float)) or not math.isfinite(old['lr']) or old['lr'] < 0:
                raise ResumeError('invalid optimizer learning rate')
            betas = old.get('betas')
            if not isinstance(betas, (tuple, list)) or len(betas) != 2 or not all(isinstance(v, (int, float)) and 0 <= v < 1 for v in betas):
                raise ResumeError('invalid optimizer betas')
            if len(old['params']) != len(group['params']):
                raise ResumeError('optimizer parameter count mismatch')
            for ident, p in zip(old['params'], group['params']):
                if ident in known:
                    raise ResumeError('duplicate optimizer state parameter')
                known.add(ident)
                state = optim['state'].get(ident)
                if state is None:  # Parameters without a gradient may legitimately have no state.
                    continue
                required = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if group.get('amsgrad') else set())
                if set(state) != required:
                    raise ResumeError('missing/unknown Adam moment state')
                for key in required - {'step'}:
                    if (not isinstance(state[key], torch.Tensor) or state[key].shape != p.shape
                            or state[key].dtype != p.dtype or not torch.isfinite(state[key]).all()):
                        raise ResumeError('optimizer moment shape/dtype/nonfinite mismatch')
                step = state['step']
                if (not isinstance(step, torch.Tensor) or step.numel() != 1
                        or not 0 < step.item() <= progress['global_step'] or not float(step.item()).is_integer()):
                    raise ResumeError('invalid optimizer step')
        if optim['state'].keys() - known:
            raise ResumeError('foreign optimizer state')
        if (scaler is None) != (saved['scaler'] is None):
            raise ResumeError('AMP scaler presence mismatch')
        if scaler is not None:
            if type(scaler) is not torch.amp.GradScaler or _qualified(scaler) != saved['scaler']['type']:
                raise ResumeError('AMP scaler type mismatch; only torch.amp.GradScaler is supported')
            state = saved['scaler']['state']
            if state.keys() != scaler.state_dict().keys():
                raise ResumeError('AMP scaler state missing/unknown fields or enabled setting differs')
            if state and (not all(isinstance(state[k], (float, int)) and math.isfinite(state[k]) and state[k] > 0
                                 for k in ('scale', 'growth_factor', 'backoff_factor'))
                          or not state['growth_factor'] > 1 or not state['backoff_factor'] < 1
                          or type(state['growth_interval']) is not int or state['growth_interval'] < 1
                          or type(state['_growth_tracker']) is not int or state['_growth_tracker'] < 0):
                raise ResumeError('invalid AMP scaler state')
        _validate_rng(saved['rng'], dict(generators or {}))

    def restore(self, path, model, optimizer, scheduler, *, normalizers, generators=None, scaler=None):
        """Validate first, strict-load all state, restore RNG LAST before iteration.

        next_epoch is the next ZERO-BASED loop index: completed100 resumes at
        range(100,total_epochs), i.e. the human-readable epoch101.
        """
        saved = self.read(path)
        self._validate(saved, model, optimizer, scheduler, normalizers, generators, scaler)
        if self._newest_committed_epoch() > saved['progress']['completed_epochs']:
            raise ResumeError('refusing continuation from an older checkpoint over a later committed epoch')
        # Preserve original objects/parameter identities; no reinitialization.
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(saved['scaler']['state'])
        optimizer.zero_grad(set_to_none=True)  # completed epoch: no partial accumulation to restore
        restore_rng(saved['rng'], generators)
        return dict(progress=saved['progress'], history=saved['history'], extra=saved['extra'],
                    saved_environment=saved['environment'], current_environment=_environment(model))
