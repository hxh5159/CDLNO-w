"""Explicit conversion helpers for audited LinearNO checkpoints.

The converter is intentionally separate from task entry points.  A caller must
name the source task because AirfRANS and ShapeNet release pickles can use the
same qualified class name while containing different classes.  No conversion
uses ``strict=False``: the normalized source key set and every tensor shape are
checked before a target model is loaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from collections import OrderedDict

import torch


TASKS = ("standard", "airfrans", "shapenet")
_EXPECTED_CLASS_NAMES = {
    "standard": ("LinearAttentionNeuralOperator", "Model"),
    "airfrans": ("LinearAttentionNeuralOperator", "AirfRANSLinearNO", "Model"),
    "shapenet": ("LinearAttentionNeuralOperator", "ShapeNetLinearNO", "Model"),
}


class ConversionError(ValueError):
    """Raised when a source checkpoint cannot be converted without guessing."""


def _validate_task(source_task):
    if source_task not in TASKS:
        raise ConversionError(f"source_task must be one of {TASKS}, got {source_task!r}")


def _state_mapping(state):
    if not isinstance(state, (dict, OrderedDict)) or not state:
        raise ConversionError("source state_dict must be a non-empty mapping")
    result = OrderedDict()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ConversionError("state_dict keys must be strings and values tensors")
        if key in result:
            raise ConversionError(f"duplicate source key: {key}")
        result[key] = value.detach().cpu().contiguous()
    prefixed = [key.startswith("module.") for key in result]
    if any(prefixed) and not all(prefixed):
        raise ConversionError("mixed module. and unprefixed state_dict keys")
    if all(prefixed):
        stripped = OrderedDict()
        for key, value in result.items():
            clean = key[len("module."):]
            if not clean or clean in stripped:
                raise ConversionError(f"duplicate key after module. normalization: {clean!r}")
            stripped[clean] = value
        return stripped
    return result


def _target_state(target):
    if isinstance(target, dict):
        state = target
    elif hasattr(target, "state_dict") and callable(target.state_dict):
        state = target.state_dict()
    else:
        raise ConversionError("target must be a model or target state_dict")
    return _state_mapping(state)


def _target_task(target):
    if isinstance(target, dict):
        return None
    module = type(target).__module__
    if module.endswith("airfrans"):
        return "airfrans"
    if module.endswith("shapenet"):
        return "shapenet"
    if module.endswith("model.LinearNO") or module.endswith("LinearNO"):
        return "standard"
    return None


def convert_state_dict(source_task, source_state, target):
    """Validate and return a CPU state dict suitable for strict target loading.

    Current release classes intentionally retain the official key spellings,
    including ShapeNet ``tempreature_q/k`` and AirfRANS' unused temperature.
    Thus conversion is identity after the reversible optional ``module.``
    normalization.  Keeping the check here makes future explicit key renames
    auditable instead of silently dropping parameters.
    """
    _validate_task(source_task)
    target_task = _target_task(target)
    if target_task is not None and target_task != source_task:
        raise ConversionError(f"source_task={source_task} conflicts with target task={target_task}")
    source = _state_mapping(source_state)
    target_state = _target_state(target)
    missing = sorted(set(target_state) - set(source))
    unknown = sorted(set(source) - set(target_state))
    if missing or unknown:
        raise ConversionError(f"state_dict key mismatch: missing={missing}, unknown={unknown}")
    converted = OrderedDict()
    for key, expected in target_state.items():
        actual = source[key]
        if tuple(actual.shape) != tuple(expected.shape):
            raise ConversionError(
                f"state_dict shape mismatch for {key}: source={tuple(actual.shape)}, "
                f"target={tuple(expected.shape)}")
        converted[key] = actual.clone()
    return converted


def add_module_prefix(state):
    """Return a reversible DataParallel-style ``module.`` state dict."""
    clean = _state_mapping(state)
    return OrderedDict((f"module.{key}", value.clone()) for key, value in clean.items())


def load_standard_state_dict(path):
    """Read an audited Standard bare state_dict without constructing a model."""
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise ConversionError("Standard checkpoint must be a bare non-empty state_dict")
    return _state_mapping(value)


def _pickle_worker():
    # Kept as a small source string so the imported official task package lives
    # only in the child interpreter and cannot contaminate the caller's modules.
    return r'''
import json, sys, torch
task, source, output, root = sys.argv[1:]
sys.path.insert(0, root)
obj = torch.load(source, map_location="cpu", weights_only=False)
name = type(obj).__name__
allowed = {
  "airfrans": {"LinearAttentionNeuralOperator", "AirfRANSLinearNO", "Model"},
  "shapenet": {"LinearAttentionNeuralOperator", "ShapeNetLinearNO", "Model"},
}[task]
if name not in allowed or not hasattr(obj, "state_dict"):
    raise ValueError("trusted object class does not match explicit source_task")
state = obj.state_dict()
if not isinstance(state, dict) or not state:
    raise ValueError("trusted object has no state_dict")
torch.save(state, output)
with open(output + ".json", "w", encoding="utf-8") as stream:
    json.dump({"source_task": task, "class_name": name,
               "module": type(obj).__module__}, stream, sort_keys=True)
'''


def extract_trusted_object(path, source_task, *, reference_root=None, trusted=False):
    """Extract a trusted AirfRANS/ShapeNet whole object in an isolated process.

    ``trusted=True`` is mandatory because loading a pickle can execute code.
    The child imports only the explicitly selected task checkout and returns a
    state_dict plus provenance; it never guesses the task from the object.
    """
    _validate_task(source_task)
    if source_task == "standard":
        raise ConversionError("Standard uses a bare state_dict, not a whole object")
    if not trusted:
        raise ConversionError("whole-object loading requires trusted=True")
    source = Path(path).resolve()
    if not source.is_file():
        raise ConversionError(f"trusted checkpoint does not exist: {source}")
    root = Path(reference_root or os.environ.get("LINEARNO_REFERENCE_ROOT", "/home/hwz/LinearNO")).resolve()
    subdir = "AirfRANS" if source_task == "airfrans" else "ShapeNetCar"
    task_root = root / subdir
    if not task_root.is_dir():
        raise ConversionError(f"audited reference task checkout is missing: {task_root}")
    with tempfile.TemporaryDirectory(prefix="linearno-convert-") as temp:
        output = Path(temp) / "state.pt"
        command = [sys.executable, "-B", "-c", _pickle_worker(), source_task,
                   str(source), str(output), str(task_root)]
        result = subprocess.run(command, cwd=task_root, env=dict(os.environ),
                                capture_output=True, text=True)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or "child extraction failed"
            raise ConversionError(message)
        state = torch.load(output, map_location="cpu", weights_only=True)
        provenance = json.loads(output.with_suffix(".pt.json").read_text())
    if provenance.get("source_task") != source_task:
        raise ConversionError("isolated extractor returned the wrong source_task")
    return _state_mapping(state), provenance


def convert_trusted_object(path, source_task, target, *, reference_root=None, trusted=False):
    state, provenance = extract_trusted_object(path, source_task, reference_root=reference_root,
                                               trusted=trusted)
    return convert_state_dict(source_task, state, target), provenance
