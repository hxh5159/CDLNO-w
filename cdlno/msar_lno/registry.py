"""Lazy new-family selection; old keys delegate to their existing selectors."""
from .config import FAMILY


def family_for_model_key(key: str) -> str | None:
    return FAMILY if key == FAMILY else None  # None delegates to the unchanged old selector.


def checkpoint_family(payload: dict) -> str | None:
    if type(payload) is not dict:raise ValueError('checkpoint metadata must be an object')
    if 'family' not in payload:return None  # Never infer MSAR from internal shape/profile/name.
    family = payload['family']
    if type(family) is not str or not family:raise ValueError('explicit family must be a nonempty string')
    return family


def core_class(model_key: str):
    """Real lifted-feature core, not a task wrapper or a placeholder constructor.

    Keep configuration/legacy imports torch-free. Industrial task selectors can
    use this same family mapping when their field adapters are integrated.
    """
    if family_for_model_key(model_key) is None:
        return None
    from .core import MSARLNO
    return MSARLNO
