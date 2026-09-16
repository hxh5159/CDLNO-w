"""M4 explicit parameter forwarding for the lifted-feature core only.

Task lift/time/graph adapters and production train/eval loops remain separate.
No old Namespace defaults become explicit MSAR architecture options.
"""
from .config import positive_int
from .options import resolve_training
from .registry import family_for_model_key


def core_model_kwargs(model_key, *, task, explicit, output_dim=1):
    """Only msar_lno receives new kwargs; other selectors receive no additions."""
    if family_for_model_key(model_key) is None:
        return {}
    positive_int('output_dim', output_dim)
    resolved = resolve_training(task, explicit)
    return dict(config=resolved.architecture, training_config=resolved.training,
                output_dim=output_dim)
