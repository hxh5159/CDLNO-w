"""Looped LinearNO V3 configuration-only contracts.

LAA1 deliberately contains no tensor, task-entry, model, or checkpoint backend
imports.  Construction is reserved for later LAA stages.
"""

from .config import resolve_config, validate_config, run_directory_id

__all__ = ("resolve_config", "validate_config", "run_directory_id")
