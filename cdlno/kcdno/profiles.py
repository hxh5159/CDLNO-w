"""New-family architecture defaults only, never task training defaults."""

from .config import KCDNOArchitectureConfig


PROFILE_NAMES = ("kcdno_v1", "transolver_shape_match")
TASKS = ("darcy", "elasticity", "airfoil", "pipe", "ns", "plasticity", "car", "airfrans")
_MAIN = {
    "darcy": (128, 8, 64, "conv_ffn"),
    "elasticity": (128, 8, 64, "point_ffn"),
    "airfoil": (128, 4, 64, "conv_ffn"),
    "pipe": (128, 4, 32, "conv_ffn"),
    "ns": (256, 8, 64, "conv_ffn"),
    "plasticity": (128, 8, 64, "conv_ffn"),
    "car": (256, 8, 64, "point_ffn"),
    "airfrans": (256, 8, 64, "point_ffn"),
}
_MATCH = {"airfoil": dict(h=8), "pipe": dict(h=8, M=64),
          "ns": dict(M=32), "car": dict(M=32), "airfrans": dict(M=32)}


def profile_values(task: str, profile: str = "kcdno_v1") -> dict:
    if task not in TASKS:
        raise ValueError(f"unknown KCDNO task: {task!r}; expected one of {TASKS}")
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown KCDNO profile: {profile!r}")
    d, h, tokens, point = _MAIN[task]
    values = dict(L=8, d=d, h=h, M=tokens, kernel_rank=16, history_mode="all", point_module=point)
    if profile == "transolver_shape_match":
        values.update(_MATCH.get(task, {}))
    # Hidden widths are resolved from the FINAL d, after explicit CLI overrides.
    return values


def resolve_profile(task: str, profile: str = "kcdno_v1", **overrides) -> KCDNOArchitectureConfig:
    return KCDNOArchitectureConfig(**(profile_values(task, profile) | overrides))
