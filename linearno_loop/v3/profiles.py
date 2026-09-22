"""Frozen V3 cost/profile tables.  This module is pure data plus stdlib checks."""

from .contracts import TASKS, COST_PROFILES, DEPTHS, V3SchemaError, integer

# Columns are TASKS order. Values are hidden_width / latent_width.
PROFILE_WIDTHS = {
    "matched_v1": {
        "d12": ((104, 704), (104, 704), (96, 320), (104, 704), (104, 720), (208, 680), (208, 672), (208, 776)),
        "d20": ((96, 736), (96, 736), (88, 312), (96, 736), (96, 744), (200, 592), (200, 592), (200, 688)),
        "d28": ((96, 648), (96, 648), (88, 272), (96, 648), (96, 656), (192, 616), (192, 616), (192, 712)),
        "d60": ((88, 728), (88, 728), (80, 288), (88, 728), (88, 736), (184, 600), (184, 600), (184, 696)),
    },
    "efficient_v1": {
        "d12": ((96, 512), (96, 512), (88, 256), (96, 512), (96, 512), (192, 512), (192, 512), (192, 512)),
        "d20": ((88, 512), (88, 512), (80, 256), (88, 512), (88, 512), (184, 512), (184, 512), (184, 512)),
        "d28": ((88, 512), (88, 512), (80, 224), (88, 512), (88, 512), (176, 512), (176, 512), (176, 512)),
        "d60": ((80, 512), (80, 512), (72, 256), (80, 512), (80, 512), (168, 512), (168, 512), (168, 512)),
    },
}


def width_for(profile, depth, task):
    if profile not in ("matched_v1", "efficient_v1"):
        raise V3SchemaError(f"unknown tabulated cost profile: {profile}")
    if depth not in DEPTHS or task not in TASKS:
        raise V3SchemaError("unknown profile depth/task")
    return dict(zip(("hidden_width", "latent_width"), PROFILE_WIDTHS[profile][depth][TASKS.index(task)]))


def validate_tables():
    for profile in ("matched_v1", "efficient_v1"):
        if set(PROFILE_WIDTHS[profile]) != set(DEPTHS):
            raise AssertionError("V3 profile table depth set drift")
        for depth in DEPTHS:
            if len(PROFILE_WIDTHS[profile][depth]) != len(TASKS):
                raise AssertionError("V3 profile table task count drift")
            for task in TASKS:
                hidden, latent = width_for(profile, depth, task).values()
                integer(hidden, f"{profile}.{depth}.{task}.hidden_width", 1)
                integer(latent, f"{profile}.{depth}.{task}.latent_width", 1)


validate_tables()
