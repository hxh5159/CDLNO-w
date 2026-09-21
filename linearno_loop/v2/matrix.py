"""Configuration-only v2 experiment matrix. No model or data imports."""

from cdlno.linearno.profiles import TASKS
from linearno_loop.contracts import PRESETS, RESIDUAL_MODES
from .config import resolve_config, run_directory_id
from .contracts import CORE_FFN_MODES


def configuration_matrix(seeds=(0, 1, 2)):
    runs = []
    for task in TASKS:
        for topology in PRESETS:
            for residual in RESIDUAL_MODES:
                for mode in CORE_FFN_MODES:
                    for seed in seeds:
                        config = resolve_config(task, options={"topology_preset": topology,
                            "residual_mode": residual, "core_ffn_mode": mode},
                            profile_overrides={"runtime.seed": seed})
                        runs.append(dict(config=config, run_id=run_directory_id(config), status="PLANNED_NOT_RUN"))
    controls = []
    for task in TASKS:
        for mode in CORE_FFN_MODES:
            config = resolve_config(task, options={"topology_preset": "p2_c2_r2_s2",
                "residual_mode": "sr_1_over_r", "core_ffn_mode": mode, "rank_multiplier": 2},
                profile_overrides={"runtime.seed": 0})
            controls.append(dict(kind="rank_x2", config=config, run_id=run_directory_id(config)))
    custom = resolve_config("darcy", options={"topology_preset": "custom", "prefix_blocks": 0,
        "recurrent_core_blocks": 2, "loop_repeats": 3, "suffix_blocks": 1,
        "residual_mode": "lb_attnres_1_over_r", "core_ffn_mode": "round_specific_latent"})
    return dict(paired_seeds=list(seeds), runs=runs, controls=controls,
                custom=dict(config=custom, run_id=run_directory_id(custom)))

