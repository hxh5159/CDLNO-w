"""Pure v2 configuration resolution; v1 remains in ``linearno_loop.config``."""

from copy import deepcopy
import hashlib

from cdlno.linearno.profiles import DEFAULT_PROFILE, PROFILES, TASKS, derived_model, validate_model
from linearno_loop.config import _attnres, _profile, _topology
from .contracts import (
    ARCHITECTURE_EXTENSION,
    CLASS_PATHS,
    CONFIG_VERSION,
    CORE_FFN_MODES,
    FAMILY,
    FORMULA_VERSION,
    OPTIONS,
    SCHEMA_VERSION,
    TOPOLOGY_FIELDS,
    LoopSchemaError,
    digest,
    exact,
    integer,
    json_value,
    require_equal,
    seal,
)


def latent_width(variant, hidden, kernel=3):
    integer(hidden, "hidden", 1)
    integer(kernel, "kernel", 1)
    if kernel % 2 != 1:
        raise LoopSchemaError("latent kernel must be odd")
    # The frozen first study uses an explicit 572 inner width for the four
    # structured variants. It is a profile value, not a derived 4H formula or
    # a public search knob (staged prompt section 4, lines 198--213).
    return 572 if variant in ("conv", "conv_temp") else hidden


def _derive_seed(label, payload):
    raw = (label + ":" + digest(payload)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)


def _model_spec(profile, loop):
    model = profile["values"]["model"]
    mapping = dict(
        space_dim="space_dim", n_hidden="hidden", n_head="heads", dropout="dropout",
        act="activation", mlp_ratio="ffn_ratio", fun_dim="fun_dim", out_dim="out_dim",
        ref="ref", unified_pos="unified_pos",
    )
    kwargs = {target: model[source] for target, source in mapping.items()}
    kwargs.update({field: loop[field] for field in TOPOLOGY_FIELDS})
    kwargs.update(
        residual_mode=loop["residual_mode"],
        core_ffn_mode=loop["core_ffn_mode"],
        linearno_rank=loop["resolved_rank"],
        latent_width=loop["latent_ffn"]["width"],
        feature_seed=loop["initialization"]["feature_seed"],
    )
    task = profile["task"]
    if task == "airfrans":
        kwargs["linear"] = True
    else:
        kwargs.update(
            Time_Input=model["time_input"], H=model["H"] if model["H"] is not None else 85,
            W=model["W"] if model["W"] is not None else 85,
        )
        if task == "car":
            kwargs["isregular"] = False
        else:
            kwargs["linearno_variant"] = model["linearno_variant"]
    return dict(class_path=CLASS_PATHS.get(task, CLASS_PATHS["standard"]), constructor_kwargs=kwargs)


def fair_seeds(config):
    profile, loop = config["profile_spec"], config["loop_spec"]
    public_seed = profile["values"]["runtime"]["seed"]
    common = dict(
        task=profile["task"], profile=profile["profile"], seed=public_seed,
        topology={field: loop[field] for field in TOPOLOGY_FIELDS},
        model=profile["values"]["model"], actual_M=loop["resolved_rank"], version=2,
    )
    data = dict(task=profile["task"], profile=profile["profile"], seed=public_seed)
    return dict(
        protocol="paired-v2-backbone-loader-v1", public_backbone_seed=public_seed,
        backbone_pair_id=digest(common),
        point_ffn_seed=loop["initialization"]["point_ffn_seed"],
        feature_seed=loop["initialization"]["feature_seed"],
        dataloader_generators={split: _derive_seed("loader-" + split, data) for split in ("train", "test")},
        rule="common_tree_identical_across_v2_modes;latent_construction_fork_rng;W2_zero_after_init",
        topology_pairs_are_parameter_matched=False,
    )


def resolve_config(task, profile=DEFAULT_PROFILE, *, options=None, profile_overrides=None):
    if type(task) is not str or task not in TASKS or type(profile) is not str or profile not in PROFILES:
        raise LoopSchemaError("unknown task/profile")
    options = deepcopy({} if options is None else options)
    json_value(options, "options")
    if type(options) is not dict or set(options) - OPTIONS:
        raise LoopSchemaError("options: unknown fields or not an object")
    mode = options.get("core_ffn_mode")
    if type(mode) is not str or mode not in CORE_FFN_MODES:
        raise LoopSchemaError("core_ffn_mode: explicit round_specific or round_specific_latent required")
    topology = _topology(options)
    residual = options.get("residual_mode")
    from linearno_loop.contracts import RESIDUAL_MODES
    if type(residual) is not str or residual not in RESIDUAL_MODES:
        raise LoopSchemaError("residual_mode: explicitly select one of the three modes")
    overrides = deepcopy({} if profile_overrides is None else profile_overrides)
    base = _profile(task, profile, overrides)
    model = base["values"]["model"]
    base_rank = model["linearno_rank"]
    if "linearno_rank" in options and "rank_multiplier" in options:
        raise LoopSchemaError("explicit actual linearno_rank and rank_multiplier are mutually exclusive")
    if "linearno_rank" in options:
        rank = integer(options["linearno_rank"], "linearno_rank", 1)
        multiplier, policy = None, "explicit_actual"
    else:
        multiplier = integer(options.get("rank_multiplier", 1), "rank_multiplier", 1)
        if multiplier not in (1, 2):
            raise LoopSchemaError("rank_multiplier: only 1 or 2")
        rank, policy = base_rank * multiplier, "profile_multiplier"
    actual_model = {**model, "linearno_rank": rank}
    try:
        validate_model(task, actual_model)
    except ValueError as error:
        raise LoopSchemaError(str(error)) from error
    P, C, R, S = (topology[field] for field in TOPOLOGY_FIELDS)
    width = latent_width(model["linearno_variant"], model["hidden"])
    seed_payload = dict(task=task, profile=profile, seed=base["values"]["runtime"]["seed"],
                        topology=topology, rank=rank, model=model, version=2)
    feature_seed = _derive_seed("latent-context-ffn-v2", seed_payload)
    point_seed = _derive_seed("round-point-ffn-v2", seed_payload)
    enabled = mode == "round_specific_latent"
    sharing = dict(
        operator="one_ln1_and_operator_per_core_position_shared_across_rounds",
        point_ffn="one_ln2_and_mlp_per_core_position_and_round",
        prefix_suffix="native_complete_disjoint_blocks",
        latent="one_per_core_position_shared_across_rounds" if enabled else "not_registered",
        recompute_qkv_context_each_visit=True, output_head="last_suffix_only_once",
        routers="independent_per_logical_receiver", history="forward_local_raw_points",
        detach=False, cross_forward=False, cross_physical_time=False, cross_member=False,
    )
    loop = dict(
        schema_version=SCHEMA_VERSION, config_version=CONFIG_VERSION, formula_version=FORMULA_VERSION,
        task=task, profile=profile, topology_preset=options["topology_preset"], **topology,
        unique_depth=P+C+S, executed_depth=P+C*R+S, residual_mode=residual,
        core_ffn_mode=mode, base_rank=base_rank, rank_multiplier=multiplier,
        resolved_rank=rank, rank_policy=policy, hidden=model["hidden"], heads=model["heads"],
        head_dim=model["hidden"] // model["heads"], variant=model["linearno_variant"],
        point_domain_attnres=True, feature_timestep_encoding=False,
        point_ffn_instance_count=P+C*R+S, operator_instance_count=P+C+S,
        operator_contract=derived_model(task, actual_model), sharing=sharing,
        latent_ffn=dict(
            enabled=enabled, position="after_KtV_before_Q_readout", instance_count=C if enabled else 0,
            sharing_scope="core_position_across_rounds", width=width,
            width_source="frozen_structured_profile_572" if model["linearno_variant"] in ("conv", "conv_temp") else "hidden",
            activation="GELU", norm="LayerNorm_affine", norm_eps=1e-5, dropout=0.0,
            linear_bias=True, second_linear_zero_initialization=True, token_axis_mixing=False,
        ),
        initialization=dict(protocol="loop-linearno-ffn-v2-init-v1", public_seed=base["values"]["runtime"]["seed"],
                            point_ffn_seed=point_seed, feature_seed=feature_seed,
                            latent_fork_rng=True, public_rng_advanced_by_latent=False),
        state_partition=dict(prefix="loop.prefix", core_operators="loop.core_operators",
                             core_ffns="loop.core_ffns", latent_ffns="loop.latent_ffns" if enabled else None,
                             suffix="loop.suffix", residual_router=residual),
        residual_contract=dict(
            prefix_suffix="native_unscaled", identity="unscaled",
            core_branch_scale="none_no_additive_residual" if residual == "rb_attnres" else "1/loop_repeats",
            sources={"sr_1_over_r": "none", "rb_attnres": "anchor_completed_raw_round_sums_and_current_raw_partial",
                     "lb_attnres_1_over_r": "anchor_and_actual_Y_minus_H"}[residual],
            receiver_before_original_layernorm=True, head_calls=1,
        ),
        attnres=_attnres(residual, C, R, model["hidden"]),
    )
    sources = {key: "derived" for key in loop}
    sources.update({field: "cli_explicit" if field in options else "preset" for field in TOPOLOGY_FIELDS})
    sources.update(
        topology_preset="cli_explicit", residual_mode="cli_explicit", core_ffn_mode="cli_explicit",
        base_rank="profile", rank_multiplier="not_applicable" if policy == "explicit_actual" else
        ("cli_explicit" if "rank_multiplier" in options else "v2_family_default"),
        resolved_rank="cli_explicit" if policy == "explicit_actual" else "derived",
        hidden=base["field_sources"]["model.hidden"], heads=base["field_sources"]["model.heads"],
        variant=base["field_sources"]["model.linearno_variant"],
        point_domain_attnres="frozen_contract", feature_timestep_encoding="frozen_contract",
    )
    result = dict(
        family=FAMILY, architecture_extension=ARCHITECTURE_EXTENSION, config_version=CONFIG_VERSION,
        request=dict(task=task, profile=profile, options=options, profile_overrides=overrides),
        loop_spec=loop, model_spec=_model_spec(base, loop), profile_spec=base,
        field_sources=dict(loop_spec=sources, profile_spec=deepcopy(base["field_sources"])),
    )
    result["fair_comparison"] = fair_seeds(result)
    return seal(result, "config_hash")


def validate_config(config):
    json_value(config)
    exact(config, {"family", "architecture_extension", "config_version", "request", "loop_spec",
                  "model_spec", "profile_spec", "field_sources", "fair_comparison", "config_hash"}, "config")
    exact(config["request"], {"task", "profile", "options", "profile_overrides"}, "request")
    request = config["request"]
    expected = resolve_config(request["task"], request["profile"], options=request["options"],
                              profile_overrides=request["profile_overrides"])
    require_equal(expected, config, "config")
    return deepcopy(config)


def run_directory_id(config):
    checked = validate_config(config)
    loop = checked["loop_spec"]
    seed = checked["profile_spec"]["values"]["runtime"]["seed"]
    P, C, R, S = (loop[field] for field in TOPOLOGY_FIELDS)
    return (f"{loop['task']}__{FAMILY}__v2__{loop['profile']}__P{P}-C{C}-R{R}-S{S}__"
            f"{loop['residual_mode']}__{loop['core_ffn_mode']}__M{loop['resolved_rank']}__"
            f"seed{seed}__cfg{checked['config_hash'][:12]}")
