"""Eight-task launcher for partial_share_feature_gate_v5."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tran_evaluate.linearno_loop.launch import TASKS, execute, plan

ARCHITECTURE = "partial_share_feature_gate_v5"
RESIDUAL = "operator_1_expert_1_over_r"
PRESETS = ("p1_c3_r2_s1", "p2_c2_r2_s2", "custom")


def _append_data_defaults(args, tokens, rest):
    flags = {token.split("=")[0] for token in rest}
    explicit_root = args.data_root is not None
    root = args.data_root or Path(os.environ.get("CDLNO_DATA_ROOT", ROOT.parent / "data"))

    def value(environment, fallback):
        return str(fallback if explicit_root else os.environ.get(environment, fallback))

    if args.task not in ("airfrans", "car") and not {"--data_path", "--data-path"} & flags:
        if args.task == "plasticity":
            tokens += ["--data_path", value("CDLNO_PLASTICITY_FILE", root / "fno" / "plas_N987_T20.mat")]
        else:
            tokens += ["--data_path", value("CDLNO_FNO_ROOT", root / "fno")]
    elif args.task == "airfrans" and "--my_path" not in flags:
        tokens += ["--my_path", value("CDLNO_AIRFRANS_DATASET", root / "AirfRANS" / "Dataset")]
    elif args.task == "car":
        if "--data_dir" not in flags:
            tokens += ["--data_dir", value("CDLNO_CAR_RAW_ROOT", root / "mlcfd_data" / "training_data")]
        if "--save_dir" not in flags:
            tokens += ["--save_dir", value("CDLNO_CAR_CACHE_ROOT", root / "mlcfd_data" / "preprocessed_data")]


def plan_v5(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("action", choices=("train", "resume", "eval", "train_eval"))
    parser.add_argument("--topology", choices=PRESETS)
    parser.add_argument("--executed-depth", type=int)
    parser.add_argument("--prefix-blocks", type=int)
    parser.add_argument("--core-blocks", type=int)
    parser.add_argument("--loop-repeats", type=int)
    parser.add_argument("--suffix-blocks", type=int)
    parser.add_argument("--expert-count", type=int)
    parser.add_argument("--expert-width", type=int)
    parser.add_argument("--linearno-rank", type=int)
    parser.add_argument("--linearno-profile", dest="profile")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("CDLNO_RUNS_ROOT", ROOT / "output")))
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--checkpoint", choices=("latest", "final"))
    parser.add_argument("--dry-run", "--preview", action="store_true")
    parser.add_argument("--print-config", "--plan-json", action="store_true")
    parser.add_argument("--print-run-dir", action="store_true")
    parser.add_argument("--then-eval", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if args.then_eval and args.action == "eval":
        parser.error("--then-eval is only for train/resume")
    action = "train" if args.action == "train_eval" else args.action
    if action != "train" and args.experiment_dir is None:
        parser.error("resume/eval requires --experiment-dir")
    forbidden = {
        "--linearno-loop", "--linearno-loop-architecture",
        "--linearno-loop-residual-mode", "--linearno-loop-dense-expert-count",
        "--linearno-loop-dense-expert-width", "--linearno-loop-topology",
        "--linearno-loop-executed-depth", "--linearno-loop-prefix-blocks",
        "--linearno-loop-core-blocks", "--linearno-loop-repeats",
        "--linearno-loop-suffix-blocks", "--linearno-rank",
    }
    if any(token.split("=")[0] in forbidden for token in rest):
        parser.error("use the V5 launcher options instead of raw architecture fields")

    values = (args.prefix_blocks, args.core_blocks, args.loop_repeats, args.suffix_blocks)
    custom_fields = any(value is not None for value in values)
    if custom_fields and (not all(value is not None for value in values) or args.topology != "custom"):
        parser.error("custom topology requires --topology custom and all P/C/R/S fields")
    if args.topology == "custom" and not custom_fields:
        parser.error("--topology custom requires all P/C/R/S fields")
    if args.executed_depth is not None and (args.topology is not None or custom_fields):
        parser.error("--executed-depth conflicts with explicit topology")

    tokens = ["--linearno-loop", "1", "--linearno-loop-architecture", ARCHITECTURE,
              "--gpu", str(args.gpu)]
    if action == "train":
        tokens += ["--linearno-loop-residual-mode", RESIDUAL,
                   "--linearno-loop-dense-expert-count", str(args.expert_count or 2)]
        if args.executed_depth is None and args.topology is None:
            tokens += ["--linearno-loop-topology", "p2_c2_r2_s2"]
    elif args.expert_count is not None:
        tokens += ["--linearno-loop-dense-expert-count", str(args.expert_count)]
    if args.topology is not None:
        tokens += ["--linearno-loop-topology", args.topology]
    if args.executed_depth is not None:
        tokens += ["--linearno-loop-executed-depth", str(args.executed_depth)]
    if custom_fields:
        tokens += ["--linearno-loop-prefix-blocks", str(args.prefix_blocks),
                   "--linearno-loop-core-blocks", str(args.core_blocks),
                   "--linearno-loop-repeats", str(args.loop_repeats),
                   "--linearno-loop-suffix-blocks", str(args.suffix_blocks)]
    if args.expert_width is not None:
        tokens += ["--linearno-loop-dense-expert-width", str(args.expert_width)]
    if args.linearno_rank is not None:
        tokens += ["--linearno-rank", str(args.linearno_rank)]
    if args.seed is not None:
        tokens += ["--seed", str(args.seed)]
    if args.profile is not None:
        tokens += ["--linearno-profile", args.profile]
    if args.experiment_dir is not None:
        tokens += ["--experiment-dir", str(args.experiment_dir.resolve())]
    if args.checkpoint is not None:
        tokens += ["--checkpoint", args.checkpoint]
    _append_data_defaults(args, tokens, rest)

    prior = os.environ.get("CDLNO_RUNS_ROOT")
    os.environ["CDLNO_RUNS_ROOT"] = str(args.output_root.resolve())
    try:
        result = plan(args.task, action, [*tokens, *rest])
    finally:
        if prior is None:
            os.environ.pop("CDLNO_RUNS_ROOT", None)
        else:
            os.environ["CDLNO_RUNS_ROOT"] = prior
    if result["config"]["architecture"] != ARCHITECTURE:
        parser.error("V5 launcher requires V5 saved metadata")
    return args, result


def main(argv=None):
    args, result = plan_v5(argv)
    if args.print_run_dir:
        print(result["run"])
        return 0
    if args.print_config or args.dry_run:
        print(json.dumps({key: value for key, value in result.items()
                          if key != "environment"}, indent=2, sort_keys=True))
        print("DRY RUN: real parser/metadata only; no data, tensor, weight, or run creation.")
        return 0
    code = execute(result)
    if code or not (args.then_eval or args.action == "train_eval"):
        return code
    follow = ["--experiment-dir", result["run"], "--gpu", str(args.gpu)]
    path_flags = {"--data_path", "--my_path", "--data_dir", "--save_dir", "--device"}
    index = 0
    while index < len(result["argv"]):
        token = result["argv"][index]
        if token.split("=")[0] in path_flags:
            follow.append(token)
            if "=" not in token:
                index += 1
                follow.append(result["argv"][index])
        index += 1
    evaluation = plan(args.task, "eval", follow, result["environment"])
    return execute(evaluation)


if __name__ == "__main__":
    raise SystemExit(main())
