#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
dataset="${CDLNO_AIRFRANS_DATASET:-}"
dataset_arg=""
run_dir=""
profile=""
args=(--model LinearNO)
while (($#)); do
    case "$1" in
        --linearno-profile) profile="${2:?missing profile}"; shift 2 ;;
        --linearno-profile=*) profile="${1#*=}"; shift ;;
        --experiment-dir|--linearno-run-dir) run_dir="${2:?missing run directory}"; shift 2 ;;
        --experiment-dir=*|--linearno-run-dir=*) run_dir="${1#*=}"; shift ;;
        --dry-run) args+=(--dry-run); shift ;;
        --my_path) dataset_arg="${2:?missing data path}"; args+=("$1" "$dataset_arg"); shift 2 ;;
        --my_path=*) dataset_arg="${1#*=}"; [[ -n "$dataset_arg" ]] || { printf 'Missing value for --my_path\n' >&2; exit 2; }; args+=("$1"); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ -z "$run_dir" ]]; then
    printf 'Evaluation requires --experiment-dir RUN_DIR; no latest-run guessing.\n' >&2
    exit 2
fi
if [[ -z "$dataset" && -z "$dataset_arg" ]]; then
    printf 'Set CDLNO_AIRFRANS_DATASET or pass --my_path to the Dataset parent.\n' >&2
    exit 2
fi
if [[ -z "$dataset_arg" ]]; then
    # main_evaluation accepts the parent directory; _data_root also accepts a
    # Dataset directory, so retain the user's configured path verbatim.
    args+=(--my_path "$dataset")
fi
if [[ -n "$profile" ]]; then args+=(--linearno-profile "$profile"); fi
[[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
args+=(--experiment-dir "$run_dir")
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" && -n "${LINEARNO_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$LINEARNO_GPU"
fi
cdlno_dispatch Airfoil-Design-AirfRANS main_evaluation.py "${args[@]}"
