#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
dataset="${CDLNO_AIRFRANS_DATASET:-}"
profile=""
run_dir=""
args=(--model LinearNO --eval 0)
[[ -z "$dataset" ]] || args+=(--my_path "$dataset")
while (($#)); do
    case "$1" in
        --linearno-profile) profile="${2:?missing profile}"; shift 2 ;;
        --linearno-profile=*) profile="${1#*=}"; shift ;;
        --experiment-dir|--linearno-run-dir) run_dir="${2:?missing run directory}"; shift 2 ;;
        --experiment-dir=*|--linearno-run-dir=*) run_dir="${1#*=}"; shift ;;
        --dry-run) args+=(--dry-run); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ -z "$run_dir" ]]; then
    : # The actual family parser resolves variant/M/evaluation hash/seed.
fi
[[ -z "$profile" ]] || args+=(--linearno-profile "$profile")
if [[ -n "$run_dir" ]]; then
    [[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
    args+=(--experiment-dir "$run_dir")
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" && -n "${LINEARNO_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$LINEARNO_GPU"
fi
cdlno_dispatch Airfoil-Design-AirfRANS main.py "${args[@]}"
