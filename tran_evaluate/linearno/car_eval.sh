#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
args=(--cfd_model LinearNO)
[[ -z "${CDLNO_CAR_RAW_ROOT:-}" ]] || args+=(--data_dir "$CDLNO_CAR_RAW_ROOT")
[[ -z "${CDLNO_CAR_CACHE_ROOT:-}" ]] || args+=(--save_dir "$CDLNO_CAR_CACHE_ROOT")
# Eval reads graph/model/profile options from the explicitly selected run.
# Resolve the run path before the dispatcher enters the benchmark directory.
forward=()
while (($#)); do
    case "$1" in
        --experiment-dir|--linearno-run-dir)
            flag="$1"; value="${2:?missing run directory}"; shift 2
            [[ "$value" = /* ]] || value="$PWD/$value"
            forward+=("$flag" "$value") ;;
        --experiment-dir=*|--linearno-run-dir=*)
            flag="${1%%=*}"; value="${1#*=}"; shift
            [[ -n "$value" ]] || { printf 'Missing run directory\n' >&2; exit 2; }
            [[ "$value" = /* ]] || value="$PWD/$value"
            forward+=("$flag" "$value") ;;
        *) forward+=("$1"); shift ;;
    esac
done
cdlno_dispatch Car-Design-ShapeNetCar main_evaluation.py "${args[@]}" "${forward[@]}"
