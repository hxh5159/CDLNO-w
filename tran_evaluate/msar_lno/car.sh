#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
action="${1:?train or eval required}"; shift
case "$action" in train) entry=main.py ;; eval) entry=main_evaluation.py ;; *) exit 2 ;; esac
args=(--cfd_model msar_lno --profile light)
[[ -z "${CDLNO_CAR_RAW_ROOT:-}" ]] || args+=(--data_dir "$CDLNO_CAR_RAW_ROOT")
[[ -z "${CDLNO_CAR_CACHE_ROOT:-}" ]] || args+=(--save_dir "$CDLNO_CAR_CACHE_ROOT")
# save_dir is preprocessed data. A run is exactly one fold. Full drag evaluation
# retains the original fixed param0 path; the preflight checks it without data IO.
exec bash "$launcher/_dispatch.sh" Car-Design-ShapeNetCar "$entry" "${args[@]}" "$@"
