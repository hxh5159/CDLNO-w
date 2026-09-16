#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
action="${1:?train or eval required}"; shift
# TRAIN my_path = Dataset (manifest.json); EVAL my_path = Dataset's PARENT.
# No --gpu in this original project; use CUDA_VISIBLE_DEVICES.
dataset="${CDLNO_AIRFRANS_DATASET%/}"
case "$action" in
 train) entry=main.py; args=(--score 0 --my_path "$dataset") ;;
 eval)
    entry=main_evaluation.py
    explicit=0
    for flag in "$@"; do case "$flag" in --my_path|--my_path=*) explicit=1 ;; esac; done
    if [[ "${dataset##*/}" != Dataset ]] && ((explicit == 0)); then
        printf 'AirfRANS eval expects a /Dataset path or explicit --my_path parent.\n' >&2; exit 2
    fi
    args=(--my_path "$(dirname -- "$dataset")") ;;
 *) exit 2 ;;
esac
exec bash "$launcher/_dispatch.sh" Airfoil-Design-AirfRANS "$entry" --model msar_lno --profile light "${args[@]}" "$@"
