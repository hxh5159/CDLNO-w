#!/usr/bin/env bash
# Explicit sequential convenience; this script never runs unless called by user.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
task="${1:?Usage: train_eval.sh TASK all|off|lrsa_matched [entry options]}"
variant="${2:?choose all|off|lrsa_matched}"; shift 2
case "$task" in darcy|elasticity|airfoil|pipe|ns|plasticity|car|airfrans) ;; *) exit 2 ;; esac
model_flag=--model; [[ "$task" != car ]] || model_flag=--cfd_model
case "$variant" in
 all|off) family=kcdno; selection=("$model_flag" kcdno --history-mode "$variant") ;;
 lrsa_matched) family=lrsa_matched; selection=("$model_flag" lrsa_matched) ;;
 *) exit 2 ;;
esac
run="${CDLNO_RUNS_ROOT:-$cdlno_repo_root/output}/$task/$family/$(date -u +%Y%m%dT%H%M%S)_${variant}_$$"
printf 'Step 1/2: training; proposed run %s (explicit user run overrides it)\n' "$run"
bash "$launcher/$task.sh" train "${selection[@]}" --kcdno-run-dir "$run" "$@"
printf 'Step 2/2: evaluation of the same run\n'
bash "$launcher/$task.sh" eval "${selection[@]}" --kcdno-run-dir "$run" "$@"
