#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
task="${1:?task required}"; action="${2:?train or eval required}"; shift 2
case "$task" in
    darcy) entry=exp_darcy.py; data="${CDLNO_DARCY_ROOT:-}" ;;
    elasticity) entry=exp_elas.py; data="${CDLNO_ELASTICITY_ROOT:-}" ;;
    airfoil) entry=exp_airfoil.py; data="${CDLNO_AIRFOIL_ROOT:-}" ;;
    pipe) entry=exp_pipe.py; data="${CDLNO_PIPE_ROOT:-}" ;;
    ns) entry=exp_ns.py; data="${CDLNO_NS_ROOT:-}" ;;
    plasticity) entry=exp_plas.py; data="${CDLNO_PLASTICITY_FILE:-}" ;;
    *) printf 'Task not integrated: %s\n' "$task" >&2; exit 2 ;;
esac
case "$action" in train) evaluation=0 ;; eval) evaluation=1 ;; *) exit 2 ;; esac
# Profile, not duplicated explicit dimensions; users can switch it last.
args=(--model kcdno --profile kcdno_v1 --eval "$evaluation")
[[ -z "$data" ]] || args+=(--data_path "$data")
# Resolve the new run flag before the old dispatcher changes working directory.
user=()
while (($#)); do
    case "$1" in
        --kcdno-run-dir)
            value="${2:?missing run directory}"
            [[ "$value" = /* ]] || value="$PWD/$value"
            user+=(--kcdno-run-dir "$value"); shift 2 ;;
        --kcdno-run-dir=*)
            value="${1#*=}"; [[ "$value" = /* ]] || value="$PWD/$value"
            user+=(--kcdno-run-dir "$value"); shift ;;
        *) user+=("$1"); shift ;;
    esac
done
cdlno_dispatch PDE-Solving-StandardBenchmark "$entry" "${args[@]}" "${user[@]}"
