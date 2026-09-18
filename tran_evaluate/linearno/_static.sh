#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
task="${1:?task required}"; action="${2:?train/eval required}"; shift 2
factory=LinearNO_Structured_Mesh_2D
case "$task" in
    darcy) entry=exp_darcy.py; data="${CDLNO_DARCY_ROOT:-}" ;;
    elasticity) entry=exp_elas.py; data="${CDLNO_ELASTICITY_ROOT:-}"; factory=LinearNO_Irregular_Mesh ;;
    airfoil) entry=exp_airfoil.py; data="${CDLNO_AIRFOIL_ROOT:-}" ;;
    pipe) entry=exp_pipe.py; data="${CDLNO_PIPE_ROOT:-}" ;;
    ns) entry=exp_ns.py; data="${CDLNO_NS_ROOT:-}" ;;
    plasticity) entry=exp_plas.py; data="${CDLNO_PLASTICITY_FILE:-}" ;;
    *) printf 'LinearNO task not integrated: %s\n' "$task" >&2; exit 2 ;;
esac
case "$action" in train) evaluation=0 ;; eval) evaluation=1 ;; *) exit 2 ;; esac
args=(--model "$factory" --eval "$evaluation")
[[ -z "$data" ]] || args+=(--data_path "$data")
user=()
while (($#)); do
    case "$1" in
        --model|--model=*|--eval|--eval=*)
            printf 'The LinearNO launcher fixes family/action; choose the train or eval script.\n' >&2; exit 2 ;;
        --experiment-dir|--linearno-run-dir)
            value="${2:?missing experiment directory}"
            [[ "$value" = /* ]] || value="$PWD/$value"
            user+=(--experiment-dir "$value"); shift 2 ;;
        --experiment-dir=*|--linearno-run-dir=*)
            value="${1#*=}"; [[ -n "$value" ]] || exit 2
            [[ "$value" = /* ]] || value="$PWD/$value"
            user+=(--experiment-dir "$value"); shift ;;
        *) user+=("$1"); shift ;;
    esac
done
cdlno_dispatch PDE-Solving-StandardBenchmark "$entry" "${args[@]}" "${user[@]}"
