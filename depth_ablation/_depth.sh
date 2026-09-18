#!/usr/bin/env bash
# Core depth-ablation helper: one LinearNO run (train or eval) at one depth.
# Not meant to be run directly; use train.sh / eval.sh / train_eval.sh.
#
# Output is fixed to:
#   <repo>/depth_ablation/<task>/seed<seed>/blocks<depth>/
# which the LinearNO entry adopts via --experiment-dir. Different seeds and
# depths therefore never collide; re-running an existing seed+depth is refused
# by the entry's atomic directory reservation (no silent overwrite).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../tran_evaluate/_common.sh"

task="${1:?usage: _depth.sh TASK DEPTH train|eval [--seed N] [entry options]}"
depth="${2:?depth (n-layers) required}"
action="${3:?train or eval required}"
shift 3

factory=LinearNO_Structured_Mesh_2D
case "$task" in
    airfoil)    entry=exp_airfoil.py; data="${CDLNO_AIRFOIL_ROOT:-}" ;;
    darcy)      entry=exp_darcy.py;   data="${CDLNO_DARCY_ROOT:-}" ;;
    elasticity) entry=exp_elas.py;    data="${CDLNO_ELASTICITY_ROOT:-}"; factory=LinearNO_Irregular_Mesh ;;
    pipe)       entry=exp_pipe.py;    data="${CDLNO_PIPE_ROOT:-}" ;;
    *) printf 'depth_ablation supports only airfoil/darcy/elasticity/pipe (got %s)\n' "$task" >&2; exit 2 ;;
esac

case "$action" in
    train) evaluation=0 ;;
    eval)  evaluation=1 ;;
    *) printf 'action must be train or eval (got %s)\n' "$action" >&2; exit 2 ;;
esac

[[ "$depth" =~ ^[0-9]+$ ]] || { printf 'depth must be a positive integer (4/8/16), got %s\n' "$depth" >&2; exit 2; }
((depth >= 1)) || { printf 'depth must be a positive integer (4/8/16), got %s\n' "$depth" >&2; exit 2; }

seed=""
user=()
while (($#)); do
    case "$1" in
        --seed) (($# >= 2)) || { printf 'missing value for --seed\n' >&2; exit 2; }; seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --experiment-dir|--experiment-dir=*|--linearno-run-dir|--linearno-run-dir=*)
            printf 'the output directory is fixed to depth_ablation/<task>/seed<seed>/blocks<depth>; --experiment-dir is not accepted\n' >&2
            exit 2 ;;
        *) user+=("$1"); shift ;;
    esac
done
[[ -n "$seed" ]] || { printf '%s\n' '--seed is required so each seed writes to its own directory' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { printf 'seed must be a non-negative integer, got %s\n' "$seed" >&2; exit 2; }

outdir="$here/$task/seed$seed/blocks$depth"
args=(--model "$factory" --eval "$evaluation" --n-layers "$depth" --seed "$seed" --experiment-dir "$outdir")
[[ -z "$data" ]] || args+=(--data_path "$data")

cdlno_dispatch PDE-Solving-StandardBenchmark "$entry" "${args[@]}" "${user[@]}"
