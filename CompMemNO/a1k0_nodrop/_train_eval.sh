#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for the eight remote A1K0/no-history-dropout launchers.
# The task wrappers below remain the user-facing, one-dataset-per-script entry.
task="${1:?task name is required}"; shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
source "$repo_root/path.sh"

case "$task" in
    airfoil|darcy|elasticity|pipe|ns|plasticity|airfrans|car) ;;
    *) printf 'Unsupported dataset: %s\n' "$task" >&2; exit 2 ;;
esac

gpu="${LINEARNO_GPU:-0}"
seed="${LINEARNO_SEED:-17}"
profile="${LINEARNO_PROFILE:-paper_table8_on_release_model}"
run_dir="${LINEARNO_RUN_DIR:-}"
extra=()

# Keep the experiment identity and innovation flags owned by this launcher.
# Other task options (for example --epochs or --data_path) are forwarded.
while (($#)); do
    case "$1" in
        --gpu)
            (($# >= 2)) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            gpu="$2"; shift 2 ;;
        --gpu=*) gpu="${1#*=}"; shift ;;
        --seed)
            (($# >= 2)) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --linearno-profile)
            (($# >= 2)) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            profile="$2"; shift 2 ;;
        --linearno-profile=*) profile="${1#*=}"; shift ;;
        --experiment-dir|--linearno-run-dir)
            (($# >= 2)) || { printf '%s requires a value\n' "$1" >&2; exit 2; }
            run_dir="$2"; shift 2 ;;
        --experiment-dir=*|--linearno-run-dir=*) run_dir="${1#*=}"; shift ;;
        --linearno_latent_attnres|--linearno-latent-attnres|\
        --linearno_history_k_conditioning|--linearno-history-k-conditioning|\
        --linearno_attnres_history_dropout_p|--linearno-attnres-history-dropout-p)
            printf 'A1K0/no-drop flags are fixed by this launcher; do not override %s\n' "$1" >&2
            exit 2 ;;
        *) extra+=("$1"); shift ;;
    esac
done

if [[ -z "$run_dir" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
    run_dir="$CDLNO_RUNS_ROOT/$task/linearno_history/${task}__${profile}__L8__A1K0__nodrop__seed${seed}__${stamp}"
fi
[[ "$run_dir" = /* ]] || run_dir="$repo_root/$run_dir"

launcher="$repo_root/tran_evaluate/linearno_history/$task.sh"
features=(
    --linearno_latent_attnres 1
    --linearno_history_k_conditioning 0
    --linearno_attnres_history_dropout_p 0
)

train_args=(train --linearno-profile "$profile" --seed "$seed" --gpu "$gpu"
    --experiment-dir "$run_dir" "${features[@]}" "${extra[@]}")
eval_args=(eval --linearno-profile "$profile" --gpu "$gpu"
    --experiment-dir "$run_dir" "${features[@]}" "${extra[@]}")

printf 'A1K0/no-drop %s run directory: %s\n' "$task" "$run_dir"
printf 'Training with latent AttnRes=1, history K=0, history dropout p=0.\n'
bash "$launcher" "${train_args[@]}" && \
    bash "$launcher" "${eval_args[@]}"
