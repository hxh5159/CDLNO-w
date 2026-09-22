#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
task="${1:?task is required}"
default_gpu="${2:?default GPU is required}"
shift 2

action="train_eval"
if (($#)) && [[ "$1" != --* ]]; then action="$1"; shift; fi
case "$action" in
    train_eval|preview|dry-run|print-run-dir) ;;
    *) printf 'Usage: %s [train_eval|preview|dry-run|print-run-dir] [--seed N] [--gpu N]\n' "$0" >&2; exit 2 ;;
esac

seed="${SEED:-0}"
gpu="${GPU:-$default_gpu}"
while (($#)); do
    case "$1" in
        --seed) (($# >= 2)) || { printf 'Missing value for --seed\n' >&2; exit 2; }; seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --gpu) (($# >= 2)) || { printf 'Missing value for --gpu\n' >&2; exit 2; }; gpu="$2"; shift 2 ;;
        --gpu=*) gpu="${1#*=}"; shift ;;
        *) printf 'Unsupported override for this fixed experiment: %s\nOnly --seed and --gpu may be overridden.\n' "$1" >&2; exit 2 ;;
    esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || { printf 'seed must be a nonnegative integer\n' >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'gpu must be a nonnegative integer\n' >&2; exit 2; }

export CDLNO_REPO_ROOT="$repo_root"
if [[ -f "$repo_root/path.sh" ]]; then source "$repo_root/path.sh"; fi
export CDLNO_REPO_ROOT="$repo_root"
export CDLNO_RUNS_ROOT="${NOADAPTER_12_SR_LOW_RUNS_ROOT:-$repo_root/output/linearno_loop_v3/noadapter_12_sr_low}"
entry="$repo_root/tran_evaluate/linearno_loop_v3/efficient_v1/$task.sh"
[[ -f "$entry" ]] || { printf 'Missing V3 task launcher: %s\n' "$entry" >&2; exit 2; }

exec bash "$entry" "$action" \
    --linearno-loop-architecture operator_latent_adapter_v3 \
    --linearno-loop-cost-profile efficient_v1 \
    --linearno-loop-topology d12 \
    --linearno-loop-residual-mode sr_1_over_r \
    --linearno-loop-latent 1 \
    --linearno-loop-adapter-mode none \
    --linearno-loop-adapter-rank 4 \
    --linearno-loop-adapter-alpha 4 \
    --linearno-profile paper_table8_on_release_model \
    --seed "$seed" --gpu "$gpu"
