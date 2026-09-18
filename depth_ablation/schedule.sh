#!/usr/bin/env bash
# Run the full depth schedule for ONE dataset on its assigned GPU(s).
# Usage: bash depth_ablation/schedule.sh TASK --seed N [--dry-run]
#
# Same-GPU depths run sequentially; different GPUs run in parallel. Each
# (task, seed, depth) writes to depth_ablation/<task>/seed<seed>/blocks<depth>,
# so seeds never collide and cross-GPU streams never touch the same directory.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

task="${1:?usage: schedule.sh TASK --seed N [--dry-run]}"
shift

seed=""
dry=()
while (($#)); do
    case "$1" in
        --seed) (($# >= 2)) || { printf '%s\n' 'missing --seed value' >&2; exit 2; }; seed="$2"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        --dry-run) dry=(--dry-run); shift ;;
        *) printf 'unknown option %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$seed" ]] || { printf '%s\n' '--seed is required so each seed writes to its own directory' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { printf 'seed must be a non-negative integer, got %s\n' "$seed" >&2; exit 2; }

# Each entry is "<gpu>:<depth> <depth> ..." — one entry per GPU stream; depths
# within an entry run sequentially. Two entries run in parallel.
case "$task" in
    airfoil)    streams=("0:4 12 16") ;;
    darcy)      streams=("1:4 12 16") ;;
    elasticity) streams=("0:4 12" "1:16") ;;
    pipe)       streams=("0:16" "1:4 12") ;;
    *) printf 'unknown task %s (airfoil/darcy/elasticity/pipe)\n' "$task" >&2; exit 2 ;;
esac

pids=()
for stream in "${streams[@]}"; do
    gpu="${stream%%:*}"; spec="${stream#*:}"
    (
        read -r -a depth_list <<< "$spec"
        for depth in "${depth_list[@]}"; do
            printf '== %s seed=%s depth=%s gpu=%s ==\n' "$task" "$seed" "$depth" "$gpu"
            if ! bash "$here/train_eval.sh" "$task" "$depth" --seed "$seed" --gpu "$gpu" "${dry[@]}"; then
                printf 'FAILED: %s seed=%s depth=%s gpu=%s\n' "$task" "$seed" "$depth" "$gpu" >&2
                exit 1
            fi
        done
    ) &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
