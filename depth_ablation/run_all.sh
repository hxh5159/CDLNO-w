#!/usr/bin/env bash
# Convenience loop over tasks x depths x seeds. Prefer train_eval.sh for a
# single run; this is only for queueing many runs sequentially.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage: bash depth_ablation/run_all.sh [--seeds "0 1 2"] [--depths "4 12 16"] [--gpu N] [--dry-run]

Runs train_eval.sh for every (task x depth x seed) in order. Tasks are fixed to
airfoil, darcy, elasticity, pipe. Defaults: seeds="0", depths="4 12 16", gpu="0".
Add --dry-run to preview all commands without executing or creating directories.
EOF
    exit 2
}

tasks=(airfoil darcy elasticity pipe)
depths=(4 12 16)
seeds=(0)
gpu=0
dry=()

while (($#)); do
    case "$1" in
        --seeds) shift; (($#)) || usage; read -r -a seeds <<< "$1"; shift ;;
        --depths) shift; (($#)) || usage; read -r -a depths <<< "$1"; shift ;;
        --gpu) shift; (($#)) || usage; gpu="$1"; shift ;;
        --dry-run) dry=(--dry-run); shift ;;
        *) usage ;;
    esac
done

for task in "${tasks[@]}"; do
    for depth in "${depths[@]}"; do
        for seed in "${seeds[@]}"; do
            printf '== %s depth=%s seed=%s ==\n' "$task" "$depth" "$seed"
            bash "$here/train_eval.sh" "$task" "$depth" --seed "$seed" --gpu "$gpu" "${dry[@]}"
        done
    done
done
