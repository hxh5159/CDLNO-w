#!/usr/bin/env bash
# Train LinearNO at one depth, then evaluate the same run on success.
# Usage: bash depth_ablation/train_eval.sh TASK DEPTH --seed N [--gpu N] [entry options]
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task="${1:?usage: train_eval.sh TASK DEPTH [--seed N] [entry options]}"
depth="${2:?depth (n-layers) required}"
shift 2
printf 'Task: %s | depth (n-layers): %s\n' "$task" "$depth"
printf 'Step 1/2: training\n'
if bash "$here/train.sh" "$task" "$depth" "$@"; then
    printf 'Step 2/2: evaluation\n'
else
    status=$?
    printf 'Training failed (exit %s); evaluation was not started.\n' "$status" >&2
    exit "$status"
fi
bash "$here/eval.sh" "$task" "$depth" "$@"
