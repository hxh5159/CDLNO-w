#!/usr/bin/env bash
# One MSAR-LNO experiment: reuse the task's train and eval launchers.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash run_msar.sh TASK [light|full] [--gpu ID] [--dry-run]

TASK: darcy elasticity airfoil pipe ns plasticity car airfrans
Defaults: light, GPU ${MSAR_GPU:-0}, coverage floor / weight 0.01 / kappa 0.2.

Examples:
  bash tran_evaluate/msar_lno/run_msar.sh darcy light --gpu 0
  bash tran_evaluate/msar_lno/run_msar.sh darcy full --gpu 1
  bash tran_evaluate/msar_lno/run_msar.sh airfrans full --gpu 1 --dry-run

Training must succeed before evaluation starts. Both use the same new run:
  output/TASK/msar_lno/PROFILE/coverage_floor/UTC_TIMESTAMP_PID
Paths follow this checkout's path.sh/environment, including CDLNO_RUNS_ROOT.
AirfRANS uses CUDA_VISIBLE_DEVICES; other tasks receive --gpu.
--dry-run creates no run and executes no Python task entry.
For custom entry options, use TASK.sh train/eval directly; see README.md.
USAGE
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

case "${1:-help}" in
    help|-h|--help) usage; exit 0 ;;
    darcy|elasticity|airfoil|pipe|ns|plasticity|car|airfrans) task="$1"; shift ;;
    *) fail "Unknown task: $1. Use --help for supported tasks." ;;
esac

profile=light
case "${1:-}" in light|full) profile="$1"; shift ;; esac
gpu="${MSAR_GPU:-0}"
dry_args=()
while (($#)); do
    case "$1" in
        --gpu)
            (($# >= 2)) || fail 'Missing value for --gpu.'
            gpu="$2"; shift 2 ;;
        --gpu=*) gpu="${1#*=}"; shift ;;
        --dry-run) dry_args=(--dry-run); shift ;;
        help|-h|--help) usage; exit 0 ;;
        *) fail "Unknown option/profile: $1. Use TASK [light|full] --gpu ID." ;;
    esac
done
[[ "$gpu" =~ ^[0-9]+$ ]] || fail 'GPU ID must be a non-negative integer (for example 0 or 1).'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../_common.sh"
run_root="$CDLNO_RUNS_ROOT"
[[ "$run_root" = /* ]] || run_root="$PWD/$run_root"
run="$run_root/$task/msar_lno/$profile/coverage_floor/$(date -u +%Y%m%dT%H%M%S%NZ)_$$"

launcher=(bash "$script_dir/$task.sh")
gpu_args=(--gpu "$gpu")
if [[ "$task" == airfrans ]]; then
    launcher=(env "CUDA_VISIBLE_DEVICES=$gpu" "${launcher[@]}")
    gpu_args=()
fi

printf 'MSAR-LNO | task=%s | profile=%s | GPU=%s | coverage=floor\n' "$task" "$profile" "$gpu"
printf 'Run directory: %s\n' "$run"
printf 'Step 1/2: training\n'
if "${launcher[@]}" train "${gpu_args[@]}" \
    --profile "$profile" --coverage-mode floor \
    --coverage-weight 0.01 --coverage-kappa 0.2 \
    --msar-run-dir "$run" "${dry_args[@]}"; then
    printf 'Step 2/2: evaluation\n'
else
    status=$?
    printf 'Training failed (exit %s); evaluation was not started.\nRun directory: %s\n' "$status" "$run" >&2
    exit "$status"
fi
"${launcher[@]}" eval "${gpu_args[@]}" --msar-run-dir "$run" "${dry_args[@]}"
