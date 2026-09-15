#!/usr/bin/env bash
# One task, then its evaluation. Reuse the existing launchers and protocols.
set -euo pipefail

cdlno_pair_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-help}" in
    darcy|elasticity|airfoil|pipe|ns|plasticity|car|airfrans) cdlno_pair_task="$1" ;;
    help|-h|--help)
        cat <<'USAGE'
Usage: bash tran_evaluate/train_eval.sh TASK [shared entry options] [--dry-run]
       [--train-args training-only options] [--eval-args evaluation-only options]

TASK: darcy elasticity airfoil pipe ns plasticity car airfrans
Runs TASK.sh train first, then TASK.sh eval only if training succeeds.
Both commands retain the existing task defaults (CDLNO L8/F2, full front).
Shared options, including custom architecture and run directory, go before
--train-args/--eval-args. These markers are consumed here, not sent to Python.
Example: train_eval.sh car --gpu 0 --train-args --preprocessed 1

Uses CDLNO_RUN_TAG if set; otherwise generates one tag for this train/eval pair.
Use --cdlno-run-dir (PDE) or --run_dir (industrial) as a shared option to select
an explicit NEW directory. Existing runs are rejected by the original helper.
Use TASK.sh eval separately to evaluate an already trained run.
--dry-run prints both commands without running Python or creating a run.

AirfRANS: use CUDA_VISIBLE_DEVICES, not --gpu. Set CDLNO_AIRFRANS_DATASET to
.../Dataset; the task launcher derives distinct train/eval --my_path arguments.
For explicit paths use --train-args --my_path .../Dataset and
--eval-args --my_path ... (the parent). Do not share --my_path for AirfRANS.
Car: full drag evaluation still requires fold0 and the original fixed raw path;
see tran_evaluate/README.md. This wrapper does not change metrics or data paths.
USAGE
        exit 0 ;;
    *) printf 'Unknown task: %s. Run train_eval.sh help.\n' "$1" >&2; exit 2 ;;
esac
shift

cdlno_pair_shared=()
cdlno_pair_train=()
cdlno_pair_eval=()
cdlno_pair_dry=()
cdlno_pair_section=shared
for cdlno_pair_arg in "$@"; do
    case "$cdlno_pair_arg" in
        --train-args) cdlno_pair_section=train; continue ;;
        --eval-args) cdlno_pair_section=eval; continue ;;
        --dry-run) cdlno_pair_dry=(--dry-run); continue ;;
        --eval|--eval=*)
            printf 'The train/eval action is controlled by this script; omit --eval.\n' >&2
            exit 2 ;;
        --run_dir|--run_dir=*|--cdlno-run-dir|--cdlno-run-dir=*)
            if [[ "$cdlno_pair_section" != shared ]]; then
                printf 'Put the run directory in shared options so both steps use the same checkpoint.\n' >&2
                exit 2
            fi ;;
        --my_path|--my_path=*)
            if [[ "$cdlno_pair_task" == airfrans && "$cdlno_pair_section" == shared ]]; then
                printf 'AirfRANS train/eval --my_path differ; use CDLNO_AIRFRANS_DATASET or separate --train-args/--eval-args.\n' >&2
                exit 2
            fi ;;
    esac
    case "$cdlno_pair_section" in
        shared) cdlno_pair_shared+=("$cdlno_pair_arg") ;;
        train) cdlno_pair_train+=("$cdlno_pair_arg") ;;
        eval) cdlno_pair_eval+=("$cdlno_pair_arg") ;;
    esac
done

# Export once for the two child processes, before their path.sh is sourced.
# A tag names the experiment; it does not set a random seed.
export CDLNO_RUN_TAG="${CDLNO_RUN_TAG:-$(date -u +%Y%m%dT%H%M%S%6NZ)}"
printf 'Task: %s | train/eval run tag: %s\n' "$cdlno_pair_task" "$CDLNO_RUN_TAG"
printf 'Step 1/2: training\n'
if bash "$cdlno_pair_dir/$cdlno_pair_task.sh" train \
        "${cdlno_pair_shared[@]}" "${cdlno_pair_train[@]}" "${cdlno_pair_dry[@]}"; then
    printf 'Step 2/2: evaluation\n'
else
    cdlno_pair_status=$?
    printf 'Training command failed (exit %s); evaluation was not started.\n' "$cdlno_pair_status" >&2
    exit "$cdlno_pair_status"
fi
bash "$cdlno_pair_dir/$cdlno_pair_task.sh" eval \
    "${cdlno_pair_shared[@]}" "${cdlno_pair_eval[@]}" "${cdlno_pair_dry[@]}"
