#!/usr/bin/env bash
set -euo pipefail

cdlno_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cdlno_repo_root="$(cd -- "$cdlno_script_dir/.." && pwd)"
source "$cdlno_script_dir/_common.sh"

case "${1:-help}" in
    train) cdlno_eval=0 ;;
    eval) cdlno_eval=1 ;;
    help|-h|--help)
        cat <<'USAGE'
Usage: bash tran_evaluate/ns.sh train|eval [original exp options] [--dry-run]

CDLNO: d256/h8/L8/F2/M64, FFN ratios2, entry CDPA, chunk0, ref8.
Original NS protocol: 64x64, 10->10, epochs500, batch2, AdamW/OneCycleLR,
lr0.001, weight_decay1e-5, NO gradient clipping by default.
Training feeds back truth; evaluation feeds back predictions.

Train: --data_path DATA_ROOT --cdlno-run-dir NEW_RUN
Eval:  --data_path DATA_ROOT --cdlno-run-dir EXISTING_RUN
Required MAT: DATA_ROOT/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat
Repeat custom architecture settings during eval; chunk may change.
Arguments supplied last override defaults. --dry-run executes no Python/data.
Use CDLNO_PYTHON=/absolute/path/to/python to select your existing environment.
--gpu is forwarded unchanged to the original CUDA_VISIBLE_DEVICES handling.
See tran_evaluate/README.md for losses, checkpoints and validation limits.
USAGE
        exit 0 ;;
    *) printf 'Expected train, eval or help; see %s help\n' "$0" >&2; exit 2 ;;
esac
shift

# Omit --max_grad_norm: the CDLNO NS preset uses null, matching original None.
# Passing 0 would zero finite gradients; it is NOT a way to disable clipping.
cdlno_dispatch PDE-Solving-StandardBenchmark exp_ns.py \
    --model CDLNO --gpu 0 \
    --n-hidden 256 --n-layers 8 --n-heads 8 --slice_num 64 \
    --mlp_ratio 2 --front-blocks 2 --latent-ffn-ratio 2 \
    --cdpa-mode entry --cdpa-source-chunk-size 0 --dropout 0 \
    --unified_pos 1 --ref 8 --downsample 1 \
    --lr 0.001 --epochs 500 --batch-size 2 --weight_decay 1e-5 \
    --eval "$cdlno_eval" "$@"
