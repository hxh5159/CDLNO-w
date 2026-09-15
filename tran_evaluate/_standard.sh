#!/usr/bin/env bash
set -euo pipefail
cdlno_task="$1"
shift
cdlno_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cdlno_repo_root="$(cd -- "$cdlno_script_dir/.." && pwd)"
source "$cdlno_script_dir/_common.sh"

case "${1:-help}" in
    train) cdlno_eval=0 ;;
    eval) cdlno_eval=1 ;;
    help|-h|--help)
        cat <<USAGE
Usage: bash tran_evaluate/$cdlno_task.sh train|eval [entry options] [--dry-run]
CDLNO defaults: L8/F2, entry CDPA, chunk0, FFN ratios2; per-task d/h/M/batch.
Paths: root path.sh; override --data_path and --cdlno-run-dir as needed.
Default: output/$cdlno_task/UTC_TIMESTAMP; eval needs an existing --cdlno-run-dir.
Each invocation performs one action. Custom architecture requires a new run
and the same architecture during eval. Arguments supplied last win.
--dry-run prints the command without Python, data or filesystem writes.
See tran_evaluate/README.md; inspect_data.sh checks remote formats read-only.
USAGE
        exit 0 ;;
    *) printf 'Expected train, eval or help.\n' >&2; exit 2 ;;
esac
cdlno_action="$1"
shift

cdlno_run="$(cdlno_run_path "$cdlno_task" "$cdlno_action" "$@")"

# Explicit values match the already accepted CDLNO task presets/launchers.
cdlno_width=128 cdlno_heads=8 cdlno_latents=64 cdlno_unified=0
cdlno_clip=(--max_grad_norm 0.1)
case "$cdlno_task" in
    darcy)
        cdlno_entry=exp_darcy.py cdlno_batch=4 cdlno_unified=1
        cdlno_data="$CDLNO_DARCY_ROOT"
        cdlno_extra=(--downsample 5 --ntrain 1000) ;;
    elasticity)
        cdlno_entry=exp_elas.py cdlno_batch=1
        cdlno_data="$CDLNO_ELASTICITY_ROOT"
        cdlno_extra=(--ntrain 1000) ;;
    airfoil)
        cdlno_entry=exp_airfoil.py cdlno_batch=4 cdlno_heads=4
        cdlno_data="$CDLNO_AIRFOIL_ROOT"
        cdlno_extra=(--downsamplex 1 --downsampley 1) ;;
    pipe)
        cdlno_entry=exp_pipe.py cdlno_batch=8 cdlno_heads=4 cdlno_latents=32
        cdlno_data="$CDLNO_PIPE_ROOT"
        cdlno_extra=(--downsamplex 1 --downsampley 1) ;;
    ns)
        cdlno_entry=exp_ns.py cdlno_batch=2 cdlno_width=256 cdlno_unified=1
        cdlno_data="$CDLNO_NS_ROOT"
        cdlno_extra=(--downsample 1)
        # Existing NS preset null resolves to None; 0 is NOT disabled clipping.
        cdlno_clip=() ;;
    plasticity)
        cdlno_entry=exp_plas.py cdlno_batch=8
        cdlno_data="$CDLNO_PLASTICITY_FILE"
        cdlno_extra=() ;;
    *) printf 'Unknown standard task: %s\n' "$cdlno_task" >&2; exit 2 ;;
esac

cdlno_dispatch PDE-Solving-StandardBenchmark "$cdlno_entry" \
    --model CDLNO --gpu 0 \
    --n-hidden "$cdlno_width" --n-layers 8 --n-heads "$cdlno_heads" --slice_num "$cdlno_latents" \
    --mlp_ratio 2 --front-blocks 2 --latent-ffn-ratio 2 \
    --cdpa-mode entry --cdpa-source-chunk-size 0 --dropout 0 \
    --unified_pos "$cdlno_unified" --ref 8 \
    --lr 0.001 --epochs 500 --batch-size "$cdlno_batch" --weight_decay 1e-5 \
    "${cdlno_clip[@]}" "${cdlno_extra[@]}" \
    --data_path "$cdlno_data" --cdlno-run-dir "$cdlno_run" \
    --eval "$cdlno_eval" "$@"
