#!/usr/bin/env bash
set -euo pipefail
cdlno_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cdlno_repo_root="$(cd -- "$cdlno_script_dir/.." && pwd)"
source "$cdlno_script_dir/_common.sh"
case "${1:-help}" in
    train) cdlno_entry=main.py ;;
    eval) cdlno_entry=main_evaluation.py ;;
    help|-h|--help)
        cat <<'USAGE'
Usage: bash tran_evaluate/airfrans.sh train|eval [entry options] [--dry-run]
CDLNO: d256/h8/L8/F2/M64, FFN ratios2, entry CDPA, chunk0.
Protocol: full/nmodel1, 398epochs/batch1/lr0.001, original sampling/metrics.
Paths from path.sh: verify CDLNO_AIRFRANS_DATASET with inspect_data.sh.
TRAIN --my_path means Dataset itself (contains manifest.json).
EVAL --my_path means Dataset's PARENT; Dataset must be named exactly Dataset.
Default: output/airfrans/UTC_TIMESTAMP; eval needs an existing --run_dir.
Use --run_dir for another run; repeat task/model/budget settings during eval.
Original entry has no --gpu; use CUDA_VISIBLE_DEVICES to choose a GPU.
Arguments supplied last win. --dry-run never calls Python/data/metrics.
USAGE
        exit 0 ;;
    *) printf 'Expected train, eval or help.\n' >&2; exit 2 ;;
esac
cdlno_action="$1"
shift
cdlno_run="$(cdlno_run_path airfrans "$cdlno_action" "$@")"
cdlno_dataset="${CDLNO_AIRFRANS_DATASET%/}"
cdlno_extra=()
if [[ "$cdlno_entry" == main.py ]]; then
    cdlno_data_arg="$cdlno_dataset"
    cdlno_extra=(--score 0)
else
    # The frozen evaluator appends '/Dataset' to --my_path.
    cdlno_explicit_path=0
    for cdlno_arg in "$@"; do
        case "$cdlno_arg" in --my_path|--my_path=*) cdlno_explicit_path=1 ;; esac
    done
    if [[ "${cdlno_dataset##*/}" != Dataset ]] && ((cdlno_explicit_path == 0)); then
        printf 'AirfRANS eval requires CDLNO_AIRFRANS_DATASET ending in /Dataset (original path contract).\n' >&2
        exit 2
    fi
    cdlno_data_arg="$(dirname -- "$cdlno_dataset")"
fi
cdlno_dispatch Airfoil-Design-AirfRANS "$cdlno_entry" \
    --model CDLNO --task full --nmodel 1 --weight 1 \
    --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64 \
    --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0 \
    --cdpa_mode entry --cdpa_source_chunk_size 0 \
    --nb_epochs 398 --batch_size 1 --lr 0.001 \
    --my_path "$cdlno_data_arg" --save_path "$CDLNO_RUNS_ROOT" \
    --run_dir "$cdlno_run" \
    "${cdlno_extra[@]}" "$@"
