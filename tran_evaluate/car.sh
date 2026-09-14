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
Usage: bash tran_evaluate/car.sh train|eval [original entry options] [--dry-run]

CDLNO: d256/h8/L8/F2/M64, FFN ratios2, entry CDPA, chunk0.
Original Car protocol: fold0, epochs200, batch1, Adam/OneCycleLR, reg0.5.

Train: --data_dir RAW_ROOT --save_dir PREPROCESSED_ROOT --run_dir NEW_RUN
Eval:  --data_dir RAW_ROOT --save_dir PREPROCESSED_ROOT --run_dir EXISTING_RUN
--save_dir means the preprocessed dataset, NOT the model checkpoint directory.
Training uses existing preprocessed data by default (--preprocessed 1).

Full original drag evaluation requires fold0 and raw data accessible at
/data/PDE_data/mlcfd_data/training_data/param0 (hardcoded upstream path).
Repeat any custom model/fold/epoch settings during eval; chunk may change.
Arguments supplied last override defaults. --dry-run executes no Python/data.
Use CDLNO_PYTHON=/absolute/path/to/python to select your existing environment.
See tran_evaluate/README.md for paths, metrics and inherited limitations.
USAGE
        exit 0 ;;
    *) printf 'Expected train, eval or help; see %s help\n' "$0" >&2; exit 2 ;;
esac
shift

cdlno_args=(
    --cfd_model CDLNO
    --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64
    --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0
    --cdpa_mode entry --cdpa_source_chunk_size 0
    --fold_id 0 --nb_epochs 200 --weight 0.5 --r 0.2 --gpu 0
)
if [[ "$cdlno_entry" == main.py ]]; then
    cdlno_args+=(--batch_size 1 --lr 0.001 --val_iter 10 --preprocessed 1)
fi
cdlno_dispatch Car-Design-ShapeNetCar "$cdlno_entry" "${cdlno_args[@]}" "$@"
