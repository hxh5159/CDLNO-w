#!/usr/bin/env bash
set -euo pipefail

# Run from Car-Design-ShapeNetCar; each invocation trains exactly one fold.
# A new unique run is printed, or pass --run_dir for an explicit new directory.
python main.py \
  --cfd_model CDLNO \
  --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64 \
  --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0 \
  --cdpa_mode entry --cdpa_source_chunk_size 0 \
  --fold_id 0 --nb_epochs 200 --batch_size 1 --lr 0.001 --weight 0.5 \
  --val_iter 10 --preprocessed 1 --gpu 0 \
  "$@"
