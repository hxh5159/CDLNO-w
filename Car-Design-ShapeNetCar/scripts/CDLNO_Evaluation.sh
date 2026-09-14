#!/usr/bin/env bash
set -euo pipefail

# Pass --run_dir <the training run printed by CDLNO.sh>.
# For a custom architecture/fold/epoch, repeat those values after the defaults.
python main_evaluation.py \
  --cfd_model CDLNO \
  --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64 \
  --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0 \
  --cdpa_mode entry --cdpa_source_chunk_size 0 \
  --fold_id 0 --nb_epochs 200 --weight 0.5 --gpu 0 \
  "$@"
