#!/usr/bin/env bash
set -euo pipefail

# Run from Airfoil-Design-AirfRANS. Training --my_path IS the Dataset directory,
# e.g. --my_path /data/naca/Dataset. Evaluation uses its parent (see other script).
# YAML supplies the preserved 398-epoch training protocol; CLI values below
# make that starting configuration explicit. User overrides are passed last.
python main.py \
  --model CDLNO --task full --nmodel 1 --weight 1 --score 0 \
  --save_path runs \
  --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64 \
  --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0 \
  --cdpa_mode entry --cdpa_source_chunk_size 0 \
  --nb_epochs 398 --batch_size 1 --lr 0.001 \
  "$@"
