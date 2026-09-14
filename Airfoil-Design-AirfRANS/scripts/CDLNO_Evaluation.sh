#!/usr/bin/env bash
set -euo pipefail

# Run from Airfoil-Design-AirfRANS. Evaluation --my_path IS the PARENT of Dataset,
# e.g. --my_path /data/naca, unlike training's /data/naca/Dataset.
# Pass --run_dir <printed training run>. Match task/nmodel/budget/architecture;
# this loads the original whole-model list, then calls the unchanged evaluator.
python main_evaluation.py \
  --model CDLNO --task full --nmodel 1 --weight 1 \
  --n_hidden 256 --n_layers 8 --n_heads 8 --slice_num 64 \
  --front_blocks 2 --mlp_ratio 2 --latent_ffn_ratio 2 --dropout 0 \
  --cdpa_mode entry --cdpa_source_chunk_size 0 \
  --nb_epochs 398 --batch_size 1 --lr 0.001 \
  "$@"
