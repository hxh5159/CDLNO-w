#!/usr/bin/env bash
set -euo pipefail
cdlno_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${cdlno_project_dir}/..${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "$cdlno_project_dir"
python exp_darcy.py \
  --model CDLNO --gpu 0 \
  --n-hidden 128 \
  --n-layers 8 \
  --n-heads 8 \
  --slice_num 64 \
  --mlp_ratio 2 \
  --front-blocks 2 \
  --latent-ffn-ratio 2 \
  --cdpa-mode entry \
  --cdpa-source-chunk-size 0 \
  --dropout 0.0 \
  --unified_pos 1 \
  --ref 8 \
  --lr 0.001 \
  --epochs 500 \
  --batch-size 4 \
  --weight_decay 1e-05 \
  --max_grad_norm 0.1 \
  --downsample 5 --ntrain 1000 --eval 1 \
  "$@"
