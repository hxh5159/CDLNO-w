#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${project_dir}/..${PYTHONPATH:+:${PYTHONPATH}}"
cd "$project_dir"
python exp_plas.py --model CDLNO --gpu 0 --n-hidden 128 --n-layers 8 --n-heads 8 --slice_num 64 --mlp_ratio 2 --front-blocks 2 --latent-ffn-ratio 2 --cdpa-mode entry --cdpa-source-chunk-size 0 --dropout 0.0 --unified_pos 0 --ref 8 --lr 0.001 --epochs 500 --batch-size 8 --weight_decay 1e-5 --max_grad_norm 0.1 --eval 0 "$@"
