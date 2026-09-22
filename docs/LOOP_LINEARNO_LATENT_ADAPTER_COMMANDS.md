# Looped LinearNO V3 Commands

LAA8 adds one shared launcher and two fixed cost-profile trees. The 16 required
entry points are:

~~~text
tran_evaluate/linearno_loop_v3/matched_v1/{airfoil,darcy,elasticity,pipe,ns,plasticity,airfrans,car}.sh
tran_evaluate/linearno_loop_v3/efficient_v1/{airfoil,darcy,elasticity,pipe,ns,plasticity,airfrans,car}.sh
~~~

matched_v1 is the default research profile. Both trees default to V3
operator_latent_adapter_v3, D12/P2-C4-R2-S2, SR, task-base M, latent on, and
the second-visit bilateral Q/K adapter with r=4 and alpha=4. A trailing
different cost-profile flag fails before native construction. User topology,
residual, feature and seed overrides are passed last and resolved by the V3
configuration layer.

## Actions

Every wrapper supports train, resume, eval, train_eval, dry-run, preview and
print-run-dir.

~~~bash
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train_eval --gpu 1 --seed 0
RUN="/absolute/output/darcy/linearno_loop/<saved-v3-run>"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh resume --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh eval --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh preview --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh dry-run --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh print-run-dir --gpu 1 --seed 0
~~~

train_eval starts eval only after train returns zero and uses the same run.
resume/eval require the exact existing run; no latest-run guessing. preview
runs the real parser and emits a JSON plan without data or weights. The run
name contains task, V3, profile, P/C/R/S and depth, residual, H/heads/M/Dz,
feature flags, adapter rank/alpha, seed and config hash. output-root and
data-root accept quoted paths containing spaces.

## matched_v1: all eight tasks

Each row is a train/resume/eval triplet. Replace RUN with that task's exact
saved path.

~~~bash
bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh resume --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh eval --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/pipe.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/pipe.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/pipe.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/ns.sh train --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/ns.sh resume --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/ns.sh eval --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/plasticity.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/plasticity.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/plasticity.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/airfrans.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/airfrans.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/airfrans.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh eval --gpu 0 --experiment-dir "$RUN"
~~~

## efficient_v1: all eight tasks

The efficient tree has the same triplets, with efficient_v1 in every path.

~~~bash
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/darcy.sh train --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/darcy.sh resume --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/darcy.sh eval --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/elasticity.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/elasticity.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/elasticity.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/pipe.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/pipe.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/pipe.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/ns.sh train --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/ns.sh resume --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/ns.sh eval --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/plasticity.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/plasticity.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/plasticity.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfrans.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfrans.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfrans.sh eval --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/car.sh train --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/car.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/car.sh eval --gpu 0 --experiment-dir "$RUN"
~~~

## Ablations, depth and custom

Flags are appended after defaults. The four ablations are latent-off/adapter-
off, latent-off/adapter-on, latent-on/adapter-off and latent-on/adapter-on.

~~~bash
bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh train_eval \
  --linearno-loop-latent 0 --linearno-loop-adapter-mode none --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train_eval \
  --linearno-loop-latent 1 --linearno-loop-adapter-mode none --gpu 1 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh train_eval \
  --linearno-loop-latent 0 \
  --linearno-loop-adapter-mode bilateral_qk_lowrank_second_visit \
  --linearno-loop-adapter-rank 4 --linearno-loop-adapter-alpha 4 --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/efficient_v1/pipe.sh dry-run \
  --linearno-loop-topology d20 \
  --linearno-loop-residual-mode lb_attnres_1_over_r --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/custom/darcy.sh dry-run \
  --linearno-loop-hidden-width 64 --linearno-loop-latent-width 128 \
  --linearno-rank 32 --linearno-loop-topology custom \
  --linearno-loop-prefix-blocks 2 --linearno-loop-core-blocks 2 \
  --linearno-loop-repeats 2 --linearno-loop-suffix-blocks 2 \
  --linearno-loop-residual-mode sr_1_over_r --linearno-loop-latent 0 \
  --linearno-loop-adapter-mode none --linearno-loop-adapter-rank 4 \
  --linearno-loop-adapter-alpha 4 --gpu 1 --seed 0
~~~

Custom requires explicit H, Dz, M, topology and adapter settings. It is
provided as an additional tree; the two required profile trees remain fixed.

## Three paired seeds

The following is a future loop, not automatic matrix execution.

~~~bash
for seed in 0 1 2; do
  bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh train_eval --gpu 0 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train_eval --gpu 1 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh train_eval --gpu 0 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/pipe.sh train_eval --gpu 0 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/ns.sh train_eval --gpu 1 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/plasticity.sh train_eval --gpu 0 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/airfrans.sh train_eval --gpu 0 --seed "$seed"
  bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh train_eval --gpu 0 --seed "$seed"
done
~~~

Use the same loop with efficient_v1 paths for the paired efficient profile. Do
not select a seed or checkpoint using test results.

## Remote checkout and paths

~~~bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin
source ./path.sh
export CDLNO_REPO_ROOT="$PWD"
export CDLNO_RUNS_ROOT="$PWD/output/linearno_loop_v3"
export CDLNO_FNO_ROOT="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno"
export CDLNO_AIRFRANS_DATASET="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/airfrans"
export CDLNO_CAR_RAW_ROOT="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/car/raw"
export CDLNO_CAR_CACHE_ROOT="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/car/cache"
bash tran_evaluate/linearno_loop_v3/matched_v1/airfoil.sh preview --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/airfrans.sh preview --gpu 0 --seed 0
bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh preview --gpu 0 --seed 0
~~~

The native output tree contains architecture.json, loop_run_manifest.json,
logs/status/epoch records, strict-pair checkpoints and weights, native
visualization/evaluation directories and the resolved config hash. preview,
dry-run and print-run-dir do not read datasets or weight tensors and do not
start a matrix.

This document describes commands only; real data, long training and remote
performance were not run.

