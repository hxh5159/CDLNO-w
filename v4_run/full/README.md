# v4 full-budget task wrappers

Each wrapper calls the existing v4 launcher and keeps the native task parser,
checkpoint, resume and evaluation path. With no action argument it runs
`train_eval`: evaluation starts only after training succeeds. The default v4
temperature mode is `latent_k_point_q`; pass `--temperature-mode
point_k_point_q` or `base` for the other registered variants.

```bash
bash v4_run/full/airfoil.sh train_eval --seed 0 --gpu 0
bash v4_run/full/darcy.sh train_eval --seed 1 --gpu 1
bash v4_run/full/elasticity.sh train_eval --seed 2 --gpu 0
bash v4_run/full/pipe.sh train_eval --seed 0 --gpu 1
```

The wrappers source the repository `path.sh` contract and inject the correct
task-specific `--data_path` when the caller does not provide one. Override
`CDLNO_DATA_ROOT` or a task-specific `CDLNO_*_ROOT` variable before sourcing
`path.sh` if the remote data layout differs.

For the remote checkout used by this repository:

```bash
REMOTE_REPO=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
cd "$REMOTE_REPO"
export CDLNO_REPO_ROOT="$PWD"
export CDLNO_RUNS_ROOT="$PWD/output/v4_full"
source ./path.sh

for seed in 0 1 2; do
  bash v4_run/full/airfoil.sh train_eval --temperature-mode latent_k_point_q --seed "$seed" --gpu 0 || break
done
for seed in 0 1 2; do
  bash v4_run/full/darcy.sh train_eval --temperature-mode latent_k_point_q --seed "$seed" --gpu 1 || break
done
for seed in 0 1 2; do
  bash v4_run/full/elasticity.sh train_eval --temperature-mode latent_k_point_q --seed "$seed" --gpu 0 || break
done
for seed in 0 1 2; do
  bash v4_run/full/pipe.sh train_eval --temperature-mode latent_k_point_q --seed "$seed" --gpu 1 || break
done
```

To run the point-K comparison, replace `latent_k_point_q` with
`point_k_point_q`; this creates distinct run directories and metadata hashes.
For a parser-only check, append `--dry-run`; it does not read data or weights.
To resume or evaluate, pass the exact printed `--experiment-dir`:

```bash
bash v4_run/full/darcy.sh resume --experiment-dir "$RUN_DIR" --gpu 1 --then-eval
bash v4_run/full/darcy.sh eval --experiment-dir "$RUN_DIR" --gpu 1
```
