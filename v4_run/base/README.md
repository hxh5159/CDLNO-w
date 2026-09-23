# v4 base-temperature wrappers

These four wrappers reuse the v4 full-budget launcher and native task
protocols, but force `--temperature-mode base`. Training is followed by
evaluation only after training succeeds when using `train_eval`.
With no action argument, `train_eval` is used. Resume and eval also assert
`base` against saved metadata; selecting a dynamic-temperature run fails.

The architecture is still `resmlp_dual_temp_v4`: eight independent operators,
five ResMLP owners, and middle raw-branch scale `1/sqrt(2)`. No dynamic
temperature predictors are registered. Existing static Q/K temperature
parameters remain trainable in the `temp`/`conv_temp` attention variants.
All four tasks retain H=128, heads=8, M=64 and FFN ratio=1.

```bash
bash v4_run/base/airfoil.sh train_eval --seed 0 --gpu 0
bash v4_run/base/darcy.sh train_eval --seed 1 --gpu 1
bash v4_run/base/elasticity.sh train_eval --seed 2 --gpu 0
bash v4_run/base/pipe.sh train_eval --seed 0 --gpu 1
```

Using the remote checkout path previously supplied by the user (sync these
scripts and the v4 launcher there before running):

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v3
export CDLNO_REPO_ROOT="$PWD"
export CDLNO_RUNS_ROOT="$PWD/output/v4_base"
source ./path.sh
```

The wrappers use the task-specific paths from `path.sh`. A different data
layout can be supplied with `CDLNO_DATA_ROOT`, `CDLNO_FNO_ROOT`, or the task
specific `CDLNO_AIRFOIL_ROOT`, `CDLNO_DARCY_ROOT`,
`CDLNO_ELASTICITY_ROOT`, and `CDLNO_PIPE_ROOT` variables.
Use `--data_path "/exact/task directory"` for an explicit per-command override.

Three seeds per task, run sequentially within each loop:

```bash
for seed in 0 1 2; do
  bash v4_run/base/airfoil.sh train_eval --seed "$seed" --gpu 0 || break
done
for seed in 0 1 2; do
  bash v4_run/base/darcy.sh train_eval --seed "$seed" --gpu 1 || break
done
for seed in 0 1 2; do
  bash v4_run/base/elasticity.sh train_eval --seed "$seed" --gpu 0 || break
done
for seed in 0 1 2; do
  bash v4_run/base/pipe.sh train_eval --seed "$seed" --gpu 1 || break
done
```

When `CUDA_VISIBLE_DEVICES` is set, `--gpu` indexes that visible-device list.
For example, `CUDA_VISIBLE_DEVICES=1` requires `--gpu 0` to use physical GPU 1.
The automatic evaluation retains the physical GPU selected for training.

Append `--dry-run` to validate the actual parser and resolved metadata without
reading data or weights. Resume and evaluation require the exact run printed
by the successful training command:

```bash
bash v4_run/base/darcy.sh resume --experiment-dir "$RUN_DIR" --gpu 1 --then-eval
bash v4_run/base/darcy.sh eval --experiment-dir "$RUN_DIR" --gpu 1
```

## Verification (2026-09-23)

- All five shell files pass `bash -n`; `git diff --check` passes.
- Actual parser previews: 4 tasks x seeds 0/1/2 x GPUs 0/1 = 24 passed,
  with unique, uncreated run paths. Task data paths and paths containing
  spaces were checked. No real data or training was used.
- Base metadata previews for resume/eval pass; dynamic-temperature metadata
  is rejected for both actions without changing the sidecar.
- `PYTHONPATH=.:cdlno:PDE-Solving-StandardBenchmark pytest -q
  tests/linearno_loop_v4/test_v4_launchers.py`: 31 passed.
- GPU regression first reproduced two failures (GPU index 1 in a mask
  already reduced to one device). The v4 launcher now uses logical index 0
  in the selected training mask for automatic evaluation. Regression checks
  retain physical GPU 1, GPU 7 from mask `4,7`, and GPU 3 from mask `3`.
  The same correction also applies to `v4_run/full`.
