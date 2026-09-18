# LinearNO propagation-kernel monitor

This directory contains an opt-in monitor for the LinearNO baseline. It does
not change `cdlno/linearno`, any benchmark entry point, losses, optimizers,
checkpoints, random seeds, or the existing visualization output. The monitor
is activated only by `monitor/run.py`, which starts the requested existing
launcher in a child process and injects a runtime hook through
`LINEARNO_MONITOR_CONFIG`.

For every captured evaluation forward, each LinearNO attention block produces
detached factors `Q_l, K_l` with shape `[B,H,N,R]`. The monitor computes

```text
P_l = Q_l K_l^T
<P_i,P_j> = tr((Q_i^T Q_j)(K_j^T K_i))
rho_ij = <P_i,P_j> / (||P_i||_F ||P_j||_F + epsilon)
```

The implementation uses only Gram matrices and therefore does not construct an
`N x N` tensor. It averages the resulting matrix over retained batch samples
and heads for the heatmap, while preserving the per-sample/per-head values in
`similarity_values.npz` and `similarity.csv`. A first snapshot is captured,
then every `--capture-every-validations` observed evaluation forward. At most
`--max-samples-per-snapshot` batch samples are retained. The monitor has no
trainable parameters and does not return auxiliary values from the model.

## Airfoil, Darcy, Elasticity and Pipe

Run on the remote host from the **LinearNO-monitor** checkout below, with your
working Python/PyTorch environment already activated. Bind the checkout and
interpreter explicitly so variables inherited from the old checkout cannot
redirect the launchers. Data defaults still come from `path.sh` at
`/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data`.
Use a new, nonexistent monitor output directory for every command. The task launcher
still owns its normal `--experiment-dir`, checkpoint and visualization paths.
The first command observes validation forwards performed during training; the
second observes the explicit evaluation process and writes a comparable
snapshot archive.

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
export CDLNO_REPO_ROOT="$PWD"
export CDLNO_RUNS_ROOT="$PWD/output"
export CDLNO_PYTHON="$(command -v python)"
source ./path.sh

PROFILE=paper_table8_on_release_model   # or official_release
GPU=0                                   # change to 1 for GPU 1
SEED=0
RUN_ROOT="$CDLNO_RUNS_ROOT/linearno/$PROFILE/seed${SEED}_$(date -u +%Y%m%dT%H%M%S%NZ)"
printf 'Keep this RUN_ROOT for evaluation: %s\n' "$RUN_ROOT"

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-airfoil-train" \
  --capture-every-validations 10 --max-samples-per-snapshot 8 -- \
  bash tran_evaluate/linearno/airfoil_train.sh \
  --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/airfoil" &&

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-airfoil-eval" -- \
  bash tran_evaluate/linearno/airfoil_eval.sh --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/airfoil"

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-darcy-train" \
  --capture-every-validations 10 --max-samples-per-snapshot 8 -- \
  bash tran_evaluate/linearno/darcy_train.sh \
  --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/darcy" &&

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-darcy-eval" -- \
  bash tran_evaluate/linearno/darcy_eval.sh --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/darcy"

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-elasticity-train" \
  --capture-every-validations 10 --max-samples-per-snapshot 8 -- \
  bash tran_evaluate/linearno/elasticity_train.sh \
  --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/elasticity" &&

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-elasticity-eval" -- \
  bash tran_evaluate/linearno/elasticity_eval.sh --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/elasticity"

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-pipe-train" \
  --capture-every-validations 10 --max-samples-per-snapshot 8 -- \
  bash tran_evaluate/linearno/pipe_train.sh \
  --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/pipe" &&

"$CDLNO_PYTHON" -m monitor.run --output "$RUN_ROOT/monitor-pipe-eval" -- \
  bash tran_evaluate/linearno/pipe_eval.sh --gpu "$GPU" \
  --experiment-dir "$RUN_ROOT/pipe"
```

The existing launchers reject a pre-existing training directory according to
the normal artifact policy. Each `&&` runs evaluation only after successful
training. Keep `RUN_ROOT` unchanged for evaluation, including after opening a
new terminal; do not generate a new timestamp for an existing checkpoint.
For repeated evaluation, use a new monitor output directory as well.

`--gpu 0`/`--gpu 1` is the task's existing selector: these Standard entries set
`CUDA_VISIBLE_DEVICES` directly to the supplied string. Use `GPU=1` to select
GPU 1 in the same allocation; do not combine `CUDA_VISIBLE_DEVICES=1` with
`--gpu 0` expecting an index remap. The monitor wrapper and task both use
`CDLNO_PYTHON` from the activated environment. On a scheduled node, use only
the device(s) assigned to the job.

The interval 10 above means the first and every tenth **eval-mode forward**,
not every tenth epoch. The sample limit applies within that forward's batch;
it does not accumulate eight separate batch-1 calls into one snapshot.

For a data-free launcher preview, append `--dry-run` to the task launcher
directly (no need to create monitor output):

```bash
for task in airfoil darcy elasticity pipe; do
  bash "tran_evaluate/linearno/${task}_train.sh" --dry-run \
    --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" \
    --experiment-dir "$RUN_ROOT/$task"
  bash "tran_evaluate/linearno/${task}_eval.sh" --dry-run \
    --gpu "$GPU" --experiment-dir "$RUN_ROOT/$task"
done
```

These commands have been checked locally for path relocation and launcher
arguments. Remote dataset availability, GPU allocation and actual training
are not verified by a dry run.

`monitor/run.py` returns the child exit code. Monitoring failures are recorded
under `monitor_error.json` and never change the benchmark's model result; a
benchmark failure still returns non-zero. Each snapshot contains
`similarity_values.npz`, `similarity.csv`, `metadata.json`, and, when
Matplotlib is available, `similarity_heatmap.png/.pdf/.svg`. Use
`--no-plots` on a training node without Matplotlib and regenerate later:

```bash
python -m monitor.plot_snapshot \
  "$RUN_ROOT/monitor-airfoil-eval/snapshots/validation_000001"
```

No real data or training was run while adding this monitor. Data-free tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s monitor -p 'test_*.py' -v
```
