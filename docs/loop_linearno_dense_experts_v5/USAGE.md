# V5 usage

The public architecture selector is `partial_share_feature_gate_v5`. Use the
dedicated launchers from the repository root. The default is
`p2_c2_r2_s2`, `K=2`, profile-derived `F`, profile-native `M/heads/C`, and
`operator_1_expert_1_over_r`. The current default is
`core_norm_mode=visit_independent`: core LN1/LN2 affine parameters are owned by
`(physical position, visit)`.

## Environment

```bash
cd /absolute/path/to/CDLNO
export CDLNO_DATA_ROOT=/absolute/path/to/data
export CDLNO_RUNS_ROOT="$PWD/output"
# Optional when a specific interpreter is required:
export CDLNO_PYTHON=/absolute/path/to/python
```

Standard tasks use `$CDLNO_DATA_ROOT/fno`, except Plasticity defaults to
`$CDLNO_DATA_ROOT/fno/plas_N987_T20.mat`. AirfRANS defaults to
`$CDLNO_DATA_ROOT/AirfRANS/Dataset`; Car raw/cache defaults are under
`$CDLNO_DATA_ROOT/mlcfd_data`. Existing task-specific path flags may be passed
after the launcher options.

## Eight entry points

| Task | Launcher |
|---|---|
| Airfoil | `tran_evaluate/linearno_loop_v5/airfoil.sh` |
| Darcy | `tran_evaluate/linearno_loop_v5/darcy.sh` |
| Elasticity | `tran_evaluate/linearno_loop_v5/elasticity.sh` |
| Pipe | `tran_evaluate/linearno_loop_v5/pipe.sh` |
| Navier-Stokes | `tran_evaluate/linearno_loop_v5/ns.sh` |
| Plasticity | `tran_evaluate/linearno_loop_v5/plasticity.sh` |
| AirfRANS | `tran_evaluate/linearno_loop_v5/airfrans.sh` |
| ShapeNet-Car | `tran_evaluate/linearno_loop_v5/car.sh` |

Each accepts `train`, `train_eval`, `resume`, and `eval`. `train_eval` starts
evaluation only after successful training and keeps the selected GPU and exact
run directory.

## Common commands

Default P2, K=2, profile-derived F, train then evaluate:

```bash
bash tran_evaluate/linearno_loop_v5/airfoil.sh train_eval \
  --seed 0 --gpu 0
```

Explicit P1 with independently selected K=3 and nondefault F=96:

```bash
bash tran_evaluate/linearno_loop_v5/darcy.sh train_eval \
  --topology p1_c3_r2_s1 \
  --expert-count 3 \
  --expert-width 96 \
  --seed 1 --gpu 1
```

The dedicated launcher accepts `--core-norm-mode visit_independent|shared`.
Omitting it for a new run selects `visit_independent`. `shared` retains the
pre-increment V5 core norm sharing semantics:

```bash
bash tran_evaluate/linearno_loop_v5/airfoil.sh train_eval \
  --topology p1_c3_r2_s1 --expert-count 3 \
  --core-norm-mode shared --seed 0 --gpu 0
```

K=2 with profile-derived F is selected by omitting `--expert-width`:

```bash
bash tran_evaluate/linearno_loop_v5/elasticity.sh train \
  --topology p2_c2_r2_s2 --expert-count 2 --seed 2 --gpu 0
```

Custom `P=1,C=2,R=3,S=1`:

```bash
bash tran_evaluate/linearno_loop_v5/pipe.sh train_eval \
  --topology custom \
  --prefix-blocks 1 --core-blocks 2 --loop-repeats 3 --suffix-blocks 1 \
  --expert-count 4 --expert-width 128 --seed 0 --gpu 0
```

The executed-depth shorthand uses `P=S=2,R=2,C=(L-4)/2`:

```bash
bash tran_evaluate/linearno_loop_v5/ns.sh train \
  --executed-depth 12 --expert-count 2 --seed 0 --gpu 1
```

Resume and evaluation require the exact existing run. Saved metadata supplies
the architecture; explicit conflicting topology/K/F/rank/profile values fail
before weights are loaded. An explicit conflicting `--core-norm-mode` also
fails before tensor loading. Current pair-v2 code intentionally rejects old
pair-v1 V5 checkpoints; continue those experiments with their original
checkout. No optimizer/RNG-perfect pair-v1 migration is provided.

```bash
bash tran_evaluate/linearno_loop_v5/airfoil.sh resume \
  --experiment-dir /absolute/path/to/run --checkpoint latest --gpu 0

bash tran_evaluate/linearno_loop_v5/airfoil.sh eval \
  --experiment-dir /absolute/path/to/run --checkpoint final --gpu 0
```

Parser-only preview performs no data read, tensor construction, weight load, or
run creation:

```bash
bash tran_evaluate/linearno_loop_v5/car.sh train \
  --dry-run --seed 0 --gpu 1

bash tran_evaluate/linearno_loop_v5/airfrans.sh train \
  --print-config --topology p1_c3_r2_s1 --expert-count 2 --seed 0 --gpu 0

bash tran_evaluate/linearno_loop_v5/plasticity.sh train \
  --print-run-dir --seed 2 --gpu 0
```

To preview all tasks:

```bash
for task in airfoil darcy elasticity pipe ns plasticity airfrans car; do
  bash "tran_evaluate/linearno_loop_v5/$task.sh" train \
    --dry-run --seed 0 --gpu 0
done
```

`K` and `F` are independent. Changing either does not change native latent rank
`M`, attention heads, hidden width, profile, or data/training protocol. The
default `F` is `C * ffn_ratio` from the selected pure LinearNO profile, not a
global task constant.
