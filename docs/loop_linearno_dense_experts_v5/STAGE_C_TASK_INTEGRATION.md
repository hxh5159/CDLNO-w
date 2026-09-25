# Stage C: eight-task native integration

## A. Authorized scope and status

Status: **PASS** for parser/factory and controlled synthetic native-task
integration. No real benchmark data was read.

## B. Files and reasons

- `cdlno/linearno/standard_entry.py`: forwards only explicitly selected V5
  constructor kwargs to the versioned loop entry.
- `cdlno/linearno_loop/{standard_entry,industrial_entry,air_entry,car_entry}.py`:
  V5 selection, saved-metadata restoration, V5 run directory ownership, and
  reuse of existing task Run classes.
- `tran_evaluate/linearno_loop_v5/launch.py`, `_task.sh`, and eight task shell
  wrappers: one shared implementation with direct task entry points.
- `tests/loop_linearno_v5/{standard_worker,test_v5_native_standard,
  test_v5_production}.py`: native experiment-AST and industrial task tests.

## C. Task semantics preserved

| Task family | V5 wrapper | Preserved native boundary |
|---|---|---|
| Airfoil, Darcy, Elasticity, Pipe | `LoopedStandardModelV5` | original input/grid/reference lifting, loss, optimizer/scheduler, relative-L2 eval and plots |
| Navier-Stokes | `LoopedStandardModelV5` | ten truth-fed training calls and prediction-fed ten-step evaluation |
| Plasticity | `LoopedStandardModelV5` | twenty time-conditioned updates and original scheduler cadence |
| AirfRANS | `LoopedAirfRANSModelV5` | x7/pos2, sampling/radius graph, weighted volume+surface loss, ensemble/member state and original metrics |
| ShapeNet-Car | `LoopedShapeNetModelV5` | `(data,geom)`, single graph, fold, `[N,4]`, surface/drag outputs and trusted whole-object compatibility boundary |

The selected pure profile remains authoritative for data, objective,
training, hidden width, heads, native variant and base latent rank. V5 only
adds topology, `expert_count`, `expert_width`, and its fixed residual rule.

## D. Commands and evidence

Eight real parser dry-runs were executed through the shell wrappers:

```bash
for task in airfoil darcy elasticity pipe ns plasticity airfrans car; do
  bash "tran_evaluate/linearno_loop_v5/$task.sh" train --dry-run --seed 23 --gpu 0
done
```

Result: **8/8 PASS**. The machine-readable plans are in
`evidence/stage_e/eight_task_dry_run.log`; no data, tensor, weight, or run
directory was accessed/created.

Controlled lifecycle tests cover all six Standard experiment ASTs and native
AirfRANS weighted loss and Car PyG graph functions. The final V5 suite result
is `48 passed` in `evidence/stage_e/v5_full_pytest_final.log`.

## E. Frozen-region evidence

The six `exp_*.py` scientific bodies, task readers, losses, normalization,
rollouts, optimizers, schedulers, and visualization math were not edited.
AirfRANS and Car retain their native train/eval modules; the V5 branches only
select a model and versioned run/checkpoint adapter. V4 launchers continue to
force V4 and were verified by `136 passed, 1 skipped`.

## F. Limits and priority review points

All task training here used controlled synthetic tensors and the original loss
functions. Real data paths, full sampling, field metrics, convergence, remote
Python 3.10/torch 2.11, distributed execution, and full epochs are NOT RUN.
Priority review points are Standard kwargs routing, NS/Plasticity call counts,
AirfRANS member independence, Car fold/single-graph handling, and saved-config
selection before tensor load.

This Stage C is complete; Stage D was executed under the user's continuous authorization.
