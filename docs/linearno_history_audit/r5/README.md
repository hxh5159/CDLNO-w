# R5 evidence

R5 is internal integration only. No task CLI/launcher, real data, GPU or long training.

- `baseline.json`: before-edit HEAD/tree, external snapshot, tracked/untracked/ignored counts.
- `environment.json`: actual CPU test software environment.
- `integration-results.json`: 20 configurations, fresh-process evaluation, six-variant gate decomposition/oracle, baseline train/eval parity and exact resume.
- `configuration_matrix.json`: completed core matrix; eight-task production rows remain NOT_STARTED. R1's original plan fixture stays unchanged.
- `numerical-summary.json`: absolute/relative numerical evidence and fixed tolerances.
- `regression-results.json`, `regression.txt`: R1–R5, pure LinearNO, original Transolver and monitor. Preserve pre-existing failing assertions rather than changing them.
- `integration-initial.txt`, `integration-second.txt`, `integration-third.txt`, `integration-final.txt`: development history; first namespace-package loader error fixed, then NumPy comparison fixture error fixed, then all tests passed. Final 11-test run includes protocol rejection and on-disk metadata corruption checks.
- `research-source.diff`: exact R5 changes to pre-existing research Python files; new config/factory/checkpoint/test files are separately listed in the report/freeze.
- `freeze.json`: end comparison to the genuine before-edit manifest, including untracked and ignored files.

Reproduce the new acceptance from the actual checkout (no GPU/data):

```bash
cd /home/hwz/CDLNO
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_integration -v
```

The complete regression module list is in `regression-results.json`; its three inherited failure assertions are already recorded in R0/R4. This audit is not a task training launch procedure.
