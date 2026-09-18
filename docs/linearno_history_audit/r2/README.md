# R2 audit record

Date: 2026-09-18. Repository: `/home/hwz/CDLNO`, HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`.

Full logs and JSON metrics are kept outside the repository at `/home/hwz/CDLNO-artifacts/linearno-history-r2-before-13jm89as/`.

## Commands

Core synthetic suite:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LINEARNO_R2_REPORT=/home/hwz/CDLNO-artifacts/linearno-history-r2-before-13jm89as/core-results.json python -B -m unittest linearno.test_history_core -v
```

It ran 11 tests in 37.074 seconds and passed all 11. The fixture does not read benchmark data or invoke a benchmark launcher.

Old regression selection:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest linearno.test_history_schema linearno.test_schema linearno.test_profiles linearno.test_attention_parity linearno.test_attention_structure linearno.test_standard_model linearno.test_standard_structure linearno.test_airfrans_model linearno.test_shapenet_model linearno.test_legacy linearno.test_rng monitor.test_monitor -v
```

This ran 66 tests: 59 passed, 4 CUDA tests skipped because CUDA was intentionally hidden, and 3 pre-existing R0 freeze assertions failed for README, `path.sh`, and the old reproduction-matrix prefix/hash. No R2 file is implicated by those failures; see the R0 baseline record.

## Scope boundary

No A/K operation, history dropout, gate, K correction, factory/CLI wiring, real data, real training, GPU, AMP, or later research phase was run. Context traces are explicit test observers and are not retained by default forward calls.

## New-file hashes

The implementation and fixture hashes at completion are recorded by:

```text
find cdlno/linearno_history PDE-Solving-StandardBenchmark/model/LinearNO_History.py tests/linearno/fixtures/history_r2_temporal.json tests/linearno/test_history_core.py -type f -print0 | xargs -0 sha256sum
```
