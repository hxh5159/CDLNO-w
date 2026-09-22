# LAA9 — V3 cost, matrix and synthetic verification

Status: **PARTIAL for the full regression scope; PASS for the new LAA9 accounting and synthetic scope.**

This phase did not read any dataset, start an experiment entry point, run a
complete epoch, or modify an old golden/tolerance. The implementation source
was the current working tree on 2026-09-22. The machine was Python 3.13.9,
Torch 2.13.0+cu130, one NVIDIA GeForce RTX 5090 Laptop GPU; `torch_cluster`
was not needed by these tensor-only checks.

## Changes

- `tools/linearno_loop_accounting.py` now has an independent `analytic_v3`
  dispatch for `loop_linearno_latent_adapter_v3`. V1/V2 branches and their
  default `B=1` semantics are unchanged. `measured_parameters` now exposes
  V3 `stem/time/prefix/shared_core/suffix/head/latent/adapter/router` groups.
- `tools/linearno_loop_laa9.py` generates the complete parse matrix, the D12
  real-module matrix, the independent 8→12/deeper comparison, a synthetic
  train/AMP/reload matrix, a shape trace and an optional canonical-N timing
  smoke. It never imports a task entry or opens data.
- `tests/loop_linearno_latent_adapter/test_laa9_accounting.py` covers the
  768-row axes, V3 accounting dispatch, feature-off state keys, router
  ownership, and V2 dispatch isolation.

Machine-readable evidence is in this directory:

- `parsed-matrix.json`: 768/768 rows (8 tasks × 2 profiles × 4 depths × 3
  residuals × 4 ablations), with resolved widths, M, unique/executed depth,
  parameter groups, matrix MAC/FLOPs, router source counts and excluded scalar
  operations.
- `d12-instances.json`: 192/192 D12 instances with actual module parameters,
  state-key counts, feature/router key presence, forward schedule and output
  shape. The measured partition equals the analytic partition in every row.
- `table-8-to-12-and-deeper.json`: exact baseline and V3 counts/ratios.
- `shape-trace.json`: real ATen shape trace for synthetic Elasticity B=1,N=6.
- `synthetic-suite.json`: controlled forward/backward/AdamW/strict-reload
  results and adapter Q/K delta dtypes.
- `performance-smoke.json`: one synchronized synthetic canonical-N/full-width
  Elasticity D12 GPU smoke; it is not an epoch estimate.

## Frozen cost comparison

The independent baseline formula reproduces the frozen original LinearNO
anchors. For the principal 8→12 comparison, the exact V3 totals are:

| Task | matched params / matrix MAC | matched param% / MAC% | efficient params / matrix MAC | efficient param% / MAC% |
|---|---:|---:|---:|---:|
| Airfoil | 1,762,177 / 93,392,581,984 | 99.7898 / 102.7607 | 1,397,281 / 80,667,249,792 | 79.1262 / 88.7589 |
| Darcy | 1,762,385 / 59,980,704,736 | 99.7871 / 102.9427 | 1,397,473 / 51,787,595,392 | 79.1256 / 88.8812 |
| Elasticity | 584,737 / 798,824,064 | 99.9180 / 98.2793 | 466,961 / 692,574,496 | 79.7928 / 85.2075 |
| Pipe | 1,762,177 / 137,746,032,544 | 99.7898 / 102.6542 | 1,397,281 / 119,004,753,792 | 79.1262 / 88.6874 |
| Plasticity | 1,797,788 / 52,889,194,112 | 99.9089 / 103.0369 | 1,416,260 / 45,582,031,360 | 78.7061 / 88.8013 |
| Navier–Stokes | 3,382,705 / 30,018,895,872 | 100.1416 / 100.0918 | 2,710,081 / 25,781,862,400 | 80.2293 / 85.9643 |
| AirfRANS | 3,353,828 / 116,267,917,312 | 99.8523 / 99.7671 | 2,695,748 / 99,906,715,648 | 80.2595 / 85.7279 |
| ShapeNet-Car | 3,848,516 / 132,807,405,376 | 99.8987 / 99.8275 | 2,967,684 / 113,934,184,192 | 77.0343 / 85.6411 |

All eight rows match the frozen 8→12 table to the reported precision. The
recomputed ranges are:

| Comparison | matched parameter ratio | matched matrix-MAC ratio | efficient parameter reduction | efficient matrix-FLOP reduction |
|---|---:|---:|---:|---:|
| 8→12 | 99.7871–100.1416% | 98.2793–103.0369% | 19.7405–22.9657% | 11.1188–14.7925% |
| 12→20 | 99.7787–100.3629% | 97.0901–103.6538% | 17.2925–24.9449% | 11.4728–16.6279% |
| 16→28 | 99.7493–100.2146% | 100.4691–104.9915% | 19.0771–23.6942% | 10.4515–14.8773% |
| 32→60 | 99.1644–100.1169% | 95.7899–100.5501% | 19.0420–27.2533% | 15.3033–19.2259% |

`matched_v1` is therefore approximate, not exact. `efficient_v1`'s headline
19.74–22.97% parameter and 11.12–14.79% matrix-FLOP reductions apply only to
the complete on/on, rank-4/alpha-4, SR 8→12 matrix. RB/LB router contraction
costs and all ablations are reported separately in `parsed-matrix.json`.

Matrix FLOPs mean `2 × MAC` for linear/convolution/einsum matrix work only.
Softmax, LayerNorm/RMSNorm, GELU, bias/residual arithmetic, dropout,
temperature/clamp, indexing and router scalar normalization are excluded and
listed under `non_matrix`. This is not a complete profiler FLOP count and is
not a latency or epoch-efficiency claim.

For SR the router count is zero. RB/LB use the frozen source schedules; the
logical router contraction is `2*B*N*H*sum(source_counts)`, while the runtime
identity source (`S=1`) can be skipped and is retained as a separate logical
versus executed accounting field. No router is registered for SR.

## Synthetic and timing evidence

- D12 actual construction and per-instance ATen trace: 192/192; all parameter
  partitions, state keys and synthetic matrix MACs agree exactly with the V3
  oracle, and no trace contains N×N/M×M attention. Prefix/core/suffix calls are
  2/8/2 for SR/LB. RB's controlled `VisitBody` path bypasses the physical
  block wrapper, so its observed wrapper calls are 2/0/2 while its logical
  core schedule remains 8 visits; this distinction is recorded rather than
  hidden.
- Synthetic optimizer/AMP/reload: 144/144 passed: CPU FP32 36, CUDA FP32 36,
  CUDA FP16 autocast 36 and CUDA BF16 autocast 36. This includes all three
  residual modes and all four ablations for the selected Elasticity, NS and
  AirfRANS matrix. Adapter Q/K delta dtypes were equal in every active case;
  RB cases exercised the existing mixed-source dtype helper. No persistent
  feature state was retained between forwards.
- Shape trace: 36,194,112 matrix MAC in both the independent oracle and ATen
  trace, 24 expected KᵀV/QC contractions, no forbidden N×N or M×M attention.
- Synchronized GPU smoke (Elasticity, canonical N=972, full matched D12,
  FP32): forward median/p90 5.194/5.651 ms; train-step median/p90
  20.395/20.982 ms; peak allocated/reserved 235,266,560/262,144,000 bytes;
  serialized state dict 2,408,973 bytes. These numbers are synthetic single-case
  measurements and cannot be extrapolated to an epoch or real data.

The optional detached diagnostic requested by the phase was not added; the
existing opt-in accounting/shape trace is observational and leaves no hooks or
graphs after return.

## Regression record

New LAA9 test file: **4/4 passed**. Existing targeted suites were run with
their required local `PYTHONPATH` isolation:

- Combined LAA1–LAA9 collection: **115 passed, 1 skipped, 1 import-order
  failure**. Adding the Standard project to `PYTHONPATH` first caused the LAA7
  Air parser helper's unqualified `cdlno_entry` import to resolve to Standard.
  A fresh isolated LAA7 run using its recorded command passed **6** and skipped
  **1** (`torch_cluster` missing). This is recorded as an invocation collision,
  not converted into a model pass.
- `tests/loop_linearno_ffn`: **34 passed**.
- `tests/loop_linearno`: **102 passed, 5 existing failures**. The failures
  are the historical source/provenance projection and exact fresh-process
  floating-point checks (`test_isolation`, launcher frozen hash,
  `test_modes`, `test_rb_core`, and standard-entry provenance), not new V3
  accounting assertions; no golden or tolerance was changed.
- `tests/linearno`: **151 passed, 4 existing failures** in 1479.00s. Two are
  old source/provenance projection mismatches for `standard_entry.py`; two are
  `FileNotFoundError` from the previously deleted
  `docs/CDLNO_EXPERIMENT_OUTPUTS.md` and `docs/kcdno_audit/audit_static.py`.
  No model calculation failed. The earlier broader accepted baseline remains
  644 passed, 9 failed and 36 skipped; its other failure/skip sources remain
  preserved in LAA0/LF7 evidence.

An initial combined pytest invocation also produced four collection errors
because LF tests import their local `support` module by directory name. This
was an invocation-path error; the isolated LF run above is the authoritative
result.

No real data, complete epoch, remote Python 3.10/Torch 2.11+cu128, real task
metric, SOTA or epoch-efficiency result was run or inferred. The full LAA9
status is **PARTIAL** because the requested old suites retain nine historical
failures across the loop and pure/history subsets; the new V3 accounting and
synthetic checks are PASS.

本 LAA9 阶段结束，未执行下一阶段。
