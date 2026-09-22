# LAA7: AirfRANS and ShapeNet-Car V3 adapters

## Result

**PASS for the LAA7 implementation and synthetic acceptance scope, with one explicit dependency-boundary skip.** V3 is routed only by the explicit V3 architecture/family metadata. AirfRANS and ShapeNet-Car retain their native data and scientific protocols; no real dataset or long training was run.

## Implementation

- `cdlno/linearno_loop/industrial_entry.py` resolves the V3 actual `M`, profile widths, topology and native industrial fields without changing the V1/V2 route. V3 rank multiplier is rejected rather than silently applied.
- `cdlno/linearno_loop/versioning.py` constructs V3 members through `build_from_config(..., initialization_seed=member_seed)`. AirfRANS members therefore receive distinct model objects, feature parameters and local initialization streams.
- `cdlno/linearno_loop/air_entry.py` keeps the native weighted objective (`MSE_weighted`, `reg=args.weight`), sampling/data manifest, member order and per-member checkpoint/RNG state. V3 metadata measures the constructed model and uses the V3 resume archive, while V1/V2 continue using their existing metadata path.
- `cdlno/linearno_loop/car_entry.py` keeps the native tuple `(cfd_data, geom)` input, single-graph and fold/surface/drag boundaries. V3 uses a strict versioned pair; the existing trusted whole-object/list compatibility remains on the old route.
- `cdlno/linearno_loop/v2_projection.py` contains the updated live source fingerprints needed to keep old V1/V2 provenance projection exact after the approved routing additions.
- The shared `linearno_loop/versioning.py` and `cdlno/linearno_loop/versioning.py` dispatch V3 only when its explicit architecture/metadata is present. The existing V3 industrial wrappers in `cdlno/linearno_loop/v3/airfrans.py` and `shapenet.py` are reused; they do not create a second industrial training framework.

## Formula and contract mapping

The V3 wrapper forwards the original AirfRANS and Car model contracts to the shared V3 construction path. The V3 core still owns full shared physical blocks, cross-round latent FFN and second-visit Q/K adapter; this phase does not alter those formulas. AirfRANS `N=32000` and Car `N=32186` appear only in `cost-matrix.json` as representative accounting points. They do not change native radius sampling, graph construction, fold selection, ensemble order, or evaluation sampling.

The analytic matrix accounting is in [cost-matrix.json](cost-matrix.json). It contains 192 rows for two tasks, two profiles, four formal depths, three residual modes and four latent/adapter ablations. Each row records stem/time, prefix, shared core, suffix/head, latent, adapter and router parameter groups; executed and unique depth; matrix MAC/FLOPs; router contraction MACs; source counts; and the excluded scalar operation inventory. `1 MAC = 2 FLOPs`. Softmax, temperature/clamp, LayerNorm, GELU, residual additions/scaling, dropout, reshaping/indexing, position/reference work and time encoding are explicitly outside the matrix-MAC total. No latency, memory or epoch-time claim is inferred from these numbers.

## Evidence

- The LAA7 test file ran 7 tests in 73.332 seconds: 6 passed, 0 failed, and 1 skipped. The skip is only the `torch_cluster` radius-graph boundary (`ModuleNotFoundError`); it was not installed or replaced.
- The parser matrix covered 2 profiles × 3 residuals × 4 ablations for both industrial tasks (48 rows). Nonzero Car `fold_id=3` was checked.
- The formal matched/efficient D12/D20/D28/D60 constructor matrix passed for both tasks. The controlled PyG checks passed V3 AirfRANS `[N,4]` output, Car tuple input and `[N,4]` output, single-graph validation, backward/AdamW, and independent ensemble member parameter sets.
- A native AirfRANS synthetic run used the actual weighted training call, wrote a checkpoint/ensemble and loaded it for evaluation. A native Car synthetic run wrote a checkpoint, restored the optimizer/scheduler boundary, and loaded the same pair for a finite `[N,4]` evaluation forward. A V3 pair was loaded in a fresh process with `strict=True`.
- Legacy industrial V2 regression passed 3/3, and the prior delivery compatibility regression passed 6/6. The existing LAA5/LAA6 reports contain the earlier wrapper and Standard evidence.

## Files

The LAA7 routing changes are in `linearno_loop/versioning.py`, `cdlno/linearno_loop/versioning.py`, `cdlno/linearno_loop/industrial_entry.py`, `cdlno/linearno_loop/air_entry.py`, `cdlno/linearno_loop/car_entry.py` and `cdlno/linearno_loop/v2_projection.py`. The V3 wrappers in `cdlno/linearno_loop/v3/airfrans.py` and `shapenet.py` are the reused versioned wrapper layer. The new focused test is `tests/loop_linearno_latent_adapter/test_laa7_industrial_entry.py`. The phase evidence is `start-manifest.json`, `results.json`, `static-checks.json` and `cost-matrix.json` in this directory. No existing task data, old checkpoint, golden or user file was deleted; the pre-existing modification to `check_checkpoints/check_pipe_loop_resume.sh` was preserved.

## Limits and review points

1. `torch_cluster` is absent in this environment, so the actual radius-graph dependency boundary is recorded as skipped. Native AirfRANS sampling was not represented as a real-data pass.
2. Synthetic PyG graphs establish shape, isolation, loss wiring and strict archive behavior, not VTK metrics, drag/field accuracy or convergence.
3. The local environment is Python 3.13.9 / Torch 2.13.0+cu130; remote Python 3.10 / Torch 2.11+cu128 remains unverified.
4. The cost matrix is an analytic forward accounting tool. It does not measure latency, peak memory, epoch time or optimizer cost.

Real AirfRANS and ShapeNet-Car data, full epochs, performance measurements and SOTA comparisons are **NOT RUN**.

本 LAA7 阶段结束，未执行下一阶段。
