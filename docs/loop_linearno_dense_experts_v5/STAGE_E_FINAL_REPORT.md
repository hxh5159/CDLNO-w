# Stage E: final acceptance and delivery

## A. Authorized scope and final status

Final status: **PASS** for the authorized no-real-data V5 implementation and
validation scope.

The new architecture is `partial_share_feature_gate_v5`. It supports all eight
task entry families, both required eight-visit topologies, custom topology,
independent `K/F`, strict train/resume/eval output handling, and data-free cost
accounting. PASS does not claim convergence, accuracy, SOTA, or training
speedup.

## B. Delivered changes

- Torch-free contracts/config/schema: `linearno_loop/v5/`.
- Operator, core, wrappers, construction, checkpoint, provenance and recording:
  `cdlno/linearno_loop/v5/`.
- Explicit gated integration in the existing version and eight-task entry
  dispatchers.
- Eight thin launchers plus one shared parser/executor in
  `tran_evaluate/linearno_loop_v5/`.
- Independent oracle, structural/numerical/native-task/checkpoint/cost tests in
  `tests/loop_linearno_v5/`.
- V5 accounting/report tools and this audit/documentation set.

Shared-file edits add only explicit V5 dispatch or V5 accounting/recording.
During final review, the V5 `train_eval` follow-up was corrected to retain the
selected GPU; a regression test was added. A provenance edge case was also
corrected so the frozen pure LinearNO fingerprint remains stable after the
reviewed V5 routing source is committed, while unknown source changes still
fail closed. No V1-V4 launcher was changed.

## C. Final architecture review

For every physical core position `p`, `in_project_x`, `to_v`, complete native
`to_out`, LN1/LN2, and all dense experts have one owner reused across rounds.
Each logical visit owns independent `to_q`, `to_k`, active native Q/K
temperatures, and `Linear(C,K,bias=True)`. Prefix and suffix are independent
single-visit blocks.

```text
Z = X + Operator[p,r](LN1[p](X))
U = LN2[p](Z)
pi = softmax(router[p,r](U), expert axis)
G = sum_j pi_j * Expert[p,j](U)
core:   Xnext = Z + G/R
prefix/suffix: Xnext = Z + G
```

Every expert is `Linear(C,F,bias) -> GELU -> Linear(F,C,bias)` and all experts
execute for every point. Native operator heads, latent rank, temperature/clamp,
K softmax over points, Q softmax over latent rank, `K^T V`, Q readout, Conv
layout, AirfRANS contiguous layout, ShapeNet spelling, and task output heads are
preserved. Shape tracing found no explicit N-by-N or M-by-M attention.

Initialization constructs and initializes the complete wrapper once, then
copies Q-to-Q, K-to-K, and active temperatures across visits into distinct
storage and zeros router weights/biases. Experts remain independently
initialized. Construction does not advance caller RNG.

## D. Verification and measured accounting

Final V5 suite after self-review:

```text
53 passed, 1 warning
```

The warning is the existing `timm.models.layers` deprecation warning. The new
provenance test simulates a committed V5 routing source and preserves the LL6
pure LinearNO source and normalized-patch hashes exactly. Eight
real parser dry-runs passed. AirfRANS/Car also passed both P1/P2 real-parser
checks with independent K/F overrides. The AirfRANS member lifecycle now has an
explicit interrupt/resume/final/eval test, and all six Standard lifecycles
assert their native visualization and result files. `compileall`, shell syntax
for all nine shell files, and `git diff --check` passed. The raw logs are under
`evidence/stage_e/`.

The original Transolver eight-task frozen fixture also passed independently:

```text
1 passed in 16.73s
```

It resolves each original selector in the task's own working directory,
compares parameters, state keys and outputs with the pre-V5 fixture, performs a
strict pure-state reload, and checks the existing local whole-object roundtrip
for ShapeNet-Car and AirfRANS. This is CPU synthetic compatibility evidence,
not a real-data result.

Default K=2 and profile-derived F parameter counts are below. V1 uses the same
profile-native M for comparison. Matrix-FLOPs count only listed dense matrix
operations with `1 MAC = 2 FLOPs`; LayerNorm, GELU, softmax, temperature,
expert weighting, residuals, reshapes, data, losses, and optimizer are excluded.
Both V5 topologies execute eight logical blocks, so their matrix-FLOPs are equal
despite different unique parameter counts.

| Task | Pure | V1 P1 | V5 P1 | V1 P2 | V5 P2 | V5/Pure matrix-FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Airfoil | 1,765,889 | 1,116,497 | 1,289,873 | 1,332,961 | 1,537,297 | 1.131056 |
| Darcy | 1,766,145 | 1,116,753 | 1,290,129 | 1,333,217 | 1,537,553 | 1.131039 |
| Elasticity | 585,217 | 378,577 | 551,953 | 447,457 | 651,793 | 1.315934 |
| Pipe | 1,765,889 | 1,116,497 | 1,289,873 | 1,332,961 | 1,537,297 | 1.131056 |
| Navier-Stokes | 3,377,921 | 2,182,145 | 3,506,961 | 2,580,737 | 4,166,417 | 1.573946 |
| Plasticity | 1,799,428 | 1,150,084 | 1,323,412 | 1,366,532 | 1,570,836 | 1.128919 |
| AirfRANS | 3,358,788 | 2,162,988 | 3,487,804 | 2,561,588 | 4,147,268 | 1.576972 |
| ShapeNet-Car | 3,852,420 | 2,459,220 | 3,784,084 | 2,923,620 | 4,509,332 | 1.508361 |

The report contains 104 parameter/accounting rows covering eight tasks, both
topologies, K=2/3/4/8, default F, and an alternate F. Analytic parameters match
live instances. Ten bounded Elasticity rows measured CPU and local CUDA FP32
forward, forward-backward-AdamW, weight bytes, and CUDA peak allocated/reserved
memory. They are microbenchmarks, not real task epoch or end-to-end latency
claims.

Legacy regression evidence:

- V4: `136 passed, 1 skipped`.
- V3 aggregate: `115 passed, 1 skipped, 1 failed`; the failure is a same-process
  top-level `cdlno_entry` import collision. The affected LAA7 suite passes in
  isolation: `6 passed, 1 skipped`.
- V2: `34 passed`.
- V1: `103 passed, 4 failed`; two failures are CPU cross-process bitwise deltas
  of `1.4901161193847656e-08` and `2.561137080192566e-09`, and two are frozen
  inventory differences in `.claude/settings.json` and a Python 3.13 pycache.
- Pure LinearNO: `153 passed, 2 failed`; both require historical files absent
  from this checkout: `docs/CDLNO_EXPERIMENT_OUTPUTS.md` and
  `docs/kcdno_audit/audit_static.py`.

The pure/loop historical provenance assertion itself passes. No old golden,
tolerance, checkpoint, or missing historical artifact was changed or fabricated.

## E. Frozen regions and evidence

Pure LinearNO and V1-V4 operator/core forward implementations were not edited.
Transolver, dataset readers, splits, sampling, fields, normalization, losses,
optimizers, schedulers, NS rollout, Plasticity time loop, AirfRANS metrics, Car
drag logic, and visualization mathematics remain unchanged. Historical source
fingerprints were restored exactly after adding isolated projection rules for
the explicit V5 branches. The original Transolver eight-task frozen numerical
and strict-reload fixture passes after all V5 changes.

## F. Limits and priority review points

**NOT RUN:** real data, complete benchmark training, three-seed convergence,
benchmark accuracy, SOTA comparison, real epoch duration, production inference
latency, distributed training, remote Python 3.10/Torch 2.11 stack, compile
mode, real AirfRANS sampling/ensemble evaluation, and real ShapeNet-Car drag
evaluation.

The five priority source reviews passed: Q/K visit ownership versus shared
operator body; expert-only `1/R` placement; all-expert execution and expert-axis
softmax; initialization/RNG lifecycle; and metadata-first strict loading before
tensor access. Remaining risk is empirical and environmental rather than a
known architecture-contract defect.

本阶段结束，未执行下一阶段。
