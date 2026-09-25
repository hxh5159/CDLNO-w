# Partial-share feature-gated dense-expert LinearNO status

## Visit-independent LayerNorm increment (2026-09-25)

Status: **PASS** for the authorized source, synthetic, parser, checkpoint and
compatibility scope. The V5 public selector and architecture version remain
`partial_share_feature_gate_v5` / 5. Its internal config/schema revision is now
6 and checkpoint format is `partial-share-feature-gate-v5-pair-v2`.

- Audited HEAD before this uncommitted increment:
  `db5394b31a1079d3bb4e9fc29642b1072a4b1544`.
- New default: `core_norm_mode=visit_independent`; each core `(position, visit)`
  owns distinct LN1/LN2 affine parameters.
- Compatibility mode: `core_norm_mode=shared`; it preserves the prior V5
  forward and state-key ownership, while using the new explicit pair-v2
  metadata contract.
- Parameter delta over `shared` is exactly `4*C*C_core*(R-1)`. Airfoil P1,
  K=3 changes from 1,456,025 to 1,457,561 parameters. Matrix MAC/FLOP counts
  are unchanged because the LayerNorm call count is unchanged.
- Old pair-v1 V5 checkpoints are rejected metadata-first. They are not strict
  optimizer/RNG resumes of the new default and no implicit migration exists.
- Focused tests: 16 passed. Complete V5 suite: 69 passed, one existing timm
  deprecation warning. V2: 34 passed; V4: 136 passed, one skipped; original
  Transolver eight-task frozen fixture: one passed. Pure LinearNO, V1 and V3
  representative numerical/checkpoint tests also passed.

See `VISIT_INDEPENDENT_LAYERNORM.md` and `evidence/visit_norms/summary.txt`.
No real data or complete benchmark training was run.

## Final status

Status: **PASS** for the authorized implementation, static checks, controlled
synthetic task lifecycles, local CUDA autocast checks, and bounded cost
measurement. This status does not cover real-data convergence or accuracy.

- Historical V5 implementation audit HEAD: `c721ed161f0b94e5293d7e43b7b55ef20ba48167`.
- User-owned untracked `PLAN_v5/` was read and preserved.
- Selector: `partial_share_feature_gate_v5`.
- Historical checkpoint format: `partial-share-feature-gate-v5-pair-v1`.
- Default: P2-C2-R2-S2, K=2, F from the selected pure LinearNO profile,
  profile-native M/heads/C, fixed residual `operator_1_expert_1_over_r`.
- Environment: Python 3.13.9, torch 2.13.0+cu130, PyG 2.3.1, CUDA available.

## Stage table

| Stage | Scope | Status | Report |
|---|---|---|---|
| A | current-tree audit and baseline freeze | PASS | `stage_a_audit.md` |
| B | independent config, operator, core and wrappers | PASS | `STAGE_B_CORE.md` |
| C | eight-task parser/factory/launcher integration | PASS | `STAGE_C_TASK_INTEGRATION.md` |
| D | strict checkpoint, outputs and recording | PASS | `STAGE_D_CHECKPOINT_OUTPUTS.md` |
| E | structural, numerical, entry and cost acceptance | PASS | `STAGE_E_FINAL_REPORT.md` |

## Acceptance summary

- V5: 53 tests passed after final launcher/output/resume/provenance self-review.
- Eight task parser dry-runs: 8/8 passed without data/tensor/weight/run access.
- AirfRANS has an explicit member interrupt/resume/final/eval lifecycle test;
  AirfRANS and Car both parse P1/P2 with independently overridden K/F.
- All six Standard synthetic lifecycles assert native visualization output,
  training results, and independent evaluation results.
- Original Transolver frozen fixtures pass for all eight tasks in fresh task
  workdirs, including strict state reload and the existing industrial local
  object roundtrip.
- V4: 136 passed, 1 skipped. V2: 34 passed.
- V3 and V1/pure aggregate suites retain documented checkout/runtime failures;
  isolated affected tests and provenance guards distinguish them from V5
  regressions. Details are in the final report.
- `compileall`, nine shell syntax checks, and `git diff --check` passed.
- Cost report: 104 exact parameter/matrix-operation rows and 10 bounded local
  timing rows. No speedup, convergence, or SOTA claim is made.

## Documents

- `USAGE.md`: eight task launchers and train/resume/eval examples.
- `REQUIREMENTS_MATRIX.md`: frozen requirement to source/test/evidence mapping.
- `evidence/stage_e/`: raw test, parser, static-check, regression and cost logs.

## Not run

Real data, complete epochs, three-seed convergence, benchmark accuracy/SOTA,
real epoch timing, production inference latency, distributed/compile, and the
remote Python 3.10/Torch 2.11 environment were not run.

本阶段结束，未执行下一阶段。
