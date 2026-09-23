# ResMLP Dual Temperature v4 status

- Branch: `main`
- HEAD at audit start: `36a2e0be9287949b06006606e4726aa77d4a2a66`
- Working tree: preserved user changes plus additive v4 implementation, evidence, scripts and docs; no reset/clean/stash/rebase/commit/push.
- Final implementation state: **PARTIAL** under the scientific verification meaning below. The authorized implementation and synthetic validation are complete. PARTIAL records that real data, long training, convergence, remote acceptance and full legacy inventory are not claims of this run.

## Stage status

| Stage | Status | Evidence |
|---|---|---|
| 0 reference audit | PASS | `REFERENCE_AUDIT.md`, `evidence/stage0/REPORT.md` |
| 1 contracts + ResMLP | PASS | `evidence/stage1/REPORT.md`, v4 schema/math tests |
| 2 eight-block core | PASS | `evidence/stage2/REPORT.md`, core tests |
| 3 adaptive temperatures | PASS | `evidence/stage3/REPORT.md`, attention/math tests |
| 4 factory/checkpoint/wrappers | PASS | `evidence/stage4/REPORT.md`, checkpoint/entry tests |
| 5 six standard tasks | PASS synthetic | `evidence/stage5/REPORT.md`, stable native matrix |
| 6 AirfRANS/Car/scripts | PASS synthetic/parser | `evidence/stage6/REPORT.md`, stable native matrix |
| 7 final audit/costs | PARTIAL by boundary | `evidence/stage7/REPORT.md`, `COSTS.md`, `REQUIREMENTS.md` |

## Frozen model contract

Public selector: `architecture=resmlp_dual_temp_v4`; family `linearno_loop`; extension `resmlp_dual_temp`; version 4. Checkpoints use `checkpoint_schema=resmlp_dual_temp_v4`, version 1, physical format `resmlp-dual-temp-v4-pair-v1`.

The model has eight independent LinearNO operators with independent LN1/LN2 and five registered point-domain ResidualMLP owners. The execution schedule is `[first,A1,B1,C1,A2,B2,C2,last]`, mapping to `[F_first,F_A,F_B,F_C,F_A,F_B,F_C,F_last]`. Owner depths are `[2,3,3,3,2]`, which means 4/5 Linear modules; task FFN ratios remain those of pure LinearNO. Middle raw operator and raw RMLP branches use `1/sqrt(2)`; identity paths and internal RMLP residuals are not scaled.

Temperature modes are `base`, `latent_k_point_q`, and `point_k_point_q`. Predictors use post-input-projection per-head features, terminal zero initialization, isolated stable seeds, and `tau_base*exp(log(2)*tanh(delta))`. K softmax is over N, Q softmax over M, followed by `K^T V` and `QZ`. No latent FFN, LoRA, AttnRes, Gumbel, history cache, N×N/M×M attention, or variable depth was added.

## Verification summary

- `pytest -q tests/linearno_loop_v4`: **133 passed, 1 skipped**, one deprecation warning.
- Focused math/checkpoint/entry tests: **74 passed, 1 skipped**.
- Launcher/reference/hardening tests: **39 passed**.
- Stable native task matrix: **56/56 rows exit 0**; all eight tasks × two dynamic modes cover interrupt/resume/eval, with latent-K uninterrupted weights exactly matching resumed weights.
- Pure LinearNO static/temporal integration: **12 passed**.
- `compileall`, all v4 shell `bash -n`, and `git diff --check`: **PASS**.
- Cost accounting: all 24 task/mode instances match parameter partitions and ATen matrix traces; see `COSTS.md`.

## Known inherited failures and limits

The old loop suite still contains pre-existing inventory/provenance failures, tiny strict floating-point fresh-process differences in old RB/mode tests, and a historical V3 test import/entry-signature failure. These are preserved in `evidence/stage7/legacy-v1-v2.txt` and `legacy-v3.txt`; no old golden or tolerance was weakened. The local environment lacks `torch_cluster`, so full AirfRANS radius-graph/sampling acceptance is skipped.

**NOT RUN:** real datasets; full production epochs; convergence, accuracy or SOTA; true epoch duration; remote Python 3.10/torch 2.11/CUDA 12.8 acceptance; distributed; `torch.compile`; multi-GPU; full AirfRANS sampling; production AMP resume. The benchmark is synthetic MSE timing only and cannot establish task speed or quality.

## Next real-experiment entry condition

Use the scripts in `docs/resmlp_dual_temp_v4/USAGE.md` after copying the code to the target environment. Run equal-seed/profile budgets in the order `base → latent_k_point_q → point_k_point_q`; preserve each run directory, metadata pair and original task metric. Do not interpret the local synthetic matrix as a result on any benchmark dataset.
