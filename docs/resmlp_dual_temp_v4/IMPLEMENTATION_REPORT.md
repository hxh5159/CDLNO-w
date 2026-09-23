# LinearNO ResMLP dual adaptive temperature v4 — final implementation report

## A. Scope and status

Stages 0–7 were completed in the current checkout under the user’s no-reset/no-real-data constraints. The final status is **PARTIAL**: the isolated v4 model, eight task routes, strict checkpoint/output handling, synthetic native train/resume/eval and cost accounting are complete; real benchmark data, long training, convergence and remote acceptance were deliberately not run.

## B. Files and change summary

The implementation is isolated in `linearno_loop/v4/` (JSON contracts) and `cdlno/linearno_loop/v4/` (model, attention, wrappers, lifecycle, checkpoint and recording). Dispatch, six standard entries, two industrial entries, accounting and v4 launchers were extended with explicit ownership checks. `docs/resmlp_dual_temp_v4/` contains the audit, usage, costs, compatibility, requirements matrix and stage evidence. Existing user directories `PLAN_linearno++/` and `download/` were preserved. The only final-stage production-code correction was the v4 unknown top-level config-field rejection in `linearno_loop/v4/config.py`.

## C. Architecture/formula mapping

The core uses eight independent operators and norms and five RMLP owners with route `[first,A,B,C,A,B,C,last]`. RMLP owner depths are 2/3/3/3/2, giving 4/5 Linear modules with GELU(tanh), conditional endpoint shortcuts and always-on hidden shortcuts. For the six middle logical blocks:

`u=x+(1/sqrt(2))*O(LN1(x))`; `x_next=u+(1/sqrt(2))*F(LN2(u))`.

First and last use coefficient 1 on both raw branches. Attention computes the existing Q/K/V projections and slice contractions. Dynamic Q/K temperatures are predicted from `[B,Hd,N,d_h]`, with latent-K `[B,Hd,1,M]` or point-K `[B,Hd,N,1]`; `tau=tau_base exp(log(2)tanh(delta))`, K softmax is over N, Q softmax over M, then `K^T V` and `QZ`. Predictors are per logical operator, head-shared, zero initialized and installed under isolated RNG after public initialization.

## D. Commands and results

```text
TMPDIR=/home/hwz/CDLNO-artifacts/v4-validation-20260923/tmp PYTHONPATH=.:cdlno:PDE-Solving-StandardBenchmark pytest -q tests/linearno_loop_v4
133 passed, 1 skipped

TMPDIR=... PYTHONPATH=... python tests/linearno_loop_v4/run_task_matrix.py --evidence docs/resmlp_dual_temp_v4/evidence/stage7/native-matrix-final
56/56 rows exit 0; latent-K resumed/uninterrupted weights exact

pytest -q tests/linearno/test_static_integration.py tests/linearno/test_temporal_integration.py
12 passed

python -m compileall ...; bash -n tran_evaluate/linearno_loop_v4/*.sh; git diff --check
PASS
```

Focused math/checkpoint/entry tests were 74 passed/1 skipped; launcher/reference/hardening tests were 39 passed. The benchmark recorded synthetic synchronized CPU/CUDA forward and train-step timings plus memory/weights; it uses synthetic MSE and is not a task-speed result. The local absence of `torch_cluster` caused only the documented sampling-dependent skip.

## E. Frozen-region evidence

Pure LinearNO static/temporal integration passed. Old v1/v2/v3 selectors and schemas remain separate; v4 artifacts are rejected by old loaders and old artifacts by v4 strict metadata. Old loop regression logs retain inherited inventory/provenance, tiny strict-float and historical import/signature failures; no old tests, golden values or tolerances were changed to hide them. Original task data readers, losses, normalizers, optimizer/scheduler protocols and evaluation metrics were not replaced.

## F. Remaining limits and review points

1. Review `REQUIREMENTS.md` against the source symbols and evidence paths.
2. Review `COSTS.md`: matrix MAC/FLOPs exclude non-matrix operators and cannot predict epoch speed; the wider task ratios can make v4 more expensive than pure LinearNO despite RMLP parameter sharing.
3. Review `CHECKPOINT_COMPATIBILITY.md` and sidecar-before-tensor-load tests.
4. Before real runs, verify remote package versions, data roots, AirfRANS `torch_cluster`, ShapeNet raw/cache paths and task-specific GPU memory.
5. Keep all real results in unique run directories and compare base, latent-K and point-K with matched seeds and budgets.

**NOT RUN:** real data, full production training, three-seed convergence, accuracy/SOTA, real epoch duration, remote stack acceptance, distributed training, `torch.compile`, multi-GPU, full AirfRANS sampled evaluation, and production AMP resume.

本阶段结束，未执行下一阶段
