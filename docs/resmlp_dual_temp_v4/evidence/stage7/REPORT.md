# Stage 7 report — final audit, costs and delivery

A. Scope/status: PARTIAL by design: the authorized implementation and synthetic verification are complete, while real-data claims remain out of scope.

B. Files: v4 accounting, final docs, requirements matrix, benchmark/evidence, and the minimal strict-config fix in `linearno_loop/v4/config.py` rejecting undeclared top-level fields even after hash resealing.

C. Cost mapping: `COSTS.md` and `evidence/stage7/costs.json` give exact parameter partitions and matrix MACs/FLOPs for all 8 tasks and three temperature modes. Non-matrix operations, router/predictor costs, and measured benchmark limitations are separate. Sharing RMLP owners reduces parameters but not executed visits.

D. Verification: component suite 133 passed/1 skipped; focused math/checkpoint/entry suite 74 passed/1 skipped; launcher/reference/hardening 39 passed; stable native matrix 56/56 rows exit 0; pure LinearNO integration 12 passed; compileall, shell syntax and `git diff --check` passed. Legacy V1/V2/V3 suites retain pre-existing inventory/provenance/strict-float/import failures documented in `legacy-v1-v2.txt`, `legacy-v3.txt`, and `preexisting-inventory.json`.

E. Frozen region: old source/checkpoint bytes and user PLAN/download directories were preserved; no reset/clean/stash/commit/push, real data, dependency installation or long training occurred.

F. Review points: verify the requirements matrix, cost comparison, and the explicit NOT RUN list before any real experiment. The v4 synthetic result is not a convergence or SOTA result.

本阶段结束，未执行下一阶段。
