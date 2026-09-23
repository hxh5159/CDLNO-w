# Stage 2 report — eight-block core

A. Scope/status: PASS. Implemented the task-independent v4 core with eight independent operators/LN1/LN2 and five registered RMLP owners.

B. Files: `cdlno/linearno_loop/v4/{core,construction,standard}.py` and core tests.

C. Formula/ownership: execution is `[first,A1,B1,C1,A2,B2,C2,last]`; RMLP route is `[F_first,F_A,F_B,F_C,F_A,F_B,F_C,F_last]`. Prefix/suffix branch scales are 1; six middle operator and RMLP raw branches use `1/sqrt(2)`. The A/B/C aliases are ordinary schedule keys, so each parameter appears once in `state_dict` and the optimizer.

D. Tests: independent core oracle, hooks, module-id/parameter-id checks, backward/optimizer and dropout/RNG checks passed. No task entry or real data was used.

E. Frozen region: no old loop class or pure LinearNO attention was modified to make v4 pass.

F. Review points: inspect `core.py` round schedule, branch scaling, and endpoint RMLP width conditions.

本阶段结束，未执行下一阶段。
