# Stage 1 report — v4 contracts and FLARE ResidualMLP

A. Scope/status: PASS. Added a torch-free JSON contract/config namespace and an independently tested FLARE-style ResidualMLP.

B. Files: `linearno_loop/v4/{contracts,config,schema}.py`, `cdlno/linearno_loop/v4/resmlp.py`, and stage-1 tests. The configuration resolves the explicit architecture selector, three temperature modes, task profile facts, seeds, RMLP route/depths, and checkpoint fields.

C. Formula mapping: `resmlp.py` implements `fc1 + L hidden same-width residual linears + fc2`; L=2/3 therefore has 4/5 Linear modules. Endpoint residuals are conditional on equal widths; hidden residuals are always enabled. GELU uses tanh approximation, with no LN/dropout/final activation.

D. Tests: JSON roundtrip/hash, schema conflicts, independent VJP/finite difference, profile field checks, and RNG isolation passed in the v4 suite. No tensors are constructed by the config package; no real data was read.

E. Frozen region: legacy profiles and old selectors are imported through their existing resolver and are not rewritten.

F. Review points: confirm the task FFN ratio table, the explicit architecture requirement, and that profile facts cannot be resealed with structural overrides.

本阶段结束，未执行下一阶段。
