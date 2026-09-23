# Stage 0 report — reference audit and baseline freeze

A. Scope/status: PASS for the read-only audit. The repository was audited at HEAD `36a2e0be9287949b06006606e4726aa77d4a2a66`, branch `main`, with user changes preserved. Pure LinearNO task paths, attention variants, initialization, old loop ownership and FLARE/Transolver++ references were recorded.

B. Files: `REFERENCE_AUDIT.md`, pinned reference sources under this directory, and `STATUS.md`. No model or task semantics were changed in this stage.

C. Contract mapping: the audit records the original Q/M and K/N softmax axes, plain/temp/conv/conv_temp/industrial/ShapeNet temperature rules, task H/head/M/ratio values, and why V1–V3 ownership cannot serve the new eight-independent-operator contract.

D. Evidence: environment snapshot and pinned FLARE/Transolver++ source excerpts are in this directory. Reference tests and later v4 component tests provide executable confirmation. Real data and long training were NOT RUN.

E. Frozen region: old pure LinearNO, V1/V2/V3, task losses, data paths, and checkpoint formats remain on their original routes; v4 is selected explicitly.

F. Review points: verify the task profile table in `REFERENCE_AUDIT.md`, the preserved ShapeNet `tempreature_q/k` spelling, and the distinction between FLARE’s internal RMLP depth and LinearNO external stem/head.

本阶段结束，未执行下一阶段。
