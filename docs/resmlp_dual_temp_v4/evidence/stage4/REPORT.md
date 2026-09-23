# Stage 4 report — factory, dispatch, checkpoint and wrappers

A. Scope/status: PASS targeted. v4 has an isolated factory, provenance, recorder, strict checkpoint pair and task-independent Standard/AirfRANS/Car wrappers.

B. Files: `cdlno/linearno_loop/v4/{construction,checkpoint,provenance,lifecycle,recording,standard,airfrans,shapenet,industrial_entry}.py`, version dispatch and recorder/accounting adapters.

C. Contract mapping: public selector is `resmlp_dual_temp_v4`; checkpoint schema/version is `resmlp_dual_temp_v4`/1 with physical pair `resmlp-dual-temp-v4-pair-v1`. Sidecar/metadata/hash/path checks precede construction and `strict=True` tensor load. State ownership records eight operators and five RMLP owners, avoiding v2 round-specific FFN misreporting.

D. Tests: metadata-first, cross-config negative loads, strict reload, provenance portability, CUDA/CPU RNG isolation and recorder schedule tests passed. Real checkpoint archives and remote stacks were NOT RUN.

E. Frozen region: v1/v2/v3 checkpoint schemas reject v4 artifacts and v4 rejects old artifacts; no migration fallback was added.

F. Review points: read `CHECKPOINT_COMPATIBILITY.md`, verify sidecar-before-tensor ordering, and inspect the source fingerprint set used for resume.

本阶段结束，未执行下一阶段。
