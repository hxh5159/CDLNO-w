# CDLNO project memory

This directory is the durable, repository-local memory for CDLNO, the planned Transolver-based model using the CDPA mechanism. It is intentionally separate from implementation code and is updated when the design, source audit, or validation status changes.

- [../docs/CDLNO_IMPLEMENTATION_STATUS.md](../docs/CDLNO_IMPLEMENTATION_STATUS.md): implementation phase authorization, standing user decisions, naming map, and phase completion reports. Read this before starting work.
- `current-state.md`: current authoritative status, constraints, and open work.
- `2026-09-13-cdpa-plan-and-reference-audit.md`: detailed notes from the CDPA v1.2 plan and LRSA/IPOT source review.
- `2026-09-13-exported-conversation-review.md`: complete 151-page conversation review, page-indexed design evolution, final choices versus superseded proposals, and mathematical evidence boundaries.

Memory rules:

1. Record facts with a source path, repository commit, or explicit status.
2. Keep planned, implemented, tested, skipped, and unknown states distinct.
3. Do not record synthetic checks as real-data training results.
4. Update `current-state.md` whenever implementation or validation status changes.
5. Design precedence is current/subsequent explicit user decisions → later confirmed conversation decisions → v1.2 → v1.1. Theory supports interpretation and checking, not new architecture/loss authorization. Historical proposals do not override later confirmed decisions.
6. Formal names are user-confirmed: model/core class CDLNO, package cdlno, mechanism CDPA. Whole-model registration CDPA is an old placeholder; mechanism identifiers remain CDPA. Historical review notes describe the evidence available at that earlier turn.
7. A historical assistant's claim that proofs, scripts, or GPU checks passed is not local verification. Record available files and checks actually performed separately.
8. Execute only the user's explicitly assigned phase. Finish its necessary work and validation, report A–F, update status/memory, and stop. Do not treat v1.2 §11 or possession of the complete plan as authorization for subsequent phases.
