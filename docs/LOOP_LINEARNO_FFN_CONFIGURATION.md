# Looped LinearNO FFN v2 Configuration Contract

**LF1 status: PASS.** This stage adds configuration and metadata contracts only.
It does not construct a tensor model or route any production task.

The historical v1 contract remains in `linearno_loop.config` and
`linearno_loop.schema`: extension `loop_linearno_v1`, config/schema version 1,
default rank multiplier 2, and the original class paths and hashes. The new
contract is isolated under `linearno_loop.v2`:

- family `linearno_loop`, extension `loop_linearno_ffn_v2`;
- config/schema version 2 and checkpoint format
  `linearno-loop-epoch-pair-v2`;
- explicit `core_ffn_mode=round_specific|round_specific_latent` selects v2;
- default rank multiplier 1, with explicit multiplier 1/2 or actual M;
- class paths under `cdlno.linearno_loop.v2` with closed constructor fields;
- run identifiers include version, P/C/R/S, residual, FFN mode, actual M, seed,
  and config hash.

The loop metadata records operator sharing, `(core, round)` point-FFN ownership,
`P+C*R+S` total point FFNs, optional `C` shared latent FFNs, residual/router
contracts, state partitions, and isolated feature initialization seed. The
first study fixes latent inner widths to 572 for the structured conv variants
and H for Elasticity, Navier--Stokes, AirfRANS and ShapeNet-Car. No public
latent-width search field exists.

The planned matrix contains 288 primary configurations: eight tasks, two
presets, three residuals, two v2 modes and three paired seeds. It additionally
contains sixteen explicit Mx2 controls and one custom P0/C2/R3/S1 example.
All entries are configuration previews and were not trained.

Verification:

- v2 LF1 tests: 9 passed;
- v1 config/metadata regression: 19 passed;
- compile and `git diff --check`: passed;
- real data, model construction, checkpoint tensors and training: NOT RUN.

Priority review points were the unchanged v1 resolver, explicit-only v2
selection, fixed 572 width from the frozen task table, distinct run IDs, and
metadata conflict rejection before any future tensor load.

本 LF1 阶段结束，已按用户授权继续执行下一阶段
