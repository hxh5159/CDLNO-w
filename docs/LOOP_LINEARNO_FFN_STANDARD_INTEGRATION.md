# Looped LinearNO FFN v2 Standard Integration

**LF5 status: PASS for the authorized no-data implementation scope.** The six
Standard entries accept explicit `--linearno-loop-core-ffn-mode`; its presence
selects v2, while omission continues through the unchanged v1 resolver and Mx2
default. Eval/resume reads the sidecar version before selecting schema,
checkpoint module or model constructor. Explicit CLI structure is assertion
only.

The existing data, normalizers, objective, metric, optimizer, scheduler,
Navier--Stokes rollout and Plasticity query loops remain the native paths.
Version dispatch only changes config/model/checkpoint/provenance selection.

Verification: 144 real parser/config combinations passed for six tasks, two
presets, three residuals and two modes. A production Elasticity in-memory run
completed forward/backward, two optimizer/scheduler updates, v2 pair save,
fresh model/optimizer/scheduler resume at epoch one and eval reconstruction.
Saved-mode conflict failed before tensor loading. LF5 tests: 3 passed.

The old Standard suite retained its LF0-known normalized provenance fixture
failure; both old `source_sha256` values remain exactly equal to their recorded
values. No new old-path failure was observed. Real data and full epochs were
NOT RUN.

本 LF5 阶段结束，已按用户授权继续执行下一阶段
