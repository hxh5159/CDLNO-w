# Looped LinearNO Round-Specific / Latent FFN Final Report

## Result

**Authorized no-data implementation status: PASS.** Both version-2 modes are
integrated across all eight tasks, both presets and custom P/C/R/S, and all
three existing residual modes. V1 Looped LinearNO remains the omitted-field
default. The final old regression has exactly the LF0 baseline result: 689
methods, 644 passed, 9 previously classified failures and 36 skips. There are
zero new unexplained failures.

Repository truth used for the work: branch `main`, base HEAD
`80ebe42d5755fc58ac6b41e2f6a0512d601ac8a8`, dirty working tree containing
the user's prior files and the authorized LF1-LF7 changes. Nothing was reset,
cleaned, stashed, committed or pushed.

## Architecture

V2 is selected only by explicit `core_ffn_mode` at train time or saved v2
metadata at resume/eval. Its extension is `loop_linearno_ffn_v2`, config/schema
version is 2, and checkpoint format is `linearno-loop-epoch-pair-v2`.

For core position p and round r:

```text
O_p(x)   = Attn_p(ln_1_p(x))                    shared across rounds
F_p,r(x) = MLP_p,r(ln_2_p,r(x))                 specific to (p,r)
```

Prefix/suffix remain native complete blocks. The last suffix is the only owner
of `ln_3 + mlp2`, so the head runs once. The loop builds physical owners
directly; it does not construct eight blocks and discard any. Every operator
visit recomputes projections, Q/K/V, `K^T V` and Q readout.

`round_specific_latent` adds one context FFN per core position, shared across
rounds:

```text
Z' = Z + W2(GELU(W1(LayerNorm_eps=1e-5(Z))))
```

It is applied to `[B,h,M,d_h]` after `K^T V` and before Q readout. It is
tokenwise, has no M-axis mixing or dropout, and uses bias. W2 weight/bias are
zero initialized after all common initialization, making the two v2 modes
initially function-equivalent. Latent construction uses an isolated seed and
does not advance public backbone RNG.

Residual formulas are unchanged:

- SR scales the operator and its selected `(p,r)` point FFN branches separately
  by `1/R`; identity, prefix and suffix are unscaled.
- RB calls an independent point-domain receiver before every core sublayer,
  accumulates only current-round raw branch outputs, records the raw round sum,
  and applies the final receiver. It has no `1/R`.
- LB applies `1/R` inside the round, records actual `Delta_r=Y_r-H_r`, and uses
  boundary/final receivers over anchor plus completed deltas. Deltas are not
  scaled again.

No receiver or latent state crosses a forward, task time query, NS rollout,
Air ensemble member or Car fold. No N x N/M x M attention was observed.

## Task Profiles

V2 defaults to M x 1; explicit multiplier two remains available. V1 retains
its historical M x 2 default.

| Task | Variant | H | Heads | Base M | Point FFN ratio |
|---|---|---:|---:|---:|---:|
| Airfoil | conv_temp | 128 | 8 | 64 | 1 |
| Darcy | conv_temp | 128 | 8 | 64 | 1 |
| Elasticity | temp | 128 | 8 | 64 | 1 |
| Pipe | conv_temp | 128 | 8 | 64 | 1 |
| NS | plain | 256 | 8 | 32 | 2 |
| Plasticity | conv | 128 | 8 | 64 | 1 |
| AirfRANS | airfrans | 256 | 8 | 32 | 2 |
| ShapeNet-Car | shapenet | 256 | 8 | 32 | 2 |

Structured `conv/conv_temp` latent FFNs use the frozen inner width 572; other
variants use H. This width is recorded in metadata and is not a public search
flag.

## Files and Responsibilities

- `linearno_loop/v2/`: closed config, schema, metadata, matrix and version-2
  constructor contracts. It has no task-entry or tensor dependency at schema
  import time.
- `cdlno/linearno_loop/v2/`: shared operator/round FFN ownership, context
  attention, latent FFN, Standard/Air/Car wrappers, construction, provenance
  and strict checkpoint pairs.
- `cdlno/linearno_loop/versioning.py` and the five task dispatch files: explicit
  version selection before construction/tensor loading.
- `cdlno/linearno_loop/v2_projection.py` and the exact LL9R composition: SHA
  guarded compatibility views for approved route-only additions.
- `tran_evaluate/linearno_loop/recording.py`: v1-preserving/v2-aware initial
  hashes, ownership partitions and first-forward schedule.
- `tran_evaluate/linearno_loop/ffn_matrix.py`: safe 288-command preview plus
  M x 2/custom controls.
- `tools/linearno_loop_accounting.py`: version-dispatched independent formula
  and actual ATen ledger.
- `tools/linearno_loop_ffn_{support,diagnostics,performance}.py`: bounded v2
  input fixtures, opt-in detached observations and measurement CLI.
- `tests/loop_linearno_ffn/`: independent oracles and LF1-LF7 acceptance.

The eight existing task data readers, losses, metrics, normalizers, optimizer/
scheduler protocols, NS 10-step loop, Plasticity query loop, Air sampling and
ensemble, and Car fold/drag protocols were reused rather than copied.

## Config and Checkpoint Safety

Train requires an explicit new mode. Resume/eval reads architecture metadata
first and treats repeated CLI structure as assertions. Family, version, task,
profile, P/C/R/S, residual, core FFN mode, H/heads, actual M, latent width,
constructor kwargs, state partition and initialization protocol are sealed.
Wrong v1/v2, mode, residual, topology, rank, task, router keys, missing keys or
optimizer shapes fail before applying weights/forward. Model loads are strict.

V2 state keys contain each shared operator and latent FFN once, and each point
FFN once per `(p,r)`. Full key lists/hashes are in `counts-final.json`.

## Verification

New-stage suite:

- LF1-LF7: 34/34 methods passed in 53.781 s after the final projection edit.
- LL9R compatibility plus v1 performance tools: 18/18 passed in 9.366 s.
- Six operator variants, both presets, custom R=3, three residuals and both v2
  modes have oracle/gradient/optimizer/strict reload coverage.
- Six Standard parser matrix, production checkpoint/resume/eval fixture, two
  industrial real-PyG synthetic matrices, Air member isolation and Car fold3
  passed.

Final LF7 matrices:

- 288 primary command previews, 16 M x 2 controls, one custom preview.
- 192 full-profile parameter/MAC constructors; 96 M x 1 strict state reloads.
- 96 reduced CPU forward/backward/AdamW/strict reload cases; max reload error 0.
- 96 CPU timing cases with median/p90 samples.
- 288 CUDA reduced-shape cases: 96 FP32, 96 AMP FP16, 96 AMP BF16, all passed.
- Formula/ATen MAC equality and zero forbidden attention in all 96 matrix rows.

Old LF0 inventory rerun:

```text
80 modules, 689 methods
644 passed, 9 failed, 36 skipped
```

This is exactly the LF0 count. The nine failures are the same four deleted
KCDNO evidence files, two legacy deleted-inventory methods, one old isolation
inventory drift, one ignored `.pyc` hash and one old normalized provenance
hash. Source hashes remain protected; no golden was refreshed. The optional
LRSA reference was rerun with `/home/hwz/LRSA-Operator` and passed 2/2. The
heavy history industrial module passed 5/5 after 1551 s, confirming its first
1000 s audit timeout was resource speed rather than a model assertion.

Static checks passed: compileall, all eight launcher `bash -n`, exact v2/LL9R
projection mutation tests and `git diff --check`.

The final freeze comparison checked all 3,031 LF0-manifest files. It found the
nine authorized existing source edits, zero unexpected source changes and zero
missing files. The required compile check regenerated two ignored Python 3.13
cache files for the edited projection modules. No exact old-cache copy exists;
the files were retained and the two hashes are disclosed in
`lf7/final-freeze.json` rather than deleting or fabricating user state.

## Evidence Index

- `docs/loop_linearno_ffn_audit/lf7/command-matrix.json`
- `docs/loop_linearno_ffn_audit/lf7/counts-final.json`
- `docs/loop_linearno_ffn_audit/lf7/matrix.json`
- `docs/loop_linearno_ffn_audit/lf7/cpu.json`
- `docs/loop_linearno_ffn_audit/lf7/cuda.json`
- `docs/loop_linearno_ffn_audit/lf7/regression-results.json`
- `docs/loop_linearno_ffn_audit/lf7/requirements.json`
- `docs/loop_linearno_ffn_audit/lf7/index.json`
- `docs/loop_linearno_ffn_audit/lf7/final-freeze.json`

## Limits

NOT RUN: real datasets, full epochs, convergence, accuracy, SOTA comparison,
real epoch throughput, remote Python 3.10/torch 2.11/cu128, distributed
training, or `torch.compile`. Local CUDA evidence is reduced synthetic smoke,
not remote acceptance or formal-width training performance.

## Priority Review Points

1. Operator/round-FFN ownership and SR/RB/LB dispatch in
   `cdlno/linearno_loop/v2/core.py`.
2. Exact latent insertion boundary in `cdlno/linearno_loop/v2/attention.py` and
   W2-zero/RNG-isolated initialization in `latent.py`/wrappers.
3. Metadata-first v1/v2 dispatch and strict rejection in Standard/industrial
   entries and `v2/checkpoint.py`.
4. V1 compatibility composition in `v2_projection.py`, `ll7_projection.py`
   and `ll9r_projection.py`.
5. Independent parameter/MAC formulas versus `counts-final.json` and
   `matrix.json`.

本 LF7 阶段结束，未执行真实实验
