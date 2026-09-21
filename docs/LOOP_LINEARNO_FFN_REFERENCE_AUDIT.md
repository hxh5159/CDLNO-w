# Looped LinearNO Round-Specific FFN / Latent FFN LF0 Reference Audit

**Status: PASS for LF0 read-only audit and design freeze.** No production code,
parser, schema, checkpoint, launcher, dependency, or existing test golden was
modified. This status does not mean that LF1 or either v2 model exists yet.

## 1. Repository and scope

The audited repository is `/home/hwz/CDLNO`, remote
`git@github.com:hxh5159/CDLNO-w.git`, branch `main`, at
`80ebe42d5755fc58ac6b41e2f6a0512d601ac8a8` with tree
`1d74a36a173ebbd3e4112d35c8143fe45054b9c3`. The tracked and staged diffs were
empty at LF0 start. Three user-owned untracked planning files were present and
were not edited:

- `PLAN_Looped_LinearNO/Looped_LinearNO_RoundFFN_LatentFFN_Codex_Staged_Prompts.md`
- `PLAN_Looped_LinearNO/Looped_LinearNO_Sparse_Expert_Continuation_Codex_Prompts_CDLNO-w.md`
- `PLAN_Looped_LinearNO/研究主线梳理 (4).md`

The start inventory records 2,023 tracked, 5 untracked (including the two LF0
scripts already present when the manifest was captured), 1,011 ignored, and
3,031 pre-existing files. The LF0 protected subset contains 1,415 files: 712
tracked and 703 ignored. See
[start-manifest.json](loop_linearno_ffn_audit/lf0/start-manifest.json) and
[scope-manifest.json](loop_linearno_ffn_audit/lf0/scope-manifest.json).

The initial collector used Git's default quoted-path output. Six existing paths
with non-ASCII names therefore appear in `start-status-ignored.txt` but were
not assigned an initial content hash. The finalizer uses
`core.quotePath=false`, treats those six paths as classification-only starting
entries, and reports them explicitly in `end-freeze.json`. None was edited.

The root `AGENTS.md` identifies `docs/CDLNO_EXPERIMENT_OUTPUTS_REPORT.md` and
`docs/CDLNO_EXPERIMENT_OUTPUTS.md` as current priority reports, but both are
absent in this checkout. `docs/output_audit/` remains present. LF0 did not
restore or synthesize these deleted historical files. The current Looped
LinearNO status, LL9R repair report, LL10 implementation report, original
LL0--LL10 prompt, new LF0--LF7 prompt, configuration, implementation,
performance, checkpoint, launcher, and task-entry sources were read instead.

No reset, clean, stash, checkout, rebase, commit, push, real data access, long
training, or dependency change was performed.

## 2. Current v1 implementation truth

### 2.1 Ownership, execution, and state paths

`cdlno.linearno_loop.core.LinearNOLoopCore` owns exactly `P+C+S` complete native
LinearNO blocks:

```text
loop.prefix[0:P] -> loop.core[0:C] visited R times -> loop.suffix[0:S]
```

The last suffix alone is constructed with `last_layer=True`; it alone owns and
executes `ln_3` and `mlp2`. A `PhysicalBlock` registers a native block once.
`LinearNOBlockBody` is a non-Module view and neither copies nor registers the
block. Its two raw branches are:

```text
O(x) = block.Attn(block.ln_1(x))
F(x) = block.mlp(block.ln_2(x))
```

Representative v1 state paths are:

```text
loop.prefix.<p>.block.ln_1.*
loop.prefix.<p>.block.Attn.*
loop.prefix.<p>.block.ln_2.*
loop.prefix.<p>.block.mlp.*
loop.core.<p>.block.ln_1.*
loop.core.<p>.block.Attn.*
loop.core.<p>.block.ln_2.*
loop.core.<p>.block.mlp.*
loop.suffix.<p>.block.{ln_1,Attn,ln_2,mlp}.*
loop.suffix.<S-1>.block.{ln_3,mlp2}.*
```

There are no round-indexed core keys in v1. The same full core block, including
`ln_2+mlp`, is physically called in every round. The wrapper property
`LoopForwardView.blocks` returns `(self.loop,)`, so the inherited pure model
forward performs the original stem/position/time processing and then calls one
loop core. It does not register the loop twice.

Source anchors:

- `cdlno/linearno_loop/core.py:38` owns P/C/S and dispatches the three modes.
- `cdlno/linearno_loop/body.py:13` exposes native, scaled, raw, and finalize calls.
- `cdlno/linearno_loop/construction.py:34` provides the non-owning forward view.
- `cdlno/linearno_loop/standard.py:15`, `airfrans.py:11`, and `shapenet.py:10`
  construct the three task-family wrappers.

### 2.2 Current residual formulas

For SR, prefix and suffix use native residuals. Every core visit applies:

```text
x1 = x + O(x) / R
x2 = x1 + F(x1) / R
```

The identity path is not scaled. For RB, the core entry is `b0`. Before every
operator and point-FFN sublayer, an independent receiver reads completed round
summaries and, after the first sublayer in the current round, the current raw
partial. The partial is the sum of raw branch outputs only. The final receiver
reads `[b0, ..., bR]`; RB has no `1/R`. The LL9R `_rb_receive` conversion is
local to a receiver call and casts temporary sources to the anchor dtype. It
does not mutate, detach, or replace the authoritative raw partial.

For LB, each round uses the same two branch-level `1/R` factors as SR. It records
the actual point states `H_r` and `Y_r` and stores `Delta_r=Y_r-H_r`. Boundary
and output receivers read `[anchor, Delta_1, ...]`; a Delta is not reconstructed
from raw branches and is not scaled again.

All RB/LB source lists and tensors are forward-local. No forward activation is
stored on the module, registered as a parameter/buffer, serialized, or shared
across batch, NS rollout call, Plasticity time query, or Air ensemble member.

### 2.3 Initialization order

Standard, AirfRANS, and Car wrappers first construct the stem/time modules and
all P/C/S blocks. They then call the task's release initializer once over the
whole current tree and draw `placeholder` afterwards. RB/LB router query/scale
parameters use direct zero/one creation and consume no RNG. This makes the
common v1 backbone equal across SR/RB/LB for the same task, topology, rank, and
seed. The shared core is initialized once because it exists once.

This ordering is a compatibility requirement for v1 archives. A future latent
module cannot simply be present during the same whole-tree `apply`, because its
Linear layers would consume draws and shift later common initialization.

## 3. Six operator variants and the safe context boundary

All six variants share this mathematical sequence:

```text
input x [B,N,H]
 -> in_project_x and reshape -> features [B,h,N,d_h]
 -> to_q / to_k / to_v
 -> Q = softmax_M(q_logits), K = softmax_N(k_logits), V
 -> context C = K^T V [B,h,M,d_h]
 -> readout = Q C [B,h,N,d_h]
 -> merge heads [B,N,H] -> to_out
```

| variant | feature projection / preserved detail | Q/K detail | output detail |
|---|---|---|---|
| `plain` | pointwise Linear | no temperature | Linear + dropout |
| `temp` | pointwise Linear | `temperature_q/k`, clamp 0.01--1 | Linear + dropout |
| `conv` | Conv2d and H/W reshape | no temperature | two Linear layers with GELU |
| `conv_temp` | Conv2d and H/W reshape | Standard temperature/clamp | two Linear layers with GELU |
| `airfrans` | pointwise Linear, contiguous head layout | registered dead `temperature` stays unused | Linear + dropout |
| `shapenet` | pointwise Linear | preserved `tempreature_q/k` spelling, clamp 0.1--2 | two Linear layers with GELU |

The common implementation is visible at `cdlno/linearno/attention.py:105`; the
context/readout lines are 134--136. The safe latent insertion boundary is after
the current call's `context=K^T V` and before the current call's `Q context`
readout. It must not touch pure LinearNO or v1 state paths.

The minimal safe v2 design is a separate context-capable operator adapter which
owns the same native projection modules and reproduces each variant's current
feature, temperature, layout, and `to_out` contract. It accepts an optional
non-owning context processor for core visits. Prefix and suffix continue using
native complete blocks. The v2 adapter must be independently checked against
the exact six formulas; changing `cdlno/linearno/attention.py` is not required
and is outside the preferred plan.

For latent mode only:

```text
C [B,h,M,d_h]
 -> transpose to [B,M,h,d_h] -> contiguous reshape [B,M,H]
 -> Z + W2(GELU(W1(LayerNorm(Z))))
 -> reshape [B,M,h,d_h] -> transpose [B,h,M,d_h]
 -> Q readout
```

The FFN acts independently on each M token and mixes only the H channels. It
does not create `[B,M,M]` or `[B,N,N]`, cache Q/K/V/context, read historical
tokens, or assume slot alignment across rounds.

## 4. Eight-task path and current ranks

| task | wrapper | H / heads | variant | base M | v1 default M | v2 frozen default M | latent width |
|---|---|---:|---|---:|---:|---:|---:|
| Airfoil | Standard | 128 / 8 | conv_temp | 64 | 128 | 64 | 572 |
| Darcy | Standard | 128 / 8 | conv_temp | 64 | 128 | 64 | 572 |
| Elasticity | Standard | 128 / 8 | temp | 64 | 128 | 64 | 128 |
| Pipe | Standard | 128 / 8 | conv_temp | 64 | 128 | 64 | 572 |
| NS | Standard | 256 / 8 | plain | 32 | 64 | 32 | 256 |
| Plasticity | Standard | 128 / 8 | conv | 64 | 128 | 64 | 572 |
| AirfRANS | industrial | 256 / 8 | airfrans | 32 | 64 | 32 | 256 |
| ShapeNet-Car | industrial | 256 / 8 | shapenet | 32 | 64 | 32 | 256 |

The rank values come from the existing LinearNO profile at resolution time.
They must not be duplicated as task constants in model or launcher code. Car's
actual M must remain an integer multiple of `head_dim`.

The current call chains are:

```text
tran_evaluate/linearno_loop/<task>.sh
 -> tran_evaluate/linearno_loop/launch.py
 -> existing pure LinearNO task launcher (dry-run/argv reuse)
 -> task cdlno_entry.py or Car models/cdlno_run.py

Standard:
 exp_*.py -> PDE cdlno_entry.parse_args
 -> cdlno.linearno_loop.standard_entry
 -> model_dict loop guard -> model_module/build_from_config
 -> LoopedStandardModel -> original training/evaluation body

AirfRANS / Car:
 main.py or main_evaluation.py -> industrial parser adapter
 -> cdlno.linearno_loop.industrial_entry
 -> independent member/model construction
 -> LoopedAirfRANSModel or LoopedShapeNetModel
 -> original task training/evaluation body
```

Only explicit loop train flags or a saved `family=linearno_loop` sidecar load
the loop adapter. Pure LinearNO, history, and other model routes remain lazy.

## 5. Current v1 configuration and archive contract

The current constants are:

```text
family = linearno_loop
architecture_extension = loop_linearno_v1
config_version = 1
schema_version = 1
checkpoint format = linearno-loop-epoch-pair-v1
default rank_multiplier = 2
```

`linearno_loop.config.resolve_config` derives topology and rank, seals the
config hash, and emits a full explicit constructor specification. The schema
checks exact section/key sets and constructor signatures. Eval/resume first
reads `architecture.json`, restores the config, and treats any explicit CLI
structure values as equality assertions. The checkpoint reader validates
pointer/manifest/file hashes, immutable metadata and resolved config before
`torch.load(weights_only=True)`. Model state is then loaded strictly; optimizer
groups, moment shapes/dtypes, progress, scheduler, and RNG/generator state are
validated/restored by the task run adapters. A shared core is stored once.

Run identifiers include task, profile, P/C/R/S, residual, resolved M, seed, and
config hash. The three residual modes cannot load each other's archives.

## 6. Frozen v2 ownership and initialization plan

LF0 freezes the following plan for LF1--LF7. It is a design contract only; no
class below exists yet.

### 6.1 Version selection

- `family` remains `linearno_loop` so the task family remains identifiable.
- New architecture extension: `loop_linearno_ffn_v2`.
- New config and metadata schema version: 2, dispatched by saved version.
- New checkpoint format: `linearno-loop-epoch-pair-v2`.
- The only new mode field is `core_ffn_mode`, with exactly
  `round_specific` or `round_specific_latent`.
- The CLI spelling is `--linearno-loop-core-ffn-mode`.
- The field must be explicitly present for a new training run to select v2.
- An old command with no field continues through the current v1 parser,
  defaults, class paths, hash, Mx2 policy, run id, and checkpoint reader.
- Resume/eval dispatches from saved metadata before model construction and
  before loading tensor weights. Explicit v2 fields are assertions only.

Separate v2 class paths are required. Reusing the current v1 wrapper class with
optional kwargs would violate its strict historical constructor signature and
make v1 archive interpretation depend on new defaults. Planned stable paths:

```text
cdlno.linearno_loop.v2.standard.LoopedStandardModelV2
cdlno.linearno_loop.v2.airfrans.LoopedAirfRANSModelV2
cdlno.linearno_loop.v2.shapenet.LoopedShapeNetModelV2
```

The exact v2 schema package will live under `linearno_loop/v2/`; v1 modules
remain authoritative for version 1.

### 6.2 Module ownership

For each core position `p`, v2 registers exactly one operator owner containing
`ln_1+LinearNO operator`. It registers `C*R` point-FFN owners indexed by
`[p][r]`, each containing its own `ln_2+native PointwiseMLP`. Prefix/suffix
remain complete native blocks. Therefore the total number of point FFNs is
`P+C*R+S` and no operator is copied per round.

Latent mode additionally registers exactly C `LatentContextFFN` modules, one
per core position. The same module is passed non-owningly to that position's
operator on all R visits. It is neither aliased under a second state path nor
attached to prefix/suffix. Round-specific mode registers no latent module/key.

Planned v2 state partitions are unambiguous:

```text
loop.prefix.<p>.block.*
loop.core_operators.<p>.{ln_1,Attn}.*
loop.core_ffns.<p>.<r>.{ln_2,mlp}.*
loop.latent_ffns.<p>.{norm,linear1,linear2}.*   # latent mode only
loop.suffix.<s>.block.*
loop.<rb_* or lb_*>.*                          # selected residual only
```

Validation will reject parameter IDs appearing at multiple paths, discarded
or unused submodules, round-specific operator copies, shared `ln_2`, and any
inactive-mode router/latent keys.

### 6.3 RNG-safe construction

The two v2 modes construct their common stem, prefix, shared operators,
round-specific point FFNs, suffix, and routers in the same order. Release
initialization is applied once to that common tree; distinct `(p,r)` FFNs get
independent draws. The placeholder is then drawn at the same point in both v2
modes.

Only after the common initialization and placeholder draw, latent mode creates
and initializes its C latent modules inside an isolated `torch.random.fork_rng`
using a versioned feature seed recorded in metadata. This does not advance the
public model/data RNG. Each latent module uses affine LayerNorm eps 1e-5,
ordinary release initialization for W1, GELU, no dropout, biased Linear layers,
and W2 weight/bias reset to exact zero after all general initialization. No
later full-tree `apply` is allowed to overwrite W2.

This ordering must prove common state tensors are bitwise equal between the two
new modes and across SR/RB/LB. It does not force independent `(p,r)` FFNs to
start equal to one another.

### 6.4 Residual compatibility

The v2 core uses a single residual-mode enum. SR selects `operator[p]` and
`ffn[p][r]`, scaling both raw branches by 1/R. RB preserves every existing
receiver position/source and raw-partial rule, but selects the corresponding
round FFN. LB preserves actual `Y-H`, boundary/final sources, and branch-level
1/R. Latent processing is internal to the operator and receives no extra 1/R.
The LL9R local dtype boundary remains unchanged.

## 7. Versioned rank and run-directory migration

The v1 default multiplier remains 2 forever. V2 config resolution defaults to
multiplier 1, permits explicit multiplier 1/2 or explicit actual M, and retains
the existing mutual exclusion rule. Metadata records base M, resolved M,
policy, source, mode, latent width, feature seed, and all construction fields.

V2 run IDs must include `core_ffn_mode`, P/C/R/S, residual mode, resolved M,
seed, and v2 config hash. Thus v1, round-specific, latent, rank controls, and
residual modes cannot share a run directory. Cross-version/mode/residual/
topology/rank/width mismatch must fail from metadata before tensor load.

## 8. Planned minimal file map

The file map is deliberately versioned. Later stages may reduce the list after
evidence, but may not move new semantics into v1 silently.

| stage | planned additions / narrow routing changes | acceptance focus |
|---|---|---|
| LF1 | `linearno_loop/v2/{contracts,config,schema,matrix}.py`, v2 config/schema tests and reports | exact types, v1 replay, v2 Mx1, hashes, metadata-first conflicts |
| LF2 | `cdlno/linearno_loop/v2/{body,core}.py`, independent whole-core oracle/tests | shared operator; `(p,r)` ln_2+MLP; all residuals; no production route |
| LF3 | `cdlno/linearno_loop/v2/{attention,latent}.py`, six-variant oracle/tests | exact context location, zero W2, no M2/N2, AMP/VJP |
| LF4 | v2 `standard.py`, `airfrans.py`, `shapenet.py`, `construction.py`, `checkpoint.py` and synthetic wrappers/tests | initialization isolation, full strict roundtrip, old archive replay |
| LF5 | version dispatcher plus narrow Standard parser/factory/run adapter changes | six Standard native synthetic train/resume/eval; old routes unchanged |
| LF6 | narrow Air/Car dispatcher/state changes | PyG synthetic members/fold/resume and old industrial archives |
| LF7 | launcher/performance extensions and final reports | 8-task matrix, costs, full regressions, shell/compile/freeze |

The current `exp_*.py`, task data, loss, metric, optimizer, scheduler, temporal
loops, pure LinearNO math, history math, Transolver/CDLNO/KCDNO/MSAR-LNO, and
old launcher outputs remain frozen unless a later stage proves a narrowly
required routing change and records an exact compatibility projection.

## 9. Risk and acceptance matrix

| risk | required control | stage evidence |
|---|---|---|
| v1 archives reinterpreted with Mx1/new kwargs | version dispatcher; leave v1 constants/signatures/hash intact | LF1, LF4, LF7 old archive replay |
| operator accidentally copied per round | unique parameter IDs and absence of round operator keys | LF2/LF4 structure tests |
| only MLP unshared while ln_2 remains shared | each `(p,r)` owner includes both ln_2 and MLP | LF2 ID/key/hook tests |
| discarded full blocks consume RNG | construct operator and FFN primitives directly | LF2 construction/RNG evidence |
| latent changes common initialization | build common tree first; isolated feature seed after placeholder | LF3/LF4 bitwise common-state/RNG tests |
| general init overwrites zero W2 | final explicit zero and no later whole-tree apply | LF3/LF4 state assertions |
| context inserted after Q readout or into prefix/suffix | six-variant oracle and call-location hooks | LF3/LF4 |
| latent mixes M or creates MxM/NxN | tokenwise shapes plus profiler/hook assertions | LF3/LF7 |
| RB/LB source timing changes | full-core independent oracle for every intermediate | LF2/LF4 |
| AMP regression at RB receiver boundary | FP32/FP16/BF16 all modes and wrappers | LF2/LF4/LF6/LF7 |
| duplicate state aliases or unused modules | named parameter IDs, active gradient, strict key partitions | LF2--LF4 |
| task protocol drift | native synthetic closure and exact source projection | LF5--LF7 |
| run collision/cross-mode resume | versioned run id and pre-load metadata conflict tests | LF1/LF4--LF7 |

No unexplained ownership, initialization, insertion-point, or archive migration
conflict remains at LF0. This is why LF0 is PASS. Production implementation is
still absent and requires separate LF1 authorization.

## 10. Numerical fixtures and regressions

The pre-change fixture set covers six variants, two presets, three residual
modes, and five device/precision plans:

| plan | cases | result |
|---|---:|---:|
| CPU FP64 | 36 | 36 PASS |
| CPU FP32 | 36 | 36 PASS |
| CUDA FP32 | 36 | 36 PASS |
| CUDA autocast FP16 | 36 | 36 PASS |
| CUDA autocast BF16 | 36 | 36 PASS |
| total | 180 | 180 PASS |

Each artifact contains config, inputs, pre-step state, output, loss, input and
parameter gradients, AdamW state/updated weights, CPU/CUDA RNG, state key/shape,
parameter count, and operator/point-FFN/head call sequence. The 65 MB fixture
root is `/home/hwz/CDLNO-artifacts/loop-ffn-lf0-baseline-attempt2`; the checked
index is [numeric-fixtures.json](loop_linearno_ffn_audit/lf0/numeric-fixtures.json).
An initial capture failed only because the evidence script attempted to view a
scalar loss as bytes without reshaping it. The retained
`numeric-fixtures-attempt1.log` records this. The LF0 script alone was fixed;
no model source changed.

The current 80-module inventory ran 689 unittest methods in 2,641.482 seconds:

```text
644 passed, 9 failed methods, 36 skipped
```

This differs from LL9R's historical `653 passed, 36 skipped, 0 failed` because
historical untracked/ignored evidence was later deleted and two old provenance
fixtures no longer describe the current checkout. The nine failed methods are:

1. Four KCDNO frozen-source methods cannot open deleted
   `docs/kcdno_audit/k4|k5|k6|k7/before.json`.
2. `linearno.test_legacy` enters one method with 218 missing-file subtest errors
   for deleted historical `docs/CDLNO_*`, `docs/KCDNO_*`, and
   `docs/kcdno_audit/**` files. A second method fails before its namespace
   assertion because deleted `docs/kcdno_audit/audit_static.py` cannot be read.
   These count as two failed methods, not 219 individual methods.
3. `loop_linearno.test_isolation` enters one method with 259 subtest failures
   from the same missing historical inventory/classification drift. This counts
   as one failed method, not 259.
4. `loop_linearno.test_launchers` checks an ignored Python bytecode hash. The
   current `cdlno/__pycache__/experiment.cpython-313.pyc` hash `e906...` differs
   from historical `72f2...`; the LF0 start manifest proves this predates LF0.
5. `loop_linearno.test_standard_entry` expects old LL6
   `normalized_patch_sha256=a535...`; current provenance reports `1933...`
   after later approved routing/observation changes.

All current core/body/SR/RB/LB/mode, pure LinearNO, history, task model,
checkpoint/resume, monitor, and slow industrial/static/temporal model modules
completed without a new math failure. LF0 did not repair missing files, bytecode
goldens, or provenance goldens. Exact module/method records are in
[regression-summary.json](loop_linearno_ffn_audit/lf0/regression-summary.json)
and raw logs under `loop_linearno_ffn_audit/lf0/regressions/`.

The numerical matrix ran locally with Python 3.13.9, torch 2.13.0+cu130, and an
NVIDIA GeForce RTX 5090 Laptop GPU; TF32 and cuDNN benchmark were disabled and
BF16 was available. The full legacy regression inventory deliberately used
`CUDA_VISIBLE_DEVICES=''`, one BLAS/OpenMP thread, `MPLBACKEND=Agg`, and
`PYTHONDONTWRITEBYTECODE=1`; its 36 skips therefore include the established CPU
suite's CUDA skips. The independent 108-case CUDA numerical matrix supplies the
LF0 GPU/AMP baseline.

Actual command forms were:

```bash
LOOP_FFN_LF0_ARTIFACT=/home/hwz/CDLNO-artifacts/loop-ffn-lf0-baseline-attempt2 \
  python -B docs/loop_linearno_ffn_audit/lf0/capture_numeric.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/hwz/CDLNO/tests:/home/hwz/CDLNO \
  /home/hwz/anaconda3/bin/python -B -c '<unittest inventory runner>' <module>

python -B docs/loop_linearno_ffn_audit/lf0/finalize_lf0.py
git diff --check
```

The exact 80 per-module argv values, environment, timings, failures, errors,
and skips are stored in `regression-results.json`, `regression-inventory.json`,
and `test-environment.json`; the placeholder above denotes that recorded inline
runner body rather than a second undocumented script.

## 11. Self-review and limits

Priority review points checked against source and numerical evidence:

1. Current v1 owns complete core blocks and therefore shares both operator and
   point FFN across rounds; the new ownership change cannot be implemented as a
   small flag inside that tree without changing keys and construction.
2. The latent insertion boundary is exactly K^T V to Q readout for all six
   variants; pure attention need not be edited.
3. A separate v2 schema/class/checkpoint path is required to retain v1 strict
   constructor replay and Mx2 hashes.
4. Common v2 initialization must finish before isolated latent construction;
   W2 must be zeroed after any generic initialization.
5. The current regression baseline is not fully green. The nine method
   failures are recorded rather than hidden, while 180 fresh model fixtures
   establish a usable numerical pre-change archive.

Four ignored `output/{airfrans,car}/<timestamp>/architecture.json` sidecars
created by regression tests were absent at LF0 start and were removed during
cleanup; no pre-existing output was deleted. The final freeze separately checks
all 3,031 hashed start files and the 1,415-file protected subset.

NOT RUN: real data loaders, real VTK metrics, real task training, full epochs,
convergence, accuracy/SOTA, real epoch throughput, remote Python 3.10 / torch
2.11 / CUDA 12.8, distributed, or `torch.compile`. LF0 makes no performance claim.

本 LF0 阶段结束，未执行下一阶段
