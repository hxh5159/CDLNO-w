# AGENTS.md

## Current phase and reading order

Latest scope (2026-09-14): the user requested a final logic check and Car/NS launch scripts in `tran_evaluate/`, without authorizing real training. This launch-preparation review is complete: see [the launch review](docs/CDLNO_TRAINING_LAUNCH_REVIEW.md) and [commands/protocols](tran_evaluate/README.md). 120 final regressions and four real-parser dry-run commands passed. Keep confirmed CDLNO M64/ratios2/2+6; the new NS preset's accidental clip0.1 was corrected to null/None to match the original NS launcher. Original entry/data/train/model/metric files were not changed. Car's frozen drag evaluator hardcodes the canonical raw root and param0; the new full-eval launcher checks/requires fold0 and that same raw dataset path. Car's inherited outer log-summary pressure/velocity swap does not affect its correct internal backward loss; report it rather than silently fixing the frozen train file. No real data, convergence, remote-new-model or full metrics acceptance has occurred.

The user requires **one explicitly assigned implementation phase per turn**. Complete necessary changes and verification autonomously within that phase, then stop for user review and an explicit instruction for the next phase. Never proceed through the entire plan automatically. This user instruction replaces the all-at-once authorization model in v1.2 §11. **Phases 0–9 were reviewed and approved. Phase10 final audit and delivery are complete, awaiting user review. No real-data training, sampling evaluation, additional research or next phase is authorized.** All eight task interfaces/configurations are integrated; no real-data training or sampling evaluation has run. Read [the final implementation report](docs/CDLNO_IMPLEMENTATION_REPORT.md) and [requirements matrix](docs/CDLNO_REQUIREMENTS_MATRIX.md), then [the phase-9 performance report](docs/CDLNO_PHASE9_PERFORMANCE.md) and [tool usage](docs/CDLNO_PERFORMANCE_TOOLS.md), then [the phase-8 AirfRANS report](docs/CDLNO_PHASE8_AIRFRANS.md), [the eight-task launch inventory](docs/CDLNO_TASK_LAUNCHERS.md), [the phase-7 ShapeNet-Car report](docs/CDLNO_PHASE7_SHAPENET_CAR.md), [the phase-6 temporal-task report](docs/CDLNO_PHASE6_TEMPORAL_TASKS.md), [the phase-5 static-task report](docs/CDLNO_PHASE5_STATIC_TASKS.md), [the phase-4 core report](docs/CDLNO_PHASE4_CORE.md), [the phase-3 CDPA report](docs/CDLNO_PHASE3_CDPA.md), and [the phase-2 module report](docs/CDLNO_PHASE2_MODULES.md) before continuing; further integration requires a new explicit instruction.

Before future CDLNO work, read [docs/CDLNO_IMPLEMENTATION_STATUS.md](docs/CDLNO_IMPLEMENTATION_STATUS.md), [memory/current-state.md](memory/current-state.md), the [v1.2 plan](PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md), and the [full-conversation review](memory/2026-09-13-exported-conversation-review.md). The [151-page conversation export](PLAN_CDLNO/比较模型架构.pdf) explains the design's evolution; it was read from start to finish on 2026-09-13. Design precedence is: **current/subsequent explicit user decisions → later confirmed conversation decisions → v1.2 complete plan → v1.1 historical supplement**. Distinguish confirmed choices from historical proposals and abandoned branches. Mathematical attachments explain/check the design; they do not authorize additional structures, losses, or research features.

The user has now explicitly confirmed **CDLNO — Cross-Depth Latent Neural Operator** as the formal model name, **`cdlno`** as the shared package, and **`CDLNO`** as the core class. Map old provisional package `cdpa_operator` → `cdlno`, core `CDPAOperator` → `CDLNO`, and whole-model registration `CDPA` → `CDLNO` when the relevant implementation phase is assigned. Existing entry-point wrappers may still export `Model`. The mechanism remains **CDPA — Cross-Depth Physics Attention**: do not globally replace mechanism names or `cdpa_*` mechanism settings.

## Standing implementation and reporting constraints

- Preserve original Transolver models/scripts and the data contracts: readers, fields, splits, sampling, point order, normalization, targets/channels, losses, time loops, optimizers/schedulers, and evaluation processing. Changes are limited to the new model and necessary model selection, argument forwarding, isolated output paths, and checkpoint integration within the assigned phase.
- Preserve the default 2 complete LRSA blocks → IPOT-style bridge → entry CDPA → 6 independent persistent latent blocks → LRSA feature decoder. Query and point residual use `H_F`; structured tasks use post-up dense 3×3 ConvFFN. Norms, Q/K norms, FFN, biases, and initialization follow the plan. Support configurable L/F, P=L−F, F=0…6 and larger L, constant M across stages, and `off`/`entry`/`every_block` plus equivalent source batching/chunking in their assigned phases. Old parser defaults must not silently override new task defaults.
- No real data downloads or real training. Validate only the assigned phase using static review, synthetic tensors, original losses, forward/backward, and checkpoint checks as applicable. Never import `exp_*.py` entry scripts that read datasets at module scope. Report passed, failed, and not-run checks separately; theory/environment checks are not model acceptance.
- Inspect and preserve the user's working CUDA 12.8 / torch 2.11 environment. Do not blindly reinstall, upgrade, or downgrade dependencies. If GPU access is unavailable, mark GPU checks not run. Use PyTorch SDPA, with no custom CUDA/Triton or reference-project training frameworks.
- Remote compatibility evidence: the user reports that Python 3.10 with the PyTorch 2.11/cu128 stack and PyG/pyg-lib installed from the matching cu128 wheel source runs the original ShapeNet-Car training successfully. Treat this as the current anchor; Python 3.11 and the other tasks remain unverified. The final `torch_geometric` version must be read from the remote environment because the reported command pins 2.4.0 and then upgrades it.
- Do not implement sparse Darcy, coordinate-query decoding, increasing M, CDPA-Slice, abandoned correction layers, new physical losses, latent convolutions, cross-physical-time latent caches, or NS 10→20/40 without matching long-trajectory labels. Do not add these as speculative extension hooks.
- Read applicable AGENTS.md and preserve existing user changes. No automatic commit/push, PR creation, training, reset, or unrelated refactoring. If a material architecture/data-protocol conflict appears, report sources and a recommendation rather than changing the design independently.
- Update [docs/CDLNO_IMPLEMENTATION_STATUS.md](docs/CDLNO_IMPLEMENTATION_STATUS.md) and memory after each phase. Final reports must include **A** completed scope; **B** files/reasons/diff summary; **C** formulas/shapes mapped to code; **D** actual commands/environment and passed/failed/not-run checks; **E** frozen-region changes and evidence; **F** unresolved issues. **Before delivery, self-review the 3–5 priority review points against source, the confirmed plan and meaningful evidence; do not hand unexamined checkpoints back to the user.** The user now asks that only remaining defects, uncertainties or decisions needing their judgment be raised after this self-review. Record passed review evidence in the report; do not invent open questions when none remain. This changes the review workflow, not the one-phase authorization boundary. End with **“本阶段结束，未执行下一阶段”** and wait for an explicit next-phase instruction.

## Repository scope

This working repository is based on the official Transolver implementation associated with **“Transolver: A Fast Transformer Solver for PDEs on General Geometries” (ICML 2024)**. It is organized as three runnable experiment projects plus a standalone reference attention implementation:

- `Physics_Attention.py`: root-level reference implementations for irregular meshes, structured 2D meshes, and structured 3D meshes.
- `PDE-Solving-StandardBenchmark/`: six PDE benchmarks (Elasticity, Plasticity, Airfoil, Pipe, Navier–Stokes, Darcy), with models under `model/`, experiment entry points `exp_*.py`, launcher scripts under `scripts/`, and metrics/normalizers under `utils/`.
- `Car-Design-ShapeNetCar/`: ShapeNet car CFD surrogate. It preprocesses VTK data into PyTorch Geometric graphs, predicts velocity and pressure, and evaluates drag coefficient and rank correlation.
- `Airfoil-Design-AirfRANS/`: AirfRANS CFD surrogate. It reads the AirfRANS manifest and VTK/NumPy data, supports full/scarce/OOD Reynolds/OOD angle-of-attack splits, and evaluates flow fields and lift-related metrics.

The repository is `https://github.com/thuml/Transolver.git`; the actual checked-out origin uses `git@github.com:thuml/Transolver.git` (same repository). Keep the upstream citation and dataset attributions in place when modifying the repository.

## Scientific model contract

This section describes the original Transolver baseline. The new CDLNO architecture follows the extension specification below; it does not inherit Transolver's slice/deslice mechanism merely because this is the base repository.

Transolver replaces point-to-point self-attention with Physics-Attention over learned physical states:

1. Embed geometry `g` and optional observed field `u` into point features `x` with shape `[B, N, C]`.
2. Project each point to `M` slice logits and apply Softmax over slices, producing `w` with shape `[B, H, N, M]` in the multi-head implementation.
3. Aggregate normalized slice tokens: `z_j = sum_i(w_ij * x_i) / (sum_i w_ij + 1e-5)`.
4. Apply scaled dot-product multi-head attention to the `M` tokens.
5. Deslice by broadcasting tokens back with the same weights: `x'_i = sum_j(w_ij * z'_j)`.
6. Use a pre-norm residual Transformer block: attention residual followed by MLP residual; the final block projects to the requested output channels.

The intended complexity is `O(N*M*C + M^2*C)`. `M` is a small fixed number (the paper commonly uses 32 or 64), so memory and compute should remain effectively linear in the number of mesh points. Do not silently replace this with canonical `N x N` attention.

The three attention classes have different contracts:

- `Physics_Attention_Irregular_Mesh`: pointwise linear projections; accepts arbitrary `[B, N, C]` point sets.
- `Physics_Attention_Structured_Mesh_2D`: reshapes `[B, N, C]` to `[B, C, H, W]`, applies local Conv2d projections, then flattens back. `N` must equal `H*W`.
- `Physics_Attention_Structured_Mesh_3D`: analogous Conv3d path; `N` must equal `H*W*D`.

Preserve the temperature parameter, its clamp in structured attention (`0.1` to `5`), orthogonal initialization of the slice projection, the `1e-5` slice normalization epsilon, and the residual/pre-norm ordering unless there is a documented reason to change the scientific method. If changing tensor layouts, verify all `einops`/`einsum` dimensions explicitly.

## Data, metrics, and experiment conventions

The paper evaluates relative L2 for physics fields:

`||prediction - target|| / ||target||`.

Car and AirfRANS additionally derive drag/lift coefficients from surface pressure, velocity/shear, normals, inlet direction, and reference area, and report relative coefficient error plus Spearman rank correlation. AirfRANS field values in the paper are MSE for volume and surface fields (the repository README records this correction), not relative L2. Preserve this distinction in evaluation code and reports.

Canonical benchmark sizes and splits used by the official experiments are:

- Elasticity: 2D point cloud, 972 points, 1000 train / 200 test, stress output.
- Plasticity: structured 2D+time, `20 x 101 x 31 x 4`, 900 train / 80 test.
- Airfoil: structured `221 x 51`, 1000 / 200, Mach output.
- Pipe: structured `129 x 129`, 1000 / 200, velocity output.
- Navier–Stokes: regular `64 x 64`, past 10 steps to future 10 steps, 1000 / 200.
- Darcy: original `421 x 421`, usually downsampled to `85 x 85`, 1000 / 200.
- ShapeNet Car: 32,186 unstructured points, 789 train / 100 test in the official split.
- AirfRANS: 32,000 points, 800 train / 200 test; full, scarce, unseen Reynolds, and unseen angle-of-attack tasks are supported.

Benchmark scripts generally use AdamW, OneCycleLR, configurable hidden width/layers/heads/slice count, optional gradient clipping, and CUDA selected through command-line arguments. Dataset paths are not bundled; never hard-code a local machine path into new code or commit downloaded datasets/checkpoints.

## Where to make changes

- Add or modify standard-benchmark architectures in `PDE-Solving-StandardBenchmark/model/`; register a new standard model in `model_dict.py` and add a launcher under `PDE-Solving-StandardBenchmark/scripts/`.
- Add AirfRANS models in `Airfoil-Design-AirfRANS/models/`, add hyperparameters to `params.yaml`, and wire the model name in `Airfoil-Design-AirfRANS/main.py`.
- Add ShapeNet Car models in `Car-Design-ShapeNetCar/models/` and select them in `main.py`.
- Keep dataset parsing/preprocessing in the relevant `dataset/` directory and metric/visualization code in `utils/`.
- The experiment projects import their local Physics-Attention copies; the root file is a reference, not a shared runtime import. Audit the actual imported path and initialization when comparing behavior; do not silently synchronize baseline copies. The new architecture is planned as one shared core with task adapters.

## Running and validating changes

When training is separately requested and the environment/data are available, run each original project from its own directory so imports such as `model.*`, `utils.*`, and `dataset.*` resolve correctly. Use the user's remote Python 3.10/3.11, PyTorch 2.11, CUDA 12.8 environment as the compatibility authority. **Do not reinstall the old requirements blindly:** the standard benchmark pins PyTorch 1.10.1, which conflicts with the planned SDPA implementation. This phase does not substitute local package versions for the remote environment.

```bash
cd PDE-Solving-StandardBenchmark
bash scripts/Transolver_Elas.sh   # or the task-specific launcher

cd ../Car-Design-ShapeNetCar
bash scripts/Transolver.sh

cd ../Airfoil-Design-AirfRANS
bash scripts/Transolver.sh
```

Before a long training run, perform a CPU/small synthetic smoke test: instantiate the changed model, pass a tensor with the expected `[B, N, C]` shape (and matching `H/W/D` for structured models), and check output shape, finite values, and a backward pass. Full experiments require the external datasets and a CUDA-capable PyTorch/PyG environment; they are not expected in ordinary code review.

When changing data loading or evaluation, test both training and evaluation paths, normalization encode/decode, checkpoint load/save, and the relevant split (`full`, `scarce`, `reynolds`, or `aoa`). Keep generated `checkpoints/`, `results/`, `metrics/`, `scores/`, preprocessed data, and downloaded archives out of version control.

## Style and compatibility

Use PyTorch tensors and the existing naming/layout conventions; avoid unrelated rewrites. The original repository documents Python 3.8, while the new plan's candidate environment uses Python 3.11 with torch 2.11/cu128; inspect the user's actual working environment before dependency changes. Keep CLI flags backward compatible where practical, especially `--data_path`, `--data_dir`, `--save_dir`, `--my_path`, `--save_path`, `--model`, `--eval`, and `--gpu`.

Document any change that alters tensor shapes, dataset splits, normalization, loss definitions, coefficient computation, or reported metrics. Cite the Transolver paper in new scientific documentation and retain the project licenses.

## CDPA extension plan and project memory

The current architecture plan is [PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md](PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md), subject to the user-defined precedence and phase boundary above. It is a design specification, not evidence that CDLNO has been implemented or trained. Only an explicitly assigned phase authorizes its implementation changes; the rest of the plan remains pending.

The plan's authoritative design is a configurable `L`-block model with `F` complete LRSA-style point/latent blocks (`P=L-F` persistent IPOT-style latent blocks), default `L=8,F=2`, and CDPA mode `entry` as the main setting. `off` and `every_block` are required for no-data validation and later ablations. Front history is captured after latent FFN2 and before up-attention; the final decoder uses the retained `H_F` point features. CDPA aligns every historical source independently with current-to-history Cross attention, then applies token-wise depth softmax over raw source candidates plus identity `R_0=Z`; its scorer is zero-initialized and depth fusion is computed in FP32. Source chunking must be mathematically and gradient equivalent and must not concatenate all source tokens into one softmax.

Additional decisions recovered from the complete export:

- A full front block still executes up-attention and point FFN after saving `T_i`; an independent bridge encodes the resulting `H_F`. Default 2+6 performs three down/bridge and three up/readout operations, eight latent SA operations, and three structured ConvFFNs.
- CDPA's zero scorer initializes a uniform mean of all raw candidates, not identity. Do not add an outer `Z + fused`, identity bias, or gate. With no history, fusion is identity. The current core omits CDPA registration at inactive locations (including F=0/entry); a standalone CDPA instance still owns its parameters when called with empty history.
- `every_block` at rear block j uses current identity `Z_(j-1)` and Cross history `[T_1,...,T_F,Z_0,...,Z_(j-2)]`. Keep the raw bridge `Z_0`; never duplicate the current state or persist history across forwards/time steps.
- Projections and normalizations are shared across sources **within one CDPA location**, but different locations have independent parameters. The early proposal to cache projected K/V across destination layers was superseded (PDF pp.145–147). Cache raw history references with gradients; no cross-location projected K/V or learned-LN cache.
- The user abandoned Gram correction (p.82), corrected point-memory retrieval to latent-history reuse (pp.103–104), chose LRSA's post-up ConvFFN (pp.128–129), and selected feature-conditioned final readout for the eight current tasks (pp.136–143). Do not revive the earlier alternatives as implementation requirements.
- NS keeps teacher-forced training and ten-step autoregressive evaluation; persistent latent refers to network depth within one forward. Plasticity keeps separate time-conditioned calls and per-time optimizer updates. Pipe's actual batch size is 8; AirfRANS uses 398 epochs. Preserve actual repository protocols rather than early approximate conversation tables.
- Theory must include the final `H_F` bypass. At a single output point, off accesses local `H_F` (one point or the final 3×3 neighborhood) plus `Z_0`; CDPA can conditionally supply historical global information absent from that access. This is not a proof of lossless compression, universal low rank, strict containment of retrained off models, or guaranteed speed/accuracy improvement. Theory attachment checks reported by the historical assistant are not checks run in this workspace.

The plan freezes the eight existing task protocols: loaders, fields, point order, splits, sampling, normalization, targets, losses, metrics, time loops, optimizers, and schedulers. It explicitly excludes sparse Darcy, latent-count schedules, NS long-horizon 10→20/40, CDPA-Slice, correction layers, extra PDE losses, latent convolutions, cross-layer tying, new geometry encoders, and automatic sweeps. Structured tasks use the LRSA-style regular `Conv2d(...,3,padding=1,groups=1)` point FFN; irregular tasks use point FFN. PyTorch SDPA is required; custom CUDA/Triton, xFormers, Lightning, Hydra, and new training frameworks are out of scope.

Reference audit snapshots reviewed for the plan:

- LRSA-Operator commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`.
- IPOT commit `18c177846267505ee9503445a146dfd7dee34c41`.

Phase-2 modules live in `cdlno/modules.py`; importing `cdlno` configuration remains independent of torch. Front/down/up and readout branch norms use RMSNorm (including independent per-head Q/K norms); bridge/rear and the explicit `LN_out` task head use LayerNorm, all eps=1e-6. ConvFFN has an internal affine LayerNorm and a regular groups=1 3x3 Conv2d with native initialization. Initialize ordinary Linear layers once with trunc_normal(std=.02), set special queries once, and do not recursively reset a constructed block. Front returns `(H_next, T)` with the live history before up; it never detaches or caches across calls. The local phase-2 tests cover independent modules, not a complete model or the user's remote environment. CPU BF16 tests disable MKLDNN locally to bypass an unsupported oneDNN backward; GPU parity tests disable TF32 locally and restore settings. Do not copy those test backend choices into production defaults.

Phase-3 fusion lives in `cdlno/cdpa.py`, imported explicitly as `from cdlno.cdpa import CDPA`; root configuration imports remain torch-independent. `CDPA(dim, heads, source_chunk_size=0)` accepts `(Z, history)` with per-call chunk override and optional FP32 `[B,M,S+1]` source weights. It projects Q once, batches sources on `[B*k,h,M,d_h]` with key length M, completes each O+b without residual, then fuses RAW candidates once over all sources plus identity. LN_q/LN_kv are affine LayerNorm eps=1e-6; Q/K/V have no bias, O has bias, no QK norm. Depth RMS scale1/no bias and zero w implement uniform initialization. The entire depth RMS/scoring/softmax/accumulation region disables autocast and uses FP32 before returning Z.dtype. Different floating history dtypes are cast to Z.dtype for Cross while preserving gradients. Empty history returns the exact input without projection or SDPA; phase 4's core now owns history timing and omits inactive CDPA modules.

Phase-4 core assembly lives in `cdlno/core.py`, imported as `from cdlno import CDLNO`. `CDLNO` accepts already-lifted `H_0[B,N,d]`, builds F independent `LRSAFrontBlock`s, one `IPOTBridge`, P=L−F independent `PersistentLatentBlock`s and one `LRSAFeatureReadout`. It creates CDPA children only at active locations: none for off, one entry location when F>0, and every location with nonempty history. On every forward it rebuilds a local history list, passes immutable tuple snapshots, keeps raw `Z_0`, appends rear identities only after use, preserves gradients and returns only `[B,N,C_out]`; no position/time encoding or task/data imports are present. Phase-4 core/configuration tests passed 19/19 (13 core + 6 configuration/loading); full local regression with the audited LRSA checkout passed 60/60, including local GPU tests. Actual SDPA spies verify default chunk-0 entry: 2 logical sources / 1 history SDPA, every: 27 / 6. Complete raw-history object IDs, two-forward graph isolation, all active gradients, real chunk changes, sidecar byte preservation, and fresh-process checkpoint/import checks passed. This is not task-wrapper or remote-environment acceptance. A follow-up self-review found the runtime chunk type-validation gap; phase 5 closed it when connecting strict task checkpoints. Runtime config now rejects bool, positive float, NaN and Inf, with validate/from_dict/save/load/compare regression coverage. This does not change the mathematical model or architecture compatibility rules.

Important source facts: LRSA's complete block is down → latent FFN1 → latent self-attention → latent FFN2 → up; its structured `dwconv` name still denotes a regular channel-mixing Conv2d (`groups=1`). IPOT's encoder uses learned query residual plus cross-attention and declares an unused encoder FFN; its processor accidentally reuses module objects when repeatedly appending layers, so new rear blocks must be independently instantiated. These reference projects' training/data frameworks are not to be copied wholesale.

Repository memory is kept in [`memory/`](memory/). Read [`memory/current-state.md`](memory/current-state.md) before future CDPA work and update it whenever implementation, validation, or scope status changes. The detailed source audit is [`memory/2026-09-13-cdpa-plan-and-reference-audit.md`](memory/2026-09-13-cdpa-plan-and-reference-audit.md); page-indexed design history and theory limits are in [`memory/2026-09-13-exported-conversation-review.md`](memory/2026-09-13-exported-conversation-review.md). Planned, implemented, tested, skipped, and unknown states must remain explicitly separate; synthetic checks never count as real-data training.


## Four static task adapters (phase 5)

Only Darcy, Elasticity, Airfoil and Pipe select `--model CDLNO` through the standard benchmark factory. The two thin `model/CDLNO_*` modules share `cdlno.standard.StaticStandardModel`; the core and phase-2/3 mathematics are unchanged. `cdlno_entry.py` supplies only new-model parser defaults/kwargs, isolated run directories and strict checkpoint branches; it is not a training framework. Four JSON presets and eight CDLNO scripts explicitly set the confirmed task/model/training defaults, with user arguments last. Legacy defaults, model kwargs and Transolver scripts remain intact.

Darcy uses the original structured index-grid reference distances (not distances recomputed from input xy), replacing xy before concatenating fx1: stem width65 at ref8. Elasticity/Airfoil/Pipe preserve existing input-coordinate and placeholder semantics; Pipe's original exp normalizes coordinates before the model. Placeholder is registered only on the three actual fx=None tasks. These static wrappers reject Time_Input=True/T input and have no time_fc. For all future wrappers, register placeholder only for actual fx=None paths and time projection only when Time_Input=True. Never recursively apply initialization after constructing the shared core.

Task checkpoint validation requires both core architecture and the mandatory `metadata.wrapper_architecture` semantics (including position/ref/placeholder/stem version). `StaticRun` reads existing sidecars first, then compares and strictly loads weights_only state_dict; generic core comparison alone is not sufficient for a wrapper. Training reserves a unique/new directory; eval requires an explicit existing run, does not resave weights, and uses a new results subdirectory. Existing normalizers, losses, optimizers/schedulers, loops and evaluation arrays are frozen. Original hardcoded plotting dimensions are retained; nondefault downsample visualizations are not newly supported.

Phase5: 13 task tests and final 73/73 full regression passed. Phase6 adds NS/Plasticity temporal wrappers and 8 focused tests; final full regression is 81/81. Real-N synthetic backward was executed at7225/972/11271/16641 with reduced d8/M4; real Darcy decode/zero-boundary/gradient-loss statements were executed through selected AST nodes. Twelve small wrapper GPU FP32/FP16 AMP/BF16 AMP cases passed locally; no real datasets, exp imports, training or remote torch2.11/cu128 run. Removing only the approved new branches/path aliases from four exp ASTs reconstructs the exact original modules; all93 non-target baseline files remain byte-identical. Self-review was completed before delivery; no remaining new implementation defect was found.


## Phase6 temporal task boundary

NS and Plasticity are now integrated under the explicitly authorized phase6. The shared temporal adapter keeps the original external `forward(x, fx, T=None)` contract while rebuilding the CDLNO core on every call. NS uses fx[B,N,10] as its rolling ten-step window, with truth feedback in training and prediction feedback in evaluation handled solely by the unchanged exp loop. Plasticity uses fx[B,N,1], T[B,1], output[B,N,4], and its original per-time optimizer updates. No temporal latent/cache crosses calls. Both industrial tasks remain pending.


## Phase7 ShapeNet-Car boundary (2026-09-14)

ShapeNet-Car is integrated through stable `models.CDLNO.Model` in its original working directory. It consumes only `cfd_data.x[N,7]` plus batch/ptr validation and returns `[N,4]` in velocity3/pressure1 order. Preserve the original active fx=None placeholder; there is no time projection, geometry encoder, label read, input mutation or node sorting. Reject multiple graphs via either batch or ptr; ordinary Data and all-zero single-graph Batch are valid.

`models/cdlno_run.py` supplies only new-model CLI/defaults, unique directories and sidecar/whole-model validation. Keep the original `train.py` and its `torch.save(model, ... model_<nb_epochs>.pth)` untouched. Eval reads and compares sidecar before the local trusted `weights_only=False, map_location=device` load, validates fold/run contract, model type/config and strict state keys/shapes. Runtime chunk changes remain allowed. Existing explicit training directories are rejected; each evaluation uses a new results directory. The only baseline eval compatibility changes are nb_epochs float→int and that local trusted whole-model load.

Phase7 final: 13 focused tests and 94 total regressions passed, with no failures/errors/skips; additional real PyG CUDA FP32 checks in all three modes passed. Full AST projection of both entry files and 37 original Car/AirfRANS file comparisons confirm frozen regions. No data, dependency installation or real training. AirfRANS is still unmodified and requires explicit next-phase authorization. See the phase7 report for commands, intermediate failures/corrections and remote-validation boundaries.


## Phase8 AirfRANS boundary (2026-09-14)

AirfRANS exports Model from its local models/CDLNO.py but serializes the unambiguous shared class `cdlno.airfrans.AirfRANSModel`. Preserve x7, pos2 and output vx/vy/p/nut; append64 reference distances from original pos, domain x[-2,4]/y[-1.5,1.5], giving stem71 plus the active fx=None placeholder. No time_fc, new geometry encoder, label read, mutation or node sorting.

Original Infer_test slices x/pos/y/surf/batch while retaining the original single-graph ptr. The Air adapter accepts this when batch is all zero and ptr=[0,original_N] with original_N>=current_N. It still rejects multiple graphs via either metadata representation. Do not modify original sampling/ptr/scatter code to accommodate the model. Car retains its previously approved stricter unsampled interface.

Only append the CDLNO YAML key, inheriting the actual Transolver fields (currently398epochs/batch1/lr.001/subsampling32000/r.05/max_neighbors64). Keep train.py, dataset, graph construction and metrics/postprocessing untouched. Train --my_path is Dataset itself; eval --my_path is its parent. New scripts document this and pass user overrides last.

AirRun preserves full-model saves (member_000/model etc) and the original models list (run-root CDLNO). Eval reads sidecar before local trusted weights_only=False/map_location loading, checks architecture/wrapper/task/nmodel/training and sampling settings, exact class/list length and strict state keys/shapes. Runtime chunk may change. Run/member/eval paths are isolated; no new resume.

Phase8: 13 focused tests and107 full regressions passed, no failures/errors/skips. Actual CUDA FP32 PyG checks for three modes passed separately. Two independent Air-cwd processes execute extracted real construction/save/evaluation-load branches without data loading. The complete entry ASTs, YAML old keys and21 frozen original Air files are checked. No actual sampling evaluation/VTK/radius_graph/metrics workflow, remote target or training has run. See phase8 report for its historical evidence limits; phase9 is now approved and phase10 delivery is complete.


## Phase9 synthetic performance boundary (2026-09-14)

`tools/cdlno_benchmark.py` and `tools/cdlno_perf/` are separate from all production model/training registries. `LRSAMatched` reuses the same task wrapper/stem and output LN/head, with L independent complete LRSA blocks; no bridge/CDPA/extra up and no F=L workaround. Original Transolver source is loaded in an isolated namespace with only local-import resolution and hardcoded cuda-to-requested-device adaptation; baseline files/math remain unchanged.

Cost auditing counts every dense Linear/Conv and actual SDPA QK/AV shape, both front latent FFNs, rear GEGLU, N projections, CDPA Q once and per-source K/V/O. It records non-matrix norm/depth/softmax/operator inventories, actual temporary materializations and saved activation/storage payloads separately from CUDA allocator peaks. Matrix MAC uses 2 FLOPs/MAC; profiler FLOPs are explicitly partial. Source batching does not reduce mathematical MACs.

Final 118/118 regressions passed (59.769s, no failures/errors/skips). Three sequential finite FP32 GPU cases produced 27 successful rows, preserving same-weight chunk0/1/2 outputs/all gradients: Elasticity matched at N972, Airfoil task presets at221x51/B4, and Airfoil matched at17x23/B2. Actual backend was efficient SDPA, not flash; all use TF32/AMP/compile off,5warmup/20samples, synchronized median/p90 and initialized AdamW state. Reports/logs are persistent under docs/performance/phase9. Earlier /tmp results disappeared at continuation and are not final evidence.

Elasticity entry reduces MACs but its synthetic train step was slower than original Transolver; additional small operators/launches and optimizer tensors are reported. Airfoil task presets differ in heads and must remain distinct from matched comparisons. Do not infer epoch speed, universal superiority or an optimal chunk from these finite noisy timings. Remote torch2.11/cu128, AMP/TF32-on/forced-other-backend/compile performance paths and true training remain unverified. No production package, task, dependency, data or training changes in this phase. See phase9 report for self-review and freeze evidence; stop for explicit phase10 authorization.


## Phase10 final audit and delivery (2026-09-14)

Read docs/CDLNO_IMPLEMENTATION_REPORT.md, docs/CDLNO_REQUIREMENTS_MATRIX.md and docs/final_audit/regression.log for the final evidence. All 119 tests passed in49.871s, zero failures/errors/skips; CPU mathematical/adapter/loss tests, real PyG2.3.1 Data/Batch integration, three original-cwd fresh-process loading, and six local GPU test methods actually ran. Local Python3.13.9/torch2.13+cu130/RTX5090 Laptop remains separate from the unrun remote Python3.10/3.11+torch2.11/cu128 target. No install or real training.

A final test audit found that the old NS synthetic test ran only3 steps and the temporal AST test only checked selected strings. Do not describe those old checks as full10-step/whole-AST evidence. Phase10 changed only tests/test_temporal_standard.py: now both feedback paths run10 steps with window assertions and one training backward/step; Plasticity uses distinct per-sample times with time_fc gradients; both complete temporal exp ASTs project exactly to baseline. The final focused9 and full119 tests pass. No production/model/task/dependency changes were needed.

README retains the original upstream text and adds CDLNO environment/eight-task/ablation/checkpoint/performance instructions. New provenance notices preserve Transolver MIT and IPOT MIT, retain Air ODbL, and explicitly record no license found in the audited LRSA snapshot. IPOT fixed-commit sources were re-read through GitHub raw; source SHA/URLs persist in docs/final_audit. Independent mathematical attachment files remain unlocated, so their full-text/proof-script validation is incomplete; this never authorizes additional structures.

Baseline71 tracked files:58 byte-identical,12 previously approved integration files unchanged this phase,1 README addition retaining all original text. Phase10-start162-file manifest:156 unchanged, only6 approved test/document files changed; no unexpected deletions/changes. Freeze evidence and cumulative/current-phase patches persist in docs/final_audit. Full exp/main AST comparisons preserve data/loss/time/optimizer/evaluation semantics. Existing Car epoch/load compatibility exceptions and inherited baseline limitations are distinguished in the final report.

No remaining known production defect was found after self-review. Unverified: real-data parsing/completeness, VTK/neighbor graphs/full sampling and coefficient evaluation, convergence/accuracy, real epoch efficiency, remote environment/editable install, full default GPU task matrix, theory attachments. Do not silently upgrade any of these from synthetic results. Any subsequent action requires an explicit user instruction; do not start training, commit/push or add research features.
