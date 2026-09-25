# V5 visit-independent LayerNorm increment

## A. Scope and status

Status: **PASS** for the authorized no-real-data implementation and validation.
The only model calculation change is the default ownership of core LN1/LN2.
The public selector remains `partial_share_feature_gate_v5`.

Before this increment, a physical core position reused one LN1/LN2 pair across
all visits. The new default is `core_norm_mode=visit_independent`; the explicit
`shared` mode preserves the previous ownership. Prefix/suffix blocks retain one
pair each and the final LN3 remains a single, once-executed module.

## B. Modified files

- `linearno_loop/v5/{contracts,config,schema}.py`: strict mode enum, default,
  config/hash/metadata fields, constructor kwargs, state partition, run ID and
  pair-v2 revisions.
- `cdlno/linearno_loop/v5/{core,standard,airfrans,shapenet}.py`: per-visit norm
  ownership and wrapper forwarding.
- `cdlno/linearno_loop/v5/{checkpoint,recording}.py`: `visit_norms` parameter
  partition and actual visit-owner call schedule.
- Shared V5 dispatch files and `tran_evaluate/linearno_loop_v5/launch.py`: raw
  CLI `--linearno-loop-core-norm-mode` and launcher option
  `--core-norm-mode`; config revision 6 is routed through existing task logic.
- `tools/linearno_loop_accounting.py`: exact extra norm parameter count; matrix
  MACs unchanged.
- `tests/loop_linearno_v5/test_v5_visit_norms.py`: focused failure-first tests.
- Exact compatibility projections were resealed for the reviewed dispatcher
  bytes; their historical projected output remains unchanged.

## C. Formula and ownership mapping

`V5PhysicalBlock.norms_for_visit(r)` selects the affine owners used by the
unchanged forward sequence:

```text
S = LN1[p,r](X)
Z = X + Operator[p,r](S)
U = LN2[p,r](Z)
pi = softmax(router[p,r](U), expert axis)
G = sum_e pi[e] * Expert[p,e](U)
core Xnext = Z + G/R
```

Visit 0 retains the historical `ln_1`/`ln_2` state keys. Independent mode
registers exactly `R-1` modules in each of `additional_ln_1` and
`additional_ln_2`; shared mode registers empty lists. No complete block is
copied. Operator Q/K/temperature/router ownership and shared V/O/projection and
expert ownership are unchanged.

Each additional visit has two LayerNorms, each with C weights and C biases:

```text
Delta parameters = 2 norms * 2 affine vectors * C * C_core * (R-1)
                 = 4*C*C_core*(R-1)
```

For P1-C3-R2-S1 this is `12C`; for P2-C2-R2-S2 it is `8C`. Airfoil P1 with
three experts has 1,456,025 parameters in shared mode and 1,457,561 in the new
default. Both have 57,329,535,744 matrix FLOPs at B=1,N=11,271 under the
repository convention. LayerNorm and other non-matrix operations remain
outside this FLOP number.

## D. Verification

Environment: Python 3.13.9, torch 2.13.0+cu130, CUDA available on an NVIDIA
GeForce RTX 5090 Laptop GPU.

- Failure-first focused test before implementation: 14 failed because the
  option and constructor field did not exist.
- Final focused test: 16 passed.
- Complete V5 suite: 69 passed, one existing timm deprecation warning.
- Real parser preview: eight tasks times both norm modes, 16 passed.
- V2 suite: 34 passed when invoked with its required `python -m pytest` layout.
  Two earlier direct `pytest` invocations produced four collection errors from
  its local imports; no test body ran in those attempts.
- V4 suite: 136 passed, one skipped.
- Original Transolver eight-task frozen fixture: one passed.
- Pure LinearNO representative six-variant attention oracle: one passed.
- V1 representative core/optimizer/dropout tests: two passed.
- V3 representative parity/checkpoint tests: two passed.
- `compileall` and `git diff --check`: passed.

### Remote Standard data-path follow-up

A remote Airfoil launch exposed an observation-layer defect after this report:
the native task launcher supplied the correct `fno/airfoil/naca` directory,
then the V5 launcher appended the generic `fno` directory. Because argparse
uses the final repeated option, dataset verification searched for
`fno/NACA_Cylinder_X.npy` and failed before model construction.

The V5 launcher now resolves each Standard task through its existing path
contract: Airfoil uses `CDLNO_AIRFOIL_ROOT`, Pipe uses `CDLNO_PIPE_ROOT`,
Plasticity uses `CDLNO_PLASTICITY_FILE`, and Darcy/Elasticity/NS use their
task root variables. Shadowed native defaults are removed from the final argv,
so execution and the automatic evaluation follow-up each receive one effective
`--data_path`. An explicit user `--data_path` remains authoritative.

The subsequent all-task review found equivalent repeated defaults for
AirfRANS `--my_path` and ShapeNet-Car `--data_dir`/`--save_dir`. Their values
were identical in the tested environment, but they are now deduplicated by the
same task-level helper for both training and automatic evaluation planning.

Failure-first coverage produced 7 Standard path failures, 2 industrial path
failures and 8 automatic-evaluation path failures. After the correction, all
17 direct cases, 29 launcher/parser/checkpoint cases, all 8 `v5_run` script
dry-runs, the complete 86-case V5 suite, provenance checks, compileall, shell
syntax and `git diff --check` passed. No real dataset was opened and no
training ran.

The V5 suite covers CPU forward/backward/AdamW, both modes, Standard,
AirfRANS and ShapeNet wrappers, strict save/load, cross-mode pre-tensor
rejection, initial output/common-tensor/RNG equality, finite norm gradients,
P1/P2/custom R3 and local CUDA FP16/BF16 autocast paths.

## E. Frozen regions

No LinearNO attention/operator code, Q/K/V/O projection, temperature, router,
expert, softmax axis, residual coefficient, topology schedule, stem, output
head, task data/loss/optimizer/scheduler/evaluation code, pure LinearNO,
Transolver, or V1-V4 model implementation was changed. Exact projection tests
and representative legacy numerical tests passed.

## F. Compatibility and limits

The internal V5 config/schema revision is 6, checkpoint version is 2, format is
`partial-share-feature-gate-v5-pair-v2`, and formula revision is
`operator1-dense-expert-1-over-r-visit-norm-v2`. Old pair-v1 V5 sidecars are
rejected before tensor loading. A new `shared` run is valid, but it is a new
pair-v2 experiment. No implicit old-checkpoint migration or optimizer-state
migration is implemented.

**NOT RUN:** real datasets, complete epochs, convergence, accuracy/SOTA,
three-seed experiments, remote Python 3.10/torch 2.11 stack, distributed or
compile mode, real epoch time, and production inference latency.

Priority source review passed for norm ownership/storage, unchanged residual
placement, common initialization/RNG, metadata-first conflict rejection, and
analytic/live parameter partition agreement. The path follow-up additionally
confirmed that the exception occurred before model construction and did not
change any architecture/config hash/checkpoint field.

本阶段结束，未执行下一阶段。
