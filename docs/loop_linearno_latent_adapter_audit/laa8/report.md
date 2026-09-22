# LAA8: V3 launcher delivery

## Result

**PASS for launcher, parser, recording and documentation scope.** No real data,
long training or automatic experiment matrix was started.

## Delivered

- `tran_evaluate/linearno_loop_v3/launcher.py` is the only public dispatcher.
  It delegates to the existing native loop parser/entry through
  `tran_evaluate/linearno_loop/launch.py`; it does not copy task training code.
- `matched_v1/` and `efficient_v1/` each contain eight thin task wrappers. An
  additional `custom/` tree requires explicit H, Dz, M, topology and adapter
  fields. All wrappers preserve quoted paths and shell error propagation.
- Actions are `train`, `resume`, `eval`, `train_eval`, `dry-run`, `preview` and
  `print-run-dir`. `train_eval` passes `--then-eval` to the existing launcher,
  which starts eval only after a zero train exit code and reuses the exact run.
- Defaults are V3 D12/P2-C4-R2-S2, SR, task-base M, latent on and adapter
  bilateral Q/K r4/a4. D20/D28/D60, RB/LB and all four feature ablations are
  trailing parser overrides. A profile mismatch is rejected before native
  construction; custom requires explicit H/Dz/M/topology/adapter fields.
- `experiment-manifest.json` has 16 rows, one per required task/profile pair,
  and records default resolved H/Dz/M, config hash seed 0, wrapper and actions.
  It is descriptive only; no matrix is launched automatically.
- `docs/LOOP_LINEARNO_LATENT_ADAPTER_COMMANDS.md` contains all task/profile
  train/resume/eval paths, ablations, custom example, paired seeds 0/1/2 and
  remote path/data-root setup.

## Recording compatibility

`tran_evaluate/linearno_loop/recording.py` now has an explicit V3 branch. V3
manifests identify `shared_complete_core_block_across_rounds`, record the V3
state partition (`shared_core`, latent FFNs, adapters and routers), and count
the actual operator/adapter/latent/MLP visit order. The V3 schedule never emits
v2 `core_ffns` names. Existing v1/v2 branches remain unchanged; the old LF7
recording/launcher regression passed 6/6.

## Evidence

- LAA8 tests: 5/5 passed in 14.780 seconds.
- Legacy v1/v2 recording and launcher regression: 6/6 passed in 7.826 seconds.
- All 25 shell files passed `bash -n`; targeted Python `compileall`, JSON
  validation and `git diff --check` passed.
- All 16 required profile/task wrappers completed real parser `preview` with
  config version 3, D12 topology, correct profile, task-base M and config hash.
- Unique run-directory checks, quoted output-root paths, custom/profile
  conflict negatives and saved-config conflict-before-load checks passed.

## Output and limits

The native run tree remains responsible for `architecture.json`,
`loop_run_manifest.json`, logs/status/epoch files, strict V3 checkpoint pairs,
weights, visualization and eval directories. Preview/dry-run/print-run-dir do
not read data or weight tensors. The wrappers do not claim actual metrics,
convergence, speed or memory. Remote Python 3.10/Torch 2.11+cu128 and all real
training/evaluation remain **NOT RUN**.

本 LAA8 阶段结束，未执行下一阶段。
