# Stage D: checkpoint, resume, evaluation, and output contract

## A. Authorized scope and status

Status: **PASS** for metadata-first checkpointing and controlled synthetic
resume/evaluation. No real benchmark data or complete production epoch was
used.

V5 uses the independent checkpoint format
`partial-share-feature-gate-v5-pair-v1`. A V5 run cannot be selected from a
V1-V4 checkpoint or from a checkpoint without explicit V5 metadata.

## B. Files and reasons

- `linearno_loop/v5/schema.py` defines exact JSON metadata, immutable sections,
  tagged optimizer/scheduler/RNG state validation, and strict saved-config
  restoration.
- `cdlno/linearno_loop/v5/checkpoint.py` measures parameter ownership, validates
  the live model, publishes atomic checkpoint/weights/metadata pairs, verifies
  hashes and pointers, and strictly loads the complete state dict.
- `cdlno/linearno_loop/v5/{provenance,recording}.py` records reviewed source
  hashes, physical/visit owners, expected and observed schedules, initialization
  hashes, parameter partitions, and all-expert call counts.
- `cdlno/linearno_loop/{standard_entry,air_entry,car_entry}.py` selects these
  facilities only for saved or explicitly requested V5 runs while retaining the
  native task output and visualization callbacks.
- `tran_evaluate/linearno_loop_v5/launch.py` provides train, resume, eval, and
  train-then-eval actions with explicit run/checkpoint selection.

## C. Stored contract and load order

`architecture.json` records the full resolved profile and constructor together
with `P/C/R/S/L`, `K`, `F`, actual `M`, native heads and variant, the fixed
operator scale `1`, core expert scale `1/R`, expert-axis routing, bias,
non-multihead/no-temperature gating, dense execution, state ownership, and
initialization policy. It also records data checksums, normalizer policy,
provenance, ensemble membership, and measured parameter partitions.

Each committed epoch consists of:

```text
checkpoints/epoch_NNNN.pt
checkpoints/epoch_NNNN.metadata.json
weights/epoch_NNNN.pt
checkpoints/epoch_NNNN.json
checkpoints/latest.json
checkpoints/final.json       # final epochs only
```

The manifest is the commit point. Relative paths and SHA-256 values are checked
before `torch.load`; immutable metadata and the epoch role are checked before
model construction; saved configuration conflicts are rejected before tensor
load; weights are applied with strict keys, shapes, and dtypes. A committed
epoch cannot be replaced or rolled back. The complete tagged resume state
contains optimizer, scheduler, Python/NumPy/CPU/CUDA RNG, loader generators,
sampler state, epoch, and global step. Optimizer ownership must equal the V5
parameter set with no duplicate owner.

The run recorder observes the first successful forward and records every
physical position and visit: shared `in_project_x/to_v/to_out`, visit-owned
Q/K/router, LN1/LN2, every dense expert, and the one final norm/head. It does
not reinterpret V5 as V2 round-specific FFNs.

## D. Commands and evidence

The V5 tests exercise a Standard interrupted run, strict resume, final eval,
AirfRANS native weighted loss plus an explicit member
interrupt/resume/final/eval lifecycle, and ShapeNet-Car native PyG graph
save/resume/eval. The six Standard workers use the original task loss and
time-call structure with synthetic tensors and assert native visualization,
training-result, and independent evaluation-result files. Evidence is in
`evidence/stage_e/v5_full_pytest_final_provenance.log`.

CUDA-local autocast tests cover FP16 and BF16 forward, finite router softmax,
dense experts, backward, and AdamW. They passed on the local CUDA environment.
The production task launchers do not add an AMP/GradScaler mode, so a complete
production AMP archive/resume was **NOT RUN** and is not claimed.

Final self-review also found and fixed one launcher-only defect: automatic eval
after `train_eval --gpu N` had selected GPU 0. It now retains `N`, with a
dedicated regression test.

## E. Frozen-region evidence

V5 has a separate schema, format, dispatcher, and output namespace. Shared
entry changes are gated by explicit V5 selection or saved V5 metadata. Pure
LinearNO and V1-V4 checkpoint formats, state keys, saved metadata, and run
directories are not migrated. Native task metric and visualization code was
not edited; evaluation creates the task's existing independent eval records
instead of rewriting training metadata.

## F. Limits and priority review points

Real filesystem power loss, concurrent writers across hosts, distributed
training, production GradScaler resume, real AirfRANS multi-member sampling,
and real Car drag evaluation were not run. Priority review points are metadata
validation before tensor load, manifest publication order, exact optimizer
ownership, ensemble member isolation, and run-sidecar immutability.

This Stage D is complete; Stage E was executed under the user's continuous authorization.
