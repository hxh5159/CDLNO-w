# Checkpoint compatibility

| Artifact | V4 loader | Original family loader |
|---|---|---|
| pure LinearNO archive / weights | reject | unchanged |
| V1 loop pair | reject | unchanged |
| V2 loop-FFN pair | reject | unchanged |
| V3 latent-adapter pair | reject | unchanged |
| V4 same complete config pair | metadata-first, strict=True | rejected by other version schemas |
| V4 different temperature mode / task / profile / route / shape | reject before tensors | no migration |
| bare user weights without sidecar/pair | no optimizer continuation | legacy rules unchanged |

V4 public checkpoint_schema=resmlp_dual_temp_v4, checkpoint_version=1; physical pair format=resmlp-dual-temp-v4-pair-v1. Each completed save writes weights/epoch_XXXX.pt, checkpoints/epoch_XXXX.pt (model and full JSON-encoded optimizer/scheduler/RNG metadata), epoch_XXXX.metadata.json, then a hashed manifest, then latest/final pointers. architecture.json is immutable. Uncommitted files and rollback are refused; committed identical epochs are idempotent.

Read order: sidecar validation → config/mode/task assertions → pointer/manifest/path/content hashes → epoch metadata comparison → model construction → tensor load with weights_only=True → strict state keys/shapes/dtypes. Training resume additionally checks original data/normalizer/optimizer-group/scheduler/provenance contracts and restores RNG last. Air members remain independent and are checked against ensemble.json and member sampler identity. Native production paths have no AMP scaler; adding AMP production training/resume is not claimed.

The model-state-only test convenience save_pair(config) uses a clearly synthetic sampler marker and an unstepped optimizer. Those fixtures establish strict model loading only; task resume evidence exclusively uses the native Run adapters and their actual optimizer/scheduler state.

V4 source fingerprints include actual config, model, dispatch, task and checkpoint files using repository-relative names, so moving an otherwise identical checkout does not change its source hash. Modifying hashed source between interrupt/resume deliberately fails. No old sidecar, archive or golden is rewritten.
