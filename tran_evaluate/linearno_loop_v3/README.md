# Looped LinearNO V3 launchers

There are two fixed cost-profile trees, `matched_v1/` and `efficient_v1/`,
with one thin wrapper per task. Both default to V3 `D12/P2-C4-R2-S2`,
`sr_1_over_r`, task-base `M`, latent on, and bilateral Q/K adapter on with
`r=4`, `alpha=4`. The wrappers delegate to one shared `launcher.py`, which in
turn delegates to the existing native loop parser/entry. No task training code
is copied here.

```bash
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train_eval --gpu 1
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh dry-run --gpu 0 --seed 17
bash tran_evaluate/linearno_loop_v3/matched_v1/ns.sh preview --data-root "/data with spaces/fno"
bash tran_evaluate/linearno_loop_v3/matched_v1/car.sh eval --experiment-dir "/runs/exact car run" --gpu 0
```

Use `resume` and `eval` only with an exact existing run. `train_eval` evaluates
only after the train subprocess returns zero, using the same run. `preview`
executes the real parser and emits its JSON plan without reading data or
weights; `dry-run` prints the resolved native command; `print-run-dir` prints a
unique parser-resolved run directory without creating it.

The optional `custom/` tree requires explicit hidden width, latent width,
actual M, topology and adapter settings. A profile wrapper rejects a different
`--linearno-loop-cost-profile` before native construction, so a profile cannot
be silently changed by a trailing override.
