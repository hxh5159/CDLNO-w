# Looped LinearNO v2 Commands

## Scope

These commands select `architecture_extension=loop_linearno_ffn_v2`. The new
field is mandatory for v2:

```text
--linearno-loop-core-ffn-mode round_specific
--linearno-loop-core-ffn-mode round_specific_latent
```

Omitting that field retains the reviewed v1 route and its historical default
rank multiplier of two. V2 defaults to the task profile's base rank (M x 1).
The existing eight task launchers still expose `train|resume|eval`, and
`--then-eval` evaluates the exact run only after train/resume succeeds.

## Train, Resume, Eval

From the repository root, a preset-A SR run is:

```bash
bash tran_evaluate/linearno_loop/darcy.sh train --then-eval \
  --linearno-loop 1 \
  --linearno-loop-topology p1_c3_r2_s1 \
  --linearno-loop-residual-mode sr_1_over_r \
  --linearno-loop-core-ffn-mode round_specific \
  --linearno-profile paper_table8_on_release_model \
  --seed 0 --gpu 1 \
  --experiment-dir "$RUN_ROOT/darcy/round_specific/p1_sr_seed0"
```

Preset B with the latent-context mode is:

```bash
bash tran_evaluate/linearno_loop/darcy.sh train --then-eval \
  --linearno-loop 1 \
  --linearno-loop-topology p2_c2_r2_s2 \
  --linearno-loop-residual-mode lb_attnres_1_over_r \
  --linearno-loop-core-ffn-mode round_specific_latent \
  --linearno-profile paper_table8_on_release_model \
  --seed 0 --gpu 1 \
  --experiment-dir "$RUN_ROOT/darcy/round_specific_latent/p2_lb_seed0"
```

Custom topology remains explicit and closed:

```bash
bash tran_evaluate/linearno_loop/darcy.sh train --then-eval \
  --linearno-loop 1 \
  --linearno-loop-topology custom \
  --linearno-loop-prefix-blocks 0 \
  --linearno-loop-core-blocks 2 \
  --linearno-loop-repeats 3 \
  --linearno-loop-suffix-blocks 1 \
  --linearno-loop-residual-mode rb_attnres \
  --linearno-loop-core-ffn-mode round_specific_latent \
  --seed 0 --gpu 1 \
  --experiment-dir "$RUN_ROOT/darcy/custom_rb_seed0"
```

Resume and eval recover all structural fields from the exact run metadata.
Repeated structural CLI fields are consistency assertions only:

```bash
bash tran_evaluate/linearno_loop/darcy.sh resume --then-eval \
  --gpu 1 --experiment-dir "$EXACT_RUN"

bash tran_evaluate/linearno_loop/darcy.sh eval \
  --gpu 1 --experiment-dir "$EXACT_RUN"
```

Use this additional flag only for the M x 2 control:

```text
--linearno-loop-rank-multiplier 2
```

An explicit `--linearno-rank M` remains an actual-M override and is mutually
exclusive with the multiplier.

## Eight Tasks

The launcher names are:

```text
airfoil.sh darcy.sh elasticity.sh pipe.sh
ns.sh plasticity.sh airfrans.sh car.sh
```

The complete planned matrix is stored in
`docs/loop_linearno_ffn_audit/lf7/command-matrix.json`: 288 primary commands
(8 tasks x 2 presets x 3 residuals x 2 v2 modes x 3 paired seeds), sixteen M x
2 controls, and one custom example. Every record contains train-then-eval,
resume-then-eval and eval strings. It is a preview and launches nothing.

Regenerate a fresh preview without running data or models:

```bash
python -B tran_evaluate/linearno_loop/ffn_matrix.py \
  --output /tmp/loop-linearno-ffn-commands.json
```

Individual native-parser validation remains safe:

```bash
bash tran_evaluate/linearno_loop/airfoil.sh train --dry-run \
  --linearno-loop 1 \
  --linearno-loop-topology p2_c2_r2_s2 \
  --linearno-loop-residual-mode sr_1_over_r \
  --linearno-loop-core-ffn-mode round_specific \
  --seed 0 --gpu 0
```

## Output Isolation

The v2 run id contains `v2`, `P-C-R-S`, residual mode, `core_ffn_mode`, actual
M, seed and config hash. Therefore the two v2 modes, v1, residuals, topologies,
ranks and seeds do not share a run directory. A normal training directory uses
the repository's existing output protocol and includes architecture/config,
logs/status, checkpoint/weight pairs, task results, and
`loop_run_manifest.json`. The latter records the measured v2 ownership
partitions and one verified call schedule. Eval writes a separate evaluation
record and does not rewrite training metadata.

No command in the generated matrix was launched against real data in LF7.

