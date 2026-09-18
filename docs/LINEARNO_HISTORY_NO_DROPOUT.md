# A1K0 without history-source dropout

The history extension now supports an explicit A-only ablation in which the
latent-summary AttnRes remains enabled while its history-source dropout is
disabled:

```text
linearno_latent_attnres=1
linearno_history_k_conditioning=0
linearno_attnres_history_dropout_p=0
```

The released research default remains A1K0 with `p=0.1`. A1K1 with `p=0` is
rejected, as are dropout values on A0K0/A0K1. The no-dropout run is identified
by `__nodrop__` in its run signature, for example
`airfoil__paper_table8_on_release_model__L8__A1K0__nodrop__seed17`.

This setting removes only the train-time sample-by-real-source mask. It does
not remove AttnRes cross-attention, source softmax, the null source, or any A
parameters, and it does not change the pure LinearNO/A0K0 checkpoint schema.
With `p=0`, the A branch does not draw the history-dropout random mask, so its
forward does not consume that additional Torch RNG stream. It does not remove
the A cross-attention FLOPs. The constructor records
`attnres_history_dropout_p=0.0` in the research `model_spec`; strict
metadata-first loading rejects a p=.1 checkpoint as structurally different.

The option is available through the existing Standard, AirfRANS and
ShapeNet-Car LinearNO history launchers. The commands below are previews of
the existing train/eval entry points; they do not download data or start a
run by themselves.

```bash
COMMON=(--linearno_latent_attnres 1 \
        --linearno_history_k_conditioning 0 \
        --linearno_attnres_history_dropout_p 0 \
        --linearno-profile paper_table8_on_release_model \
        --n-layers 8 --seed 17)

# Standard tasks (replace paths and add --gpu as required)
bash tran_evaluate/linearno_history/airfoil.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/airfoil.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/darcy.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/darcy.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/elasticity.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/elasticity.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/pipe.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/pipe.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/ns.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/ns.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/plasticity.sh train "${COMMON[@]}" --gpu 0
bash tran_evaluate/linearno_history/plasticity.sh eval  --experiment-dir /ABS/RUN_DIR "${COMMON[@]}" --gpu 0

# Industrial tasks use their existing native adapters. Keep --model fixed by
# the launcher and provide the task data roots through the existing env vars.
bash tran_evaluate/linearno_history/airfrans.sh train \
  --linearno_latent_attnres 1 --linearno_history_k_conditioning 0 \
  --linearno_attnres_history_dropout_p 0 --linearno-profile paper_table8_on_release_model \
  --seed 17 --gpu 0
bash tran_evaluate/linearno_history/airfrans.sh eval --experiment-dir /ABS/RUN_DIR \
  --linearno_latent_attnres 1 --linearno_history_k_conditioning 0 \
  --linearno_attnres_history_dropout_p 0 --gpu 0
bash tran_evaluate/linearno_history/car.sh train \
  --linearno_latent_attnres 1 --linearno_history_k_conditioning 0 \
  --linearno_attnres_history_dropout_p 0 --linearno-profile paper_table8_on_release_model \
  --seed 17 --gpu 0
bash tran_evaluate/linearno_history/car.sh eval --experiment-dir /ABS/RUN_DIR \
  --linearno_latent_attnres 1 --linearno_history_k_conditioning 0 \
  --linearno_attnres_history_dropout_p 0 --gpu 0
```

These are configuration-level commands only. Real-data convergence, metrics,
GPU/remote parity and full training acceptance remain to be measured in the
user's data environment.
