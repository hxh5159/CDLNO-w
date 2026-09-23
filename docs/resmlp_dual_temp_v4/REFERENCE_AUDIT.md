# ResMLP Dual Temperature v4 Reference Audit

## A. Scope and baseline

This audit was performed against the current checkout at `36a2e0be9287949b06006606e4726aa77d4a2a66` on branch `main`. The working tree contains user-owned untracked planning/download files; they were not changed. No model, parser, task entry, test, or checkpoint code was edited for this audit.

The implementation truth is the current source tree. The proposed public v4 contract is `architecture=resmlp_dual_temp_v4`, `architecture_family=linearno_loop`, `architecture_extension=resmlp_dual_temp`, `architecture_version=4`, with checkpoint schema `resmlp_dual_temp_v4` version 1.

## B. Pure LinearNO task facts

The profile resolver in `cdlno/linearno/profiles.py` and `_profile_data.py` supplies the following release model facts. All tasks use 8 native blocks, 8 heads, and independent block parameters. H is embedding width and M is the absolute LinearNO rank.

| task | H | heads | M | FFN ratio | variant | input/output contract |
|---|---:|---:|---:|---:|---|---|
| airfoil | 128 | 8 | 64 | 1 | conv_temp | 221x51 structured, field to output |
| darcy | 128 | 8 | 64 | 1 | conv_temp | 85x85 structured, coefficient field |
| elasticity | 128 | 8 | 64 | 1 | temp | 972 irregular points, stress output |
| pipe | 128 | 8 | 64 | 1 | conv_temp | 129x129 structured, velocity output |
| ns | 256 | 8 | 32 | 2 | plain | 64x64, ten truth-fed input frames and ten-step rollout |
| plasticity | 128 | 8 | 64 | 1 | conv | 101x31 time-conditioned grid |
| airfrans | 256 | 8 | 32 | 2 | airfrans | Data.x[N,7], reference distances, output [N,4] |
| car | 256 | 8 | 32 | 2 | shapenet | tuple (data, geom), single graph, output [N,4] |

The native implementation is `cdlno/linearno/attention.py`. After `in_project_x`, features have shape `[B,heads,N,d_h]`; `to_q`, `to_k`, and `to_v` produce `[B,heads,N,M]`, `[B,heads,N,M]`, and `[B,heads,N,d_h]`. Q applies `softmax(dim=-1)` over M, K applies `softmax(dim=-2)` over N, then `context=einsum('bhnm,bhnd->bhmd',K,V)` and `readout=einsum('bhnm,bhmd->bhnd',Q,context)`. Structured variants enforce `N=H*W`; AirfRANS makes the feature layout contiguous. `temp`/`conv_temp` use `temperature_q/k.clamp(.01,1)`, ShapeNet preserves the misspelled `tempreature_q/k` with clamp `[.1,2]`, and AirfRANS' declared temperature is inert in the current forward.

The task wrappers are `cdlno/linearno/standard_entry.py`, `shapenet.py`, and `airfrans.py`; task data/training remains in the original experiment entries. Initialization is one outer `initialize_release_weights` application followed by the non-persistent placeholder, and checkpoint metadata is handled by the existing `cdlno/linearno/checkpoint.py` and experiment recorder.

## C. Existing loop versions

* V1 (`cdlno/linearno_loop/core.py`) owns prefix, physical core, and suffix complete blocks. Core visits reuse physical blocks according to topology. Its SR controller scales raw operator and raw point-MLP branches by `1/R`; RB/LB register only their active receiver modules.
* V2 (`cdlno/linearno_loop/v2`) owns a shared core operator but registers `ln_2` and point FFN per `(core position, round)`. Optional latent context FFN is per core position and shared across rounds. This round-specific point ownership is explicitly excluded by v4.
* V3 (`cdlno/linearno_loop/v3`) shares complete core blocks and adds optional per-core latent FFN and second-visit bilateral Q/K adapter. Its public selector is `operator_latent_adapter_v3`, its checkpoint format is v3, and its residual/topology contracts are the old v3 contracts. It cannot be reused as v4 because v4 has five shared FLARE ResidualMLP owners, eight independent operators, three temperature modes, `1/sqrt(2)` residual scaling, and a distinct schema.

Version dispatch and metadata are currently concentrated in `linearno_loop/versioning.py`, the v1/v2/v3 construction and schema modules, `cdlno/experiment.py`, `tran_evaluate/linearno_loop`, and `tools/linearno_loop_accounting.py`. v4 must be additive and selected by its explicit architecture field.

## D. External references

The FLARE commit `4e053784fcb8b803c4459cba1dd2bd5566fe68a1` defines `ResidualMLP(num_layers=L)` as `fc1`, `L` same-width hidden linear layers, and `fc2`; consequently L=2 has four Linear modules and L=3 has five. Each hidden layer has GELU(tanh) followed by a same-width residual; input/output residuals are conditional on matching dimensions. There is no LayerNorm, dropout, or final activation. v4 adapts this topology while retaining LinearNO's outer initialization and stem/head.

Transolver++ uses input-conditioned temperature prediction with positive/clamped temperature and Gumbel routing. v4 inherits only the input-conditioned predictor idea. It deliberately uses separate K and Q predictors, two K granularities, bounded multiplicative temperature `exp(log(2)*tanh(delta))`, zero-initialized predictor output, and no Gumbel/noise.

## E. v4 frozen mapping

The v4 logical sequence is `[first,A1,B1,C1,A2,B2,C2,last]`. Eight operators, LN1 and LN2 are independent. Exactly five point ResidualMLP owners are registered: `F_first,F_A,F_B,F_C,F_last`, called by `[first,A,B,C,A,B,C,last]`. Middle branch coefficients are `1/sqrt(2)` for the raw operator and outer ResMLP; first/last coefficients are one. Dynamic temperature multiplies the legal static base temperature; raw logits are divided by the resulting temperature before their original softmax axis. The model never forms N-by-N or M-by-M attention.

## F. Baseline evidence

Commands and results:

```text
PYTHONPATH=.:cdlno pytest -q tests/linearno/test_attention_structure.py tests/linearno/test_attention_parity.py tests/linearno/test_profiles.py tests/linearno/test_standard_model.py tests/linearno/test_shapenet_model.py tests/linearno/test_airfrans_model.py
38 passed

PYTHONPATH=.:cdlno pytest -q tests/loop_linearno/test_block_body.py tests/loop_linearno/test_sr_core.py tests/loop_linearno/test_rb_core.py tests/loop_linearno/test_lb_core.py tests/loop_linearno_ffn/test_lf1_config.py tests/loop_linearno_ffn/test_lf2_core.py tests/loop_linearno_latent_adapter/test_laa3_attention.py tests/loop_linearno_latent_adapter/test_laa4_core.py
63 passed
```

The first collection attempt without `PYTHONPATH=.:cdlno` failed with import errors; this is an invocation issue and is retained as pre-existing evidence, not repaired by changing old code. No real data, remote Python 3.10/torch 2.11 stack, or long training was run.

