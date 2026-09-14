# CDPA plan and LRSA/IPOT audit — 2026-09-13

## Plan facts

Source: `PLAN_CDLNO/CDPA_Transolver_Implementation_Plan_v1_2.md`.

The plan is an execution specification, not an implementation report. It freezes the eight existing task protocols and introduces a new model beside the original Transolver. The main configuration is 2 complete LRSA-style front blocks plus 6 IPOT-style persistent latent blocks, with `entry` CDPA as the primary mode. `off` and `every_block` are required for later ablations and no-data validation. A scalar M is shared across front, bridge, and rear in a run.

The front history point is precise: save each front block's latent representation after latent FFN2 and before up-attention. The history must retain gradients. `H_F` is also retained for the final feature-conditioned decoder. The bridge works for F=0 and must not register unused parameters.

CDPA has two distinct axes: each historical source performs its own current-to-history token attention, and the resulting source candidates are fused with a per-current-token depth softmax. The identity candidate is raw current Z, not passed through Cross or output projection. FP32 is required for depth scoring/fusion; dropout is zero in the first version. The source-batched path changes execution grouping only and must match the source-loop reference in output and gradients.

The plan's efficiency accounting distinguishes logical sources from SDPA calls. For default 2+6: `entry` has F historical sources at one CDPA location; `every_block` has `P*F + P*(P-1)/2` historical sources over P locations. Source batching reduces calls/Python overhead, not MACs. All point projections, FFNs, dense ConvFFNs, CDPA projections, norms, activation memory, and actual SDPA calls must be counted in benchmarks.

The model must keep original wrapper contracts:

- Standard tasks call `Model(x, fx, T=None)` and return `[B,N,C_out]`.
- ShapeNet-Car calls `forward((cfd_data, geom_data))` and returns `[N,4]`; original batch=1 behavior remains and multi-graph batches must fail clearly rather than mixing samples.
- AirfRANS calls `forward(data)` and returns `[N,4]`; masks, sampling, scatter/reorder, and metric postprocessing remain outside the model.
- NS retains the current 10-input/10-step training/test loop; no long-horizon data is fabricated.
- Plasticity retains per-time conditional calls and no autoregressive state.

## LRSA source observations

Source: `/tmp/LRSA-Operator-remote`, commit `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`.

Relevant files:

- `src/perceiverforpde/modeling/layers/attn.py`
- `src/perceiverforpde/modeling/perceiver.py`
- `src/perceiverforpde/modeling/perceiver_structured.py`
- `src/perceiverforpde/modeling/layers/mlp.py`
- `src/perceiverforpde/modeling/transolver_plus_adaptor.py`

`PerceiverAttention` owns independent learned latent queries, down projection, optional latent channel mixing 1, latent self-attention, channel mixing 2, and up projection. Its forward order is down -> latent FFN1 -> latent self-attention -> latent FFN2 -> up. `PerceiverDownProject` uses learned latent Q with point K/V; `PerceiverUpProject` uses point Q with latent K/V and normalizes latent context. The adaptor's `Transolver_plus_block` is a separate, simpler Transolver-style model and is not the complete LRSA block to copy.

The structured variant `ConvNextConv` does a regular `Conv2d(in_features,in_features,3,padding=1)` with default groups=1, then channels-last LayerNorm, pointwise linear expansion/activation/projection. It is not depthwise convolution despite the `dwconv` name. The CDPA plan explicitly follows this cost and ordering for structured tasks and forbids latent-sequence convolutions.

LRSA attention supports SDPA, per-head Q/K normalization, optional gates/RoPE/mass, and configurable biases. The CDPA v1.2 defaults intentionally disable optional RoPE/gates/mass and choose explicit norm/bias settings instead of importing every task YAML preset. LRSA's model uses `reset_parameters` paths and structured shape metadata; the new package must make device handling and initialization deterministic without relying on accidental wrapper order.

## IPOT source observations

Source: `/tmp/IPOT-remote`, commit `18c177846267505ee9503445a146dfd7dee34c41`.

Relevant files:

- `models/ipot/ipot_encoder.py`
- `models/ipot/ipot_processor.py`
- `models/ipot/ipot_decoder.py`
- `models/ipot/layers.py`
- `models/epd.py`

`IPOTEncoder` creates learned latents and a pre-norm cross-attention from latent queries to flattened input; by default it adds the query residual. It also declares `encoder_ff`, but its `forward` does not call it. The CDPA plan therefore keeps the query residual and omits an unused bridge FFN.

`IPOTProcessor` applies pre-norm self-attention plus pre-norm FeedForward with residual for each processor layer. In this commit, the same attention/FFN module objects are appended repeatedly even when `weight_tie_layers=False`; this means naive copying would share parameters. The CDPA plan requires independently instantiated rear blocks and uses the paper-consistent pre-LN self-attention path with Q/K/V derived from the same normalized latent.

`FeedForward` is GEGLU: linear to `2 * mult * channel`, split value/gate, `value * GELU(gate)`, dropout, then linear back to channel. The CDPA plan fixes rear ratio r=2 for the first version and keeps block independence.

`IPOTDecoder` supports arbitrary output queries and optional positional encodings. CDPA v1.2 does not implement sparse Darcy or arbitrary output-query mode; for the eight current one-to-one tasks it deliberately uses the LRSA-style `H_F` feature-conditioned final up/readout. This distinction must not be advertised as generic super-resolution.

## Source/plan divergences to preserve

- Do not import LRSA's complete YAML presets or IPOT's full training/data framework.
- Do not silently use LRSA's optional RoPE, attention gates, mass weighting, or latent widths.
- Do not copy IPOT's accidental module sharing.
- Do not replace CDPA's per-source Cross + depth fusion with concatenated-source attention, ordinary AttnRes, or optimal transport.
- Do not modify the original Transolver files or data paths for the documentation-only phase.

## Environment notes

The plan targets an existing user environment around CUDA 12.8 / PyTorch 2.11 and says not to downgrade it to the old Transolver `torch==1.10.1` requirements. Candidate pins are documented in the plan, but no remote GPU probe or joint PyG/VTK/PyVista validation has been run in this session. The source review used no training data.
