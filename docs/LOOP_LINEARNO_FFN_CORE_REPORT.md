# Looped LinearNO FFN v2 Core Report

## LF2: round-specific point FFN

**Status: PASS.** The new task-independent core registers one `ln_1+Attn` per
physical core position and one `ln_2+MLP` per `(position, round)`. Prefix and
suffix remain complete native blocks, and the final suffix alone owns the
head. SR selects each round FFN with branch scale `1/R`; RB preserves raw
partial and receiver sources without `1/R`; LB preserves actual `Y-H`, its
boundary sources and branch scale `1/R`.

The independent full-core oracle covered P1/C3/R2/S1, P2/C2/R2/S2 and
P0/C2/R3/S1 across all three residuals. Structure tests proved shared operator
IDs, distinct round FFN IDs, exact visit counts, live second-round gradients,
inactive router absence, all six operator variants and strict state roundtrip.
LF2 targeted tests: 4 passed, 0 failed.

## LF3: latent context FFN

**Status: PASS.** `LatentContextFFN` performs tokenwise
`Z + W2(GELU(W1(LN(Z))))` after `K^T V` and before Q readout. It uses affine
LayerNorm eps 1e-5, biased Linear layers, GELU, no dropout and no M-axis
mixing. The second Linear is zero initialized after its isolated initialization.
One module is installed per core position and called by all rounds.

The context-capable attention delegates its ordinary forward to the existing
LinearNO implementation. The enhanced path was checked against an independent
formula for plain/temp/conv/conv_temp/AirfRANS/ShapeNet, including temperature
clamps, structured reshape and output projection. Zero initialization produced
exact output and input-gradient equality; nonzero W2 matched forward and VJP.
Token permutation, B/N/M/head shapes, staged gradients and public-RNG isolation
were verified. LF3 targeted tests: 5 passed, 0 failed.

No production task parser, launcher, v1 core, pure model or checkpoint path was
modified in LF2/LF3. Real data and long training were NOT RUN.

本 LF3 阶段结束，已按用户授权继续执行下一阶段
