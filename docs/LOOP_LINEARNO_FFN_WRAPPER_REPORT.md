# Looped LinearNO FFN v2 Synthetic Wrapper Report

**LF4 status: PASS.** Three versioned classes now cover Standard, AirfRANS and
ShapeNet-Car without production parser routing. They reuse the native forward
contracts and stem/head behavior while replacing only block ownership with the
v2 P/C/R/S loop.

Construction order is common tree allocation, one native release initializer,
placeholder draw, then optional latent construction in `torch.random.fork_rng`
with the metadata feature seed. Tests proved common tensor equality across both
v2 modes and SR/RB/LB, unchanged external Torch RNG, distinct `(p,r)` FFNs,
one latent FFN per core position, and zero W2 after final construction.

The v2 checkpoint module uses format `linearno-loop-epoch-pair-v2`, validates
metadata and constructor before state loading, checks a freshly constructed
state template, and loads with strict state semantics. A v2 pair was saved,
read and reloaded; the v1 reader rejected its format before tensor loading.

Targeted LF4 tests: 4 passed, 0 failed. The matrix covered six operator
variants, both modes, all three residuals, forward/backward/AdamW, presets,
custom R=3 and strict state/checkpoint roundtrips. These are reduced synthetic
shapes, not full-width or real-data training.

本 LF4 阶段结束，已按用户授权继续执行下一阶段
