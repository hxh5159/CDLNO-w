# Parameters and matrix FLOPs

B=1, canonical spatial N, one model forward. NS is one network call (a ten-step rollout makes ten calls); Plasticity is one time query (the native outer training batch makes twenty calls). 1 MAC = 2 matrix FLOPs. Bias, LayerNorm, GELU/tanh/exp/softmax, residual additions, positional distances and time sinusoid are excluded from the matrix number and recorded in each ATen inventory. These are not full-operation FLOPs or measured speeds.

| Task | N | Pure parameters | V1 P1C3R2S1 M=base parameters | V4 base | V4 latent-K | V4 point-K |
|---|---:|---:|---:|---:|---:|---:|
| airfoil | 11271 | 1,765,889 | 1,116,497 | 1,881,473 | 1,932,681 | 1,899,921 |
| darcy | 7225 | 1,766,145 | 1,116,753 | 1,881,729 | 1,932,937 | 1,900,177 |
| elasticity | 972 | 585,217 | 378,577 | 700,801 | 752,009 | 719,249 |
| pipe | 16641 | 1,765,889 | 1,116,497 | 1,881,473 | 1,932,681 | 1,899,921 |
| ns | 4096 | 3,377,921 | 2,182,145 | 6,003,713 | 6,029,321 | 6,021,137 |
| plasticity | 3131 | 1,799,428 | 1,150,084 | 1,915,012 | 1,966,220 | 1,933,460 |
| airfrans | 32000 | 3,358,788 | 2,162,988 | 5,984,580 | 6,010,188 | 6,002,004 |
| car | 32186 | 3,852,420 | 2,459,220 | 6,478,212 | 6,503,820 | 6,495,636 |

| Task | Pure GFLOPs | V1 GFLOPs | V4 base GFLOPs / ratio | V4 latent-K GFLOPs / ratio | V4 point-K GFLOPs / ratio |
|---|---:|---:|---:|---:|---:|
| airfoil | 45.441787 | 45.441787 | 53.567005 / 1.1788× | 55.137305 / 1.2134× | 56.706295 / 1.2479× |
| darcy | 29.133050 | 29.133050 | 34.341523 / 1.1788× | 35.348361 / 1.2133× | 36.353888 / 1.2479× |
| elasticity | 1.625619 | 1.625619 | 2.326330 / 1.4310× | 2.462350 / 1.5147× | 2.597060 / 1.5976× |
| pipe | 67.092252 | 67.092252 | 79.088682 / 1.1788× | 81.406830 / 1.2134× | 83.723667 / 1.2479× |
| ns | 29.991371 | 29.991371 | 77.236011 / 2.5753× | 77.789921 / 2.5937× | 78.343307 / 2.6122× |
| plasticity | 12.832591 | 12.832591 | 15.089717 / 1.1759× | 15.526408 / 1.2099× | 15.961788 / 1.2438× |
| airfrans | 233.078784 | 233.078784 | 602.177536 / 2.5836× | 606.503174 / 2.6021× | 610.828288 / 2.6207× |
| car | 266.073680 | 266.073680 | 637.317818 / 2.3953× | 641.668598 / 2.4116× | 646.018853 / 2.4280× |

All 24 V4 configurations match measured parameter partitions and ATen matrix MACs exactly at these N values. Each trace contains eight K^T V and eight QZ contractions, and 16 softmax operations, with no N×N or M×M attention. Pure baseline parameter counts are also checked against constructed models. V1 representative counts use unchanged existing accounting.

ResMLP ownership has 23 unique Linear modules and 38 executed Linear visits per network forward. Each middle owner has H→W, three W→W, and W→H, W=H×task ratio. Sharing saves three such owner parameter sets compared with eight independent ResMLPs; it does not remove any of their forward visits. Ratios remain 2 for NS/AirfRANS/Car, so their ResMLP matrices are much larger than the original two-Linear FFNs.

Per-operator predictor parameters: point = d_h M + 2M + 1; latent-K = d_h M + M² + 2M. Eight Q predictors always execute per point; latent-K first averages N and executes its MLP once per head/sample. Thus latent-K has more parameters but fewer MACs than point-K.

CPU/GPU measurements are in evidence/stage7/benchmark.json when available; they use a synthetic mean-squared-output objective for timing only, not the original task objective. They cannot predict production epoch time, convergence or accuracy.
