# Looped LinearNO v2 Cost and Synthetic Measurements

## Accounting Contract

The independent formula and actual module/ATen ledger split parameters into:

```text
stem | prefix | shared_operators | round_specific_point_ffns
suffix_body | head | routers | latent_ffns
```

For hidden width H, point-FFN ratio f, C core positions and R repeats, one
shared core operator owns `ln_1 + LinearNO attention`, while `C*R` point FFNs
each own `ln_2 + MLP`. Prefix and suffix are complete native blocks. RB adds
`2H(2CR+1)` router parameters and LB adds `2HR`; SR adds none.

The latent mode adds C tokenwise residual FFNs. For latent inner width Dz, each
has `2*H*Dz + Dz + 3H` parameters. Its executed dense-matrix cost is:

```text
2 * B * C * R * M * H * Dz MAC
```

This is applied after `K^T V` and before Q readout. It contains no M x M
operation. Dense matrix FLOPs are reported as `2*MAC`; norm, softmax, GELU,
bias, residual and other scalar/tensor operations are excluded and separately
described in the JSON scope.

## Full-Profile Parameters

The table is M x 1, SR, seed 17. RB/LB add only their router counts. Values are
actual `model.parameters()` counts and equal the independent formula.

| Task | H | M | Topology | round_specific | round_specific_latent |
|---|---:|---:|---|---:|---:|
| Airfoil | 128 | 64 | P1-C3-R2-S1 | 1,216,337 | 1,658,501 |
| Airfoil | 128 | 64 | P2-C2-R2-S2 | 1,399,521 | 1,694,297 |
| Darcy | 128 | 64 | P1-C3-R2-S1 | 1,216,593 | 1,658,757 |
| Darcy | 128 | 64 | P2-C2-R2-S2 | 1,399,777 | 1,694,553 |
| Elasticity | 128 | 64 | P1-C3-R2-S1 | 478,417 | 578,257 |
| Elasticity | 128 | 64 | P2-C2-R2-S2 | 514,017 | 580,577 |
| Pipe | 128 | 64 | P1-C3-R2-S1 | 1,216,337 | 1,658,501 |
| Pipe | 128 | 64 | P2-C2-R2-S2 | 1,399,521 | 1,694,297 |
| NS | 256 | 32 | P1-C3-R2-S1 | 2,972,417 | 3,368,705 |
| NS | 256 | 32 | P2-C2-R2-S2 | 3,107,585 | 3,371,777 |
| Plasticity | 128 | 64 | P1-C3-R2-S1 | 1,249,924 | 1,692,088 |
| Plasticity | 128 | 64 | P2-C2-R2-S2 | 1,433,092 | 1,727,868 |
| AirfRANS | 256 | 32 | P1-C3-R2-S1 | 2,953,260 | 3,349,548 |
| AirfRANS | 256 | 32 | P2-C2-R2-S2 | 3,088,436 | 3,352,628 |
| ShapeNet-Car | 256 | 32 | P1-C3-R2-S1 | 3,249,492 | 3,645,780 |
| ShapeNet-Car | 256 | 32 | P2-C2-R2-S2 | 3,450,468 | 3,714,660 |

`counts-final.json` contains all 192 task/preset/residual/mode/Mx1-or-Mx2
rows, every state key and key hash, parameter partitions, unique/executed calls
and MACs. All 96 primary M x 1 states were strictly loaded into fresh models.
M x 2 increases LinearNO projection/context compute and is not compute-matched.

## Bounded Measurements

Environment: Python 3.13.9, torch 2.13.0+cu130, Intel Core Ultra 9 290HX Plus,
one CPU intra/inter-op thread. CPU measurements use reduced synthetic tensors
(H=8, heads=2, M=4), two warmups and five samples across all 96 primary cases.

| Mode | Median of case medians, forward | Case range | Median of case medians, forward+backward+AdamW | Case range |
|---|---:|---:|---:|---:|
| round_specific | 1.412 ms | 0.858-2.346 ms | 6.493 ms | 4.327-8.483 ms |
| round_specific_latent | 1.619 ms | 1.066-2.694 ms | 7.537 ms | 5.252-10.130 ms |

These are local small-tensor medians/p90 samples, not real dataset throughput.
System load and every sample are in `cpu.json`.

Local synthetic CUDA smoke used an RTX 5090 Laptop GPU with TF32 disabled. All
96 configurations passed in FP32, AMP FP16 and AMP BF16: 288/288. The largest
process allocator peak observed was 68,516,864 bytes. These are reduced-shape
smoke/peak measurements, not formal-width timing or epoch memory forecasts.

`matrix.json` covers 96 reduced CPU forward/backward/AdamW/strict-reload cases
with diagnostics explicitly enabled. Formula and actual ATen matrix MACs match,
strict reload max error is zero, one final head is called, and no N x N or M x M
attention is observed. Diagnostics retain only detached scalar summaries and
are excluded from timing.

## Commands and Evidence

```bash
python -B tools/linearno_loop_ffn_performance.py counts --output COUNTS.json
CUDA_VISIBLE_DEVICES='' python -B tools/linearno_loop_ffn_performance.py \
  matrix --diagnostics --output MATRIX.json
CUDA_VISIBLE_DEVICES='' taskset -c 0 python -B \
  tools/linearno_loop_ffn_performance.py cpu --warmup 2 --steps 5 --output CPU.json
CUDA_VISIBLE_DEVICES=0 python -B tools/linearno_loop_ffn_performance.py \
  cuda --output CUDA.json
```

Machine-readable results are under `docs/loop_linearno_ffn_audit/lf7/`.
No real data, full epoch, convergence, accuracy, SOTA or remote target-stack
measurement was run.

