# V3 efficient D12 + SR + latent, no adapter

八个入口固定为 `efficient_v1`、D12 `P2-C4-R2-S2`、SR、latent on、adapter
off。执行深度为12，独立完整block数为8。未注册 Q/K adapter 参数或调用。

| Task | H/Dz/M | Parameters | Matrix FLOPs | vs original LinearNO params/FLOPs |
|---|---:|---:|---:|---:|
| Airfoil | 96/512/64 | 1,394,849 | 159,580,190,976 | 78.9885% / 87.7937% |
| Darcy | 96/512/64 | 1,395,041 | 102,450,633,984 | 78.9879% / 87.9162% |
| Elasticity | 88/256/64 | 464,561 | 1,347,824,192 | 79.3827% / 82.9114% |
| Pipe | 96/512/64 | 1,394,849 | 235,419,369,216 | 78.9885% / 87.7223% |
| Navier--Stokes | 192/512/32 | 2,708,289 | 51,328,843,776 | 80.1762% / 85.5727% |
| Plasticity | 96/512/64 | 1,413,828 | 90,189,394,944 | 78.5710% / 87.8519% |
| AirfRANS | 192/512/32 | 2,693,956 | 198,895,927,296 | 80.2062% / 85.3342% |
| ShapeNet-Car | 192/512/32 | 2,965,892 | 226,945,531,392 | 76.9878% / 85.2942% |

不带 action 时自动执行 `train_eval`，训练成功后评估同一 run：

```bash
bash succeed_run/noadapter_12_sr_low/airfoil_train_eval.sh --seed 0 --gpu 0
bash succeed_run/noadapter_12_sr_low/darcy_train_eval.sh preview --seed 0 --gpu 1
```

默认输出根为 `output/linearno_loop_v3/noadapter_12_sr_low`，可用
`NOADAPTER_12_SR_LOW_RUNS_ROOT` 覆盖。成本是冻结代表性 batch/N 的单次
forward 矩阵口径，`1 MAC=2 FLOPs`，不是实测速度。
