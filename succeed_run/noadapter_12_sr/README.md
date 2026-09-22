# V3 matched D12 + SR + latent, no adapter

八个入口固定为 `matched_v1`、D12 `P2-C4-R2-S2`、SR、latent on、adapter
off。执行深度为12，独立完整block数为8。未注册 Q/K adapter 参数或调用。

| Task | H/Dz/M | Parameters | Matrix FLOPs | vs original LinearNO params/FLOPs |
|---|---:|---:|---:|---:|
| Airfoil | 104/704/64 | 1,759,713 | 185,007,772,352 | 99.6503% / 101.7828% |
| Darcy | 104/704/64 | 1,759,921 | 118,822,055,872 | 99.6476% / 101.9650% |
| Elasticity | 96/320/64 | 582,305 | 1,559,825,664 | 99.5024% / 95.9527% |
| Pipe | 104/704/64 | 1,759,713 | 272,867,845,952 | 99.6503% / 101.6764% |
| Navier--Stokes | 208/680/32 | 3,380,849 | 59,794,522,112 | 100.0867% / 99.6862% |
| Plasticity | 104/720/64 | 1,795,324 | 104,790,895,872 | 99.7719% / 102.0750% |
| AirfRANS | 208/672/32 | 3,351,972 | 231,585,562,624 | 99.7971% / 99.3593% |
| ShapeNet-Car | 208/776/32 | 3,846,660 | 264,659,015,296 | 99.8505% / 99.4683% |

不带 action 时自动执行 `train_eval`，训练成功后评估同一 run：

```bash
bash succeed_run/noadapter_12_sr/airfoil_train_eval.sh --seed 0 --gpu 0
bash succeed_run/noadapter_12_sr/darcy_train_eval.sh preview --seed 0 --gpu 1
```

默认输出根为 `output/linearno_loop_v3/noadapter_12_sr`，可用
`NOADAPTER_12_SR_RUNS_ROOT` 覆盖。成本是冻结代表性 batch/N 的单次 forward
矩阵口径，`1 MAC=2 FLOPs`，不是实测速度。
