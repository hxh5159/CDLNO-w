# V3 efficient D12 + SR + bilateral Q/K adapter

本目录提供低计算预算的八任务脚本。它们与 `succeed_run/lr_12_sr/` 使用完全
相同的 V3 架构、训练入口和任务科学协议，只切换到文档规定的
`efficient_v1` profile。

固定结构：

- architecture: `operator_latent_adapter_v3`
- cost profile: `efficient_v1`
- executed depth: D12 = `P2-C4-R2-S2`，即 `2 + 4*2 + 2 = 12`
- unique complete blocks: `P+C+S = 8`
- residual: `sr_1_over_r`
- latent FFN: on，每个 core 位置一套、两轮共享
- adapter: `bilateral_qk_lowrank_second_visit`，只在第二轮启用
- adapter rank/alpha: `4/4`
- heads: 8
- model profile: `paper_table8_on_release_model`
- default seed: 0

## 低预算解析配置和成本关系

下表来自 V3 的 `efficient_v1` D12 profile。参数量和矩阵 MAC 是冻结的
representative batch/N forward 口径；`1 MAC = 2 FLOPs`，所以 MAC 比例也是
矩阵 FLOPs 比例。softmax、LayerNorm、GELU、bias、残差、温度/clamp 和其他
标量操作不计入矩阵 MAC。

| Task | H/Dz/M | V3 params | 原始 LinearNO params | 参数比例 | V3 matrix MAC | 原始 LinearNO matrix MAC | FLOPs 比例 | 矩阵 FLOPs 减少 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Airfoil | 96/512/64 | 1,397,281 | 1,765,889 | 79.1262% | 80,667,249,792 | 90,883,573,248 | 88.7589% | 11.2411% |
| Darcy | 96/512/64 | 1,397,473 | 1,766,145 | 79.1256% | 51,787,595,392 | 58,266,099,200 | 88.8812% | 11.1188% |
| Elasticity | 88/256/64 | 466,961 | 585,217 | 79.7928% | 692,574,496 | 812,809,728 | 85.2075% | 14.7925% |
| Pipe | 96/512/64 | 1,397,281 | 1,765,889 | 79.1262% | 119,004,753,792 | 134,184,503,808 | 88.6874% | 11.3126% |
| Navier--Stokes | 192/512/32 | 2,710,081 | 3,377,921 | 80.2293% | 25,781,862,400 | 29,991,370,752 | 85.9643% | 14.0357% |
| Plasticity | 96/512/64 | 1,416,260 | 1,799,428 | 78.7061% | 45,582,031,360 | 51,330,365,440 | 88.8013% | 11.1987% |
| AirfRANS | 192/512/32 | 2,695,748 | 3,358,788 | 80.2595% | 99,906,715,648 | 116,539,392,000 | 85.7279% | 14.2721% |
| ShapeNet-Car | 192/512/32 | 2,967,684 | 3,852,420 | 77.0343% | 113,934,184,192 | 133,036,839,936 | 85.6411% | 14.3589% |

相对于相同任务的原始 8-block LinearNO，低预算 D12 V3 参数量为
**77.0343%–80.2595%**，矩阵 FLOPs 为 **85.2075%–88.8812%**，即参数减少
约 **19.7405%–22.9657%**，矩阵 FLOPs 减少约 **11.1188%–14.7925%**。
这些是解析矩阵口径，不是实际 epoch 速度、显存或精度结果。

直接换算后的单次 forward 矩阵 FLOPs 如下：

| Task | V3 matrix FLOPs | 原始 LinearNO matrix FLOPs | V3/LinearNO |
|---|---:|---:|---:|
| Airfoil | 161,334,499,584 | 181,767,146,496 | 88.7589% |
| Darcy | 103,575,190,784 | 116,532,198,400 | 88.8812% |
| Elasticity | 1,385,148,992 | 1,625,619,456 | 85.2075% |
| Pipe | 238,009,507,584 | 268,369,007,616 | 88.6874% |
| Navier--Stokes | 51,563,724,800 | 59,982,741,504 | 85.9643% |
| Plasticity | 91,164,062,720 | 102,660,730,880 | 88.8013% |
| AirfRANS | 199,813,431,296 | 233,078,784,000 | 85.7279% |
| ShapeNet-Car | 227,868,368,384 | 266,073,679,872 | 85.6411% |

## 使用

默认执行 seed 0 的训练，训练成功后评估同一 run：

```bash
cd /path/to/looplin
bash succeed_run/lr_12_sr_low/airfoil_train_eval.sh
```

覆盖 seed 或 GPU：

```bash
bash succeed_run/lr_12_sr_low/darcy_train_eval.sh --seed 1 --gpu 1
```

真实训练前只做 parser/metadata 预览：

```bash
bash succeed_run/lr_12_sr_low/elasticity_train_eval.sh preview --seed 0 --gpu 0
```

默认 GPU：Airfoil/Elasticity/Plasticity/AirfRANS/Car 为 0，Darcy/Pipe/NS 为
1。可用 `--gpu` 覆盖。输出根独立于 matched 目录：

```text
output/linearno_loop_v3/lr_12_sr_low
```

也可通过 `LR12_SR_LOW_RUNS_ROOT` 指定输出根。每个 run 的目录名包含 task、
profile、P/C/R/S、residual、H/heads/M/Dz、latent/adapter、seed 和 config
hash，并带唯一时间标识，因此不会覆盖 `lr_12_sr` 或同目录下其他 run。

NS 的十步训练/预测反馈、Plasticity 的二十次时间条件查询、AirfRANS 的
weighted loss/sampling/ensemble，以及 Car 的 fold/surface/drag 协议均由原有
任务入口保留。该目录只提供 V3 参数与调用入口，不复制训练逻辑。
