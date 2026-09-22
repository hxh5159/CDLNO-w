# V3 matched D12 + SR + bilateral Q/K adapter

本目录的八个脚本固定使用以下模型配置：

- architecture: `operator_latent_adapter_v3`
- cost profile: `matched_v1`
- executed depth: D12 = `P2-C4-R2-S2`，即 `2 + 4*2 + 2 = 12`
- unique complete blocks: `P+C+S = 8`
- residual: `sr_1_over_r`
- latent FFN: on，每个 core 位置一套、两轮共享
- adapter: `bilateral_qk_lowrank_second_visit`，只在第二轮启用
- adapter rank/alpha: `4/4`
- heads: 8
- model profile: `paper_table8_on_release_model`
- default seed: 0

任务解析值和相同任务原始 8-block LinearNO 的解析成本对照如下。MAC 是冻结
representative batch/N 下的 forward matrix MAC，`1 MAC = 2 FLOPs`；不包含
softmax、norm、GELU、bias、残差和其他标量操作。

| Task | H/Dz/M | V3 params | LinearNO params | Param ratio | V3 matrix MAC | LinearNO matrix MAC | MAC/FLOPs ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Airfoil | 104/704/64 | 1,762,177 | 1,765,889 | 99.7898% | 93,392,581,984 | 90,883,573,248 | 102.7607% |
| Darcy | 104/704/64 | 1,762,385 | 1,766,145 | 99.7871% | 59,980,704,736 | 58,266,099,200 | 102.9427% |
| Elasticity | 96/320/64 | 584,737 | 585,217 | 99.9180% | 798,824,064 | 812,809,728 | 98.2793% |
| Pipe | 104/704/64 | 1,762,177 | 1,765,889 | 99.7898% | 137,746,032,544 | 134,184,503,808 | 102.6542% |
| Navier--Stokes | 208/680/32 | 3,382,705 | 3,377,921 | 100.1416% | 30,018,895,872 | 29,991,370,752 | 100.0918% |
| Plasticity | 104/720/64 | 1,797,788 | 1,799,428 | 99.9089% | 52,889,194,112 | 51,330,365,440 | 103.0369% |
| AirfRANS | 208/672/32 | 3,353,828 | 3,358,788 | 99.8523% | 116,267,917,312 | 116,539,392,000 | 99.7671% |
| ShapeNet-Car | 208/776/32 | 3,848,516 | 3,852,420 | 99.8987% | 132,807,405,376 | 133,036,839,936 | 99.8275% |

因此 `matched_v1` 匹配的是参数量和矩阵计算量的近似量级，不是要求每项严格
相等，也不代表实际运行时间相等。这里的原始 LinearNO 基线深度为 8；V3 的
执行深度为 12，但只存储 8 个完整 block，因为 4 个 core block 被调用两轮。

训练协议仍来自各任务的 `paper_table8_on_release_model` profile：

| Task | Variant/FFN ratio | Epochs | Batch | Optimizer | LR/weight decay | Scheduler | Objective |
|---|---|---:|---:|---|---|---|---|
| Airfoil | conv_temp/1 | 500 | 4 | AdamW | 1e-3/1e-5 | OneCycleLR | relative L2 |
| Darcy | conv_temp/1 | 500 | 4 | AdamW | 1e-3/1e-6 | OneCycleLR | relative L2 |
| Elasticity | temp/1 | 500 | 1 | AdamW | 1e-3/1e-5 | CosineAnnealingLR | relative L2 |
| Pipe | conv_temp/1 | 500 | 4 | AdamW | 1e-3/1e-5 | OneCycleLR | relative L2 |
| Navier--Stokes | plain/2 | 500 | 2 | AdamW | 1e-3/1e-6 | OneCycleLR | relative L2 |
| Plasticity | conv/1 | 500 | 8 | AdamW | 1e-3/1e-6 | OneCycleLR | relative L2 |
| AirfRANS | airfrans/2 | 400 | 1 | Adam | 1e-3/0 | OneCycleLR | native weighted MSE train |
| ShapeNet-Car | shapenet/2 | 200 | 1 | Adam | 1e-3/0 | OneCycleLR | native volume/surface MSE |

NS 继续使用原十步训练/预测反馈评估，Plasticity 继续使用二十次时间条件查询；
AirfRANS 的 sampling/ensemble/field metric 和 Car 的 fold/surface/drag 协议均由
原任务入口负责。上表只概括已解析字段，不替代任务实现。

## 使用

脚本不带参数时以 seed 0 训练，训练成功后评估同一 run：

```bash
cd /path/to/looplin
bash succeed_run/lr_12_sr/airfoil_train_eval.sh
```

仅允许覆盖 seed 和 GPU：

```bash
bash succeed_run/lr_12_sr/darcy_train_eval.sh --seed 1 --gpu 1
```

真实训练前可通过同一个真实 parser 预览，不读取数据或权重：

```bash
bash succeed_run/lr_12_sr/elasticity_train_eval.sh preview --seed 0 --gpu 0
```

默认 GPU：Airfoil/Elasticity/Plasticity/AirfRANS/Car 为 0，Darcy/Pipe/NS 为
1。可用 `--gpu` 覆盖。默认输出根是：

```text
output/linearno_loop_v3/lr_12_sr
```

每个实际 run 仍由正式记录器生成任务目录、完整结构字段、seed、config hash
和唯一时间标识，不会互相覆盖。可以用 `LR12_SR_RUNS_ROOT` 覆盖输出根。

数据路径沿用根目录 `path.sh`：Standard 六任务使用相应 `CDLNO_*` 变量，
AirfRANS 使用 `CDLNO_AIRFRANS_DATASET`，Car 使用 `CDLNO_CAR_RAW_ROOT` 和
`CDLNO_CAR_CACHE_ROOT`。脚本不复制任务训练、loss、metric 或 eval 逻辑。
