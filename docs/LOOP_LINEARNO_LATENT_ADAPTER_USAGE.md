# Looped LinearNO V3 使用说明

本文档对应显式架构 `operator_latent_adapter_v3`。V3 共享每个物理 core
位置的完整 LinearNO block，在每个 core 位置跨轮共享一个可选 latent FFN，
并可在第二轮启用 Q/K 双端低秩 adapter。省略 architecture 字段的命令仍按
旧 V1/V2 规则解析。

## 默认配置

- `cost_profile=matched_v1`
- `D12=P2-C4-R2-S2`
- `residual=sr_1_over_r`
- latent FFN 开启
- `adapter=bilateral_qk_lowrank_second_visit`，`rank=4`，`alpha=4`
- Airfoil、Darcy、Elasticity、Pipe、Plasticity 使用 `M=64`
- Navier--Stokes、AirfRANS、ShapeNet-Car 使用 `M=32`

规范深度的拓扑固定为：D12=`P2-C4-R2-S2`、D20=`P2-C8-R2-S2`、
D28=`P2-C12-R2-S2`、D60=`P2-C28-R2-S2`。每个配置的 unique depth 是
`P+C+S`，executed depth 是 `P+C*R+S`。

## Profile 宽度

表内每格为 `hidden_width/Dz`。两个规范 profile 均使用 8 heads。

| 任务 | matched D12/D20/D28/D60 | efficient D12/D20/D28/D60 |
|---|---|---|
| Airfoil | 104/704, 96/736, 96/648, 88/728 | 96/512, 88/512, 88/512, 80/512 |
| Darcy | 104/704, 96/736, 96/648, 88/728 | 96/512, 88/512, 88/512, 80/512 |
| Elasticity | 96/320, 88/312, 88/272, 80/288 | 88/256, 80/256, 80/224, 72/256 |
| Pipe | 104/704, 96/736, 96/648, 88/728 | 96/512, 88/512, 88/512, 80/512 |
| Plasticity | 104/720, 96/744, 96/656, 88/736 | 96/512, 88/512, 88/512, 80/512 |
| Navier--Stokes | 208/680, 200/592, 192/616, 184/600 | 192/512, 184/512, 176/512, 168/512 |
| AirfRANS | 208/672, 200/592, 192/616, 184/600 | 192/512, 184/512, 176/512, 168/512 |
| ShapeNet-Car | 208/776, 200/688, 192/712, 184/696 | 192/512, 184/512, 176/512, 168/512 |

规范 profile 不允许普通 H/Dz/M/head 参数静默覆盖。需要调整这些字段时，
必须使用 `custom` profile；`hidden_width` 必须能被 head 数整除，adapter
开启时 `R` 必须为 2。

## 八任务入口

两套正式入口分别位于：

```text
tran_evaluate/linearno_loop_v3/matched_v1/{airfoil,darcy,elasticity,pipe,ns,plasticity,airfrans,car}.sh
tran_evaluate/linearno_loop_v3/efficient_v1/{airfoil,darcy,elasticity,pipe,ns,plasticity,airfrans,car}.sh
```

每个入口支持 `train`、`resume`、`eval`、`train_eval`、`dry-run`、
`preview` 和 `print-run-dir`。`train_eval` 仅在训练成功后评估同一 run。
`resume`/`eval` 必须给出精确 RUN，不搜索最近目录。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin
source ./path.sh
export CDLNO_REPO_ROOT="$PWD"
export CDLNO_RUNS_ROOT="$PWD/output/linearno_loop_v3"

# matched 默认训练并评估
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh train_eval \
  --gpu 1 --seed 0

# efficient D20 + LB
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh train_eval \
  --linearno-loop-topology d20 \
  --linearno-loop-residual-mode lb_attnres_1_over_r \
  --gpu 0 --seed 0

# 精确恢复和评估；RUN 是已有 V3 run 的绝对路径
RUN="/absolute/path/to/existing/v3/run"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh resume \
  --gpu 1 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/matched_v1/darcy.sh eval \
  --gpu 1 --experiment-dir "$RUN"

# efficient 的恢复/评估使用相同的确切目录规则
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh resume \
  --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop_v3/efficient_v1/airfoil.sh eval \
  --gpu 0 --experiment-dir "$RUN"
```

三种 residual 的字段值是 `sr_1_over_r`、`rb_attnres`、
`lb_attnres_1_over_r`。四种消融示例：

```bash
ENTRY=tran_evaluate/linearno_loop_v3/matched_v1/elasticity.sh

# latent off / adapter off
bash "$ENTRY" preview --linearno-loop-latent 0 \
  --linearno-loop-adapter-mode none --gpu 0 --seed 0

# latent off / adapter on
bash "$ENTRY" preview --linearno-loop-latent 0 \
  --linearno-loop-adapter-mode bilateral_qk_lowrank_second_visit \
  --linearno-loop-adapter-rank 4 --linearno-loop-adapter-alpha 4 --gpu 0 --seed 0

# latent on / adapter off
bash "$ENTRY" preview --linearno-loop-latent 1 \
  --linearno-loop-adapter-mode none --gpu 0 --seed 0

# latent on / adapter on
bash "$ENTRY" preview --linearno-loop-latent 1 \
  --linearno-loop-adapter-mode bilateral_qk_lowrank_second_visit \
  --linearno-loop-adapter-rank 4 --linearno-loop-adapter-alpha 4 --gpu 0 --seed 0
```

custom 低成本配置必须显式给出 H、Dz、M、topology 和 adapter 参数：

```bash
bash tran_evaluate/linearno_loop_v3/custom/darcy.sh train_eval \
  --linearno-loop-hidden-width 64 --linearno-loop-latent-width 128 \
  --linearno-loop-heads 8 --linearno-rank 32 \
  --linearno-loop-topology custom \
  --linearno-loop-prefix-blocks 2 --linearno-loop-core-blocks 2 \
  --linearno-loop-repeats 2 --linearno-loop-suffix-blocks 2 \
  --linearno-loop-residual-mode sr_1_over_r \
  --linearno-loop-latent 1 \
  --linearno-loop-adapter-mode bilateral_qk_lowrank_second_visit \
  --linearno-loop-adapter-rank 4 --linearno-loop-adapter-alpha 4 \
  --gpu 1 --seed 0
```

三 paired seeds 应按相同任务/profile/depth/residual/消融运行，不能按测试集
结果选择 seed 或 checkpoint：

```bash
for seed in 0 1 2; do
  for task in airfoil darcy elasticity pipe ns plasticity airfrans car; do
    bash "tran_evaluate/linearno_loop_v3/matched_v1/${task}.sh" train_eval \
      --seed "$seed" --gpu 0
  done
done
```

## 输出和恢复

run id 记录 task、V3、cost profile、P/C/R/S、executed depth、residual、
H/head/M/Dz、latent/adapter、adapter rank/alpha、seed 和 config hash。同一
目录包含 `architecture.json`、run manifest、配置、日志、状态、epoch 记录、
`weights/`、`checkpoints/`、原任务可视化和独立 eval 目录。eval 不重写训练
sidecar 或 normalizer。

| 保存格式 | V1 loader | V2 loader | V3 loader |
|---|---:|---:|---:|
| V1 strict pair | 是 | 否 | 否 |
| V2 strict pair | 否 | 是 | 否 |
| `linearno-loop-epoch-pair-v3` | 否 | 否 | 是 |
| 旧本地 trusted whole-object | 保留原任务边界 | 保留原任务边界 | 不用于推断 V3 结构 |

V3 先验证 sidecar、family/extension、完整构造参数、config/hash、normalizer、
optimizer/scheduler/scaler、RNG/DataLoader/ensemble，再加载 tensor；state dict
始终 `strict=True`。跨版本、residual、topology、H/head/M、task 或特性配置
冲突在权重应用前拒绝。

## 成本口径和限制

`matched_v1` 是近似参数/矩阵成本匹配，不是逐项完全相同；
`efficient_v1` 是必做的低成本 profile。成本表按 `1 MAC = 2 FLOPs` 统计
Linear/Conv/einsum 矩阵运算；softmax、LayerNorm/RMSNorm、GELU、bias、残差、
dropout、温度/clamp 和 router 标量归一化另列。主表只对应 on/on、r4/a4、
SR；RB/LB router 和关闭模块后的成本需按实际配置重新计算。MAC 不能推出
真实延迟、显存或 epoch 时间。

截至 LAA10，真实数据、完整训练、三 seed 收敛、SOTA、真实 epoch 时长、
远端 Python 3.10/Torch 2.11+cu128、distributed 和 `torch.compile` 均未运行。
这些命令是生产入口交付，不是精度或效率结果。
