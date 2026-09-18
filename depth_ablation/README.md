# LinearNO 深度消融（depth_ablation）

对 LinearNO 在四个标准数据集（airfoil / darcy / elasticity / pipe）上做不同 block 数（深度）的训练+评估，验证深度对精度的影响。所有脚本与输出都在本目录 `depth_ablation/` 下，不改动任何模型、数据、训练或 checkpoint 代码。

## 输出结构

每个（任务 × 种子 × 深度）组合独占一个目录，互不覆盖：

```text
depth_ablation/
  airfoil/
    seed0/
      blocks4/      ← 一次完整训练+评估的全部产物
      blocks8/
      blocks16/
    seed1/ ...
  darcy/ ...
  elasticity/ ...
  pipe/ ...
```

每个 `blocks<N>/` 目录内部与仓库其它模型族的 run 目录一致（由 `cdlno.experiment` 统一记录），主要文件：

```text
blocks4/
  architecture.json      # LinearNO 模型/checkpoint 元数据（含 n_layers=4）
  config.json            # 解析后配置：profile/variant/M/d/L/seed/命令/环境/参数量
  train.log              # 训练日志
  train_history.jsonl    # 每个 epoch 的训练/验证指标
  train_results.json     # 训练结果
  checkpoints/           # strict epoch pair（epoch_XXXX、latest、final 指针）
  weights/               # 独立权重
  model.pt               # 便捷权重（权威校验以 pair 为准）
  training_artifacts/    # 训练期场图
  evaluations/<ts>/      # 每次评估（results.json、eval.log、场图）
  eval_results.json      # 评估索引
```

## 深度与种子

- 深度（block 数）由 `--n-layers` 控制，可设为 **4 / 8 / 16**（8 是当前默认，作为对照）。
- 种子由 `--seed` 控制，**必须提供**；不同种子写入不同目录，绝不互相覆盖。
- 同一（任务、种子、深度）重复运行会被入口的原子目录预留拒绝（不会静默覆盖）；如需重跑，先删除对应的 `blocks<N>/` 目录。

## 用法

```bash
# 训练
bash depth_ablation/train.sh airfoil 4 --seed 0 --gpu 0

# 评估（读取已完成的训练 run）
bash depth_ablation/eval.sh airfoil 4 --seed 0 --gpu 0

# 训练成功后自动评估（推荐）
bash depth_ablation/train_eval.sh darcy 16 --seed 1 --gpu 0

# 预览命令（不执行、不建目录）
bash depth_ablation/train_eval.sh elasticity 8 --seed 0 --gpu 0 --dry-run
```

默认 profile 为 `paper_table8_on_release_model`，也可显式 `--linearno-profile official_release`。其余入口参数（`--epochs`、`--batch-size`、`--lr` 等）按原入口语义透传。

```bash
# 批量：四任务 × 深度 × 种子（依次执行）
bash depth_ablation/run_all.sh --seeds "0 1 2" --depths "4 8 16" --gpu 0 --dry-run
```

## 数据路径

沿用仓库根 `path.sh`（由 `CDLNO_FNO_ROOT` 派生 `CDLNO_DARCY_ROOT` / `CDLNO_ELASTICITY_ROOT` / `CDLNO_AIRFOIL_ROOT` / `CDLNO_PIPE_ROOT`）。运行前确认 `path.sh` 中的 `CDLNO_DATA_ROOT` / `CDLNO_FNO_ROOT` 指向实际数据，再用去掉 `--dry-run` 的命令。

## 说明

- 沿用 LinearNO 各任务官方配置：airfoil/darcy/pipe 为 `conv_temp`、d128、h8、M64、ratio1；elasticity 为 `temp`、d128、h8、M64、ratio1。深度消融只改 `n_layers`，其余保持不变。
- 脚本仅负责 train/eval 的组织、深度/种子透传与输出目录固定；模型、数据读取、损失、优化器、scheduler 与 checkpoint 协议全部复用现有 LinearNO 入口。
