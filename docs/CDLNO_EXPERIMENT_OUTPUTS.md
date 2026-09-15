# CDLNO 实验目录与记录

本次范围：八任务 CDLNO 的实验路径、配置、日志和结果记录。适用于 `full/no_sa/identity`、`off/entry/every_block` 及已有深度设置；模型数学和正式任务预设不变。旧 Transolver 入口及输出协议保留。该工作不代表可视化/断点续训 V2–V5 已接入。

## 默认目录和生命周期

默认根目录跟随实际 checkout：本地 `/home/hwz/CDLNO/output`，用户远端为 `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w/output`。八个数据集目录为 `darcy/elasticity/airfoil/pipe/ns/plasticity/car/airfrans`。每次新训练使用 UTC 时间戳；Python 含微秒，shell 含纳秒。原子 `mkdir(exist_ok=False)` 防止覆盖；不复用旧实验目录。

```text
output/<dataset>/<timestamp>/
  config.json
  architecture.json
  train.log
  train_history.jsonl
  train_results.json
  model.pt                        # 六标准任务，原 state_dict
  model_200.pth                   # Car 示例，原末轮整模型
  CDLNO                           # AirfRANS 原整模型列表
  member_000/model                # AirfRANS 原成员整模型
  training_artifacts/             # 原入口需要的训练侧输出位置
  eval_results.json               # 各次评估的索引及结果
  evaluations/<eval_timestamp>/
    eval.log
    results.json
    ...原任务图、数组、score.json/VTK...
```

三个 checkpoint 形式按任务实际出现，并非每个实验都同时生成。已有保存频率、序列化方式和严格加载协议未改动。没有把旧权重文件宣称为含 optimizer/scheduler/RNG 的完整续训包。

有效 CLI 参数解析完成后、数据加载前，创建目录、`config.json`、运行中结果记录和日志。此时实际模型未构造，参数量明确为 `null`、`pending_model_construction`。在实际模型构造后、第一训练 batch 前写入 `measured`、总参数量/可训练参数量、架构、wrapper 配置、设备/dtype；不构造额外随机模型估算参数。

`config.json` 还记录最终 CLI 参数、数据路径、超参数、代码 commit/dirty 状态、Python/torch/PyG 版本、启动命令和工作目录；实际 split 数量/时间条件和优化器参数组在原入口对应步骤补充。AirfRANS YAML 在数据加载前只读解析，原训练读取/合并逻辑保留；多成员模型的参数分别列于 `members`，累计已构造成员参数位于 `constructed_members_parameters`。

训练每个已完成 epoch 追加 `train_history.jsonl`，并更新 `train_results.json` 中最近结果。`epoch=1` 指完成原循环 `ep=0`。训练成功且原 checkpoint 保存完成后才标记 `completed`。未捕获异常为 `failed`，Ctrl-C 为 `interrupted`，未走显式完成出口的正常进程退出为 `incomplete`；硬杀/断电可能保持 `running`，这不表示成功。参数解析、依赖导入失败发生在正式记录启动前，不保证产生目录。

单独评估必须明确指定已有 run 或明确 `CDLNO_RUN_TAG`，不猜测最近一次训练。每次评估新建子目录和日志；配置及 sidecar 只读，`eval_results.json` 保留各次结果。索引写入使用文件锁及原子替换。实际权重形状、mode 和架构冲突继续由原严格校验拒绝。旧 `runs/CDLNO/...` 可通过显式 run 路径加载，即使不存在新 `config.json` 也不补写它。

## 命令

以下命令均在仓库根执行，每一行是一个独立选择。本次没有执行这些真实训练命令。

```bash
# 如先前手动 export 过固定 tag，取消后恢复每次自动时间戳。
unset CDLNO_RUN_TAG

# 推荐先预览，不加载数据、不创建输出目录。
bash tran_evaluate/train_eval.sh darcy --gpu 0 --dry-run

# 每个任务：训练成功后，使用同一目录和配置评估。
bash tran_evaluate/train_eval.sh darcy --gpu 0
bash tran_evaluate/train_eval.sh elasticity --gpu 0
bash tran_evaluate/train_eval.sh airfoil --gpu 0
bash tran_evaluate/train_eval.sh pipe --gpu 0
bash tran_evaluate/train_eval.sh ns --gpu 0
bash tran_evaluate/train_eval.sh plasticity --gpu 0
bash tran_evaluate/train_eval.sh car --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans

# 三种前段模式共用参数。选择一个模式即可；每次新训练自动新目录。
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-latent-mode no_sa
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-latent-mode identity
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-latent-mode full

# 用户此前指定的 F3/L8、完整前段、关闭 CDPA。
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-blocks 3 --n-layers 8 --cdpa-mode off

# 单独训练仍自动创建 timestamp。
bash tran_evaluate/ns.sh train --gpu 0

# 复评：替换下列时间戳为控制台打印的真实目录，重复自定义架构参数。
bash tran_evaluate/darcy.sh eval --gpu 0 --front-blocks 3 --cdpa-mode off \
  --cdlno-run-dir "$PWD/output/darcy/实际时间戳"
bash tran_evaluate/car.sh eval --gpu 0 --run_dir "$PWD/output/car/实际时间戳"
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/airfrans.sh eval --run_dir "$PWD/output/airfrans/实际时间戳"
```

默认依然 F2/L8/full/entry，Pipe M32，其余 M64；epochs PDE500、Car200、AirfRANS398。评估省略 mode 可从已有 sidecar 恢复；其他自定义模型深度、宽度、Car fold/epochs、Air task/nmodel/budget 等仍需重复，不自动重建全部训练参数。

`CDLNO_RUNS_ROOT` 可显式改变输出根；`CDLNO_RUN_TAG` 可选自定义唯一标识。指定 tag 后三种模式不再自动添加后缀，每个新实验必须选择新 tag，避免碰撞。`--cdlno-run-dir`（六标准任务）/`--run_dir`（工业任务）仍最后覆盖默认路径；显式路径必须是新目录，复评则用已有目录。不要对同一目录重跑 `train_eval.sh` 来复评。

Car 的 `--save_dir` 始终是预处理数据目录，不是模型目录。Air 的 `--save_path` 保留旧分支意义，CDLNO 的运行位置由 `--run_dir`/统一输出根决定。Air 训练 `--my_path` 指 Dataset，评估指其父目录；根脚本继续自动处理。Car 阻力评估固定 raw 路径/fold0 限制保持，详见 [启动说明](../tran_evaluate/README.md)。

## 指标来源

| 任务 | 训练记录 | 独立评估记录 |
|---|---|---|
| Darcy | 原已归一分母的 train_loss、derivative_regularizer、validation_relative_l2 | 原解码后 relative_l2 |
| Elasticity/Airfoil/Pipe | 原 train_loss、validation_relative_l2 | 原 relative_l2 |
| NS | 原 train_step/full、test_step/full 平均 loss | 原打印的 test_full_loss；10→10 不变 |
| Plasticity | 原 train_step、test_step/full loss | 原 test_step/full；20 次独立时间前向不变 |
| Car | 原 train 返回的 pressure/velocity 分项；原外层日志另用 upstream_* 字段标记 | 原压力/速度 relative L2、RMSE、阻力误差、Spearman、原计时 |
| AirfRANS | 原 MSE_weighted 的 train_loss、surface/volume 及通道分项；有实际验证的 epoch 才记录验证值 | 原 `score.json` 的结构化指标直接拷入 results；原数组/VTK 留在该次评估目录 |

Car 原外层日志的 pressure/velocity 名称对调仍保留；新分项按真实返回值命名，不改内部损失。Air 原验证标量存在 `MSE_weigthed` 拼写分支，且抽样后分项均值与最后一次 `val_loss` 不等价；故将原标量称为 `upstream_validation_log_value`，独立记录原已计算的 surface/volume 均值，不伪称修正后的验证目标。Car/Air 原 forward 计时未加 GPU 同步，不作为本次可靠性能测量。

## 验证与边界

实际结果见 [交付报告](CDLNO_EXPERIMENT_OUTPUTS_REPORT.md)。本次只运行无真实数据的检查：24 个 task×mode 记录/合成步/checkpoint、多次评估与三工作目录新进程、失败记录、路径预览、冻结 AST。原损失/时间循环/GPU 合成回归复用既有套件。新增合成 MSE 检查只说明模型与记录接口；不称为原数据集训练。

无真实数据训练、收敛、精度、完整物理指标或远端目标环境验收。当前机器缺 `torch_cluster`，Air 完整抽样 epoch 的新增测试跳过；其可安全复用的原加权 loss/梯度和真实记录片段有单独验证。没有安装替代依赖。远端可在既有环境运行：

```bash
PYTHONPATH=tests:. python -B -m unittest test_experiment_records -v
CDLNO_LRSA_ROOT=/path/to/audited/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
```

上述快照AST回归依赖真正修改前的 `output-before-kql2cnpe/source`；远端可同步该保留快照并设置 `CDLNO_OUTPUT_BASELINE=/path/to/output-before-kql2cnpe`。缺快照则该项明确跳过，不能在修改后重新生成“旧基准”。此前A1/LRSA等回归的外部快照也需按原说明提供。
