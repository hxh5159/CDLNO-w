# LinearNO：六个 Standard 任务

L4 接入 Airfoil、Darcy、Elasticity、Pipe；L5 增加 NS、Plasticity。AirfRANS、ShapeNet 尚未接入。脚本调用原 `exp_*.py`，显式传 `LinearNO_Structured_Mesh_2D` 或 `LinearNO_Irregular_Mesh`，不依赖旧入口不存在的默认模型 key。

从远端实际 checkout 根目录运行，路径自动跟随脚本位置。数据位置沿用根 `path.sh` / `CDLNO_*_ROOT`，或给每条命令追加 `--data_path /实际路径`。任何命令加 `--dry-run` 只显示解析后的 shell 命令，不访问数据。没有为这些命令执行真实训练。

## 四个静态任务的两组配置（L4）

默认 `paper_table8_on_release_model`；可以显式选择 `official_release`。两者这四题均为 d128/L8/heads8/M64/FFN ratio1；Elasticity temp/batch1，其余 conv_temp/batch4。Darcy weight_decay=1e-6，其余1e-5。Airfoil221×51、Darcy85×85、Pipe129×129；Elasticity为不规则972点。正式配置不因测试而缩小。

**本阶段明确的协议决定**：两组接入均按用户 L4 要求使用逐样本 flatten、无 epsilon 后取 batch mean 的 rL2；官方发布代码的训练 batch sum 与此有1/B的损失缩放差异。其余 hxh 数据/decode/训练节奏保持；Airfoil沿用每epoch验证（release每10epoch），Pipe保留原 first1200 split。config/metadata 记录这些差异；`official_release` 表示发布结构和超参，当前不是这些行为的逐字 release-exact。后续不能把它们混同或悄悄改回。Darcy 保留0.1导数项，只有导数分支预测边界置零，zero-padding central difference，dx=1/85。

以下选一组初始化；每次新训练使用新 RUN_ROOT。运行名包含 family/profile/seed，实际各任务目录由 `--experiment-dir` 明确指定，已有目录拒绝覆盖。

```bash
# 默认论文配置
PROFILE=paper_table8_on_release_model
SEED=0
GPU=0
RUN_ROOT="$PWD/output/linearno/$PROFILE/seed${SEED}_$(date -u +%Y%m%dT%H%M%S%NZ)"
```

```bash
# 官方结构/超参配置（上述 L4 协议差异同样明确保留）
PROFILE=official_release
SEED=0
GPU=1
RUN_ROOT="$PWD/output/linearno/$PROFILE/seed${SEED}_$(date -u +%Y%m%dT%H%M%S%NZ)"
```

训练命令互相独立，按需选择一个；每条默认500 epoch，需要用户自行启动。`--gpu 0/1` 选择远端GPU。不要创建 RUN_ROOT 中的任务目录再运行，程序会原子保留新目录。

```bash
bash tran_evaluate/linearno/airfoil_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/airfoil"
bash tran_evaluate/linearno/darcy_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/darcy"
bash tran_evaluate/linearno/elasticity_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/elasticity"
bash tran_evaluate/linearno/pipe_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/pipe"
```

训练成功后，评估同一目录的 final checkpoint。评估先读 metadata 再构造，自动恢复 profile/variant/M/d/L；显式传错 profile、variant、rank 会报错。

```bash
bash tran_evaluate/linearno/airfoil_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/airfoil"
bash tran_evaluate/linearno/darcy_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/darcy"
bash tran_evaluate/linearno/elasticity_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/elasticity"
bash tran_evaluate/linearno/pipe_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/pipe"
```

新终端中请将 RUN_ROOT 设置成已存在的确切目录；程序不猜“最近一次运行”。eval 可追加 `--linearno-profile "$PROFILE"` 作一致性断言。没有 final 时默认拒绝；仅诊断中间epoch可显式 `--checkpoint epoch_0001`，该权重不标为 final。

## 续跑、参数与产物

从当前任务最新**已提交**epoch继续，保持原总epoch和scheduler计划，不把 `--epochs` 改成剩余epoch。例如：

```bash
bash tran_evaluate/linearno/darcy_train.sh --resume --experiment-dir "$RUN_ROOT/darcy" --gpu "$GPU"
# 其余同理替换为 airfoil_train.sh / elasticity_train.sh / pipe_train.sh。
```

eval 默认 `--checkpoint final`，resume 默认 `--checkpoint latest`；也支持显式 `epoch_XXXX`，resume拒绝回退到已有更晚存档之前。旧裸权重没有训练状态，不能用作 resume。完成500 epoch的final不能继续追加epoch；变更epoch属于新实验。

Darcy `official_release` scheduler 固定500，对非500 `--epochs` **明确拒绝**；paper配置按显式epochs同时解析循环/scheduler。所有 OneCycle 元数据保存 resolved epochs/steps_per_epoch/total_steps；Elasticity保持Cosine每epoch一次，其他三题OneCycle每batch一次。

允许显式 `--n-hidden`、`--n-layers`、`--n-heads`、`--mlp_ratio`、`--batch-size`、`--lr`、`--weight_decay`、`--max_grad_norm` 等覆盖并记录来源；LinearNO独立使用 `--linearno-rank` 与 `--linearno-variant`，拒绝 `--slice_num`。不要把改小的实验称为论文正式配置。用户显式 `--save_name` 保留；不会采用或覆盖旧Transolver默认save_name。

运行目录包含现有 hxh 的 `config.json`、`train.log`、`train_history.jsonl`、`train_results.json`、`visualizations/` 及独立 `evaluations/<timestamp>/`、`eval_results.json`。新增/适配：

- `architecture.json`：不可变初始完整 metadata，真实构造器字段、resolved profile、数据checksum、保存的normalizer数值、来源hash和初始训练状态。
- `checkpoints/epoch_XXXX.pt`、`weights/epoch_XXXX.pt`、`checkpoints/epoch_XXXX.metadata.json`：完整训练payload、独立纯权重、该epoch完整metadata。
- `checkpoints/epoch_XXXX.json`：以上三文件哈希提交清单；`latest.json`/`final.json`仅在清单提交后更新。`model.pt`为hxh便利纯权重；eval/resume以已校验的pair为准。

完整存档沿用四入口现有 `ep % 100 == 0` 和 final 时机，即完成epoch1、101、201、301、401及500后保存；不是每100个已完成epoch。中断只能恢复到最后已保存epoch，之后未存档更新会重算。训练曲线/日志追加保留，eval不改训练config/sidecar、不重新拟合normalizer、不按test选best。周期场图继续复用现有每50epoch与final逻辑。

无真实数据的验收命令和限制见 [L4报告](../../docs/LINEARNO_L4_REPORT.md)。

## NS 与 Plasticity（L5）

两个 profile 均使用下表正式结构；默认仍为 `paper_table8_on_release_model`。旧 Transolver 默认值不会覆盖它们。

| 任务 | variant / M / d / L / heads / FFN ratio | 位置、输入输出 | 参数量 |
|---|---|---|---|
| NS | plain / 32 / 256 / 8 / 8 / 2，无温度 | unified on/ref10，64×64，输入10帧、每次输出1帧 | 3,377,921 |
| Plasticity | conv / 64 / 128 / 8 / 8 / 1，无温度 | H101/W31，fx1、T[B,1]、输出4通道 | 1,799,428 |

NS 默认 batch2；Plasticity batch8；两者500epoch、lr0.001、AdamW weight_decay=1e-6；NS无clip、Plasticity clip1。evaluation batch1。以下是正式实验模板，未在本机执行真实训练。GPU编号按远端机器修改，每次新实验使用新的目录：

```bash
PROFILE=paper_table8_on_release_model   # 或 official_release
GPU=0
SEED=0
RUN_ROOT="$PWD/output/linearno/$PROFILE/seed${SEED}_$(date -u +%Y%m%dT%H%M%S%NZ)"

bash tran_evaluate/linearno/ns_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/ns"
bash tran_evaluate/linearno/ns_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/ns"

bash tran_evaluate/linearno/plasticity_train.sh --linearno-profile "$PROFILE" --seed "$SEED" --gpu "$GPU" --experiment-dir "$RUN_ROOT/plasticity"
bash tran_evaluate/linearno/plasticity_eval.sh --gpu "$GPU" --experiment-dir "$RUN_ROOT/plasticity"
```

每条 eval 应在对应 train 成功后单独执行。两种 profile 可分别运行上述完整命令；无需在 eval 重填结构参数。NS `--data_path` 是包含 `NavierStokes_V1e-5_N1200_T20/` 的 fno 根目录；Plasticity `--data_path` 是 **MAT 文件完整路径**。默认分别读取 `CDLNO_NS_ROOT`、`CDLNO_PLASTICITY_FILE`。`--dry-run` 可预览，不读取数据。

```bash
# 新终端先恢复已存在的确切 RUN_ROOT；总 epochs 不改成剩余 epoch。
bash tran_evaluate/linearno/ns_train.sh --resume --gpu "$GPU" --experiment-dir "$RUN_ROOT/ns"
bash tran_evaluate/linearno/plasticity_train.sh --resume --gpu "$GPU" --experiment-dir "$RUN_ROOT/plasticity"
```

时间任务保持实际入口的 **batch sum** rL2，epoch指标再除样本数；没有沿用 L4 静态任务的 batch mean 改动。NS训练10次真值回填、累加10步loss后一次backward/optimizer/scheduler；eval为10次预测回填。Plasticity原 `random_collate_fn` 对**每个样本调用 torch.randperm(20)**，随后20次各自forward/backward/optimizer，scheduler只更新一次。L0 C23已确认提示词里的NumPy描述不符当前及固定官方源码；Torch、NumPy、Python RNG仍全部保存/恢复。

Plasticity保持原 meshgrid、field flatten 与 Conv2d(H101,W31) 历史点序。normalizer仍仅作用于输入fx；eval/resume恢复训练存下的mean/std。其 `global_step` 计实际optimizer更新数，是scheduler步数的20倍；NS两者相同。metadata分别记录两个计数口径。保存时机仍为ep%100==0及final；仅从已提交epoch恢复，不改变原步进或增加真实时间缓存。

沿用L4严格规则：完整存档/纯权重配对、metadata先于构造、strict=True、final默认评估、eval不改训练metadata。源码变化时resume会按既有规则明确拒绝，不能把不同版本轨迹冒充连续训练；纯eval不要求源码hash完全相同。

无数据验收：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest linearno.test_temporal_integration -v
```

此测试在内存合成数据上执行原main AST，真实空间尺寸、小模型、3个合成epoch。真实loader、真实mini-run/长训练、远端CUDA12.8/Torch2.11均未执行，详见 [L5报告](../../docs/LINEARNO_L5_REPORT.md)。


## ShapeNet-Car（L7/L8）

官方配置可用 `car_train.sh --linearno-profile official_release`、`car_train.sh --resume`、`car_eval.sh`。
均接受 `--data_dir RAW --save_dir CACHE --experiment-dir RUN --gpu 0`；eval必须显式给RUN。
默认 paper/official 训练均采用用户确认的官方代码目标：所有点三通道速度 normalized-MSE，
加表面压力 normalized-MSE 的 `0.5` 倍；论文表中的 physical field rL2、drag 与 Spearman
仍作为独立评价轴记录。具体完整命令与边界见 `docs/LINEARNO_L7_REPORT.md` 和
`docs/LINEARNO_L8_REPORT.md`。

AirfRANS 的默认训练合同同样固定为官方 normalized 四通道 `volume MSE + 1×surface MSE`；
论文表中的 pressure/field rL2 不会偷偷替换训练目标。八任务 profile 的实际 resolved
字段、split、normalizer、评价聚合和命令见 `docs/LINEARNO_REPRODUCTION_MATRIX.md`。

## L8 外部 checkpoint 与 profile 复现

Standard 只接受由用户明确指定任务的裸 `state_dict`；AirfRANS 和 ShapeNet 的 whole-object
pickle 只在 `trusted=True` 且固定官方 checkout 的隔离子进程中读取。两任务可能共享官方
限定类名，转换器不会从反序列化结果猜任务；`module.` 前缀可逆归一化，所有未知、缺失、
重复或 shape 不一致的键都会拒绝，目标最终使用 `strict=True`。实现位于
`cdlno/linearno/converter.py`，不改变旧 Transolver checkpoint 路径。

无数据配置/闭环检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
  tests/linearno/test_converter.py tests/linearno/test_profiles.py \
  tests/linearno/test_schema.py
```

真实数据、官方外部 checkpoint、完整论文 epoch 和远端 Python3.10/Torch2.11/CUDA12.8
仍需单独执行；本阶段没有下载或运行它们。
图构造默认保持原入口，只有用户显式 `--cfd_mesh` 时使用已有mesh edges。
