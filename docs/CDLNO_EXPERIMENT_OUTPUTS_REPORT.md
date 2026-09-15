# 统一实验目录和记录交付报告（2026-09-15）

本轮只完成用户批准的八任务 CDLNO 输出归档与配置/日志/结果记录。新实验默认 `output/<dataset>/<UTC timestamp>`；训练启动即预留目录并写配置，实际模型构造后补全参数量。训练与评估结果分别记录，重复评估不覆盖历史。模型、数据和训练计算没有修改。

## A. 完成范围

- 六标准任务、Car、AirfRANS 的实际训练/评估入口均已接入，共用 `cdlno.experiment`；三个前段模式及已有 CDPA 模式无需单独实现。
- 数据加载前写 `config.json`、`train.log`/`eval.log`、运行状态；模型构造后、首 batch 前记录实测 total/trainable 参数、完整架构/adapter、设备/dtype、训练超参数、优化器组配置、实际数据 split 数量/时间条件。
- `train_history.jsonl` 保存原 epoch 指标；`train_results.json` 保存状态、各成员最近/末轮指标；每次 eval 的 `results.json` 和 `eval.log` 位于独立 `evaluations/<timestamp>`，根 `eval_results.json` 聚合索引。
- 顺序脚本只生成一个 run tag，训练成功后评估同一目录；显式 run 参数最后覆盖。独立 eval 必须明确已有 run/tag，不猜 latest。旧 run 不迁移、不删除、不补写 config/sidecar。
- 原 Transolver 路径/命令保留。原 checkpoint 保存频率、state_dict/整对象/列表形式、strict 校验均保留。本轮没有接入 V2–V5 的周期图/完整恢复训练功能。

## B. 文件和增量 diff

实际基线：`main`，HEAD `769fa333742f73c132868cf560bce5ec21529362`。工作区在本轮开始前已有大量接受的未提交 A1–A4/V1/脚本修改，完整留存在 `/home/hwz/CDLNO-artifacts/output-before-kql2cnpe/source`（312 文件），附 hashes、HEAD、status。不能用对 HEAD 的累计 diff 冒充本轮 diff。工作期间另外出现的用户 `PLAN_KCDNO/KCDNO_Model_Specification_v1.md` 保持原样，不属于本轮实现，也不计入本轮patch。

本轮可审查 patch：[output_audit/incremental.patch](output_audit/incremental.patch)，清单及行数：[output_audit/changes.json](output_audit/changes.json)。

| 文件/区域 | 本轮用途 |
|---|---|
| 新 `cdlno/experiment.py` | 原子预留、初始配置、实测模型附加、Python stdout/stderr 日志、epoch/result 写入、异常状态、并发复评索引 |
| 标准 `cdlno_entry.py`、Car `models/cdlno_run.py`、Air `cdlno_entry.py` | 只允许认领本进程该 args 已预留目录；关联 recorder；保留原 sidecar 和 checkpoint 校验/序列化 |
| 六个 `exp_*.py` | CDLNO 分支启动/结束；原模型构造后和 scheduler 建立后记录配置；epoch/最终 eval 只读取已有指标 |
| 工业 `main.py/main_evaluation.py`（4 文件） | 同上；Air 各成员统计、训练后可选 score 分开记录；原输出导向当前 run |
| 工业 `train.py`（2 文件） | 增加默认 None 的可选 record 参数；epoch 后读取既有指标；旧调用参数及计算不变 |
| `path.sh`、根 common/standard/car/airfrans/train_eval shell | 默认 output 根和时间戳；明确 eval 路径；余参仍最后覆盖；dry-run 不创建文件 |
| `.gitignore` | 忽略根 `/output/` |
| 新 tests + 5 个既有测试 | 新目录/记录/往返/失败验证；既有冻结检查只移除精确记录语句后比较完整 AST，不跳过训练冻结检查 |
| README、命令/交付文档、STATUS、AGENTS、memory | 当前目录规则、八任务命令、结果与边界；保留历史文档 |

## C. 模型公式/形状与记录对应

本轮无新数学模块；原 `full/no_sa/identity` 的前段公式、`T_i`、CDPA 两级融合、bridge/rear/readout 均字节不变。原输出六标准任务 `[B,N,C_out]`，Car/Air `[N,4]`。参数量直接取实际 `model.parameters()` 的 `numel()` 及 `requires_grad`，不是理论 MAC 或另建随机模型的估计。

记录只接收原循环已计算的 loss/metric，必要时 `detach().cpu().tolist()` 用于 JSON；不替换用于 backward 的 Tensor。Darcy 保留原 normalizer/decode/导数项；NS 保留10次 forward 后1次 backward/step；Plasticity 保留20次独立时刻更新和原 scheduler 节奏。详细逐任务字段见 [指标来源表](CDLNO_EXPERIMENT_OUTPUTS.md#指标来源)。

Car 外层旧日志变量名互换：新记录分项对应真实返回 `pressure, velocity`，旧组合日志值明确命名 `upstream_*`，不改真实训练损失。Air 保留 `MSE_weighted`；原验证 typo 分支和抽样聚合原样保留，新记录明确区分分项均值与原外层 val_loss，不声称修正旧指标。

## D. 实际命令、环境和结果

环境：[environment.json](output_audit/environment.json)：Python3.13.9，torch2.13.0+cu130，CUDA build13.0，PyG2.3.1，RTX5090 Laptop GPU。保留用户远端 Python3.10/torch2.11/cu128 作为目标，未安装/升级/替换任何依赖。未在远端执行。

```bash
PYTHONPATH=tests:. python -B -m unittest test_experiment_records -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
```

| 范围 | 实际结果 |
|---|---|
| 新记录测试 | 9 个 test methods，12.872s，OK，1 个 Air sampled-epoch 子项跳过；[日志](output_audit/record-tests.txt) |
| 最终完整套件 | 212 tests，142.652s，0 failures/errors，1 skipped 子项；[日志](output_audit/final-regression.txt) |
| 24 task×mode | 初始 pending config→真实参数、合成 MSE forward/loss/backward/AdamW step、原格式 save/load、chunk0→1、每次两次 eval、sidecar/config 字节保留通过 |
| 原损失 | 既有 A3 等原 loss/normalizer/时间循环/PyG 测试实际重跑通过；另通过 Air 原 `MSE_weighted` loss 梯度及真实 epoch record AST 片段，未混称完整抽样训练 |
| Car 完整合成 epoch | 真实 PyG、原 train.main、2epochs；有/无 observer 同权重、同 torch/Python/NumPy RNG 完全一致；记录分项等于原函数实际返回 |
| 工业/标准新进程 | 从三个原工作目录加载包、执行记录生命周期及 state_dict/整对象/列表加载，输出完全一致 |
| 静态冻结 | 86 个原已有核心/模型/配置/数据/utils/脚本/性能文件字节相同；12 个 entry/train 完整 AST 去除精确观察语句后等于本轮前版本 |
| 原 upstream 冻结 | 7 项定向检查通过，0.175s；含基线投影、数据/指标/旧模型原文件；工业 train.py 为精确观察语句剔除后 AST 比较，其他仍字节比较 |
| 路径 | 8任务×默认/显式路径=16个顺序预览对，两步目录一致，支持空格、最后参数覆盖、dry-run零输出文件；8个独立无路径eval清晰拒绝；既有48个模式/parser检查重跑通过 |
| 失败/复评 | FileNotFoundError/Ctrl-C/无显式完成退出均留记录；已有目录拒绝；旧无config run可eval；重复eval各有独立日志/结果 |
| GPU | 完整套件中既有实际 GPU 合成/精度/梯度/保存恢复检查通过；本轮没有真实任务GPU训练或性能跑分 |
| 语法 | 13个shell通过 bash -n；变更Python通过3.10语法解析；diff空白检查通过 |

初次完整回归（203 tests）出现8失败/1错误：[初次日志](output_audit/first-regression.txt)。原因是原目录命名断言和禁止任何入口/训练源码变化的旧冻结检查尚未适配授权记录逻辑；修订仅限新目录规则和精确 AST 观察语句剔除，并新增对312文件快照的冻结证据。另一个定向检查曾漏导入 projection，已修正测试导入。Air 完整 sampled epoch 探测发现本机 `torch_cluster` 缺失，明确跳过且不伪造依赖。最终无失败。

## E. 冻结区域证据

[freeze.json](output_audit/freeze.json) 包含全部冻结清单。既有 `cdlno/config.py/core.py/modules.py/cdpa.py/standard.py/airfrans.py/checkpoint.py`、工业 Model、八个 JSON/YAML、原 Transolver 模型/脚本、dataset/utils、V1 续训基础、性能工具与快照一致。

全部10个实际 exp/main 及2个工业 train 的完整 AST，经 [精确记录投影](../tests/output_recording_projection.py) 后与本轮前版本相等。不是只检查几个 loss 字符串；能检测数据读取、点序、归一化、原模型调用、optimizer/scheduler、时间回填/训练循环、指标和 checkpoint 计算被意外修改。

运行记录不会注册模型参数、改变 forward 签名、初始化权重、消耗 torch/Python/NumPy 模型 RNG。实际 synthetic 参数/训练结果和旧 full 回归已验证；固定保存协议和旧 checkpoint 接入未被替换成新 resume 协议。

## F. 自审及剩余限制

交付前已自行审查：

1. 目录提前创建与保护：先 config pending，后真实参数；只能原 args 身份认领本进程预留目录，外来已存在目录拒绝。失败/重复评估和干跑均有实际证据。
2. train/eval 路径：组合脚本只生成一次时间戳；最后显式路径一致；单独eval不能猜最新；旧run无config仍加载，sidecar/config字节不变。
3. 记录值与训练计算：完整 AST 相等，Car记录分项来源正确、训练权重/RNG精确一致，Air真实加权loss与记录片段通过。
4. 模型与兼容：冻结核心字节一致，24模式接口/检查点及旧基准回归通过；没有把weights文件称为完整续训状态。
5. 证据归因：真实PyG对象不等于真实数据；本机GPU合成通过不等于远端验收；指标数组与原score文件保留，不从stdout拼凑精度。

当前未发现本轮范围内待修复缺陷。具体未验证项：本机无torch_cluster，Air完整抽样epoch记录测试未执行；原抽样/图/完整物理评价不变但本轮未用真实数据运行。未验证远端完整数据读取、收敛、实际精度、真实epoch效率、磁盘断电恢复或并发跨网络文件系统锁行为。硬杀可能留下running状态，不能将其当作成功。独立V1的完整恢复基础尚未接入八任务，V2–V5仍未执行。

八任务可运行命令、复评和自定义目录见 [使用说明](CDLNO_EXPERIMENT_OUTPUTS.md#命令)。无需审批新的架构决定；本次按批准的输出管理方案交付供审查。

本阶段结束，未执行下一阶段。
