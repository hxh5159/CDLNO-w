# LAA6：六个 Standard 任务的 V3 接线

## 结果

**PASS（仅 LAA6 范围）**。V3 `operator_latent_adapter_v3` 已通过统一 Standard 入口显式接入 Airfoil、Darcy、Elasticity、Pipe、Navier--Stokes 和 Plasticity。原六个 `exp_*.py` 的数据读取、normalizer、objective、metric、optimizer/scheduler、NS 十步 teacher-forced/prediction-fed 时间循环、Plasticity 二十次 time-conditioned 更新和可视化调用没有改写。

## 实现

- `linearno_loop/versioning.py` 增加 V3 版本识别、配置/schema、construction/checkpoint/provenance 的延迟分派。旧 v1/v2 配置仍按原 schema、默认 rank 和 pair format 处理。
- `cdlno/linearno_loop/standard_entry.py` 增加唯一 `--linearno-loop-architecture operator_latent_adapter_v3` 选择器及 V3 cost/profile/topology/residual/消融字段。没有 architecture 时，旧 `--linearno-loop 1` 仍解析为 v1/v2；cost profile 单独不能选择 V3。
- `cdlno/linearno/standard_entry.py` 提供仅在 V3 family 下启用的 legacy constructor kwargs bridge，保留六个 exp 的现有 `linearno_model_kwargs(args, H=..., W=...)` 调用形状。V3 resolved `hidden_width`, `actual_M`, variant、grid 和 topology 是唯一模型真值。
- `cdlno/linearno_loop/standard_entry.py` 对 V3 使用已有 V3 checkpoint API，保存 `linearno-loop-epoch-pair-v3`，先 metadata/config/data/normalizer/provenance 检查再 strict state 应用；recorder、epoch、visualization 和指标接口仍由原任务主干调用。
- `cdlno/linearno_loop/v3/provenance.py` 为 V3 记录当前 V3/入口源码 fingerprint，避免调用只接受旧入口字节的 LF5 projector。`v2_projection.py` 对旧 fingerprint 仍把精确新增入口投影回 LF5 基线。

## 公式到代码

V3 模型仍由 LAA5 的 `LoopedStandardModelV3`/`LinearNOLoopCoreV3` 执行：P/C/S 物理 block、三种 residual、latent 和第二轮 Q/K adapter 的数学没有在 LAA6 改动。LAA6 只把解析后的 constructor kwargs、run family 和独立 pair format 接入原任务入口。V3 默认仍是 `matched_v1 + D12 + P2-C4-R2-S2 + SR + latent on + bilateral Q/K adapter r4/a4`；V3 profile 表统一在 `linearno_loop/v3/config.py` 解析，六个 exp 没有复制 H/Dz 表。

## 证据

- LAA6 专项入口测试：4/4 通过。覆盖显式 architecture、旧 v1 无 architecture、V3 sidecar family 选择、cost profile 不得隐式选 V3。
- 六任务 × 两 cost profile × 三 residual × 四消融 parser 矩阵：144/144 通过。
- 六任务小网格 custom 合成 forward/backward/AdamW：72/72 通过。该矩阵使用合成输入，不代表真实数据指标。
- V3 synthetic custom run 完成 metadata 写入、`linearno-loop-epoch-pair-v3` pair 保存和新构造 strict reload；参数/optimizer/RNG archive 由既有 V3 checkpoint 合同校验。
- LAA1--LAA5 合并回归：97/97 通过；LF5/LF6 旧 Standard/industrial 定向回归：6/6 通过；旧 parser 在阻断 V3 模块导入的测试进程中仍成功解析 v1：1/1。
- Python `py_compile` 通过。旧用户修改 `check_checkpoints/check_pipe_loop_resume.sh` 未触碰。

## 冻结区与限制

本阶段未修改六个 `exp_*.py` 的科学主体，也未接入 AirfRANS/ShapeNet-Car 的生产选择分支。没有修改旧 model key、旧 v1/v2 checkpoint 内容、golden 或容差。真实数据、完整 epoch、远端 Python 3.10/Torch 2.11/cu128、实际任务 resume/eval、收敛/精度、延迟、显存和 SOTA 均 **NOT RUN**。因此 PASS 只表示 V3 六任务入口、配置分派和合成严格闭环范围成立。

优先复核点：

1. 显式 architecture 是唯一 V3 入口，旧 parser 在 V3 包不可用时仍保持 v1 解析。
2. `model_kwargs` bridge 不让旧 exp 的默认 128/256 覆盖 V3 resolved hidden width；V3 actual M 和 profile variant 来自保存 config。
3. V3 metadata-first / strict pair 与旧 LF5/LF6 provenance projection 的边界。
4. NS/Plasticity 原生时间循环仍在 exp 主体，LAA6 没有复制或替换 objective。
5. 合成 72 行只证明可构造/反传，不是数据集训练结果。

本 LAA6 阶段结束，未执行下一阶段。
