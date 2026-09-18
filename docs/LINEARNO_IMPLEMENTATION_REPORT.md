# LinearNO 最终实施报告（L10）

日期：2026-09-18。目标仓库为 `/home/hwz/CDLNO`，目标基线 `bb73b3099d3b8ce45bd939156b737453b9ca5454`。固定参考为 `thuml/Transolver@75e0f67643806a81cd1d3f6adc88dd8c02416fe7`、`HiPRL/LinearNO@3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269` 和论文 `2511.06294v3`。

## 结论

LinearNO 的纯因式线性注意力、Standard 六任务完整模型、AirfRANS/ShapeNet-Car 专属模型、三组 profile、严格 checkpoint 元数据和任务接线已经实现，并通过源码独立审计、固定官方源码同权重 parity、合成训练/评估闭环和旧 Transolver 回归。AirfRANS 与 Car 的默认训练目标采用用户确认的官方标准化 MSE 合同；论文表中的 physical rL2 保留为独立评价轴，不能称为论文训练公式逐字复现。

完整真实数据训练、真实 VTK/MAT 读取、收敛/精度、完整论文 epoch 时长，以及远端 Python 3.10/Torch 2.11/CUDA 12.8 尚未运行。因此本报告证明的是实现和无数据验收，不是论文数值复现成功。

阶段状态：L0 PASS，L1 PASS，L2 PASS，L3 PASS，L4 PASS，L5 PASS，L6 PASS，L7 PASS，L8 PASS，L9 PASS，L10 PASS（资源门控项按 NOT RUN 记录）。

## 实现映射

| 规格 | 实际实现 | 独立证据 |
|---|---|---|
| `Q=softmax_M(XWq)`, `K=softmax_N(XWk)`, `C=K^T V`, `Y=QC` | `cdlno/linearno/attention.py`；Q/K/V 为独立、跨 head 共享的小投影 | `test_attention_parity.py` 独立 oracle、softmax 轴和官方同权重 parity |
| Standard plain/temp/conv/conv_temp | `PDE-Solving-StandardBenchmark/model/LinearNO.py` 与本地 attention 包装 | L3 六配置、逐 block 输出/梯度/AdamW 一步/strict load |
| AirfRANS 7+64 reference-distance 输入 | `cdlno/linearno/airfrans.py` | 固定边界/中心距离 golden test、真实 PyG 合成对象、3,358,788 参数 |
| ShapeNet 7 维输入、`M=key_ratio*head_dim`、拼写键 | `cdlno/linearno/shapenet.py` | 温度、M32、逐 block parity、真实 PyG 单图和多图拒绝、3,852,420 参数 |
| strict metadata/checkpoint | `cdlno/linearno/schema.py`, `checkpoint.py`, `converter.py` | metadata roundtrip、未知/缺失/shape/family/profile 拒绝、L8 转换器 |
| 任务 profile 和目标来源 | `cdlno/linearno/profiles.py`, `_profile_data.py` | 8×3 profile 往返、CLI 显式值优先、用户确认的 Air/Car MSE contract |

正常 attention 路径只物化按 head 的 Q/K/V、`[B,H,M,d_h]` context 和 `[B,H,N,d_h]` readout；没有 `N×N` logits、slice self-attention 或长期诊断缓存。第 8 个 block 仍完整执行 attention 和 FFN，之后才进行最后 LN/head。温度、placeholder、LayerNorm/Linear/Conv 初始化和官方 state_dict 键保持任务专属语义；AirfRANS dead temperature、ShapeNet `tempreature_q/k` 均保留。

## 任务配置与目标

正式构造参数量：Darcy `1,766,145`，Elasticity `585,217`，Airfoil/Pipe 各 `1,765,889`，NS `3,377,921`，Plasticity `1,799,428`，AirfRANS `3,358,788`，ShapeNet-Car `3,852,420`。

默认可选 profile 为 `paper_table8_on_release_model`、`official_release` 和 `transolver_matched`。结构、训练预算和评价字段分开保存，实际 M 与原始 `key_ratio`/`slice_num` 映射写入 metadata。

AirfRANS 默认目标是已标准化标签上的四通道 `volume MSE + 1*surface MSE`，Car 默认目标是全点三速度 normalized MSE 加 `0.5*surface pressure MSE`。两者均沿用用户确认的 Transolver/LinearNO 官方目标。论文表中写出的 rL2、drag 和 Spearman 仍作为独立 evaluation_spec 记录；统一 MSE 只解决训练目标一致性，不构成完整公平性或精度证明。

## Checkpoint 与命令

Standard 使用严格 state_dict；AirfRANS 和 ShapeNet 保留各自整对象/成员列表协议，并由 LinearNO metadata 先识别 family、profile、constructor kwargs 后构造，最终 `strict=True` 加载。外部 whole-object 只允许用户明确提供且可信的文件，在固定官方 checkout 的隔离子进程中导出 state_dict；没有外部 pickle 在本次验收中被加载。Standard 裸 state_dict、可逆 `module.` 前缀、Air dead 参数和 ShapeNet 拼写键均有显式转换检查。

实际 launcher 位于 `tran_evaluate/linearno/`。典型命令如下，路径和数据参数需替换为用户已有数据；本次没有启动这些命令：

```bash
cd /home/hwz/CDLNO
PROFILE=official_release
GPU=0
RUN=/absolute/path/to/new/run
bash tran_evaluate/linearno/darcy_train.sh --linearno-profile "$PROFILE" --gpu "$GPU" --experiment-dir "$RUN/darcy"
bash tran_evaluate/linearno/darcy_eval.sh --gpu "$GPU" --experiment-dir "$RUN/darcy"
bash tran_evaluate/linearno/ns_train.sh --linearno-profile "$PROFILE" --gpu "$GPU" --experiment-dir "$RUN/ns"
bash tran_evaluate/linearno/ns_eval.sh --gpu "$GPU" --experiment-dir "$RUN/ns"
bash tran_evaluate/linearno/airfrans_train.sh --linearno-profile "$PROFILE" --gpu "$GPU" --experiment-dir "$RUN/airfrans"
bash tran_evaluate/linearno/airfrans_eval.sh --gpu "$GPU" --experiment-dir "$RUN/airfrans"
bash tran_evaluate/linearno/car_train.sh --linearno-profile "$PROFILE" --gpu "$GPU" --experiment-dir "$RUN/car"
bash tran_evaluate/linearno/car_eval.sh --gpu "$GPU" --experiment-dir "$RUN/car"
```

其余任务使用同目录下的 `airfoil_*`、`elasticity_*`、`pipe_*`、`plasticity_*` 脚本。训练和评估目录必须显式区分；eval 先读 metadata，不根据“最近一次运行”猜路径。GPU 选择通过各脚本的 `--gpu N`，AirfRANS/Car 的单图 batch 约束和原数据路径语义保持不变。

## 实际验证

| 检查 | 结果 | 限制 |
|---|---:|---|
| L10 抽查：attention、Standard、AirfRANS、ShapeNet、converter、legacy | `28 passed`，46.91 s | 合成输入/固定源码 parity |
| L10 全 LinearNO 回归 | `84 passed, 2 warnings`，394.61 s | warning 为既有 timm/pytest collection warning |
| L9 综合套件 | `118 passed, 2 warnings`，419.44 s | 包含 Air/Car/前段/旧模型回归和 GPU smoke |
| profile/schema/converter | `13 passed`，13.97 s | 无外部 checkpoint |
| shell/编译/diff | `bash -n`、compileall、`git diff --check` PASS | 只读检查 |
| 本机 GPU smoke | RTX5090 Laptop，FP32/math，warmup3/measure5 | 小型合成模型，不作论文规模速度结论 |

L9 记录的合成效率 smoke：plain `N=64,d=32,L=8` median `3.004 ms`、peak `33.9 MB`；conv_temp `N=15` median `3.552 ms`、peak `36.6 MB`；窄模型 `N=1024/4096` median `0.662/1.227 ms`。这不是与 Transolver、matched LRSA 或 KCDLNO 的公平速度比较，也不是完整训练 step/epoch 预测。

旧 Transolver 的 factory、默认 CLI、模型 key、state_dict/whole-object 路径和已有回归均保持通过；L0/L6/L7 freeze 证据显示旧 Physics-Attention/Embedding 核心未被 LinearNO 改写。当前 checkout 没有 `LINEARNO/PropagationMonitor`，因此 monitor 检查记为 N/A，没有创建替代包。

## 未验证边界与人工审查点

尚未验证真实数据文件 checksum/完整性、真实八任务一批训练、500/400/200 epoch 收敛、论文表格精度、实际 epoch 时长、远端 Python3.10/Torch2.11/CUDA12.8/PyG cu128、AMP/compile 全矩阵和外部官方 pickle。没有下载数据、启动真实训练、安装依赖、commit 或 push。

用户最应人工审查的五点：

1. AirfRANS/Car 的 MSE 训练目标与论文 rL2 evaluation_spec 是否按实验目的分别选择。
2. 真实数据下 AirfRANS 32k 随机采样、Car fold0/预处理目录与 force 输入路径。
3. 远端环境中的 conv 非方形布局、PyG 单图约束和正式内存占用。
4. 真实训练时的 OneCycle steps、最终 checkpoint 角色和三 seed 汇总方式。
5. 论文结果对照必须在同 split、metric、采样、seed、设备和预算下进行，不能用本地 synthetic smoke 替代。

完整训练建议顺序：先在目标环境做只读 preflight 和一批 forward/backward，再按 `official_release` 三个预声明本地 seed 逐任务训练并保存 final，随后用同一目录独立 eval；需要论文指标时另跑明确的 evaluation_spec，不用 test 选择 best，不覆盖旧 Transolver 结果。

本L阶段结束，未执行下一阶段。
