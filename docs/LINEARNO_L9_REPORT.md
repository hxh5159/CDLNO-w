# LinearNO L9 报告

## 状态

**PASS（真实数据和完整训练 NOT RUN）。** L9 只运行已有测试、合成 forward/backward、
checkpoint、旧模型回归和有限 GPU smoke；没有修改生产模型、factory、训练协议、数据或依赖。

## 综合矩阵

| 检查 | 结果 | 边界 |
|---|---:|---|
| `tests/linearno` + Air/Car + 前段/旧模型回归 | **118 passed** | 419.44 s；2 个既有 warning，无 failure/error |
| 转换器/profile/schema | **13 passed** | 13.70 s；另含 converter 2 项于总套件中 |
| shell 语法 | **PASS** | `bash -n tran_evaluate/linearno/*.sh` |
| 正式参数量/结构构造 | **PASS** | 八任务正式 kwargs，见 L3/L6/L7 数值 |
| strict state/metadata/native roundtrip | **PASS** | 合成任务闭环，未用外部 checkpoint |
| GPU smoke | **PASS（有限）** | 本机 RTX5090 Laptop，FP32、batch1、warmup3/measure5 |
| 真实数据/VTK/MAT | **NOT RUN** | 当前无真实数据授权/文件 |
| 远端 Python3.10/Torch2.11/CUDA12.8 | **NOT RUN** | 本机为 Python3.13/Torch2.13+cu130 |

此前第一次综合运行的唯一失败是旧 `test_legacy` 对复现矩阵历史后缀的冻结断言；矩阵已恢复
为 L1 原文，失败用例单独重跑通过，最终综合套件为上述 118/118。

## 结构和效率

Standard、AirfRANS、ShapeNet 的 attention 只生成按 head 的 Q/K/V、`K^T V` 和 `Q C`；
代码没有 `N×N` logits、slice self-attention 或持久 Q/K/C。已有 L2/L3/Air/Car 独立 oracle
和 forward hook 逐层验证了这一点，L9 复查了 `LinearNO_Attention.py`、Air/Car attention
包装和完整 block 残差顺序。合法的 `[M,d_h]` context 未被误判为 token self-attention。

参数量（正式构造）为：Darcy 1,766,145；Elasticity 585,217；Airfoil/Pipe 1,765,889；
NS 3,377,921；Plasticity 1,799,428；AirfRANS 3,358,788；ShapeNet-Car 3,852,420。
LinearNO 的 attention 主项为 `O(B*N*H*d_h*M + B*H*M*d_h*d_h)`，卷积分支另有每 block
`O(B*H*W*d*k^2*d)`；FFN 为 `O(B*N*d*(ratio*d))`。这些是实现路径的结构复杂度，不是
实测速度承诺。

本机合成 smoke（FP32/math backend 默认、无 AMP/compile）结果详见
`linearno_audit/l9/performance.json`：小模型 `plain N64 d32 L8` 中位 3.004 ms、峰值
33.9 MB；`conv_temp N15 d32 L8` 中位 3.552 ms、峰值 36.6 MB；窄模型 `N1024/N4096`
分别中位 0.662/1.227 ms。GPU kernel 启动和窄模型配置使这些数字不能用于 4090 论文规模
或与 Transolver/KCDLNO 的公平速度比较；没有运行 CPU 推断替代 GPU 延迟，也没有真实训练
step/epoch 性能结论。

## 冻结和可视化边界

`git status --porcelain -- LINEARNO/` 为空，仓库没有既存 `LINEARNO/` 目录；旧
Transolver/Physics-Attention/Embedding 文件的既有 L0/L6/L7 hash 证据保持。PropagationMonitor
在当前 checkout 不存在，记为 **N/A**；没有创建假 monitor 或写入 `LINEARNO/monitor/output`。
现有可视化/recorder 和 resume 产物由 L1/L6/L7 合成入口测试覆盖；LinearNO 不支持的内部
诊断不会让旧模型路径失败。

`paper_table8_on_release_model` 与 `official_release` 的结构字段在八任务 profile 中一致时，
目标/评价字段仍分开保存；AirfRANS 和 Car 的官方 MSE 训练合同与论文 rL2 评价描述差异
已在 profile/metadata 中显式记录。完整训练耗时、磁盘、收敛、精度仍留到真实数据阶段估算，
本 L9 不启动。

**本 L9 阶段结束，未执行下一阶段。**
