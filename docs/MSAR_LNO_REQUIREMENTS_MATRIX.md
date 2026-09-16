# MSAR-LNO 总控规格独立反查（M9）

依据：用户最新确认的总控26条及M1—M9要求，其次[分阶段提示词](../PLAN_MSAR_LNO/MSAR_LNO_Codex_Staged_Prompts.md)。[架构讨论原文](../PLAN_MSAR_LNO/MRSA_LNO_arch.md)包含已经放弃的specialization、双向JS、skip dropout等候选；它们不构成当前授权。逐项从需求查实际源码，再找独立数值、hook、负向或冻结证据，不以文件/类名称作为结构正确的证据。

M8证据见[覆盖表](MSAR_LNO_M8_ACCEPTANCE.md)与[结果](msar_lno_audit/m8/summary.json)。M9重跑M2/M3的36个数学/core方法，并增加不依赖原reference的手算例及性能核对；见[M9报告](MSAR_LNO_IMPLEMENTATION_REPORT.md)。下表“通过”只在对应证据范围内成立，不表示真实数据实验通过。

| 总控条款 | 实际文件 / 符号 | 独立证据（测试位于tests/） | 状态 |
|---|---|---|---|
| 1 完整浏览 | `MSAR_LNO_REFERENCE_AUDIT.md`及M0 inventory/hash分组 | M0受版本控制源码/配置/脚本/README覆盖与排除清单；M9重新核查相关生产/性能/规格文件 | 已审计；M0为M1后补审，历史顺序已披露，不倒填 |
| 2 实际映射 | M0八任务表；`standard_entry/industrial_entry`、真实model_dict/main分支 | M5—M8安全AST构造、真实parser、新cwd与原loss检查 | 通过 |
| 3 保存用户工作 | M9 `before.json/preexisting.patch`及独立源快照 | 全量起点hash比较、阶段diff、HEAD/status保留 | 通过；无reset/commit/push/真实训练 |
| 4 分阶段授权 | 独立STATUS/M0—M8报告 | 各阶段冻结/证据，用户补审M0并接受后续阶段；M9只改工具/测试/说明 | 本轮通过；不隐去M1先于补M0的历史 |
| 5 旧模型/实验保留 | 旧`cdlno`、K家族、原Transolver、matched及原入口 | M8旧157方法及75份同权重fixture；M9生产文件全字节冻结、旧15性能方法通过 | 已验证范围通过；GUNet原依赖缺失未执行 |
| 6 独立家族与参数 | `cdlno/msar_lno/{registry,core,standard,temporal,industrial}.py` | M3/M8对象/storage唯一、独立core；M9实际全参数分组覆盖 | 通过；组合公共PlainFFN/SA原语，不共享实例 |
| 7 strict/legacy | `metadata.MSARMetadata/resolve_evaluation`、各Run | M4—M8缺family、缺key/shape、错family/行为与strict往返；旧pickle原路径 | 通过；没有自动转换或strict=False |
| 8 数据/训练冻结 | 八个既有exp/main/train中的显式新family分支；`objective.py` | M5—M7完整AST精确剥离、M8同权重/原loss/time-loop；M9字节冻结 | 通过；真实读取/全指标另列未验证 |
| 9 四级、无卷积、逐点头 | `core.MSARLNO`、三个wrapper的`preprocess/core.output` | M3执行trace、独立完整tensor oracle；M8/M9实际模块计数/参数形状 | 通过，包括规则网格 |
| 10 Light/Full固定 | `profiles.profile_values`、`config.MSARArchitectureConfig` | M1非法/优先级；M8真实profile全任务反传；M9实际16模型参数和N成本 | 通过；不按N截断，M1>N合法 |
| 11 Down learned Q | `modules.LearnedQueryDown` | M2逐batch/head显式oracle和double梯度；M9二维单位投影手算softmax、mask；无P residual | 通过；P为直接投影后Q，无额外Wq，四级独立 |
| 12 FFN–SA–FFN | `modules.LatentFFNSAFFNBlock`；原`PlainFFN/_SelfAttention` | M2三残差oracle/hook；M3/M8 SA12、FFN24；M9两FFN/全部norm/projection实测参数 | 通过；2d GELU，两FFN独立 |
| 13 deepest | `core.forward: decoders[3](encodings[3])` | 独立`msar_core_reference.core`及完整执行顺序hook | 通过；无第四fusion/额外bottleneck |
| 14 Up Q=E / branch-only | `QueryAlignedUpCross`→原`_ProjectedAttention.forward`；`core.ups[level](encoder,decoded)` | M2 explicit-head oracle、query排列、batch隔离、零投影分支测试；M9单位投影手算 | 通过；receiver heads，独立Q/K/V/O |
| 15 两源scale-preserving | `PairwiseAttnResFusion.forward` | M2独立sigmoid差值公式/double梯度；M8三fusion w0/Jacobian/slot排列；M9手算alpha=(.75,.25)输出(1.5,.5) | 通过；raw values、source轴2、固定2、无gamma，只有3d参数 |
| 16 final无E0 skip | `core.final_up(e0,decoded) → output_norm → output` | M3 `test_final_branch_has_no_e0_skip`及oracle，M8 N35处无SAhook | 通过；LayerNorm+逐点Linear沿M0任务head约定 |
| 17 无旧候选机制 | 新core独立ModuleList组合 | M3/M8不存在Conv/CDPA/kernel/history/bridge/persistent参数；M9实际全参数无未归属项 | 通过；历史讨论不能授权新增候选 |
| 18 coverage仅Down、均匀mu | `core`四次Down后`CoverageFloorLoss`；首层mask/measure | M2 oracle/部分mask/归一测度；M8 raw对decoder/Up/Pair梯度为None；工业不把surf/y当mu | 通过；没有额外loss或新数据权重 |
| 19 coverage公式与默认 | `CoverageFloorLoss.forward`、`training_objective/rollout_objective` | M2 per-source手算、独立double；M9 H2/M2/N3非均匀p手算；M6真实10/20步语义 | 通过；source求和、batch/层均值；.2/.01/eps1e-6，日志四项 |
| 20 off/weight0真实关闭 | `coverage_enabled`、`training_forward`、core分支 | M2禁止Tensor.softmax并确认Down SDPA；M4同一个PDE Tensor；M8 off不请求A；M9真实A个数0/4与训练内存 | 通过（diagnostics默认关闭）；SDPA内部attention不等于额外coverage A |
| 21 CLI/metadata/纯eval | `options.py`、真实入口的`parse_args/model_kwargs`、Run/metadata | M5—M8真实parser/freshcwd/strict/sidecar字节不变；M9 32种真实train parser+32种eval预览 | 通过；coverage不构成纯eval形状冲突，未新增resume |
| 22 可选no-grad诊断 | `diagnostics.observe_down/observe_fusion`、`coverage_diagnostics` | M3 CPU同步禁用检查、重用A/显式重算/no-grad/RNG不变、M8第二次backward无残留 | 通过；显式diagnostics会临时重算A，非off默认开销承诺 |
| 23 环境边界 | `environment.json`/GPU原始结果；标准torch/SDPA | 本机Torch2.13/cu130/PyG2.3.1/RTX5090 Laptop；无安装/替换依赖 | 本机通过；目标Torch2.11/cu128未运行 |
| 24 无数据检查 | 既有安全helper/合成Data、工具Case输入 | M8原loss/optimizer/eval/checkpoint；M9只synthetic MSE有限计时，不import exp/main | 通过；无真实数据下载或训练 |
| 25 shape/mask/图隔离 | `_tokens/_mask/_single_graph`及wrapper检查 | M2负向/mask排列/batch隔离；M7真实Data/Batch及多图拒绝；M8变N/非方形边界 | 通过；工业仍单图，可变N |
| 26 报告/停点 | 独立STATUS、最终报告、原README新增独立章节 | M9实际结果索引、阶段diff、命令/链接核查和限制 | M9交付；不自动开始实验 |

独立性复核：`tests/msar_reference.py`只import math/torch，不import生产MSAR/共享模块，也不调用SDPA或待测forward；它逐batch/head计算归一化权重，以logistic差值实现两源融合，以逐样本sum实现coverage。`msar_core_reference.py`只组合这些tensor公式。M9手算例既不调用这两个reference，也不复用待测forward求expected；完整拓扑再由执行trace及零分支测试约束，避免仅靠两个相同循环互证。没有发现需要改确认架构/数据协议的冲突。

未证实的是经验效果：coverage-floor只约束低覆盖源点，不保证所有latent不重复；Up Q=E提供目标slot地址对齐，不具有MoNo的共享OT计划/质量守恒/逆映射性质；无卷积不等于无attention。收敛、精度、真实epoch时长及SOTA均需要后续真实实验，目前不作结论。
