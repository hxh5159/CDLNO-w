# CDLNO 最终需求追溯矩阵

日期：2026-09-14，阶段10。按实际生产源码、测试断言及本轮执行日志反查 v1.2 §§0、8、13、14；不是把历史阶段报告当作测试结果。用户当前决定优先于计划：正式名称CDLNO、分阶段授权、保留已工作远端依赖。数学附件不授权新增结构。

**状态口径**：通过=该行代码/静态或合成门槛实际满足；未运行=不能从其他成功推导；边界=只约束实现/论述，不是已证明的数学或实验结论。最终 [regression.log](final_audit/regression.log) 为119项通过、0失败/错误/跳过，49.871秒；[temporal-focused.log](final_audit/temporal-focused.log) 为补齐时间验收后的9项通过。最终环境见 [environment.json](final_audit/environment.json)。

## 代码与证据索引

表内符号指下列实际文件中的类/方法，测试缩写后跟**精确test方法名**（无参数扫描省略号）。

| 索引 | 文件与具体定位 |
|---|---|
| M | [modules.py](../cdlno/modules.py)：LRSAFrontBlock.forward、IPOTBridge.forward、PersistentLatentBlock.forward、LRSAFeatureReadout.forward_features、ConvFFN、RMSNorm、_init_linear |
| C | [core.py](../cdlno/core.py)：CDLNO.__init__/forward，活跃CDPA注册及局部history |
| D | [cdpa.py](../cdlno/cdpa.py)：CDPA.forward/_depth_fusion |
| Q | [config.py](../cdlno/config.py)：CDLNOArchitectureConfig/RuntimeConfig；[checkpoint.py](../cdlno/checkpoint.py)：save/load/validate_sidecar |
| W | [standard.py](../cdlno/standard.py)：StaticStandardModel/TemporalStandardModel；[Car Model](../Car-Design-ShapeNetCar/models/CDLNO.py)；[AirfRANSModel](../cdlno/airfrans.py) |
| E | [标准StaticRun](../PDE-Solving-StandardBenchmark/cdlno_entry.py)、[CarRun](../Car-Design-ShapeNetCar/models/cdlno_run.py)、[AirRun](../Airfoil-Design-AirfRANS/cdlno_entry.py)，各parse_args/model_kwargs/load/validate |
| P | [性能models](../tools/cdlno_perf/models.py)：LRSAMatched/build；[costs](../tools/cdlno_perf/costs.py)：audit；[measure](../tools/cdlno_perf/measure.py)：benchmark |
| TM / TL | [test_modules.py](../tests/test_modules.py) / [test_lrsa_reference.py](../tests/test_lrsa_reference.py) |
| TD / TC / TQ | [test_cdpa.py](../tests/test_cdpa.py) / [test_core.py](../tests/test_core.py) / [test_core_config.py](../tests/test_core_config.py) |
| TS / TT | [test_static_standard.py](../tests/test_static_standard.py) / [test_temporal_standard.py](../tests/test_temporal_standard.py) |
| TG / TA / TP | [test_shapenet_car.py](../tests/test_shapenet_car.py) / [test_airfrans.py](../tests/test_airfrans.py) / [test_performance.py](../tests/test_performance.py) |
| Freeze | [freeze-check.json](final_audit/freeze-check.json)：逐文件基线/阶段10起点SHA256；整入口AST回归由TS、TT、TG、TA执行 |
| Provenance | [三仓来源与许可核对](CDLNO_THIRD_PARTY_NOTICES.md)：固定commit/source SHA、继承差异 |

## §0.1 必须遵守的14项

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| 1 Transolver基仓库，旧模型脚本保留，新名称/配置/目录/checkpoint | E；各configs/CDLNO和scripts/CDLNO；Q.model_name | Freeze；TS.test_defaults_and_explicit_overrides_leave_legacy_parser_unchanged；各run隔离测试 | 通过；旧真实训练未重跑 |
| 2 仅架构和必要选择/参数/checkpoint接入，冻结任务协议 | E+六exp/两组main的有限分支 | TS.test_legacy_ast_recovered_exactly_from_limited_branches；TT.test_complete_temporal_entries_project_to_baseline_ast；TG.test_original_entries_project_to_baseline_ast；TA.test_complete_entry_ast_preserves_original_protocol；Freeze | 通过；数据运行完整性未测 |
| 3 八任务且不统称PDEBench数据库 | W+model_dict；8份任务JSON | TP.test_all_eight_task_wrapper_and_original_baseline_synthetic_calls；README/任务清单逐名列出 | 通过 |
| 4 默认完整2+6、独立bridge、入口CDPA、feature readout | C.__init__/forward；M | TC.test_default_and_42_depth_mode_point_grid_forward_backward_cases；test_operation_counts_and_constant_m_all_front_depths | 通过 |
| 5 规则后置dense ConvFFN；不规则point；无slice/latent卷积 | M.ConvFFN/LRSAFrontBlock/LRSAFeatureReadout；C | TM.test_conv_5_by_7_row_major_channel_mixing；TC.test_registration_storage_independence_and_single_initialization | 通过 |
| 6 T在FFN2后/up前 | M.LRSAFrontBlock.forward | TM.test_front_order_history_object_and_live_gradients；TC.test_history_objects_timing_readout_and_two_forward_graph_isolation | 通过 |
| 7 CDPA Cross；无Slice/Gram | D；Q.cdpa_mode | TD数学reference；Q只接受三模式；生产源码审查无校正分支 | 通过；整网正式名改CDLNO |
| 8 全阶段M相同，按任务设置M | C统一m；W/Q | TC.test_operation_counts_and_constant_m_all_front_depths；TP.test_task_preset_comparison_matches_current_sources | 通过 |
| 9 F/L可配置，不动态删增层 | Q.validate/P；C构造 | TC.test_extended_boundary_and_front_above_six_all_modes；TQ.test_illegal_architecture_and_core_policy_rejected | 通过，含F7/F8 |
| 10 entry默认，off/every可验收不启动扫描 | C/D；P仅合成CLI | TC三模式；阶段9持久结果；执行日志无真实训练 | 通过 |
| 11 稀疏Darcy/递增M仅记录 | Q固定feature/标量M；W八任务接口 | TQ.test_illegal_architecture_and_core_policy_rejected；源码审查无额外loader/decoder | 通过（排除项） |
| 12 无数据，数学/合成/损失/checkpoint验收 | tests及P.synthetic_inputs | 最终119项日志；无exp直接导入；无下载数据命令 | 通过；不等于训练成功 |
| 13 PyTorch SDPA，无自定义核/新训练框架 | M._ProjectedAttention/_DownAttention；D.forward；pyproject | TM.test_sdpa_matches_explicit_attention_and_gradients；生产import及依赖diff审查 | 通过 |
| 14 明确确认决定和实现默认区别 | Q/W/任务JSON；README/报告 | M的norm/bias/init测试；规格与来源差异表 | 通过；不称原论文统一preset |

## §0.2 六对实现/排除边界

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| 八原任务接入；稀疏Darcy不接 | W/E | TS/TT/TG/TA接口与TP八任务factory | 通过 |
| 配置L/F；不递增latent数 | Q/C | TC配置矩阵/constant M | 通过 |
| off/entry/every；无CDPA-Slice | D/C | TD、TC数学与模式检查 | 通过 |
| 无数据工具；无物理残差损失/校正层 | P/tools；原loss冻结 | TP成本/CLI，Freeze与原loss AST | 通过 |
| NS10→10；不做无长标签20/40 | W.TemporalStandardModel；exp_ns原循环 | TT.test_ns_teacher_forcing_and_prediction_windows_batched（本轮完整10步）及整AST | 通过 |
| Plasticity独立条件调用；不展平时空/跨时间latent推进 | W.TemporalStandardModel；exp_plas原循环 | TT.test_plasticity_twenty_independent_time_steps_and_loss；TC两forward隔离 | 通过 |

## §8.1 静态合同（逐项）

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| 文件/字段/轴交换/采样/点序/fold/split冻结 | 六exp、两工业main、dataset | Freeze+TS/TT/TG/TA完整AST投影 | 通过（静态） |
| normalizer/decode/通道/loss/权重/mask/梯度项冻结 | 原utils/train/exp；W输出 | TS.test_darcy_exact_training_decode_and_gradient_loss_ast；TG.test_original_train_mask_loss_backward_optimizer_scheduler；TA.test_original_weighted_loss_train_and_test；Freeze | 通过（静态+合成连接） |
| NS真值/预测回填，Plasticity更新节奏 | 两时间exp原loop | TT完整10步及20次更新；test_complete_temporal_entries_project_to_baseline_ast | 通过 |
| optimizer/scheduler/clip/预算/指标/边界后处理 | E新默认；原优化器/指标区段 | 各AST投影+Freeze；TP.test_task_preset_comparison_matches_current_sources | 通过；真实指标值未测 |
| 不import exp触发真实I/O，不重写训练器 | tests的AST抽取；E辅助模块 | TQ.test_three_task_cwd_fresh_imports_without_entry_or_install；TS/TT/TG/TA测试源码审查 | 通过 |

## §8.2 十三类合成输入

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| CPU小模型形状/有限前反向 | M/C | TM各公式测试；TC.test_default_and_42_depth_mode_point_grid_forward_backward_cases | 通过 |
| 六adapter：fxNone/fx1/fx10/T/out1/out4 | W | TS.test_four_interfaces_three_modes_backward_placeholder_and_no_time；TT.test_defaults_dimensions_and_time_registration及条件/窗口测试 | 通过 |
| 5×7点序、错误N报错 | M.ConvFFN；Q/grid | TM.test_conv_5_by_7_row_major_channel_mixing；TQ.test_invalid_input_and_chunk_rejected_before_computation | 通过 |
| 原N布局（减小d/M），全配置GPU可选 | W | TS.test_real_n_with_smaller_width_latents_preserves_layout；TT实际4096/3131；TG.test_large_synthetic_point_count_and_input_gradients；TA.test_32000_synthetic_points_small_width_backward | CPU通过；八任务默认完整配置GPU矩阵未运行 |
| NS10次调用、两种回填 | W/exp_ns | TT.test_ns_teacher_forcing_and_prediction_windows_batched：B2、每路10步、窗口逐步检查、训练一次backward/step | 通过；本轮补齐原3步缺口 |
| Plasticity B>1/不同T/时间梯度/无反馈 | W/exp_plas | TT.test_plasticity_time_changes_output_and_time_gradients：T=[[.1],[.3]]与+0.4、time_fc梯度；20次独立更新测试 | 通过 |
| Darcy原normalizer/TestLoss/边界与梯度项 | exp_darcy原表达/W | TS.test_darcy_exact_training_decode_and_gradient_loss_ast | 通过 |
| 工业原surf true/false mask损失backward | 原train.train/test；W | TG.test_original_train_mask_loss_backward_optimizer_scheduler；TA.test_original_weighted_loss_train_and_test（真实PyG） | 通过 |
| 改y无影响、输入无原地改动 | Car/Air W | TG.test_variable_n_single_batch_no_mutation_no_label_geometry_read；TA.test_variable_nodes_batch_and_immutable_fields_no_y_leak | 通过 |
| 工业单图可变N/多图拒绝/通道点序 | Car/Air W | TG.test_multigraph_batch_ptr_and_malformed_inputs_rejected；TA.test_sampled_single_graph_ptr_and_multiple_graph_rejection；前一行可变N测试 | 通过 |
| point版置换等变，grid不要求任意置换 | M/C/W | TM.test_point_permutation_latent_permutation_and_batch_isolation；TC.test_point_order_batch_independence_and_variable_n；工业可变节点测试 | 通过 |
| 三原cwd新进程导入wrapper | W/E | TQ.test_three_task_cwd_fresh_imports_without_entry_or_install；TS.test_fresh_process_wrapper_checkpoint_restore；TG/TA独立进程测试 | 通过；PYTHONPATH导入，未执行editable安装 |
| 同配置/独立进程checkpoint；M/F/L/错误权重拒绝 | Q/E | TQ.test_strict_weights_and_existing_sidecar_in_fresh_process；TS.test_strict_roundtrip_sidecar_no_overwrite_and_separate_outputs；TG.test_checkpoint_rejects_wrong_object_and_missing_weights；TA.test_checkpoint_list_type_length_and_strict_keys | 通过（键/形状/架构语义校验，非密码学完整性） |

## §8.3 CDPA十四项及正常forward约束

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| 1 空历史恒等；F0/entry无参数 | D空sources/C.cdpa_at | TD.test_empty_history_is_exact_identity_without_computation；TC.test_f0_entry_off_same_weights_exact_outputs_gradients_and_calls | 通过 |
| 2 w0=RAW候选均值，无外残差 | D.__init__/_depth_fusion | TD.test_uniform_initialization_raw_values_and_no_outer_residual | 通过 |
| 3 token与source两独立softmax轴 | D.forward/_depth_fusion | TD.test_two_softmax_axes_and_tokenwise_weights，含错误concat反例 | 通过 |
| 4 独立显式reference输出/关键梯度 | D；[cdpa_reference.py](../tests/cdpa_reference.py) | TD.test_reference_does_not_call_production_or_sdpa；test_sdpa_reference_output_and_all_gradients_matrix（48case） | 通过；浮点容差，非逐位等同 |
| 5 来源交换不变 | D共享投影/scorer | TD.test_source_reorder_and_historical_token_permutations | 通过 |
| 6 历史token重排不变 | D每源独立M | 同上 | 通过 |
| 7 当前token重排等变 | D | TD.test_current_token_permutation_equivariance | 通过 |
| 8 B>1样本独立 | D的[B*k,h,M,d_h]折叠 | TD.test_batch_isolation_forward_and_gradient；test_chunk_layout_counts_q_once_and_single_global_fusion | 通过 |
| 9 T/bridge/后段/readout/depth路径梯度可达 | C局部活history；M/D | TC.test_direct_history_readout_gradients_zero_w_then_updated_w；test_explicit_raw_state_schedule_output_and_all_gradient_parity | 通过；非仅单次CDPA测试 |
| 10 零w不误判norm零梯度 | D.depth_norm/w | TD.test_zero_w_gradient_then_updated_w；TC同项 | 通过；首步scale可0，更新w后检查 |
| 11 raw Z0/过去完整Z时序，不含当前/未来/跨forward | C.forward | TC.test_history_objects_timing_readout_and_two_forward_graph_isolation；test_explicit_raw_state_schedule_output_and_all_gradient_parity | 通过；检查对象ID与图隔离 |
| 12 极端值/FP32 depth/autocast/GPU半精度 | D._depth_fusion | TD.test_extreme_magnitudes_and_all_gradients_finite；test_depth_fp32_dispatch_under_cpu_autocast；3项test_gpu_* | 通过；实际本地GPU，远端未运行 |
| 13 chunk0/1/2/>S，B>1/S1,2,5/非整除、Q一次、统一融合 | D.forward | TD.test_sdpa_reference_output_and_all_gradients_matrix；test_chunk_layout_counts_q_once_and_single_global_fusion；TC.test_actual_history_sdpa_counts_chunks_and_each_location_projections | 通过；mixed dtype另有test_mixed_history_dtypes_preserve_cast_gradients |
| 14 chunk可载；mode/F/L拒绝；eval不覆盖 | Q/E | TQ.test_sidecar_runtime_allowed_architecture_rejected_bytes_preserved；TS/TG/TA实际checkpoint同chunk0→1和错误配置测试 | 通过 |
| 正常forward只回Tensor、不保存巨大注意力/retain_grad/CPU统计 | C.forward；M/D SDPA | TC输出类型和两forward隔离；源码审查；仅D显式return_weights=True可返回小depth权重 | 通过 |

## §8.4 配置与计数八项

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| L8 F0..6×三模式 | Q/C | TC.test_default_and_42_depth_mode_point_grid_forward_backward_cases：point和grid各21种 | 通过 |
| L12/F2、L16/F6、L1/F0 | Q/C | TC.test_extended_boundary_and_front_above_six_all_modes，另含L8/F7、L10/F8 | 通过 |
| 非法F/dh/N清楚报错 | Q.validate；C/M检查 | TQ.test_illegal_architecture_and_core_policy_rejected；test_invalid_input_and_chunk_rejected_before_computation | 通过 |
| P=L−F独立对象/storage | C.latent_blocks/cdpa_at | TC.test_registration_storage_independence_and_single_initialization；TM.test_parameter_identity_and_storage_are_independent | 通过 |
| down/up各F+1；latentSA=L | C/M | TC.test_operation_counts_and_constant_m_all_front_depths；TP.test_live_counts_formulas_and_chunk_invariant_matrix_cost | 通过；默认3/3/8 |
| ConvFFN=F+1；后段/CDPA无卷积 | C/M | TC计数及模块类型检查；TP.test_projection_ffn_and_dense_conv_all_counted_against_closed_form | 通过；默认3 |
| entry总F；every PF+P(P−1)/2；次数与份数分开 | C/D/P | TC.test_actual_history_sdpa_counts_chunks_and_each_location_projections；TP计数测试 | 通过；默认entry2/1、every27/6（chunk0） |
| off不存T；entry无后段历史；every只存待消费状态 | C.forward局部列表 | TC.test_off_discards_t_and_entry_releases_history_after_only_use；test_history_objects_timing_readout_and_two_forward_graph_isolation | 通过 |

## §8.5 工具、执行与证据约束

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| 建议测试分类、数学/布局/adapter/时间/checkpoint/无I/O | 实际10个test文件及cdpa_reference | 最终119项，采用unittest并覆盖对应类别；文件名为建议而非固定API | 通过 |
| 建议editable安装与环境候选文件 | pyproject、tools/cdlno_environment_preflight.py、README | 可安装最小包已建；用户已确认保留现有依赖，未新增锁定/重装文件；本次未运行pip | 实现可用；目标安装未验证 |
| 工业真PyG，不伪造模块 | TG/TA Data/Batch | 最终PyG2.3.1测试均运行无skip；CPU原mask损失连接 | 通过 |
| 预检不访问数据，不等于模型验收 | tools/cdlno_environment_preflight.py | 源码审查；本轮只读environment.json不调用训练入口 | 通过（范围声明） |
| 不从CPU/理论升级为真实训练/收敛 | README/最终报告 | 将真实I/O、远端与精度、epoch明确列未运行 | 边界已落实 |

## §13 全对话20行逐项复核

| 要求 | 代码位置 | 验证证据 | 状态 |
|---|---|---|---|
| Transolver基仓、LRSA/IPOT参考 | M/W及固定来源 | Provenance+TL两项实际外部参考对照 | 通过 |
| 只改模型，原数据处理不变 | E有限分支、原dataset/train/utils | Freeze+§8.1 | 通过（静态） |
| 不依靠真实数据验收 | tests/P | 最终日志；§8.2 | 通过 |
| 完整2个LRSA+6个persistent | M/C | TC默认矩阵、TM完整顺序 | 通过 |
| F0..6及扩展L、P派生 | Q/C | TC扩展、TQ非法配置 | 通过 |
| 历史来自latent而非点 | M.LRSAFrontBlock.forward | TM history对象；TC historyID | 通过 |
| bridge后entry一次，F0无闲置参数 | C/D | TC同权重F0路径/计数 | 通过 |
| every读前段与更早完整Z，不跨forward | C.forward | TC完整时序/两图隔离 | 通过 |
| 逐历史Cross，再tokenwise RAW depth融合 | D.forward/_depth_fusion | TD reference、两softmax轴 | 通过 |
| 机制CDPA，无Slice | Q/D；整网CDLNO | 模式白名单+生产源码审查 | 通过；遵守后续正式命名 |
| feature H_F query/残差，非任意输出网格 | M.LRSAFeatureReadout | TM.test_readout_query_context_residual_and_output_norm；TC history-readout | 通过；不声称超分辨率 |
| 规则dense3×3后置卷积，无latent卷积 | M.ConvFFN/点分支 | TM5×7与TP独立MAC公式 | 通过 |
| 全阶段M固定，Pipe32其余64 | Q/C/任务JSON | TC constantM；TP presets | 通过 |
| 原epoch/训练/评价，Pipe batch8/Air398 | E defaults/YAML/scripts | TP presets；TS/TT/TG/TA冻结 | 通过 |
| NS10→10 rollout，不无标签20/40 | W/exp_ns | TT完整两种10步+全AST | 通过 |
| 稀疏Darcy仅未来记录 | Q.feature/W白名单 | 源码无loader/decoder扩展；§0.2 | 排除正确 |
| 争取效率，完整成本、matched LRSA实测 | P；独立性能CLI | TP11项及阶段9持久27行GPU结果 | 工具/有限实测通过；收益不保证 |
| every额外成本；Q一次/来源批处理 | D/P | TD chunk布局spy；TC counts；TP MAC | 通过 |
| 不跨位置KV缓存/不detach/不省来源 | C/D | TC参数/storage、完整历史梯度及SDPA次数 | 通过 |
| 远端CUDA12.8/torch2.11保留，不盲重装 | pyproject无运行依赖；预检/README | 原requirements未动；本轮无安装；远端Car证据仅用户报告 | 保留通过；CDLNO远端未运行 |

## §14 数学支持与边界

文件核查：本轮以计划列出的5个精确文件名在PLAN_CDLNO及/home/hwz下复查，仍未定位独立中文理论、英文tex/pdf、NumPy脚本及结果JSON。现有聊天导出末尾理论摘要和v1.2第14节可读，但**不是附件全文或附件检查复现**。因此下列数学主张只映射计算图、记录适用边界；不标作新证明通过。

| §14要求/主张 | 代码位置 | 本轮证据与边界 | 状态 |
|---|---|---|---|
| 附件正文、编译PDF、8项NumPy核查 | 无对应实现文件要求 | 历史文本报告这些产物；本地未定位，未编译/执行；[阶段0记录](CDLNO_REFERENCE_AUDIT.md)一致 | 附件全文核验未完成 |
| 14.1 少数点域往返/latent可压缩性 | C前段/bridge/rear | TC计数通过；未测POD谱，不保证任意PDE/M64充分 | 实现对应已核；理论未复验 |
| 完整独立LRSA可累积不同方向 | M/C | TM/TL完整block、TC参数独立；不称后层天然容量更大 | 同上 |
| down/up解耦 | M.down/up独立 | TM参数/storage；未强制双正交/求逆/inf-sup | 实现对应已核 |
| feature decoder局部H_F/全局latent依赖 | M.LRSAFeatureReadout | TM/readout+TC对象检查；point本地点或最终3×3邻域，不声称访问全部HF | 实现对应已核；理论未复验 |
| 逐历史对齐的token重编号性质 | D | TD历史置换/当前等变；不保证物理对应或最优传输 | 合成性质通过；强结论不宣称 |
| 来源共享/评分置换性质和初始化路径 | D | TD来源换序/梯度；没有depth embedding，不识别绝对来源编号 | 合成性质通过 |
| 历史复用的条件用途/盲方向 | C/D | raw历史梯度可达；冻结前端理论不等于重训练函数类严格包含 | 实现对应已核；未做JVP诊断 |
| w0/RAW融合：均匀与凸组合 | D._depth_fusion | TD均值/FP32/source和为1；不称完整网络非扩张/守恒稳定 | 合成性质通过；理论未复验 |
| every扩展状态与成本 | C/P | TC每层ID/来源数，TP成本；无新观测，不保证多历史必增精度 | 实现/计数通过 |
| 14.2 不为理论增删norm/decoder/正交/quadrature/gate | M/C/D/Q | 源码核对固定公式；未出现上述新增结构，FFN内GEGLU不等于外加融合gate | 排除正确 |
| 不宣称固定M普适逼近、普适低秩/无损、误差必降、守恒/网格一致、严格类包含、GPU必加速 | README/报告 | 明确保留条件性；阶段9还有变慢的实测，不包装为普遍优势 | 边界已落实 |
| 14.3 历史理论审阅/数值/编译不升级工程状态 | 全报告证据分类 | 只把本轮119项与已持久阶段9结果标实测；缺附件项未完成 | 通过（记录准确） |
| POD/JVP/VJP等未来可选，不加研究/损失；CFD全局后处理冻结 | 生产core/原metrics | 无新研究诊断/损失；Freeze原指标字节相同 | 排除/静态冻结通过 |

