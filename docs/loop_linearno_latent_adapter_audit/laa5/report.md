# LAA5：V3 wrappers、checkpoint 与 output adapter

## 状态

**PASS（仅本阶段范围）**。本阶段在 `main @ c02e671506f706910e0a1d58f03c310abf188345` 上完成 V3 三类 wrapper、独立 V3 checkpoint pair 和当前 recorder 的薄适配。没有修改八任务真实生产选择分支，没有真实数据或长训练。

## 实现映射

- `cdlno/linearno_loop/v3/standard.py` 继承现有 Standard `Model` forward，保留 irregular、structured、temporal 的输入检查、位置距离、time embedding、placeholder 和输出 shape；仅把 `blocks` 视图替换为一个 V3 loop core。
- `cdlno/linearno_loop/v3/airfrans.py` 继承 AirfRANS native `Data -> [N,4]` forward，保留 x7、原 reference、可变 N、单图校验和 dead temperature；每个构造实例拥有自己的 core、feature modules 与 state。
- `cdlno/linearno_loop/v3/shapenet.py` 继承 ShapeNet native `(cfd_data, geom) -> [N,4]` forward，保留 x7、单图、fold 相关输入合同；V3 允许 `H=208, heads=8, d_h=26, M=32`，旧 wrapper 的 M/head_dim 约束未改。
- `cdlno/linearno_loop/v3/construction.py` 只允许经过已验证 V3 config 构造，使用 forked public seed；分配完整 stem/core/head 后一次 native release initialization，再生成 placeholder，最后用 feature-local seeds 安装 latent/adapter。V3 不进入旧 dispatch。
- `cdlno/linearno_loop/v3/checkpoint.py` 使用 `linearno-loop-epoch-pair-v3`。先读并验证 sidecar、manifest、config、路径和 hash；恢复前验证 state shape、optimizer ownership、scheduler/scaler、normalizer、sampler、RNG/DataLoader generator 和 ensemble checksum，最后 `strict=True` 应用 tensor。V1/V2 pair reader 和格式不变。
- `cdlno/linearno_loop/v3/output.py` 只委托仓库已有 recorder 的 `attach_model`、training setup、epoch、visualize 和 metrics 方法，不拥有 trainer、指标数学或新目录。
- `linearno_loop/v3/*.py` 是独立 schema、profile、cost、matrix 和 metadata 合同。`architecture=operator_latent_adapter_v3` 必须显式存在；cost profile 不选择 V3。

## 数学与所有权

V3 core 由 P+C+S 个完整 native block 构成，core block 按物理位置跨 R 轮共享完整 operator、`ln_1`、`ln_2` 和 point FFN。latent FFN 按 core 位置跨轮共享，第二轮 Q/K adapter 也按 core 位置独立；关闭特性时不注册对应参数。prefix/suffix 保持 native residual，最后 suffix head 只运行一次。attention 的 Q/K/V、K^T V 和 Q readout 每次 visit 重新计算；没有 activation cache 或跨 forward history。

ShapeNet 的实际 M 直接传给 V3 attention，因而 `M=32, d_h=26` 合法；该放宽只存在于 V3 wrapper。AirfRANS ensemble 的成员在测试中分别构造，参数 id 与 state 独立。

## 测试证据

命令均在仓库根目录、`PYTHONPATH=tests:.`（旧回归另加 `tests/loop_linearno_ffn`）运行，线程设为 1；环境见 `results.json`。

- LAA5 新测试：15/15 通过（26.47s），0 失败、0 错误、0 跳过；包括三类 wrapper 的 forward/backward/AdamW、metadata-first 冲突、V3 pair hash/strict replay、scaler/RNG/generator/normalizer/ensemble 校验、V1/V2 新进程 replay 和 output delegation。最终日志为 `new-final-after-matrix.log`。
- 完整 wrapper 矩阵：144/144 通过，覆盖 6 attention variants × 2 cost profiles × 3 residual modes × 4 feature ablations，正式 D12 H/Dz/M/head 表配合合成 B/N。
- 旧 wrapper/entry/checkpoint 定向回归：33/33 通过。
- LAA1–LAA3 配置/原语/attention 回归：68/68 通过；LAA4 core 新测试重跑：14/14 通过。
- pure/v1/v2/LL9R 旧定向回归：72/72 通过。
- Python AST 内存编译 50/50、shell `bash -n` 通过；V3 符号未进入旧 parser/factory/entry。

坏 generator 名称合同的负向测试确认恢复会在权重应用前失败，且目标模型 state 未被改写。初次误写的“当前 generator 随机状态必须相同”测试已删除；恢复语义正确地允许当前 generator 状态不同，再从 archive 恢复保存状态。

## 冻结与限制

唯一 tracked diff 是用户原有 `check_checkpoints/check_pipe_loop_resume.sh`，本阶段未编辑该文件。所有 V3 与 LAA5 测试/证据是新增文件；旧 V1/V2/pure/history/Transolver/CDLNO/KCDNO/MSAR-LNO 文件、golden、容差和输出结果未改。

LAA0 的旧全仓基线仍是 644 passed、9 historical failures、36 skips；本阶段没有重写它。9 个失败包含已记录的 legacy/provenance 与 KCDNO frozen projection/environment 项，LAA5 新增 0 failures。真实数据、完整 epoch、精度、远端 Python 3.10/Torch 2.11/cu128、真实任务恢复/ensemble 长程断点、延迟、峰值显存和 SOTA 均 **NOT RUN**。因此 PASS 只表示本阶段的合成 wrapper/checkpoint/output 合同成立，尚不能提供 V3 生产训练命令。

## 优先复核点

1. wrapper 只通过 V3 config 构造，且 native forward 的继承关系和 ShapeNet M/d_h 解耦边界。
2. `inspect_checkpoint`/`restore_training_state` 的 metadata-first 顺序、完整 archive 校验与 strict state application。
3. 144 行完整交叉矩阵是否覆盖六 variant、两 profile、三 residual、四消融。
4. output adapter 是否只委托现有 recorder，未引入新训练或指标逻辑。
5. LAA0 九个历史失败和 36 个 skip 的来源未被本阶段测试结果掩盖。

本 LAA5 阶段结束，未执行下一阶段。
