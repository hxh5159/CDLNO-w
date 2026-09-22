# Looped LinearNO V3 最终实施报告

## 结论

**最终状态：PARTIAL。** 授权的 V3 架构、配置、八任务入口、strict
checkpoint、成本工具和无真实数据验证已经完成；V3 新增范围没有发现模型
逻辑失败。整体不能标 PASS，因为旧 V1/pure/history 回归仍保留 9 个既有
失败方法，且当前工作树缺少用户先前删除的旧文档/审计日志。没有修改旧
golden、容差或恢复这些删除内容来制造全绿结果。

本结论只覆盖无真实数据的实现与合成验证，不代表收敛、精度、加速或 SOTA。

## 来源和版本

- checkout：`/home/hwz/CDLNO`
- branch/HEAD：`main@c02e671506f706910e0a1d58f03c310abf188345`
- origin：`git@github.com:hxh5159/CDLNO-w.git`
- 本地环境：Python 3.13.9、Torch 2.13.0+cu130、PyG 2.3.1、
  NVIDIA GeForce RTX 5090 Laptop GPU；`torch_cluster` 缺失
- 设计来源：冻结 V3 staged prompt、研究主线梳理、当前 V1/V2/纯 LinearNO
  源码及 LAA0 参考审计。当前工作树是实施真值。

## 架构反查

forward 路径是 native stem/time/placeholder → P 个独立完整 block → R 轮
重复访问同一组 C 个完整 core block → S 个独立完整 block → 最后 suffix
的 `ln_3 + mlp2` 一次。每个 core block 的 `ln_1 + LinearNO operator +
ln_2 + point FFN` 全部共享；V3 没有 `core_ffns` 或 round-specific point
FFN。prefix/core/suffix 物理对象彼此独立。

每个 core 位置可拥有一套跨轮共享 latent FFN：

```text
context = K^T V                         [B,h,M,d_h]
Z = merge_heads(context)                [B,M,H]
Z' = Z + W2 GELU(W1 LayerNorm(Z))
context' = split_heads(Z')
output = Q context'
```

LayerNorm `eps=1e-5`，无 dropout，不混 M；W2 和 b2 零初始化。关闭时不
注册模块或 state key。

第二轮 adapter 对 Q/K 独立计算：

```text
delta(X) = ((X A^T) B^T) * (alpha / rank)
A: [r,d_h], B: [M,r]
```

矩阵在 heads 间共享；Q/K 各自拥有 A/B，V 不变。A 使用隔离 seed 初始化，
B 为零；第一轮不访问 adapter，第二轮先形成 `base_logits + delta`，再执行
原 variant 的 temperature/clamp 和 softmax。Q softmax 沿 M，K softmax 沿
N。attention 只有 `K^T V` 和 `Q context`，没有 N×N/M×M attention。

三种 residual 复用 V1 冻结公式：SR 对 core 每个 operator/MLP raw branch
分别乘 `1/R`，identity 和 prefix/suffix 不缩放；RB 在每个 sublayer 前由
PointDepthAttnRes 混合 anchor、completed round raw summaries 和当前 raw
partial，无 `1/R`，并复用低精度来源到 anchor dtype 的局部 AMP 边界；LB
轮内 branch 乘 `1/R`，保存实际 `round_output-round_entry`，在 boundary/final
receiver 混合 anchor+deltas。router 状态只属于当前 forward，没有跨 forward
或物理时间 cache。

## 配置和任务接入

V3 唯一选择字段是 `architecture=operator_latent_adapter_v3`；cost profile
不能选择版本。`matched_v1` 默认 D12/P2-C4-R2-S2，`efficient_v1` 同样是
正式 profile；二者支持 D12/D20/D28/D60、三 residual 和四消融。custom
要求显式 H/Dz/M/topology/adapter，所有会改变 state 或数学的字段进入 config
hash 和 run id。八任务宽度表、命令和输出说明见
`docs/LOOP_LINEARNO_LATENT_ADAPTER_USAGE.md`。

六个 Standard 任务继承纯模型 forward，AirfRANS 保留 Data→`[N,4]`、
reference distance、原 weighted loss/metric/sampling/ensemble，ShapeNet-Car
保留 `(cfd_data,geom)`→`[N,4]`、7 通道、fold/surface/drag 边界。V3 ShapeNet
允许 actual M 独立于 `d_h`；旧 ShapeNet 限制和 checkpoint 没有改变。
Standard 的 exp 科学主体、NS 十步循环和 Plasticity 二十查询未改写。

V3 采用独立 `linearno-loop-epoch-pair-v3`。metadata-first 校验发生在 V3
wrapper import/构造及 `torch.load` 前；模型 key/shape 预检后严格加载。
archive 保存 optimizer、scheduler、可选 scaler、Python/NumPy/Torch RNG、
DataLoader generators、sampler、normalizer 和 ensemble manifest。输出继续
委托当前 recorder、visualization 和 metric 接口。

## 文件映射

| 边界 | 文件 | 原因 |
|---|---|---|
| 纯 V3 合同 | `linearno_loop/v3/{contracts,profiles,config,schema,costs,matrix}.py` | 无 tensor 副作用的解析、hash、metadata 和解析成本 |
| 数学原语 | `cdlno/linearno_loop/v3/{adapter,latent,attention,core}.py` | adapter、latent 复用、visit attention、完整 block loop |
| Wrapper/构造 | `cdlno/linearno_loop/v3/{construction,standard,airfrans,shapenet,initialization}.py` | 三类 native forward、隔离初始化和 config-first 构造 |
| Checkpoint/输出 | `cdlno/linearno_loop/v3/{checkpoint,output,provenance}.py` | V3 strict pair、现有 recorder 委托和独立 provenance |
| 版本路由 | `linearno_loop/versioning.py`、`cdlno/linearno_loop/versioning.py`、`cdlno/linearno_loop/{standard_entry,industrial_entry,air_entry,car_entry}.py`、`cdlno/linearno/standard_entry.py` | 显式 V3/saved metadata 分派与八任务复用桥 |
| 旧投影/记录 | `cdlno/linearno_loop/v2_projection.py`、`tran_evaluate/linearno_loop/recording.py` | 保持旧 projection，同时正确记录 V3 完整共享 core |
| Launcher | `tran_evaluate/linearno_loop_v3/` | 两 profile 各八任务及 custom 薄入口，共用一个 parser |
| 成本工具 | `tools/linearno_loop_accounting.py`、`tools/linearno_loop_laa9.py` | V3 独立 analytic/measured 分派与 LAA9 矩阵 |
| 测试/证据 | `tests/loop_linearno_latent_adapter/`、`docs/loop_linearno_latent_adapter_audit/laa0..laa10/` | 独立 oracle、合成闭环、阶段原始日志与交付材料 |

LAA10 本身没有修改上述生产文件，只新增最终文档/证据并更新独立 V3 状态。

## 成本结果

LAA9 覆盖 768 个解析配置和 192 个 D12 实际实例。参数分为 stem/time、
prefix、shared core、suffix/head、latent、adapter、router；解析值、实际
parameter/state keys 和 ATen shape trace 一致。主 8→12、on/on、r4/a4、SR
对照如下：

| 范围 | matched | efficient |
|---|---:|---:|
| 参数比 | 99.7871%–100.1416% | 减少 19.7405%–22.9657% |
| 矩阵 MAC 比 | 98.2793%–103.0369% | FLOPs 减少 11.1188%–14.7925% |

`matched_v1` 只是近似匹配。更深对照、逐任务绝对值和 RB/LB/消融成本在
`laa9/table-8-to-12-and-deeper.json` 与 `parsed-matrix.json`。矩阵口径为
`1 MAC=2 FLOPs`，不含 softmax、norm、GELU 等标量操作，不能外推速度。

唯一 canonical-N 性能 smoke 是合成 Elasticity D12 FP32：forward
median/p90 5.194/5.651 ms，train-step 20.395/20.982 ms，peak allocated/
reserved 235,266,560/262,144,000 bytes。它不是实际 epoch 测量。

## LAA0--LAA9 交付汇总

| 阶段 | 交付 | 阶段结论 |
|---|---|---|
| LAA0 | 源码/文献/旧模型基线、成本复算、冻结清单 | PASS audit；旧基线 644/9/36 |
| LAA1 | 无 torch 副作用的 V3 config/schema/profile/cost oracle | PASS，30 新 + 49 旧定向 |
| LAA2 | bilateral adapter 与复用 latent 原语、独立 oracle | PASS，30 新 + 72 旧定向 |
| LAA3 | 六 variant V3 attention、温度/轴/AMP/无密集 attention | PASS，16 新 + 52 前序 + 72 旧定向 |
| LAA4 | 完整 block 共享 core、三 residual oracle | PASS，14 新 + 68 前序 + 72 旧定向 |
| LAA5 | 三 wrapper、V3 strict pair、output adapter | PASS，15 新、144 wrapper matrix、旧回放 |
| LAA6 | 六 Standard 任务显式生产接线 | PASS，144 parser、72 合成训练行 |
| LAA7 | AirfRANS/Car 接线 | PASS，6 通过、1 `torch_cluster` 跳过 |
| LAA8 | 两 profile 的 16 个正式入口、记录器和命令 | PASS，16 parser preview、25 shell |
| LAA9 | 参数/MAC/AMP/性能工具和回归 | 新范围 PASS；总回归 PARTIAL |

各阶段原始日志、失败尝试、source map、start/end freeze 和 JSON 证据保留在
`docs/loop_linearno_latent_adapter_audit/laa0` 至 `laa9`。LAA10 逆向矩阵
位于 `laa10/requirements-matrix.json`。

## LAA10 最终验证

- V3/LAA：按各套件记录的原生 `PYTHONPATH` 隔离执行，权威汇总为 116 项
  通过、1 项因缺少 `torch_cluster` 跳过。首次合并收集在 106 项通过后因
  LAA6/LAA7 的任务本地 `cdlno_entry` 同名导入发生 1 个调用环境错误；隔离
  复测 LAA6 4/4、LAA7 6/7 且 1 项依赖跳过。该错误不是模型失败，也没有
  被包装成一次全绿运行。
- V2 FFN：首次调用缺少测试目录 `PYTHONPATH`，23 项中 4 个收集错误；按
  原生命令复测 34/34 通过，91.368s unittest、99.95s wall。
- V1 loop：107 个方法中 102 通过、5 个方法失败，175.687s unittest。
  失败是旧文件/provenance/hash 与新进程零容差差异；没有 V3 assertion
  或新的模型计算失败。
- LAA8/LAA9 交付定向：9/9，35.581s unittest、39.81s wall。
- compileall、25 个 V3 shell 的 `bash -n`、全部 LAA JSON 解析和
  `git diff --check` 通过。
- pure/history LinearNO 全套执行 155 个测试方法：151 通过、2 个失败方法、
  2 个错误方法；后两者合计展开为 219 个 `FileNotFoundError` 子用例。
  unittest 用时 4310.054s，wall 4323.57s；原始输出保存为 LAA10
  `linearno-full-regression.log`，结构化结果保存为 `results.json`。没有改
  golden、容差或缺失文件清单。

旧 V1 loop 的 5 个失败方法是：大量已删除旧 docs/log 导致的 isolation
fixture、旧 launcher hash、两项新进程零容差差异（最大约 `1.49e-8` 和
`2.56e-9`）及旧 standard-entry provenance hash。pure/history 的 4 个非
通过方法由两个 source/provenance projection mismatch 和两个 legacy 文件
审计方法组成。第一个 legacy 方法因 218 个已删除历史文件报错，具体分为
30 个 `docs/CDLNO_*`、15 个 `docs/KCDNO_*` 和 173 个
`docs/kcdno_audit/*`；第二个因已删除的
`docs/kcdno_audit/audit_static.py` 报错。共计 219 个 error 子用例。这些
问题已在 LAA9 前存在，不表示 V3 forward 失败，但它们阻止最终全局 PASS。

## 冻结证据与自审

1. 完整 core 共享和特性时机：对象/parameter id、hooks、独立 oracle、
   state key 和 call schedule 均通过；无 round-specific FFN/cache。
2. 数学和精度：三 residual、六 attention variant、温度/softmax 轴、RB
   FP16/BF16、zero-init、两步梯度和 RNG 配对通过。
3. 配置/脚本：8×2×4 depth 表、三 residual、四消融共 768 行解析；16 个
   正式入口及 custom 冲突通过真实 parser preview。
4. checkpoint/output：三 wrapper 合成 forward/backward/AdamW、strict
   save→fresh-process resume/eval、V1/V2 pair replay 和 recorder 委托通过。
5. 冻结科学区：六个 exp 科学主体和工业 loss/metric/sampling/fold/drag
   文件没有被 V3 重写；LAA6 AST projection 与旧定向测试通过。未用真实
   loader 验证端到端数据行为。

逆向矩阵逐条给出源码、测试、证据和状态。没有发现需要改变冻结数学的
新冲突；当前未解决项均来自旧回归基线或未授权的真实环境验证。

## NOT RUN 和真实实验下一步

明确 **NOT RUN**：真实数据、完整训练、三 seed 收敛、SOTA、真实 epoch
时长、远端 Python 3.10/Torch 2.11/cu128、distributed、`torch.compile`、
真实 AirfRANS radius graph/VTK 全评估。缺少本机 `torch_cluster` 的单项
边界被跳过，没有安装替代依赖。

真实实验应先在远端目标栈对每个任务执行 `preview`，确认数据路径和 resolved
config；再用相同任务/profile/depth/residual/消融运行 seeds 0/1/2，保留
每个完整 strict archive 和独立 eval。只能汇总预先定义的三 seed，不按 test
结果挑 seed/checkpoint。真实性能需独立记录同步后的 epoch 时长、峰值显存和
任务指标，不能从本报告 MAC 或单个合成 smoke 推断。

本 LAA10 阶段结束，未执行真实实验。
