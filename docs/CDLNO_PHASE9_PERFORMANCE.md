# CDLNO 阶段9：无数据性能工具与 matched LRSA

日期：2026-09-14。阶段0–8已审查通过；本阶段只执行用户授权的 v1.2 §9。工具用法、成本口径和远端命令见 [CDLNO_PERFORMANCE_TOOLS.md](CDLNO_PERFORMANCE_TOOLS.md)。本报告的性能数据仅来自合成张量，不构成真实训练或LRSA论文复现。

## A. 完成范围

实现独立性能工具，覆盖 `transolver`、`lrsa_matched`、`cdlno_off`、`cdlno_entry`、`cdlno_every_block`，支持八任务合成接口、明确分开的task/matched配置、chunk0/1/正整数、参数/完整矩阵MAC/非矩阵算子及存储账、同步计时和CUDA峰值显存。每个CLI只运行一个明确配置，不自动扫描任务/深度/网格。

`LRSAMatched`仅在benchmark内：沿用现有任务wrapper和stem，串联L个独立完整 `LRSAFrontBlock`，末端仅原CDLNO输出LN/head。没有bridge/CDPA/final-up；没有以F=L绕过CDLNO的P≥1约束。不是LRSA训练入口或论文实验复现。

## B. 文件及diff

| 新增文件 | 理由 |
|---|---|
| `tools/cdlno_benchmark.py` | 一个配置的CLI、结果隔离、共同精度设置、同权重chunk比较、失败记录 |
| `tools/cdlno_perf/models.py` | 薄matched组合、八任务preset/合成输入、原Transolver源码隔离加载 |
| `tools/cdlno_perf/costs.py` | 实际Linear/Conv/SDPA形状计MAC、原slice/deslice、norm/depth/临时内存/保存激活账 |
| `tools/cdlno_perf/measure.py` | 同步median/p90、初始化AdamW state、实际backend及未计时诊断；可选共同AMP/TF32/compile |
| `tools/cdlno_perf/__init__.py` | benchmark辅助模块命名空间，不修改生产包导出 |
| `tests/test_performance.py` | 11项结构、闭式MAC、来源/call、权重/梯度、独立性、原配置和计时协议测试 |
| 本报告、使用文档、`docs/performance/phase9/` | 持久化实际结果、原始样本、测试日志及冻结证据 |

既有文件只更新 `AGENTS.md`、`docs/CDLNO_IMPLEMENTATION_STATUS.md` 和 `memory/current-state.md` 三份状态文档。没有修改 `cdlno/`、三个任务子项目、旧模型/脚本、依赖或训练注册项。

原Transolver用实际模型源码及真实依赖运行。benchmark独立namespace只解决标准项目的 `model.Embedding/Physics_Attention` import，所有无参 `.cuda()` 在内存AST中改为 `.to(指定device)`，并记录源文件SHA256和修改行。未改动原文件、数学或初始化；没有把原手写attention改成SDPA。原AirfRANS的闲置 `mlp_new` 照实计参数、列missing-grad，未删除。

## C. 公式、张量与代码核对

```text
任务输入 → 原wrapper stem → H0[B,N,d]
lrsa_matched: L×完整LRSA(H) → HL[B,N,d] → LN/head → [B,N,Cout]
CDLNO: F×完整LRSA → HF与Ti[B,M,d] → bridge/raw Z0
       → 原off/entry/every调度 → ZP[B,M,d]
       → 原readout(HF,ZP) → [B,N,Cout]
```

以下由真实hook/SDPA调用验证，不只按构造参数推算：

| 默认L8/F2/P6、front=full，结构化任务 | down/bridge | up/readout | latent SA | dense ConvFFN | 逻辑历史S | 历史SDPA chunk0/1/2 |
|---|---:|---:|---:|---:|---:|---|
| lrsa_matched | 8 | 8 | 8 | 8 | 0 | 0 / 0 / 0 |
| off | 3 | 3 | 8 | 3 | 0 | 0 / 0 / 0 |
| entry | 3 | 3 | 8 | 3 | 2 | 1 / 2 / 1 |
| every_block | 3 | 3 | 8 | 3 | 27 | 6 / 27 / 15 |

CDLNO非历史SDPA共14次，entry/chunk0整次forward是15次、every/chunk0为20次。一次API调用不是一个GPU kernel。原Transolver的手写Physics Attention有非零slice/deslice/QK/AV成本，SDPA API调用数0不能解释为无attention。

`matrix_macs`完整计入stem/output、N规模Q/K/V/O、两次前段latent FFN、每次点FFN及dense3×3卷积、后段GEGLU、CDPA一次Q及每来源K/V/O、所有attention矩阵乘法。独立闭式参考和实际记录逐项相等：

```text
matched attention = 2Bd(2LNM+LM²)
CDLNO attention = 2Bd(2(F+1)NM+(L+S)M²)
CDPA projection = BMd²(A+3S)
Sentry=F；Severy=PF+P(P−1)/2
```

1 MAC记2 FLOPs；这项只命名为matrix FLOPs。bias、norm/QK norm、GELU/GEGLU乘法、depth、softmax/reduction、reference/time计算和复制以单独实际算子/元素及payload账说明，**不声称得到精确全指令FLOPs**。融合SDPA内部softmax按实际Q/K恢复逻辑元素数；通用profiler FLOPs单列partial，不能拿缺项数当整网成本。

来源stack、Q expand/reshape可能产生的clone、dtype副本和FP32候选/keys/weights/weighted-values全部进入内存账。H_F payload为BNd×元素bytes，every最后历史列表有F+P−1份，加当前identity为F+P份；训练还有完整保存激活图。去重storage账和临时payload不是同时存活峰值，不能相加冒充实测CUDA allocated峰值。

## D. 实际验证与性能结果

完整回归：

```bash
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator \
  python -B -m unittest discover -s tests -p 'test_*.py' -v
```

**118/118通过，0失败/错误/跳过，59.769秒。** [原始回归日志](performance/phase9/regression.log)。其中本阶段11项，包含八任务真实wrapper/原模型合成调用（工业为真实PyG Data）、完整matched结构及同权重stem/head、无共享参数/storage、非方形5×7、MAC闭式与实际调用、chunk0/1/2/大于S、扩展L/F、非法输入、CPU计时与state初始化、新进程CLI/不覆盖输出、Python3.10 AST语法。

本地环境：Python3.13.9、torch2.13.0+cu130、CUDA build13.0、PyG2.3.1、RTX5090 Laptop 24GB、driver591.86。该环境不同于用户远端Python3.10/torch2.11/cu128，不作为远端兼容验收。

正式结果统一FP32输入/参数，无AMP、TF32关闭、compile关闭、SDPA auto（实际算子另记录）、cuDNN benchmark关闭、CPU线程1、5次warmup/20次正式样本。GPU逐样本前后同步，median/p90都是毫秒。合成训练步为一次模型调用+FP32 MSE+backward+AdamW(lr=.001,decay=1e-5,foreach=False)，测量前state已初始化；峰值包含相应模型/输入、激活、梯度及AdamW state，无数据I/O、scheduler或真实时间rollout。

### 原任务配置比较：Airfoil

B4，N=221×51=11271，d128/L8；原Transolver h8/M64，另外四种h4/M64；新模型F2。[完整JSON](performance/phase9/airfoil-task.json)。

| 模型 | 参数量 | 矩阵G MAC | forward median / p90 ms | 训练步 median / p90 ms | 峰值MiB forward / 训练 |
|---|---:|---:|---|---|---|
| transolver | 3,073,985 | 146.329 | 41.436 / 50.384 | 135.965 / 138.911 | 362.43 / 3769.28 |
| lrsa_matched | 4,315,009 | 114.319 | 35.977 / 36.995 | 187.903 / 190.992 | 323.40 / 3891.86 |
| cdlno_off | 2,449,089 | 44.037 | 14.563 / 15.014 | 77.473 / 78.699 | 316.27 / 1619.86 |
| cdlno_entry | 2,515,521 | 44.075 | 14.468 / 14.653 | 77.641 / 78.851 | 316.64 / 1623.52 |
| cdlno_every_block | 2,847,681 | 44.516 | 15.205 / 15.773 | 84.716 / 86.271 | 317.91 / 1658.78 |


### 同配置结构比较：Elasticity

B1，N972，d128/h8/M64/L8；新模型F2，point FFN。[完整JSON](performance/phase9/elasticity-matched.json)。

| 模型 | 参数量 | 矩阵G MAC | forward median / p90 ms | 训练步 median / p90 ms | 峰值MiB forward / 训练 |
|---|---:|---:|---|---|---|
| transolver | 976,833 | 1.127 | 4.393 / 5.568 | 24.300 / 25.817 | 73.44 / 156.84 |
| lrsa_matched | 3,133,569 | 1.441 | 10.012 / 11.735 | 56.801 / 72.756 | 80.79 / 181.37 |
| cdlno_off | 2,006,113 | 0.617 | 4.484 / 8.271 | 36.490 / 55.706 | 76.48 / 121.43 |
| cdlno_entry | 2,072,545 | 0.627 | 4.737 / 5.826 | 34.961 / 42.853 | 76.76 / 122.88 |
| cdlno_every_block | 2,404,705 | 0.737 | 6.351 / 16.567 | 48.750 / 68.669 | 78.03 / 133.95 |


### 同权重chunk比较：Elasticity

每行同一模型恢复相同state_dict/hash和输入；各chunk矩阵MAC相同，输出及所有参数梯度比较通过。表中逻辑来源数不能和调用次数混淆。

| 模型 | chunk | S / 历史SDPA次数 | forward median / p90 ms | 训练步 median / p90 ms | 训练峰值MiB |
|---|---:|---|---|---|---:|
| cdlno_entry | 0 | 2 / 1 | 4.737 / 5.826 | 34.961 / 42.853 | 122.88 |
| cdlno_entry | 1 | 2 / 2 | 5.227 / 14.020 | 32.242 / 42.979 | 122.88 |
| cdlno_entry | 2 | 2 / 1 | 4.582 / 4.978 | 41.207 / 53.887 | 122.88 |
| cdlno_every_block | 0 | 27 / 6 | 6.351 / 16.567 | 48.750 / 68.669 | 133.95 |
| cdlno_every_block | 1 | 27 / 27 | 8.999 / 10.453 | 60.218 / 81.080 | 133.96 |
| cdlno_every_block | 2 | 27 / 15 | 7.905 / 8.531 | 58.774 / 72.290 | 133.95 |


### 同配置结构比较：Airfoil缩小网格

B2，N=17×23=391，所有模型d128/h4/M64/L8，新模型F2；这是缩小合成网格，不是原Airfoil规模。[完整JSON](performance/phase9/airfoil-matched.json)。

| 模型 | 参数量 | 矩阵G MAC | forward median / p90 ms | 训练步 median / p90 ms | 峰值MiB forward / 训练 |
|---|---:|---:|---|---|---|
| transolver | 3,100,577 | 2.567 | 7.344 / 8.666 | 27.424 / 30.074 | 97.75 / 167.08 |
| lrsa_matched | 4,315,009 | 2.242 | 9.882 / 10.478 | 64.425 / 76.155 | 84.81 / 199.44 |
| cdlno_off | 2,449,089 | 0.970 | 5.261 / 6.380 | 33.280 / 41.058 | 77.68 / 132.67 |
| cdlno_entry | 2,515,521 | 0.989 | 5.285 / 6.126 | 33.007 / 37.787 | 78.00 / 134.87 |
| cdlno_every_block | 2,847,681 | 1.209 | 7.484 / 8.599 | 43.016 / 47.286 | 80.02 / 154.41 |

最终GPU命令（顺序执行，无并发测试）：

```bash
python -B tools/cdlno_benchmark.py --task elasticity --device cuda:0 \
  --comparison matched --warmup 5 --iterations 20 \
  --output docs/performance/phase9/elasticity-matched.json
python -B tools/cdlno_benchmark.py --task airfoil --device cuda:0 \
  --comparison task --chunks 0 --warmup 5 --iterations 20 \
  --output docs/performance/phase9/airfoil-task.json
python -B tools/cdlno_benchmark.py --task airfoil --device cuda:0 \
  --comparison matched --grid 17 23 --B 2 --warmup 5 --iterations 20 \
  --output docs/performance/phase9/airfoil-matched.json
```

上述27行模型/chunk结果全部通过；同权重chunk最大输出绝对差`1.788139e-7`、最大参数梯度绝对差`1.430511e-6`。这只是三个明确配置组（两个任务），没有完整网格矩阵/超参扫描。各行均完成前反向、有限值、参数使用检查；同一模式的chunk使用完全相同初始权重hash，训练前输出及参数梯度比较通过。测量期间每个活跃参数都执行25次AdamW更新（warmup5+正式20），另有明确分开的诊断step。实际新模型SDPA自动选择`aten::_scaled_dot_product_efficient_attention`；未将auto称为flash。JSON包含每行20个原始forward和训练样本、参数/state bytes、前后峰值、dtype、norm/depth/复制/存储清单、实际调用与工具/核心源码hash。


本阶段前半段也运行了CPU11项定向测试和本地GPU初测，工具输出曾显示不同运行间明显延迟波动。用户续接后原 `/tmp` 日志/结果不可用，故这些初测不作为最终表格或验收依据；未完成/无法找回结果的运行不算通过。最终结果直接归档在项目文档目录；未计时诊断与正式计时分离。不使用初测中的某个更快数字替换正式记录，也不从forward推导真实epoch时间。

未运行：远端目标栈、真实数据/轨迹训练、实际图构造/采样/指标/epoch性能、八任务完整大网格矩阵和超参数扫描。AMP/TF32开启、强制其它SDPA backend和torch.compile路径在本阶段没有实测；工具统一转发这些可选设置、分离编译首调用与稳态并具备eager核对，但CLI支持不等于这些组合已验收。

## E. 冻结区段证据

阶段开始基线是main/75e0f67643806a81cd1d3f6adc88dd8c02416fe7，已有阶段0–8未提交改动原样保留。最初145文件hash检查在状态文档更新前显示既有文件无变化；该 `/tmp` manifest在续接时不可用，不伪造原manifest。续接时新建持久化 [152文件manifest](performance/phase9/continuation-baseline.json)，SHA256 `288568f43d659168d9acf52e5523931ea01ef3a8191ceb0a4bd7a4b03400699d`，用于最终文件核对。最终[冻结检查](performance/phase9/freeze-check.json)确认：152个既有文件中仅AGENTS/STATUS/memory三份状态文档变化，**149文件完全相同，意外改动0**。

本阶段从未编辑三个原任务目录或cdlno数学/接口/config/checkpoint，也未变更pyproject/requirements。118项回归再次通过前阶段的整模块AST投影、原文件字节比较、原损失/时间循环和加载检查。没有import顶层读数据的exp/main，没有假数据文件、下载、依赖替换、真实训练、commit/push/PR。

## F. 已完成自审及剩余边界

1. matched LRSA的完整两次latent FFN、独立down/up、每层point FFN/ConvFFN以及末层只接LN/head：源码与直接手工顺序/参数storage/梯度测试通过。
2. 全MAC和临时内存账：实际Linear/Conv/SDPA形状对独立闭式逐项通过，原Transolver按实际每头投影与手写attention计数；softmax/norm/复制没有冒充零开销或精确总FLOPs。
3. 来源数与调用数、chunk同权重和图：测试及合成运行检查通过；未修改CDPA公式、历史调度、参数独立性或H_F decoder，没有detach历史/跨层KV缓存。
4. 计时公平条件与加载状态：统一精度、同步、warmup/quantile、同权重初态和新AdamW，通过实际成功step计数验证state初始化；诊断不进入正式计时，已有JSON拒绝覆盖。
5. 既有区域与研究范围：生产包、任务数据/训练入口及依赖未改；对照只在tools内，未接真实训练。

当前授权范围内未发现待修复的架构或接口问题。实际效率边界如下：

- Elasticity下entry矩阵MAC约为原Transolver的55.6%，但训练步中位数34.96ms高于原模型24.30ms；相对matched LRSA的56.80ms仍更快。这个结果不能写成“减少MAC必然加速”。
- 未计时训练诊断中，原Transolver/entry/matched LRSA分别有169/200/321个参与AdamW更新的参数tensor，7234/8218/14186次ATen调用；cudaLaunchKernel调用为1893/2504/4781。前段完整RMS/QK norm、两次latent FFN、后段GEGLU、CDPA和逐tensor AdamW产生更多小算子和发射开销，是小规模差异的可见来源。profiler自身有开销，这些事件数不是严格的因果耗时分解；未独立归因的部分仍保留不确定性。
- Airfoil原任务配置下entry的44.075G MAC中dense Conv约19.944G、点FFN线性约8.864G、前段/bridge/readout投影约8.939G，不能只看4.474G的attention矩阵项。此次entry训练步约77.64ms，低于原Transolver135.96ms和matched LRSA187.90ms；原Transolver heads不同，结论限于各自任务preset的这次合成运行。
- chunk0没有减少数学计算量；every中相对chunk1减少的主要是Python/SDPA调用组织。entry只有两份历史，0与2组织相同，但实测也有波动，不能据单轮微小差异宣布最优chunk。显存payload和实测峰值分别保留，不把来源batch折叠解释为免费存储。
- 目标机器、其他精度/backend/compile和真实任务耗时仍未验证。工具未调大warmup或自动扩大扫描来追求某个排名，没有改公式、decoder、缓存策略或生产优化器。用户下一步只需审查本阶段交付；这些限制不授权继续下一阶段。


本阶段结束，未执行下一阶段。
