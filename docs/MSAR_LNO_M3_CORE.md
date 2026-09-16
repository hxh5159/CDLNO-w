# MSAR-LNO M3：四级 core 与模型级验证

2026-09-16。依据 [M0复用映射](MSAR_LNO_REFERENCE_AUDIT.md)、[M1配置](MSAR_LNO_M1_CONFIGURATION.md)、[M2原语](MSAR_LNO_M2_PRIMITIVES.md)及本轮明确公式，只完成M3。**未接八任务factory、数据、loss或生产训练脚本，未执行M4。**

## A. 交付与边界

新增稳定类路径 `cdlno.msar_lno.core.MSARLNO`。输入为**已经提升**的浮点 `E0[B,N,d]`，输出为 `prediction[B,N,output_dim]`。此选择沿用M0中共享core接收lifted features的约定；坐标/fx/reference/时间处理及输入MLP留给后续任务wrapper。本core包含最终LayerNorm+Linear逐点输出头，不应由未来wrapper再重复添加。

新增文件：

- [core.py](../cdlno/msar_lno/core.py)：`MSARLNO`、显式aux结果 `MSARAuxOutput`。
- [diagnostics.py](../cdlno/msar_lno/diagnostics.py)：独立no-grad的Down与fusion观测，无注册参数或状态。
- [msar_core_reference.py](../tests/msar_core_reference.py)：只组合M2 tensor-only数学reference，显式四级流程/LayerNorm，不调用生产forward。
- [test_msar_core.py](../tests/test_msar_core.py)：16项新模型级验证；辅助捕获/核查脚本及JSON/log在 `docs/msar_lno_audit/m3/`。
- 本报告、MSAR STATUS、旧总STATUS/memory增量记录。旧报告原文保留。

分支 `main`，HEAD `9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`，原有23个tracked修改及未跟踪工作全部保留。开始时720文件hash与源码快照：`/home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/source`。**M1/M2、旧共享模块、旧模型、factory、八任务、数据/训练/评估、依赖和已有测试均未修改**。没有commit/push/PR、依赖安装或真实训练。

## B. 公式、层次与参数归属

计算直接为：

```text
E0 = 输入的lifted features
对l=1..4：S_l = Down_l(E_(l-1)); E_l = EncoderStack_l(S_l)
D4 = DecoderStack4(E4)
对l=3,2,1：U_l = Up_l(E_l, D_(l+1))
            Z_l = Fusion_l(E_l, U_l)
            D_l = DecoderStack_l(Z_l)
U0 = final_up(E0, D1)
prediction = Linear(LayerNorm(U0))
```

| 对应公式 | 真实属性（0起始索引） | 参数/计算边界 |
|---|---|---|
| Down1—4 | `downs[0..3]` | M2 learned P直接Q；每尺度独立P/K/V/O及QK norm |
| Encoder1—4 | `encoders[0..3]` Sequential | 独立实例，depths=[3,1,1,1]，各尺度自己的heads |
| D4 | `decoders[3](encodings[3])` | 只有最深DecoderStack，无第四个fusion、额外Up、Bridge/history/processor |
| Up3→2→1 | `ups[2]`→`ups[1]`→`ups[0]` | Q恰来自同尺度E；KV来自更深D；receiver heads；M2纯分支，无内部query残差 |
| Fusion3→2→1 | `fusions[2]`→`fusions[1]`→`fusions[0]` | 每处仅w[d]；raw E/U，来源softmax，固定2，w0初始等于E+U |
| Decoder3→2→1 | `decoders[2]`→`decoders[1]`→`decoders[0]` | 在同尺度fusion之后，depths对应[1,1,3]，不交换顺序 |
| U0 | `final_up` | receiver heads=heads[0]；Q为原E0行，KV=D1；没有E0横向skip |
| task-independent head | `output_norm`、`output` | LayerNorm(eps1e-6)→Linear(d,output_dim)，无卷积/点域残差；output_dim显式正整数，默认1 |

所有模块在构造器中各自实例化，没有重复把同一个对象放进ModuleList，没有递归初始化。P正交、Linear trunc_normal/.02、norm scale1、fusion w0均由已验证原语保留。SA仅在12个latent block内，无N点SA。严格state_dict中没有reader/writer/CDPA/Bridge/persistent/Conv参数。`output_dim`由后续任务合同提供；M1的core架构配置仍不包含任务通道，后续元数据接入必须记录该合同，本轮未伪造八任务sidecar支持。

Light/Full精确采用原M1：M=[512,256,128,64]/[1024,512,256,128]、d96/192；heads=[4,4,8,8]；encoder/decoder depths均[3,1,1,1]。不依N裁剪M。实际模块数：**Down4、Up4、encoder blocks6、decoder blocks6、latent SA12、latent FFN24、fusion3；旧history/CDPA/Bridge/Conv=0**。底层Linear125、RMSNorm76、输出LayerNorm1；不是把所有cross当latent SA计数。

实测参数（含4通道输出头，**不含未来任务input lift**）：

| 参数归属 | Light | Full |
|---|---:|---:|
| 4 Down | 203,280 | 812,064 |
| 6 encoder latent blocks | 669,552 | 2,666,208 |
| 6 decoder latent blocks | 669,552 | 2,666,208 |
| 3尺度 Up | 111,000 | 443,184 |
| 3 fusion w | 288 | 576 |
| final Up | 37,008 | 147,744 |
| 输出LayerNorm+Linear | 580 | 1,156 |
| **总数** | **1,691,260** | **6,737,140** |

来自实际构造的 `named_parameters`，不是论文估计或删除分支的局部成本。coverage和diagnostics无可训练参数。各任务通道/lift尚未接入，不能将本表当作八任务最终参数表或推理提速证据。

## C. 实际 shape trace

完整hook事件、输入输出shape、参数key/shape在 [core-capture.json](msar_lno_audit/m3/core-capture.json)。以下是实际合成输入trace的空间维；全程保持batch和d不变：

| 事件 | Light B2/d96/N35 | Full B1/d192/N7 |
|---|---:|---:|
| E0 | 35 | 7 |
| Down1→Encoder1 | 512→512 | 1024→1024 |
| Down2→Encoder2 | 256→256 | 512→512 |
| Down3→Encoder3 | 128→128 | 256→256 |
| Down4→Encoder4→Decoder4 | 64→64→64 | 128→128→128 |
| Up3→Fusion3→Decoder3 | 128→128→128 | 256→256→256 |
| Up2→Fusion2→Decoder2 | 256→256→256 | 512→512→512 |
| Up1→Fusion1→Decoder1 | 512→512→512 | 1024→1024→1024 |
| final Up→输出head | `[2,35,96]`→`[2,35,4]` | `[1,7,192]`→`[1,7,4]` |

Full中1024>7是真实latent扩张，未把正式M缩到N。16项测试另用d8/M=[7,5,3,2]/heads=[2,2,4,4]缩小数值检查，**保留完整12块拓扑**。N3/9/11/35、B1/B2和非方形点数都覆盖，未用sqrt(N)或网格卷积。mask只作用于第一个Down的原source；后面latent全有效。mask不是输出mask：final query仍为N行E0，调用方按其任务合同解释无效输出行，不将图batch混成一个场。

## D. coverage、diagnostics与forward合同

默认 `model(e0)` 始终返回预测Tensor；无论train/eval或配置diagnostics=True，未显式请求aux就不计算loss/诊断。训练需要 `model(e0, return_aux=True)`，返回 `MSARAuxOutput`：

| 字段 | 内容 |
|---|---|
| `prediction` | 原预测Tensor `[B,N,output_dim]` |
| `coverage_per_level` | `[4]` raw FP32 loss，顺序scale1/2/3/4；未乘coverage_weight |
| `coverage_mean` | 四层raw loss均值，保留训练梯度 |
| `coverage_active` | 本次是否training+显式aux+coverage有效开启 |
| `diagnostics` | 未请求时None；请求时仅每样本小型no-grad统计，无A/E/D原张量 |

每次forward可通过 `training_config=MSARTrainingConfig(...)` 显式覆盖本次训练目标；不会修改model保存的配置或权重。floor且weight>0时只从四次Encoder Down收集A→raw coverage；decoder/fusion/final不加辅助损失。source求和、batch均值在M2完成，然后层均值。仅第一个Down可接受原输入 `source_measure`；其余latent均匀mu，未推导面积或新增数据字段。

off、weight0、普通forward或纯eval均不向Down请求aux A、不调用CoverageFloorLoss；显式aux返回4个无图FP32零及零均值。caller后续只应为新family把`weight * coverage_mean`加到原PDE loss，本轮没有训练器接线。κ0保持M2语义：raw loss零，但不是另一个off结构开关。同权重off/floor的预测与预测梯度已作backend容差对照，不把aux损失梯度混入该比较。

`training_config.diagnostics=True`且**同时**`return_aux=True`时，按需返回：

- `diagnostics['encoder']`四份每样本coverage_ratio/diversity_ratio `[B]`；有训练A时直接no-grad复用，没有时`observe_down`单独no-grad重算QK权重，**不修改原预测Down的SDPA调用或训练态**。
- `diagnostics['fusion']`三份每样本`alpha_mean[B,2]`（E/U来源顺序）及`entropy_mean[B]`，对token取均值，自然对数；条目按scale1/2/3排列。w0均为[.5,.5]，entropy=ln2；非零w也独立公式验证。

因此coverage off的**默认路径**无aux A；显式diagnostics仍有观察权重/特征值分解成本，但没有aux梯度图，不声称“开启诊断也没有矩阵开销”。这是本轮第8条的独立观察路径，没有放宽M2生产Down返回A的条件。输出预测与RNG在开/关诊断时精确一致。统计全在原device，不做默认CPU记录，不跨batch存储。显式有效mask的全空校验沿用M2布尔检查；本轮没有增加常规无mask路径的CPU同步，CPU/item拦截测试通过。

AMP中输入E0可能仍是FP32、decoder为低精度：final query在进入M2 Up前转为decoder dtype，等价于投影输入随autocast使用该精度，无新可学习变换或残差。默认同dtype时是原对象；M2严格Q/KV同dtype合同保留。FP32/double/AMP的原norm与coverage规则不变。

## E. 验证与实际命令

- **16/16 M3测试，7.110s**：独立四级公式前向/解析梯度；模块/执行顺序/shape/对象引用；无N点SA；所有stage有限梯度、一次合成MSE+coverage AdamW step；coverage只连接encoder Down依赖；off/weight0无aux图；诊断no-grad/RNG/预测不变；无跨调用状态；置零final Up后严格没有E0 bypass；非法输入/配置拒绝。
- **20/20 M2原语，1.946s；13/13 M1配置，0.344s**。共49测试，无失败/错误/skip。M1 torch-free配置导入继续通过；没有修改旧测试逻辑。
- **真实Light/Full构造与小N前向、Full严格state_dict往返通过**。新增两份post-M3同权重固定输入夹具，capture后在新进程replay，eval预测、training态floor预测及raw coverage均atol=rtol=0；不含训练权重或真实数据。
- 小模型state_dict缺key/错误output_dim明确拒绝。真实保存本地整对象/对象列表后，从三个原子项目cwd的新进程加载，稳定类路径与预测精确一致。只加载本测试刚写入的可信临时对象，未扩大任何生产`torch.load`边界；不声称工业完整训练checkpoint接入完成。
- **41个K0 Transolver/CDLNO +6个pre-M1 KCDNO/matched核心，本轮47/47精确回放**。其余28个旧wrapper/其他模型复用M2有效证据及未变源码；没有把它们写成本轮重新运行。M0索引182个旧夹具文件hash均核对不变。
- 本机Python3.13.9/torch2.13+cu130/CUDA13.0/RTX5090 Laptop/PyG2.3.1。M3小core在CUDA MATH SDPA下FP32、FP16 AMP、BF16 AMP均完成有限forward/backward，off/floor预测对照通过；输出dtype与coverage FP32正确。真实Light/Full本轮的配置/前向/夹具检查为CPU，不当作大模型GPU验收。

容差：独立core FP32预测atol5e-6/rtol2e-4，梯度1e-4/2e-3；double预测3e-10/3e-8，梯度6e-9/3e-7；raw FP32 coverage对double reference 2e-7/2e-4。double同权重off/floor采用同double容差。GPU FP32预测3e-6/2e-4，FP16/BF16 AMP预测3e-3/.03；不是低精度逐值相同的承诺。w0融合、诊断开关、同设备夹具及严格保存加载采用零容差。

```bash
python -B -m unittest discover -s tests -p 'test_msar_core.py' -v
python -B -m unittest discover -s tests -p 'test_msar_modules.py' -v
python -B -m unittest discover -s tests -p 'test_msar_config.py' -v
python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/msar_lno_audit/m3/k0-replay.json
python -B docs/msar_lno_audit/m1/replay_current.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures \
  --result docs/msar_lno_audit/m3/current-core-replay.json
python -B docs/msar_lno_audit/m3/core_fixtures.py capture \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/post-m3-core \
  --result docs/msar_lno_audit/m3/core-capture.json
python -B docs/msar_lno_audit/m3/core_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m3-before-jx4v53vh/post-m3-core \
  --result docs/msar_lno_audit/m3/core-replay.json
python -B docs/msar_lno_audit/m3/verify_delivery.py
```

capture拒绝已有夹具，后续用replay，不覆盖本次参考。远端前三条可直接在现有环境执行；replay需迁移真实夹具并替换路径。新二进制约34MB仅放仓库外，不自动提交。证据：[summary](msar_lno_audit/m3/summary.json)、[freeze](msar_lno_audit/m3/freeze.json)、[fixture-index](msar_lno_audit/m3/fixture-index.json)、[测试日志](msar_lno_audit/m3/core-tests.log)。

交付检查脚本初次把表中代码`decoders[3](encodings[3])`误识别为Markdown链接；仅修正新增审计脚本的链接解析，排除代码后复核。模型/配置测试没有出现失败；该问题不涉及生产计算。

## F. 自审与剩余依赖

五项自审已对应源码/实际测试：①四级profile/shape/参数独立和计数；②deepest及Up→fusion→Decoder顺序、final无E0skip；③只有Encoder Down收集loss、off/eval无aux图；④诊断显式且无梯度/跨batch状态，不改变预测/RNG；⑤稳定新类路径、严格保存加载、旧源码/夹具冻结。没有发现需要用户改变确认公式的阻断冲突。

尚未完成且不越阶段实施：八任务真实lift/head通道合同及metadata/factory/loss接入、工业训练保存流程、真实数据读取/训练/收敛/准确率、远端Python3.10/torch2.11/cu128、完整Light/Full实际N训练和其他backend性能。没有拿合成MSE称为原任务loss，没有以GPU小模型通过代替远端或真实数据验收。

本M阶段结束，未执行下一阶段。
