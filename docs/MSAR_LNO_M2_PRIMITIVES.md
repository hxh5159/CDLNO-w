# MSAR-LNO M2：独立数学原语

2026-09-16。只完成 M2，依据已接受的 [M0 实际映射](MSAR_LNO_REFERENCE_AUDIT.md)、[M1 配置](MSAR_LNO_M1_CONFIGURATION.md)和本轮明确公式。**未组装四级 core，未注册任务模型或接入训练入口。** M0 补审与 M1 的实际时间顺序保持原记录。

## A. 范围与源码边界

开始时分支 `main`，HEAD `9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`，已有23个tracked修改及未跟踪工作保持。先核对 AGENTS、M0/M1、MSAR staged prompts、公共原语和现有测试，保存695个既有文件的hash及源码快照：

```text
/home/hwz/CDLNO-artifacts/msar-m2-before-bovueh8k/source
```

新增生产文件仅 [cdlno/msar_lno/modules.py](../cdlno/msar_lno/modules.py)。复用 `cdlno.modules.RMSNorm/PlainFFN/_SelfAttention/_ProjectedAttention/_init_linear`，**旧共享文件保持字节不变**。新 Down 自行实现可选权重及 mask；没有修改旧 Down 的调用、参数名或数值行为。新类各自构造参数，没有复用同一个模块实例。

新增 [独立reference](../tests/msar_reference.py)、[20项针对性测试](../tests/test_msar_modules.py)、本报告及 [m2证据](msar_lno_audit/m2/summary.json)。MSAR STATUS、旧总STATUS和memory仅增量记录。M1配置、包根torch-free导出、factory、任务wrapper、数据、loss、训练/评估、依赖及已有测试均未修改。无安装、下载、commit/push、PR或真实训练。

## B. 公式 → 类/方法 → 独立证据

| 公式/语义 | 生产符号与轴 | reference / 定向测试 |
|---|---|---|
| learned query cross：`Q=P`，`K=X Wk`，`V=X Wv`；标准head尺度，读出后O | `LearnedQueryDown.forward`；P参数`[M,d]`，Q/K per-head RMS；A`[B,H,M,N]`，输出`[B,M,d]`；**无Wq、P残差或隐藏source pre-norm** | `oracle.down/explicit_heads`按batch/head逐份矩阵公式；`test_down_explicit_oracle_outputs_masks_and_gradients`、`test_down_matches_accepted_old_primitive_same_weights` |
| `X1=X+FFN1(N1(X)); X2=X1+SA(Nsa(X1)); Y=X2+FFN2(N2(X2))` | `LatentFFNSAFFNBlock.forward`；两个独立d→2d→d GELU，3个独立pre-RMS；SA为原标准多头原语 | `oracle.latent/ffn`使用erf GELU、显式attention；`test_latent_three_residual_formula_and_calls`验证六模块各一次及零输出投影后的恒等残差 |
| Up：Q来自当前E，KV来自更深D | `QueryAlignedUpCross.forward`；独立Q/K/V/O、receiver heads；纯`[B,Mcurrent,d]`读取分支，无query残差/额外pre-norm/FFN | `oracle.up`；`test_up_formula_permutation_branch_and_batch_isolation`，O置零输出严格零、query同排列、batch隔离 |
| `sE=w·RMS(E); sU=w·RMS(U); Z=2*(αE E+αU U)` | `PairwiseAttnResFusion.forward`；来源轴`[B,M,2]`，只注册`w[d]`，无norm scale、bias、gate、QKV、sqrt(d)或外层残差；raw values | `oracle.fusion`用二源logistic差值而非相同stack实现；`test_fusion_zero_initialization_jacobian_and_learnable_w`、`test_fusion_raw_values_slot_alignment_and_oracle` |
| `p=mean_head mean_latent A`，`sum_j relu(κ μ_j−p_j)^2/(μ_j+eps)` | `CoverageFloorLoss.forward`；有效source **sum**，batch **mean**；`mean_layers`再对层mean；返回raw FP32标量，无参数 | `oracle.coverage`逐样本先选有效列再手算；`test_coverage_hand_values_and_layer_reduction`、`test_coverage_reference_gradient_mask_and_measure` |
| coverage仅经A反传到query/key | Down与coverage的独立loss链，不使用预测loss | `test_isolated_coverage_gradient_to_learned_q_and_key_projection`：P、K投影、Q/K norm和输入有有限有效梯度；V/O梯度为None，符合纯coverage依赖 |
| 可选覆盖/多样性诊断，不作为loss | `coverage_diagnostics`，`@torch.no_grad`，返回device上的每样本指标；无模块缓存/默认调用 | `test_diagnostics_no_grad_and_not_automatic`；均匀相同行的CoverageRatio=1、DiversityRatio=1/M，单位矩阵的DiversityRatio=1 |

Down没有额外Q投影，因此“coverage梯度到Q”具体指其直接可学习P，而不是新增Wq。初始化沿用已验证规则：Linears trunc_normal(std=.02)，bias=0；Q/K/V无bias、O有bias；RMS scale=1、eps=1e-6；P最后单独orthogonal，允许M>d（列正交）。没有递归apply重新覆盖初始化。Fusion仅w严格为0，初始输出E+U、对E/U的直接Jacobian严格为1，w有非零有限梯度。不同层/实例storage独立。

**以下是实际构造的单个组件参数数，不是完整MSAR参数或性能：**

| 组件 | Light第1尺度：d96/h4/M512 | Full第1尺度：d192/h4/M1024 |
|---|---:|---:|
| LearnedQueryDown | 76,944 | 307,488 |
| LatentFFNSAFFNBlock | 111,600 | 444,384 |
| QueryAlignedUpCross | 37,008 | 147,744 |
| PairwiseAttnResFusion | 96 | 192 |
| CoverageFloorLoss | 0 | 0 |

实际state keys/shapes在 [component-parameters.json](msar_lno_audit/m2/component-parameters.json)。每个fusion只有d个参数；未来三处为3d，不能把旧affine RMSNorm实例化进去再声称只有w。

## C. 调用、mask、损失与dtype

独立原语使用示例（小型合成张量，不是训练命令）：

```python
import torch
from cdlno.msar_lno.config import MSARTrainingConfig
from cdlno.msar_lno.modules import LearnedQueryDown, CoverageFloorLoss

cfg = MSARTrainingConfig()  # floor / weight=.01 / kappa=.2 / eps=1e-6
down = LearnedQueryDown(dim=8, heads=2, num_latents=5).train()
x = torch.randn(2, 7, 8)
valid = torch.ones(2, 7, dtype=torch.bool)
features, attention = down(x, valid_mask=valid, return_aux=True, coverage=cfg)
raw_coverage = CoverageFloorLoss(cfg)(attention, valid_mask=valid)
weighted_coverage = cfg.coverage_weight * raw_coverage

down.eval()
features_only = down(x, valid_mask=valid)  # Tensor，没有aux权重
```

- 默认返回Tensor。只有`self.training && return_aux && coverage.coverage_enabled`才显式计算A并返回`(output,A)`。`return_aux=True`但条件不满足时返回`(output,None)`。传`coverage=None`表示此调用未请求coverage；后续core需显式传M1训练配置。off、weight0、eval或未请求aux均走SDPA，测试拦截显式softmax确认未进入aux路径。SDPA内部是否物化矩阵由PyTorch所选backend决定；这里承诺不额外构造/保留aux A，不宣称所有backend都无内部attention矩阵。
- mask必须为同设备bool`[B,Nsource]`、每个样本至少一个有效source，禁止batch广播。无mask时无额外host归约；显式mask的全空拒绝会作一次布尔检查。padding在投影前置零、softmax前置`-inf`，A对应列严格0；invalid输入梯度为0。不会把surf误当valid mask，不新增工业多图方式。
- coverage默认mu在各样本有效source上均匀；B2不同有效点数独立归一。可显式传`source_measure[B,N]`，仅表示外部可靠提供的非负测度，不从坐标/标签估计；有效项必须有限且有正总质量。先用最大值缩放再归一保证大有限权重之和不溢出；数学测度不变。无数据格式改动。
- mask列不进入mu和loss；即使独立loss测试给padding填NaN，也先屏蔽再聚合，前向与梯度有限。source项**不再除N**。每层先batch平均，再对有效层平均，不能把不同N的层按source数量加权混合。weight只用于off分支判断；raw loss不乘weight，调用者将来只乘一次。
- `CoverageFloorLoss(off/weight0)(None)`返回无图CPU FP32零；传Tensor时零在其device。此处是原语测试合同，后续task的off训练分支应直接保留原PDE loss，不额外调用loss或构造图。κ0 loss为0，但按当前约定不会把κ0自动改成coverage_mode=off。
- aux scores/softmax/AV禁用autocast，float16/bfloat16/float32输入用FP32，double保留double；attention为工作dtype，AV后转回投影V dtype。Fusion评分/softmax/融合同样低精度升FP32、double保留，最后转回输入dtype。**Coverage无论输入dtype均强制FP32并关闭autocast**。
- independent reference不调用任何待测forward/共享模块/SDPA；保留double。Coverage double gradcheck只对reference做，不拿生产float强转路径冒充double。生产/参考的解析梯度另行比较。所有权重/attention只作为本次局部变量/显式返回，不在module上长期保存，没有原地修改输入。
- `coverage_diagnostics(A)`仅显式调用：CoverageRatio=`exp(H(p))/Nvalid`；DiversityRatio=`erank(Abar Abarᵀ)/M`，以归一化非负Gram特征值定义erank。没有默认CPU统计、长期attention图或熵正则；没有specialization/MI、严格JS/L2、Sinkhorn、CoTAP、decoder balance。

## D. 实际验证、命令与边界

[测试日志](msar_lno_audit/m2/modules-tests.log)：**20/20，1.940s，无失败/错误/skip**。覆盖B1/B2、N3/5/7/35、M大于N、d8/12及double gradcheck d4、h1/2/3/4、非方形点数、不同mask与measure、残差/branch、slot对应、batch隔离、参数/storage独立、原地保护、严格state_dict参数往返。组合原语完成一次明确的**合成MSE+coverage AdamW step**，不是完整四级模型或原任务loss。

参考容差：CPU FP32输出atol3e-6/rtol1e-4、梯度atol1.5e-5/rtol5e-4；double输出2e-10/2e-8、梯度1e-9/1e-7。double off-SDPA/aux梯度按同一容差一致；A显式参考FP32 atol1e-7、double atol1e-12；FP32 coverage与double oracle atol2e-7/rtol2e-5。w0直接输出/Jacobian、旧Down同权重对照均零容差。没有为了通过而修改旧回归容差。

实际GPU：RTX5090 Laptop，Python3.13.9、torch2.13.0+cu130/CUDA13.0、PyG2.3.1。小B2/d8案例用MATH SDPA，分别运行FP32、FP16 AMP、BF16 AMP，五组件前向/反传有限；A与coverage FP32，实际拦截loss运算确认autocast关闭。该检查是有限AMP执行/dtype验证，**不是AMP与FP32严格逐值等价或完整任务性能验收**。CPU attention使用当前默认SDPA，不强行冒称所有平台相同kernel。

旧模型：M0索引的41个旧Transolver/CDLNO、6个真正pre-M1 K/matched core、24个K/matched任务wrapper、4个其他旧模型，共**75/75同权重精确回放**，工业保留真实PyG与whole/list格式、三个原cwd新进程。182个既有夹具文件hash保持。GUNet图链仍因缺torch_cluster未执行。另复跑**18/18共享原语测试（1.485s）**、**13/13 M1配置测试（0.332s）**，含M1 torch-free导入/metadata；未重复无关的全部316项，M0有效证据保留。

```bash
python -B -m unittest discover -s tests -p 'test_msar_modules.py' -v
python -B -m unittest discover -s tests -p 'test_modules.py' -v
python -B -m unittest discover -s tests -p 'test_msar_config.py' -v
python -B docs/kcdno_audit/make_regression_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures \
  --result docs/msar_lno_audit/m2/k0-replay.json
python -B docs/msar_lno_audit/m1/replay_current.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m1-before-n7y0eg1j/current-fixtures \
  --result docs/msar_lno_audit/m2/current-core-replay.json
python -B docs/msar_lno_audit/m0/wrapper_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/wrappers \
  --result docs/msar_lno_audit/m2/wrappers-replay.json
python -B docs/msar_lno_audit/m0/other_model_fixtures.py replay \
  --artifacts /home/hwz/CDLNO-artifacts/msar-m0-after-m1-q12zvb3s/other-models \
  --result docs/msar_lno_audit/m2/other-replay.json
python -B docs/msar_lno_audit/m2/verify_delivery.py
```

远端可直接执行前三条无数据测试命令，使用现有环境；其余replay需要迁移真实外部夹具并替换路径，不重生成“旧基准”。测试日志/JSON见m2目录，[freeze](msar_lno_audit/m2/freeze.json)列出695个起点文件及允许的3个状态文档增量。保留用户既有diff。

## E. 自审与未完成项

已自审五点：①P直接Q、无隐藏Down/Up残差；②三个独立latent残差子层完整；③fusion仅w、raw值/来源轴/固定2正确且slot敏感；④coverage归约/mask/FP32/off路径与独立oracle一致；⑤旧生产及已有测试冻结、同权重输出/夹具hash不变。有可执行证据支持以上结论，没有需要更改确认公式的阻断冲突。

未执行：M3 core与完整MSAR保存/加载、八任务aux loss接入、真实数据读取/训练/收敛/准确率、完整图采样、远端Python3.10/torch2.11/cu128、其他SDPA backend与大规模性能。原公共norm的极端幅值数值限制不因本轮改变；不以小随机张量有限性保证任意有限输入不溢出。M2仅报告原语合成训练，不能标记“MSAR八任务训练通过”。

本M阶段结束，未执行下一阶段。
