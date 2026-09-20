# Looped LinearNO LL2：点域 AttnRes 与 block body

日期：2026-09-19。**唯一阶段状态：PASS。** 本阶段只实现独立原语和原 block 子模块调用适配，没有完整 loop 模型、任务入口或 checkpoint 接线。

## A. 实际范围与起点

仓库 `/home/hwz/CDLNO`，origin `https://github.com/hxh5159/CDLNO-w.git`，分支 `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，HEAD tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。起点 tracked diff 为空，LL0/LL1交付及用户两个设计文件属于已有 untracked 内容。

本轮先读当前阶段状态、LL1合同、LL0插入点和回归清单、冻结模型规格与 LL2 提示词，再核对实际 Standard `LinearNOBlock`、AirfRANS `AirfRANSBlock`、Car `ShapeNetBlock` 及共享 attention 的公式/初始化。适用根 AGENTS 已读并遵守，当前用户授权仅 LL2。

在写原语之前，先新增独立 oracle，并对 `[3,4]` 与 `[0,2]` 两来源做标量手算校验，FP64/FP32都通过；没有调用生产 AttnRes。FP64手算输出 `[2.0574578354044544, 3.3716385569363028]`，来源权重 `[0.6858192784681514, 0.3141807215318486]`。

完整起点清单见 [start-manifest](loop_linearno_audit/ll2/start-manifest.json)，共 **2005文件：1549 tracked、69 untracked、387 ignored**；以实际分类和内容hash判断本轮变化，不把未提交工作当成本轮新增。

## B. 实际文件与 diff 摘要

| 文件 | 改动 |
|---|---|
| `cdlno/linearno_loop/__init__.py` | 新增空的独立原语包入口，不自动 import torch/模型/任务 |
| `cdlno/linearno_loop/attnres.py` | 新增 `PointDepthAttnRes(hidden)`，仅 query/norm_scale 两参数；点域来源融合、输入验证、显式权重诊断 |
| `cdlno/linearno_loop/body.py` | 新增 `LinearNOBlockBody(block)`，只保留已有block引用，暴露raw/native/scaled/finalize调用 |
| `tests/loop_linearno/point_attnres_oracle.py` | 独立显式 RMS/exp/softmax/加权数学 oracle，保留FP64 |
| `tests/loop_linearno/test_point_attnres.py` | 12项：oracle/VJP/step/gradcheck、手算、置换/隔离、初始化、负例、dtype/device、trace、singleton、CPU BF16 |
| `tests/loop_linearno/test_block_body.py` | 5项：六变体60组同权重对照、raw/scaled/identity、head时序、参数身份/RNG/错误输入 |
| `tests/loop_linearno/test_artifacts.py` | 更新LL1阶段边界断言：允许LL2原语目录存在，仍禁止完整模型/core文件和launcher；未删旧schema/metadata断言 |
| 本文、`LOOP_LINEARNO_IMPLEMENTATION_STATUS.md`、`docs/loop_linearno_audit/ll2/` | 合同映射、报告、原始日志/误差、冻结与复算工具 |

源码增量见 [source.diff](loop_linearno_audit/ll2/source.diff)。顶层 `linearno_loop/` 纯schema、LL1配置/矩阵/metadata及所有旧模型/入口/launcher/checkpoint/monitor均保持原字节。没有修改旧 Transolver/LinearNO/history 测试或golden。

## C. 公式 → 实现 → 测试

设每个来源 `V_s[B,N,H]`，H为点特征完整隐藏宽度，不按 operator heads 分组：

\[
k_s={V_s\over\sqrt{\operatorname{mean}_H(V_s^2)+10^{-6}}}\odot g,
\quad \ell_s=\sum_H w\odot k_s,
\quad \alpha_s=\operatorname{softmax}_s(\ell_s),
\quad y=\sum_s\alpha_s V_s.
\]

| 冻结要求 | 实现/映射 | 验证 |
|---|---|---|
| 单receiver，query0、scale1 | `PointDepthAttnRes.query/norm_scale`，各 `[H]`，参数恰2H；constructor不抽随机数 | inventory、receiver对象独立、RNG、common initializer后仍0/1 |
| key RMSNorm最后轴keepdim/eps1e-6 | `_weights_and_values`；stack `[S,B,N,H]`，mean(-1,keepdim=True) | 手算、独立oracle、RMS幅值抑制、double gradcheck |
| 单来源轴softmax，raw value | logits `[S,B,N]`，softmax(dim=0)，乘原values再sum(0) | 均匀权重、raw幅值反例、来源/点/batch/channel置换、B2隔离 |
| 没有额外结构或缩放 | 无bias、value/Q/K/V投影、depth heads、sqrtH、source-count或1/R、dropout、外层residual | 参数/子模块清单、手算/oracle、实际dispatch trace |
| 来源合法性 | 非空sequence；各tensor严格同 `[B,N,H]`、dtype、device，B/N>0且H匹配；receiver device/dtype检查 | 空/错形状/B/N/H/dtype/device/meta device/非法hidden负测 |
| 数学梯度、无detach | 多来源图保持至每个输入、query、scale；所有中间值只在函数局部 | 独立VJP逐输入/参数、一步SGD对照、有限差分 |
| 单来源 | 验证后直接返回原tensor，严格恒等 | 输出同对象、输入梯度1；router梯度None，即无有效梯度，符合首RB receiver约定 |
| 不缓存history | 无buffer、无forward tensor属性；诊断权重不挂self | 连续变B/N、异常后恢复、参数/buffer/属性清单不变 |
| 原operator raw branch | `body.operator(x) = block.Attn(block.ln_1(x))` | 六变体直接子模块结果对照 |
| 原MLP raw branch | `body.mlp(x) = block.mlp(block.ln_2(x))` | 六变体直接子模块结果对照 |
| native residual body | `x1=operator(x)+x; y=mlp(x1)+x1`，不调用head | 与原非末层forward逐值/梯度/优化器/RNG对照 |
| scaled residual body | `x1=x+a*operator(x); y=x1+a*mlp(x1)`，identity不缩放 | a=1等native，a=1/3显式两branch对照，zero branch时y=x |
| 显式finalize | 仅允许last_layer=True，`block.mlp2(block.ln_3(x))` | body后head计数0，finalize后LN/head各1；body+finalize等原末层forward |

oracle使用另一种布局 `[B,N,S,H]`，显式 sum/H、sqrt、稳定化exp/归一、einsum 加权；不 import 或调用生产模块。FP64不会被强制转FP32。生产AttnRes内部关闭autocast，FP64保留，FP32保持；FP16/BF16源在FP32计算后转回源dtype。source必须同dtype，低精度source允许FP32 master参数；其它参数dtype不匹配明确拒绝。仅CPU BF16路径作了补充检查，不把它当作CUDA AMP或完整模型验收。

`source_weights(sources)` 是显式诊断重算接口，返回 `[S,B,N]`（计算dtype）；正常 `forward` 只返回 `[B,N,H]`。不在self保留权重/图，也没有CPU同步统计。实际dispatch仅出现 `[3,2,17]` 来源softmax，**没有mm/bmm/matmul或N×N张量注意力**。

body adapter 是普通Python对象，不是 `nn.Module`；`__slots__=('block',)`，不另行注册/复制/初始化block，不保存历史。原body的Q/K/V、温度、softmax轴、conv、输出投影、MLP和norm全部由旧子模块原样执行。native/scaled/raw是内部调用形式，没有新增第四种正式 residual mode。

LL2没有拓扑，`finalize`只能验证原block的last_layer/head；“该block属于最后suffix且只finalize一次”由后续LL3拓扑负责，本阶段没有假装实现这一层约束。adapter不调用完整 `block.forward`，因此将来统计visit须hook实际Attn/MLP子模块或新core的显式visit事件，不能把旧整个block.forward hook当作已触发。

## D. 实际命令、环境与数值

本机 Python3.13.9 / torch2.13.0+cu130 / CUDA13.0 / RTX5090 Laptop；见 [environment](loop_linearno_audit/ll2/environment.json)。新CUDA检查仅小型FP32原语/block；body关闭TF32与cuDNN benchmark并使用确定性cuDNN，检查后恢复选项。没有安装依赖、访问真实数据或训练任务。

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll2/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll2/body-report.json \
python -B -m unittest discover -s tests/loop_linearno -v

PYTHONDONTWRITEBYTECODE=1 \
python -B docs/loop_linearno_audit/ll2/run_existing_regressions.py

PYTHONDONTWRITEBYTECODE=1 \
python -B docs/loop_linearno_audit/ll2/finalize_evidence.py
```

新原语/body17方法 + LL1配置24方法 = **41/41通过，5.856s，0failure/error/skip**。详见 [loop-tests.log](loop_linearno_audit/ll2/loop-tests.log)。CUDA可用，本轮2个新CUDA测试方法确实执行。

AttnRes每类3个fixture（S/N=1/1、2/7、4/11，B2/H5），均比较output、weights、每个输入和query/scale VJP、一步更新。下表汇总所有逐tensor误差；“mean上界”是单项mean误差的最大值，完整逐项数据见 [attnres-report](loop_linearno_audit/ll2/attnres-report.json)。相对误差分母floor=1e-12。

| dtype/device | atol / rtol | 最大abs | mean abs上界 | 最大relative | mean relative上界 |
|---|---|---:|---:|---:|---:|
| CPU FP64 | 1e-12 / 1e-10 | 2.665e-15 | 1.044e-15 | 5.682e-14 | 8.147e-16 |
| CPU FP32 | 1e-6 / 1e-5 | 2.861e-6 | 1.383e-6 | 4.215e-5 | 7.541e-7 |
| CUDA FP32 | 1e-5 / 1e-4 | 9.537e-7 | 4.768e-7 | 5.274e-5 | 8.557e-7 |

全部满足逐元素 `abs(error) <= atol + rtol*abs(reference)`；不把最大abs必须低于单独atol或最大relative必须低于单独rtol作为不同门槛。没有放宽以上阈值。double finite-difference gradcheck另用eps1e-6、atol1e-6、rtol1e-4，测试非零query/scale与全部source输入。

六variant × 末层/非末层 × CPU FP32/FP64 × train/eval =48组；另CUDA FP32 train12组，总计 **60组body对照，output/loss/input grad/全部活跃参数grad/AdamW一步权重最大与mean误差全部0**。dropout=.25；每次两侧forward前恢复同一CPU/CUDA RNG，之后RNG精确相等；使用测试方复制的同一份实际权重，adapter自身不复制。Air已知dead `Attn.temperature` 两侧均无梯度，单独登记；Car `tempreature_q/k`拼写保留。conv采用非方形3×5网格，N错误保留原错误消息。详见 [body-report](loop_linearno_audit/ll2/body-report.json)。

首轮新测试的一处来源置换诊断错误地要求逐位相等；观察到softmax分母来源求和顺序产生FP64最大1.11e-16舍入差。已将该断言改为本轮预先使用的FP64 1e-12/1e-10阈值，与output/oracle门槛一致；没有改生产数学、旧阈值或旧测试。原始失败保留在 [attnres-first-run.log](loop_linearno_audit/ll2/attnres-first-run.log)，最终41项通过另存。

## E. LL0回归与冻结

重跑LL0完全相同的24组测试模块、同一CPU环境参数与测试名单，**153方法：147通过、2个失败方法（4条已存在文档冻结断言）、4 CUDA skip、0error**。结果逐组test/failure/error/skip与已批准LL0精确一致；还逐行比对失败断言文本/hash，未产生新失败。[完整结果](loop_linearno_audit/ll2/regression-results.json)、[对照摘要](loop_linearno_audit/ll2/regression-summary.json)。

其中包含纯LinearNO官方CPU FP32/FP64原语与三套整模型parity、输入/参数梯度、一步optimizer、strict checkpoint、八任务Transolver/RNG/原生合成闭环、history核心59方法、monitor10方法。旧GPU方法按LL0原CPU协议跳过；与本轮确实执行的新CUDA原语检查分开记录。

两个历史失败方法仍是 `LegacyRegression.test_all_preexisting_files_and_absent_frozen_directories`（README/path.sh/旧复现矩阵文档差异）和 `StaticIntegration.test_all_new_launcher_actions_profiles_and_old_scripts_static`（同一path.sh差异）。来源解释在LL0审计及原patch，本轮未修改这些文件/golden。**不声称全仓测试全绿。**

冻结复算覆盖本轮起点全部tracked/untracked/ignored文件；除本研究状态与LL1阶段边界测试外，其余2003个已有文件不变。纯配置包、所有旧模型/入口/data/训练/评价/checkpoint/launcher/monitor及旧LL0回归文件均逐字节一致。允许新增ignored产物仅 `docs/loop_linearno_audit/ll2/*.log`，不新增run/cache目录。见 [end-freeze](loop_linearno_audit/ll2/end-freeze.json) 和 [delivery-review](loop_linearno_audit/ll2/delivery-review.json)。

## F. 自审与边界

| 本轮最应复核的点 | 自审结果 |
|---|---|
| 是否仍为raw value、source-only softmax，而非旧latent A/K | 公式、参数清单、scalar/oracle及trace相互对应；没有旧history import |
| query/norm初始化与singleton梯度 | 初始0/1不消耗RNG；多来源query非零梯度、scale初始零梯度；singleton路由梯度无效符合合同 |
| adapter是否重复残差或提前head | native两残差、scaled只缩branch、raw无残差；hook验证head只在finalize执行；60组数值误差0 |
| 是否重注册/复制/重初始化旧block | 普通对象只持原block；parameter/module IDs、state键值与构造前后RNG精确不变 |
| 回归失败是否由本轮引入、是否越阶段 | LL0失败文本/hash精确一致；全量文件冻结；没有core/wrapper/生产接线 |

无新增待用户裁定的模型语义冲突。**NOT RUN/未实现**：完整loop拓扑和三种完整residual组合、task wrapper/CLI/factory、loop checkpoint/resume、loop性能/真实数据/精度/长训练、远端Python3.10/torch2.11/cu128环境、完整模型CUDA AMP。参数共享与suffix唯一归属将在后续已授权阶段验证；本轮不能宣称完整loop family已可训练。

**本 LL2 阶段结束，未执行下一阶段。**
