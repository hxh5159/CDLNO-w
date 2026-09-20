# Looped LinearNO LL4：点域 Block AttnRes

**唯一状态：PASS。仅完成 LL4，未执行 LL5–LL10。**

## A. 本阶段范围与来源

仓库 `/home/hwz/CDLNO`，remote `https://github.com/hxh5159/CDLNO-w.git`，branch `main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。2026-09-20，用户明确批准 LL3 后只授权本 LL4。

核对根 AGENTS、LL0–LL3 报告/状态及当前源码。重新阅读冻结规格 §4.2、LL4 提示词、Attention Residuals v1 PDF 第3–5页 Eq.2–6/Figure2，及固定官方 README 的 Block AttnRes 伪代码。论文 PDF SHA-256 `444673994328f7be8aee9d96fb240596b6f254f06ebaa53a2673413a244198c9`；官方材料 commit `85e22310fe5ee860b4a023de312d791de8a5a5e6`。[本轮来源复核](loop_linearno_audit/ll4/source-recheck.json) 与 LL0 台账 hash 相同。环境没有 `pdftotext`，改用已安装 `fitz` 直接读取 PDF；未安装依赖、未重新下载来源。

本阶段在同一通用主干增加 `rb_attnres`。一次 core pass 对应一个 AttnRes block，内部有 2C 个 residual sublayer；这是论文 Block AttnRes 在对齐点域 `[B,N,H]` 上的适配。prefix/suffix、LinearNO 的 Q/K/V/温度/卷积/输出投影、native FFN/LN/head 都沿用 LL3 和纯模型。没有引入旧 A/K、latent history 或 LB。

固定官方材料是公式来源；本项目的 LinearNO 点域适配和 PCRS 拓扑不是官方发布的 LinearNO 实验。论文 Figure2 的“在边界转换 completed/partial 列表”和本实现“前轮末尾入 completed、次轮从空 partial 开始”传递同一组张量；以 Eq.6 的来源定义和本项目冻结规格逐项验证。

## B. 实际文件与最小接入

| 文件 | 实际修改 |
|---|---|
| `cdlno/linearno_loop/core.py` | 允许 SR/RB 两个已实现 mode；只在 RB 注册 receiver；新增 forward-local `_rb_forward`，补结构独立性检查；原 SR 数学分支不变 |
| `cdlno/linearno_loop/construction.py` | 合成 config 构造允许 RB，LB 继续在构造前拒绝；更新公共初始化字典的说明 |
| `tests/loop_linearno/test_sr_core.py` | 仅把“未实现 mode 拒绝”的断言范围收窄至仍未实现的 LB；SR 数值、初始化、计数及 strict 测试不改 |
| `tests/loop_linearno/rb_oracle.py` | 先于生产 RB 改动写入的独立全 core 公式 |
| `tests/loop_linearno/rb_support.py` | 测试专用 live hook/权重 trace、非零 query fixture，无生产诊断缓存 |
| `tests/loop_linearno/test_rb_core.py` | 9 项专项测试，含 36 个数学 oracle 案例和 24 个训练 AdamW 案例 |
| 本报告、研究 STATUS、`docs/loop_linearno_audit/ll4/` | 命令、数值、原文件备份、来源、diff、冻结和自审 |

没有修改三套 wrapper 文件、LL2 PointDepthAttnRes/body、顶层无 torch schema，或任何旧模型/八任务入口/factory/训练/eval/checkpoint/launcher/monitor。内部调用链仍为：

```text
LL1 config -> validate_config -> require_implemented -> validate_constructor
  -> 原 LoopedStandard/AirfRANS/ShapeNet 合成 wrapper
  -> 原 stem / 原输入位置时间处理
  -> prefix native -> shared core 的 RB 子层调度 -> suffix native -> 唯一 head
```

仅 RB 新增 state 键：`loop.rb_receivers.<round>.<sublayer>.query/norm_scale` 与 `loop.rb_output.query/norm_scale`。core 算子仍位于 `loop.core.<physical_index>.block.*`，没有按 round 复制算子。SR 没有任何 router 参数。router 是全零 query/全一 scale，构造不消耗 RNG；原 wrapper 的一次 whole-tree apply 只识别 Linear/LN/Conv，不会覆盖这两类自有 Parameter。16 组同 seed 的八任务×两 preset 构造证明公共主干逐键值及构造后 RNG 完全一致。

## C. 公式与调度

prefix 输出为 anchor (a=b_0\)。第 r 轮开始 `partial=None`，完成轮列表为 `[b0,...,b(r-1)]`。第 j 子层输入来源为：

\[
V_{r,j}=\begin{cases}[b_0,\ldots,b_{r-1}],&j=1\\
[b_0,\ldots,b_{r-1},p_{r,j-1}],&j>1.\end{cases}
\]

每个 `(r,j)` 独立 receiver 使用 LL2 的 eps1e-6、单 pseudo-query、RMS key、source softmax 和 raw value：

\[
h_{r,j}=\mathrm{AR}_{r,j}(V_{r,j}),\quad
u_{r,2k-1}=\mathrm{Attn}_k(\mathrm{LN}_{1,k}(h_{r,2k-1})),\quad
u_{r,2k}=\mathrm{MLP}_k(\mathrm{LN}_{2,k}(h_{r,2k})).
\]

`partial` 只累加这些 raw u，(b_r=\sum_{j=1}^{2C}u_{r,j}\)。轮末入库一次，随后清空下一轮 partial。最终 `rb_output((b0,...,bR))` 才进入 suffix。**没有 h+u、anchor+partial、1/R、来源数补偿、depth/value 投影。**

completed/partial/h/raw 都是一次 `_rb_forward` 内的局部变量，传 receiver 时使用 tuple 快照，既不改变此前 source 列表，也不 detach。没有 forward tensor 属性、buffer、跨 batch/time/member 的缓存。注册在 self 上的只有固定 receiver 参数和既存物理 blocks。

| 拓扑 | receiver 数 | router 参数 | 每个 receiver 的来源数（`|`分轮/最终） |
|---|---:|---:|---|
| P1/C3/R2/S1 | 13 | 26H（fixture H8 实测208） | `1,2,2,2,2,2 | 2,3,3,3,3,3 | 3` |
| P2/C2/R2/S2 | 9 | 18H（fixture H8 实测144） | `1,2,2,2 | 2,3,3,3 | 3` |
| custom P0/C2/R3/S1 | 13 | 26H（fixture H8 实测208） | `1,2,2,2 | 2,3,3,3 | 3,4,4,4 | 4` |

H 表示 hidden width，不是 head 数/网格高。receiver 的 query/norm 在 round、sublayer、output 之间均独立，原 attention/MLP 在 round 之间共享。SR 的参数/MAC报告与 LL3 完全相同；本轮只实测新增 router 参数，不声称 RB 与 SR FLOPs 相同、内存更少或训练更快。

## D. 验证、数值与实际命令

本轮 CPU 合成，Python3.13.9、torch2.13.0+cu130、PyG2.3.1；完整版本见 [environment](loop_linearno_audit/ll4/environment.json)。未执行真实数据、GPU、远端或任务训练。

最终命令（仓库根目录）：

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll4/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll4/body-report.json \
LOOP_LL3_PARITY_REPORT=docs/loop_linearno_audit/ll4/sr-parity.json \
LOOP_LL3_ACCOUNTING_REPORT=docs/loop_linearno_audit/ll4/accounting.json \
LOOP_LL4_REPORT=docs/loop_linearno_audit/ll4/rb-report.json \
python -B -m unittest discover -s tests/loop_linearno -p 'test_*.py' -v \
> docs/loop_linearno_audit/ll4/loop-tests.log 2>&1

PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_audit/ll4/run_existing_regressions.py \
> docs/loop_linearno_audit/ll4/regression-progress.jsonl 2>&1

PYTHONDONTWRITEBYTECODE=1 python -B docs/loop_linearno_audit/ll4/finalize_evidence.py
```

**最终 loop suite：64 方法，62 pass，0 failure/error，2 CUDA skip，44.342s。** 9 项 LL4 新测试全通过；两 skip 是本轮 CPU 环境下原 LL2 CUDA 检查，不把历史 LL2 GPU 结果当作本轮 GPU 验收。首次 7 项专项也通过，保留 [首轮日志](loop_linearno_audit/ll4/rb-first-tests.log)。最终 [测试日志](loop_linearno_audit/ll4/loop-tests.log)、[逐张量报告](loop_linearno_audit/ll4/rb-report.json)、[数值摘要](loop_linearno_audit/ll4/numeric-summary.json)。

| 检查 | 覆盖和实际结果 |
|---|---|
| 完整数学 oracle | 六 variant × 三拓扑 × FP64/FP32，36 案例；独立 LN/erf GELU、独立 LinearNO 因子公式和独立 source AR，不调用生产 forward |
| 中间量 | 每个 source、实际 source weights、h、raw u、每步 partial、每个 b_r、最终 route 和完整输出逐项比较；LN 实际收到的对象就是相应 h |
| 梯度 | 输入与所有活动算子/router 参数逐项对照；非零 query 下每个活动 router 梯度 finite/nonzero；首 singleton query/norm 无有效梯度，Air dead temperature 单独记录 |
| raw 累加反例 | C1/R1/零 query，手算 operator=2x、MLP=3x、suffix identity，结果为3.75x而非普通 residual 的12x；证明无 anchor/identity 注入和来源数补偿 |
| 训练更新 | 24 组 native 算子独立展开、FP64/FP32，原 attention dropout=.2；forward、输入/参数梯度、AdamW权重/state、RNG逐位全等。此 dropout 是原 attention dropout，不是 history dropout |
| 来源和执行时机 | 三拓扑来源序列完全匹配；第一 receiver 返回 anchor 原对象；query0 为均匀平均，非零 query 权重逐点不同；router→对应LN→raw顺序精确；Q/K/V 每轮重算；最终LN/head各一次 |
| 共享与反传 | 16组SR/RB实际公共初值/RNG逐位相等；第二轮 isolated raw branch VJP 回到第一轮 b1、输入、共享to_v参数，均finite/nonzero；无future/current raw source |
| 无N×N/M×M attention | TorchDispatch实际bmm仅KTV/QC；N11/M4/dh4 fixture明确区分合法4×4 context与slot self-attention；core native/full-block.forward设为必抛错仍通过，证明未隐含普通 residual |
| 隔离 | B>1单样本对照、非conv点置换、异常发生在已有history后再调用、更换B/N、旧输入autograd无路径、重复forward一致；NS/Plasticity/Air/Car调用与新对象对照，第一source列表每次只含anchor |
| strict | Standard/Air/Car三套config先读、三次临时cwd全新进程、weights_only=True/strict=True，输出逐位相同；SR/RB state互载拒绝；router别名/非法mode结构拒绝 |

独立数学 oracle 的一步更新采用 SGD，沿用 LL3 对独立 FP32 reduction 舍入与 AdamW 分母敏感性的分开验证方法；AdamW 由同数值算子、独立控制流对照验证。没有改变生产 optimizer，也没有放宽 LL2/LL3 容差。

| 独立数学对照 | 案例数 | atol / rtol | 最大 abs | 各张量mean abs最大值 | 最大 relative / mean relative |
|---|---:|---|---:|---:|---:|
| FP64 | 18 | 1e-12 / 1e-10 | 8.88178e-16 | 2.49583e-16 | 4.09564e-10 / 4.84628e-11 |
| FP32 | 18 | 1e-6 / 1e-5 | 9.45525e-7 | 4.18833e-7 | 1.65630 / 0.211761 |
| native 独立展开 AdamW | 24 | 0 / 0 | 0 | 0 | 0 / 0 |

每个dtype有5210项逐张量误差记录。relative 使用 `abs(error)/max(abs(reference),1e-12)`；最大 relative 来自 near-zero `plain/P2/rb_receivers.0.1.query` 梯度，该张量 max abs 仅6.84452e-12。FP32最大绝对误差来自 `conv_temp/P2/core.0.mlp.linear_post.bias` 梯度，仍在既定 atol+rtol 内；没有把 relative 的大比值隐去。

## E. SR 与旧模型兼容证据

SR 的完整 [数值报告](loop_linearno_audit/ll4/sr-parity.json) 和 [参数/MAC报告](loop_linearno_audit/ll4/accounting.json) 与已验收 LL3 JSON 全等，不仅是测试总数相同。SR 无新增 parameter/buffer；公共 backbone 构造、初始化、placeholder RNG 不因 RB receiver 的存在而改变。既有 LL3 冻结数值、容差、原始失败诊断全部保留。

重跑 LL0 原 24 模块、153 方法：147 pass、2 个已审查历史失败方法（4 条文档冻结断言）、4 CUDA skip、0 error。逐模块结果、失败ID与实际断言文本均与 LL0 一致，无新增失败，没有改旧 golden。旧失败涉及 README/path.sh/纯 LinearNO 复现矩阵历史内容，不能据此宣称全仓测试全绿。[回归对照](loop_linearno_audit/ll4/regression-summary.json)。

起点 [manifest](loop_linearno_audit/ll4/start-manifest.json) 保存2142个 tracked/untracked/ignored 文件分类、size、SHA-256，含全部旧模型与历史用户修改。最终 [freeze](loop_linearno_audit/ll4/end-freeze.json)：仅上列4个已有研究文件改变，其他 **2138个文件保持内容与分类不变**，无意外新增/删除/ignored变化。HEAD/tree、tracked/staged diff 不变；未reset/clean/stash/checkout/commit/push。新ignored仅本阶段证据 `.log`，不生成缓存/运行目录。完整增量见 [source.diff](loop_linearno_audit/ll4/source.diff)。

首次冻结脚本沿用 LL3 时漏改了证据目录的 lowercase 前缀，将本阶段 `ll4/` 产物误报为意外新增；原有文件检查当时已无意外改变。[初次结果](loop_linearno_audit/ll4/initial-freeze.json) 保留。已将本阶段允许证据路径精确改为 `ll4/` 并重跑，不扩大生产文件变更范围、不修改起点manifest或旧冻结规则。

## F. 自审、未运行项与状态理由

五项交付前自审已经逐源码/公式/证据核对：

1. **来源时序**：无初始零partial作为额外source；轮首只读completed，每个后续子层加且仅加当前raw partial，最终读取全部completed。
2. **数学边界**：core不存在普通 residual、1/R 或额外来源数缩放；prefix/suffix仍native；query0的RB不声称等于SR。
3. **两种共享**：算子跨轮共享，router按逻辑位置独立；参数数目/别名负例、初值/RNG及分支梯度证据齐全。
4. **状态生命周期**：生产路径无trace/history属性；hooks仅在测试，保留活图；异常、变batch/点数、连续物理时间/成员调用不串history。真实任务循环未运行。
5. **回归和声明**：SR两个JSON精确相等、旧153回归与原历史结果一致、2142文件全冻结；没有用测试通过替代论文精度/真实训练/远端验收。

没有新待裁定架构冲突，LL4所有非资源门控要求完成，唯一状态 **PASS**。NOT RUN/未实现：LB、生产parser/factory/launcher/monitor与统一输出接线、loop完整optimizer/RNG resume archive、真实数据/训练、GPU/AMP/compile、远端Python3.10/torch2.11 cu128、完整性能/显存和精度。strict测试仅是合成state_dict新进程往返，不称生产train/resume/eval闭环。

**本 LL4 阶段结束，未执行下一阶段。**
