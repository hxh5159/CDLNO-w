# R4 — history-conditioned compression K

## A. 范围与实际审计

**PASS（R4 范围）。** 实现并验证独立 K-only；不依赖 A，不支持联合 A+K，不接 benchmark CLI/factory/launcher。未读取真实数据、下载 checkpoint、运行 GPU/长训练或 commit/push。

仓库 `/home/hwz/CDLNO`，branch `main`，HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`，tree `d74a1b07aa405009992879847aca0c48c75c31f0`。只有根 AGENTS.md，按当前 R4 用户授权限定工作范围。开始时运行 `rg --files -g AGENTS.md -g CLAUDE.md`、`git status --short --branch --untracked-files=all`、`git rev-parse HEAD HEAD^{tree}`、`git ls-files` 的 tracked/untracked/ignored 分类及 diff；对1640文件保存 size/SHA-256，并复制已有非ignored源码到外部快照：

`/home/hwz/CDLNO-artifacts/linearno-history-r4-before-p9d57o29/`

重读冻结提示词 K.1–K.3/R4/R5边界、R1 schema/fair initialization、R2 context/core、R3 A/三套 wrapper、STATUS、测试和当前未提交修改。纯 LinearNO 温度/投影行为和官方来源沿用已审查的 R0/L0–L10 证据；本次复跑其现有 CPU parity，未重新下载或 vendor 官方代码。

## B. 决定与冻结公式

`cdlno/linearno_history/history_k.py:HistoryConditionedK` 实现唯一 K 模块，独立持有 `uq/uk/uv`，没有 A import、mask、dropout 或派生投影缓存。

```text
LN0(x) = (x - mean_last(x)) / sqrt(var_last(x, unbiased=False) + 1e-6)
bank = concat(old C_raw_i, token_axis)
E = uq(LN0(current_base_to_k.weight))                 # [M,d_h]
K_hist = uk(LN0(bank)), V_hist = uv(LN0(bank))          # [B,H,S,d_h]
A_K = softmax_S(E @ K_hist.T / sqrt(d_h))              # [B,H,M,S]
G = A_K @ V_hist                                      # [B,H,M,d_h]
Delta = Z @ G.T                                      # [B,H,N,M]
Delta = Delta - mean_N(Delta, keepdim=True)
eta = tanh(raw_gate_l)                                # [1,H,1,1]
combined_logits = base_K_logits + eta * Delta
K = softmax_N(combined_logits / original_tau_K)
```

LN0 无可训练参数，所有 mean/var 均沿最后一维且 keepdim=True。共享 Uq/Uk/Uv 是无 bias `Linear(d_h,d_h)`；query 直接使用当前层真实 `to_k.weight` 的 M 行，未创建 learned slot query。所有旧历史 token 拼成一个 bank，只在总 S 上一次 softmax，不做 source softmax。Delta 无第二次 sqrt 缩放，不是点恒定 bias。

每个 receiver l>0 独立 raw gate，严格零初始化；第一层无 gate且直接返回同一个 base logits 对象，不执行归一化、投影、matmul、softmax 或随机采样。K 投影遵守普通 Linear 的 trunc_normal std=.02 初始化，非全零；只对 K 子树 apply，之后再把 gate 清零。CPU fork_rng 和显式 feature_seed 不推进主干 RNG，不触碰 CUDA。

代码先修正 raw K logits，再执行既有温度/clamp/softmax_N。Standard temp/conv_temp 的 clamp[.01,1]、ShapeNet `tempreature_*` 拼写与 clamp[.1,2]、AirfRANS dead temperature、plain/conv 无温度均保留。Q/V、输出重建和点残差/FFN/head 不改。K-only 中 raw 已可受此前 K-conditioning 影响；raw 指 pre-AttnRes，不能误称为完全不受历史影响。

## C. 改动文件及插入点

新增：

- `cdlno/linearno_history/history_k.py`：LN0、K算子和按调用的可关闭 trace。
- `tests/linearno/history_k_reference.py`：独立显式 sample/head/point/slot oracle；不导入生产数学 helper。
- `tests/linearno/test_history_k.py`：13个测试方法。
- 本报告和 `docs/linearno_history_audit/r4/` 证据。

增量修改研究文件：

- `core.py:attention_factors` 在基础 Q/K/V 投影后、任何温度之前调用 K；额外暴露 combined logits，仍保留 base logits。`LinearNOHistoryCore.forward` 从局部 context 传递恰好 l份旧raw；A+K 同时传入时明确报错，留待 R5。
- Standard `model/LinearNO_History.py` 新增 `HistoryKModel`；工业 `models.py` 新增 `AirfRANSHistoryKModel`、`ShapeNetHistoryKModel`。先构造原主干，再初始化 `.history_k`。内部 K 类继承 no-op wrapper，不继承 A-only 类。
- 研究 `__init__.py` docstring、STATUS、RESEARCH_MATRIX。

原纯 LinearNO、R3 `attnres.py`、`RawHistoryContext`、R1 schema、原测试、monitor、所有任务 parser/factory/训练/评估/checkpoint/数据/依赖/launcher 均未改。未产生任何生产 K 配置或研究 checkpoint schema；内部 state-dict 往返不等同于生产 metadata-first resume。纯 A0K0 仍走原类和原 loader。

## D. 实际命令、数值与结果

从仓库根运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_k -v
```

实际运行另设 `LINEARNO_R4_REPORT=/home/hwz/CDLNO-artifacts/linearno-history-r4-before-p9d57o29/history-k-results.json`。最终 **13/13 passed，11.026 s**。日志和逐项误差：[最终日志](linearno_history_audit/r4/history-k-final.txt)、[完整数值](linearno_history_audit/r4/history-k-results.json)、[摘要](linearno_history_audit/r4/numerical-summary.json)。

第一次12项检查中11通过，唯一失败来自旧 `Trace` 只采集 bmm：当前 PyTorch 把共享 E 的广播乘法展开成 mm。新增测试用 MatrixTrace 同时捕获 mm/bmm，并核对真实 softmax/Delta shape；未改变模型、公式或容差。保留 [首轮日志](linearno_history_audit/r4/history-k-initial.txt)。之后补充实际query对象/因果时序/M边界检查。

| 验证 | 实际结果 |
|---|---|
| FP64独立oracle，前向/全部梯度/AdamW一步 | combined logits max/mean abs=4.44e-16/3.07e-17；全指标max abs=2.11e-15；atol1e-12/rtol1e-10 |
| FP32独立oracle，同上 | logits max/mean abs=2.38e-7/1.70e-8；全指标max abs=9.54e-7；atol1e-6/rtol1e-5 |
| double finite-difference gradcheck | Z、base slot weight、旧raw和活跃K参数通过；eps1e-6/atol1e-6/rtol1e-4 |
| 六变体，zero gate完整模型 | L4 eval/train（原输出dropout=.2）及L8 eval共18例：逐block/final、loss、输入/主干梯度、AdamW主干更新均误差0，初始化/forward RNG相同 |
| 温度顺序/Q/V | 六变体、温度-3/.5/8逐项独立公式核对，Q/V逐值相同，Q/K沿正确轴归一，Air dead温度无梯度 |
| 点恒定bias反例 | 可精确表示的logits/bias样例在softmax_N后误差严格0 |
| 真正Delta | 同时随n/m变化；mean_N最大abs=9.52e-17；固定Z改变历史后K最大变化0.0248948 |
| 对称性 | 全历史token跨source联合置换G不变；非conv点置换等变；B2/H3分拆计算一致 |
| raw时序/生命周期 | 当前层真实to_k.weight对象传入、只见已完成的l份raw；L4 bank长度8/16/24，每层只条件化一次；连续backward、B/N变化、异常和重入通过 |
| 空历史/初始化/参数 | 第一层同一base对象返回，0投影/0matmul/0softmax/0RNG；无第0层gate，Uq/Uk/Uv非零且全局共享 |
| actual M边界 | M1/32/64、N7（含M>N）前反向通过；完整wrapper保持其既有rank含义 |
| checkpoint与A独立性 | K模块/完整模型strict往返；三个子项目cwd新进程非零gate加载结果误差0，构造/forward后均未import A模块 |

隔离 VJP 从 `eta*Delta` 与固定随机cotangent的乘积求和出发，base logits、未发生的当前 C、未来 C 不参与。当前 Z、基础 slot weight、Uq/Uk/Uv、三份旧raw的梯度L1分别为：27.0767、6.34386、22.3404/19.7245/25.1832、10.0381/12.5785/13.6041，均finite非零；当前/未来C和base logits梯度为None。gate=0时只要求接收gate的新增梯度非零，其余活跃K投影梯度为0；不是通过完整主干loss偷换历史梯度证据。

执行shape检查采用 N7/M5/S9/d_h4，记录的交互是 `[B,H,5,9]` bank softmax 和 `[B,H,7,5]` Delta，没有N×N。常规投影还产生[M,d_h]/[S,d_h]，聚合产生[M,d_h]；没有把这些合法张量误判成attention。共享query的matrix实现使用flattened `mm [B*H*S,d_h]@[d_h,M]`，并在softmax处恢复完整M×S语义。

K参数增量为 `3*d_h² + (L-1)*H`。d_h4/H3下L4=57、L8=69；没有per-edge参数或独立slot参数，没有cache/mask状态。

旧回归命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_latent_attnres linearno.test_history_core \
    linearno.test_history_schema linearno.test_schema linearno.test_profiles \
    linearno.test_attention_parity linearno.test_attention_structure \
    linearno.test_standard_model linearno.test_standard_structure \
    linearno.test_airfrans_model linearno.test_shapenet_model \
    linearno.test_legacy linearno.test_rng monitor.test_monitor -v
```

**90方法：85通过、4 GPU跳过、1既有失败方法（3个subtest断言），0 errors，92.242 s。** R2、R3、原LinearNO官方parity、八任务旧Transolver固定权重/checkpoint/选择、随机流和monitor均通过。失败仍是R0以来README、path.sh、旧reproduction-matrix冻结快照，不是R4引入；未修旧测试来制造全绿结果。日志见 [regression.txt](linearno_history_audit/r4/regression.txt)。

本机 Python3.13.9/Torch2.13.0+cu130/PyG2.3.1；实际仅CPU，没有安装依赖。不是远端Torch2.11/cu128或GPU验收。

## E. 兼容性与自审证据

1. 源码冻结：开始时1242 tracked、68 untracked、330 ignored；所有tracked/ignored和所有旧纯模型字节不变。R3 A模块、R2 raw context原文件保持hash；新改动只落在研究源码/文档。
2. 公式独立核对：LN0 oracle显式均值和平方差，K oracle使用sample/head/point/slot循环；另有有限差分与点bias无效反例，未仅对照同一函数。
3. 插入点：实际hook确认传入本层to_k.weight、先K后当前block完成；温度前相加，Q/V/原重建未改；原点残差/FFN/head AST与R3快照相同。
4. 两机制独立：K参数和中间量不与A共享；fresh-process证明不import A，故没有A参数/投影/dropout；A+K接口显式拒绝，R3本身全部回归通过。
5. 生命周期/公平：未detach、未存self history/derived cache；feature seed隔离主干初始化；K无RNG采样，因此zero gate的train/eval都保留原RNG演化。

具体hash、新增/改动列表及source diff见 [freeze.json](linearno_history_audit/r4/freeze.json)。未覆盖用户tracked/untracked修改，未reset/clean/stash/commit/push。

## F. 未执行、限制与风险

未执行A+K联合、R5内部factory/config/metadata-first checkpoint、八任务生产launcher/loader闭环、研究monitor接入、真实数据/训练/收敛/精度、GPU/AMP/compile、性能或远端环境。完整模型的loss为合成MSE；工业Data来自真实PyG，但不是实际数据/采样/后处理验证。并未升级R1全20组合或八任务生产矩阵为完成。

无已知新增数学失败或需要裁定的规格冲突。全套旧测试并非全绿：上述历史冻结断言和未运行GPU项继续保留。

## G. 阶段状态

**PASS。** 本 R4 阶段结束，未执行 R5 或后续阶段。
