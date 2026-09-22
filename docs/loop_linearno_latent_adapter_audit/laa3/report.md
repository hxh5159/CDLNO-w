# LAA3：独立 V3 attention 路径

**状态：PASS，仅独立 attention 与合成验证范围。LAA4 未执行。**

## A. 来源、范围与实际变更

工作树 `/home/hwz/CDLNO`，`main@c02e671506f706910e0a1d58f03c310abf188345`，origin `git@github.com:hxh5159/CDLNO-w.git`。实施依据是当前纯 `LinearNOAttention` 六 variant、已通过 LAA1 的字段合同、LAA2 两个原语、冻结提示词 §2.3–2.5/§4.1/LAA3，以及本轮用户的同名阶段指令。延续 LAA0 对旧 AGENTS/记忆的历史性解释；没有恢复已删除文件或改写旧审计统计。

起点 hash 包含 4164 个已有文件（tracked2130/untracked675/ignored1359）。保留用户 `check_checkpoints/check_pipe_loop_resume.sh` 已有修改。无依赖安装、真实数据读取、reset/clean/stash/checkout/rebase/commit/push。

| 文件 | 修改理由 |
|---|---|
| **新增** `cdlno/linearno_loop/v3/attention.py` | 唯一新增生产文件，继承原构造器；native 委托和显式 round visit；两处 active 特性插入 |
| **新增** `tests/loop_linearno_latent_adapter/attention_oracle.py` | 无生产依赖的卷积展开、逐 head 注意力和完整中间状态 oracle |
| **新增** 同目录 `attention_support.py` | 确定性权重/输入、误差记录 |
| **新增** `test_laa3_oracle.py`、`test_laa3_attention.py`、`test_laa3_amp.py` | 手算、数学、结构、兼容、非法输入和 CUDA 验证 |
| **新增** 本证据目录 | 原始日志、JSON、test-first/freeze/diff/source-map、复现脚本及报告 |
| **更新** `docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md` | 追加 LAA3 状态，保留原历史文本 |

没有修改纯 LinearNO、v1/v2 attention、LAA2 原语、三残差、旧 golden/容差、task wrapper/入口、schema/config、版本路由、checkpoint backend、launcher、记录器或可视化。新 attention 尚未接入生产模型；本阶段不交付可运行 V3 训练命令。

## B. API、初始化与共享边界

`V3LinearNOAttention.__init__` 就是继承的 `LinearNOAttention.__init__`，没有第二次 backbone 构造，也没有新的公共初始化算法。参数 `rank` 就是实际每 head 的 M，`H/W` 仍为空间网格尺寸。ShapeNet 的旧 task-local `key_ratio*dim_head` 语义以及 v1/v2 M%dh 验证不变；V3 直接调用共享底层构造器，以 M 决定 `to_q/to_k.out_features`。

安装顺序：先创建公共 attention、完成原 release 初始化，再调用一次 `install_features(...)`。接口使用 LAA1 字段名 `latent_enabled/latent_width/adapter_mode/adapter_rank/adapter_alpha` 和显式 `latent_seed/adapter_seed`。复用 LAA2 的隔离初始化，adapter 与 latent 先在局部构造成功，再注册；非法参数不会留下半套模块。off 没有相应 module/state key。安装后不能再次全树 apply，未来 wrapper 必须遵守此顺序。

`forward(x)` 完全调用原 native forward，供非 recurrent 调用使用。`forward(x, round_index=0/1)` 显式选择 core visit，无内部计数器。第一轮不读取/调用 adapter 参数；只有第二轮调用。adapter-off 允许 custom 的任意非负 round index；adapter-on 拒绝 index>1。R=2 的完整 topology 合同仍由已完成的 LAA1 负责，本类不组装或推断 P/C/R/S。

每个 attention 实例只安装一套 latent/adapter。两次显式 visit 的 latent module id 相同；不同实例的所有 Parameter id/storage 独立。prefix/suffix 在后续 core 中必须继续使用 native blocks；本阶段验证 native 委托，不声称已经构造完整 prefix/core/suffix。

## C. 冻结公式 → 实现 → 测试

令 X 为 `in_project_x` 后的 `[B,h,N,dh]`：

```text
Lq = to_q(X); Lk = to_k(X); V = to_v(X)
round_index == 1 且 adapter-on：
    Lq += (X Aq^T) Bq^T * alpha/r
    Lk += (X Ak^T) Bk^T * alpha/r
Q = softmax_M(original_temperature_rule(Lq))
K = softmax_N(original_temperature_rule(Lk))
C = K^T V
latent-on：C = SplitHeads(Z + W2 GELU(W1 LN(Z) + b1) + b2)
    其中 Z = MergeHeads(C)
readout = Q C
output = original_to_out(MergePointHeads(readout))
```

没有 point residual、1/R、额外归一化、来源缩放、V 增量或 Q-only/K-only 生产开关。

| 顺序/边界 | 实现 | 实际验证 |
|---|---|---|
| 无特性 | `V3LinearNOAttention.forward` 返回 `super().forward(x)` | 六 variant、CPU双精度/单精度、CUDA三精度：输出/公共梯度/dropout RNG 逐位一致 |
| structured reshape | 原 N=H*W 检查、Conv2d、head layout | 展开 patch oracle、非法 N、非 contiguous 输入 |
| Q/K/V 与增量 | 各一次原投影，第二轮两端 delta 加到原 logits | 六variant×四消融×两轮，全部输入/参数 VJP；adapter首轮无调用/梯度 |
| 温度 | temp/conv_temp clamp `[.01,1]`；Car `tempreature_*` clamp `[.1,2]` | 内区间/越界夹紧；错误“先温度后delta”oracle必须与正确结果区分 |
| softmax | Q=-1(M)、K=-2(N) | 逐点/逐slot显式 exp/sum oracle，以及 ATen 轴跟踪 |
| latent位置 | KtV后、QC前；两轮同一个原v2 LatentContextFFN | 对照raw/context/readout，非零W2，merge/split和所有参数梯度 |
| Air特殊性 | features.contiguous；dead temperature、scale、softmax保留 | to_q pre-hook检查；改变dead temperature输出不变且其梯度None；惰性dropout/softmax不调用 |
| 访问计数 | 仍为两次原einsum | 每visit恰一次Q/K/V/KtV/QC；BMM形状仅 `[Bh,M,N]@[Bh,N,dh]` 和 `[Bh,N,M]@[Bh,M,dh]` |
| native数学冻结 | active路径仅两个特性插入 | 精确移除use_adapter/use_latent两段后，剩余计算体AST与原forward相同；原构造器对象相同 |

无 N×N/M×M attention；latent 混合 token 的通道/head，不混 M。未加入诊断缓存；全部 features/logits/Q/K/V/context 是局部变量。无新 buffer、history 或 detach。weakref、连续独立反传、不同 batch、非法输入/错误 processor 后再次调用、state 不变检查通过。

## D. Oracle 独立性和先失败证据

`attention_oracle.py` 不导入生产 attention/adapter/helper，不调用被测 forward。Conv 使用 unfold 和展平核矩阵乘法；按 batch/head 循环计算；adapter先组成 BA矩阵而非生产两级linear；softmax为稳定exp/sum；KtV/QC为逐head矩阵乘法。latent复用LAA2独立数学oracle的手动均值/方差/LN affine/erf GELU，未复用生产helper。

oracle自检两项先通过：2点/2slot/非零B的手算概率矩阵，以及非零W2的GELU(1)常数更新。之后在生产 `v3/attention.py` 尚不存在时运行12个目标方法，记录428个预期 `ModuleNotFoundError` 子项；hash与不存在性见 `test-first-manifest.json`。随后才新增生产代码。最终另外补充两项自审测试：公共AST一致、不同实例独立及processor失败局部性，共16个方法。

首轮实现验证14个方法出现244个错误子项，全部来自**新测试夹具** `fill()` 忘记将一维确定性权重reshape成参数shape，尚未进入数学比较；只修正该新fixture的reshape，未改生产数学或任何容差。第二轮14/14通过（11.304s）；两项补充自审加入后最终16/16通过。原始失败和成功日志均保留。

## E. 实际命令、环境和结果

Python3.13.9、Torch2.13.0+cu130、NumPy2.2.6、PyG2.3.1、timm1.0.28，Linux WSL2，RTX5090 Laptop GPU。验证进程使用 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；最终runner显式Torch线程1。CUDA oracle矩阵临时禁用TF32并恢复原值；autocast保持启用，无全forward强转FP32。不是远端Python3.10/torch2.11/cu128验收。

| 实际命令（根目录） | 结果 | 耗时 |
|---|---|---:|
| `PYTHONPATH=tests:. python -B -m unittest -v loop_linearno_latent_adapter.test_laa3_oracle` | 生产实现前2 passed | 0.489s |
| `PYTHONPATH=tests:. python -B -m unittest -v loop_linearno_latent_adapter.test_laa3_attention loop_linearno_latent_adapter.test_laa3_amp` | 实现前12方法，428预期缺模块子项 | 0.090s |
| `PYTHONPATH=tests:. python -B -m unittest -v loop_linearno_latent_adapter.test_laa3_oracle loop_linearno_latent_adapter.test_laa3_attention loop_linearno_latent_adapter.test_laa3_amp` | 首轮fixture244错误；修正后14/14通过 | 3.814s / 11.304s |
| `python -B docs/loop_linearno_latent_adapter_audit/laa3/run_checks.py new` | **16 passed / 0 failed / 0 error / 0 skipped** | 13.843s含加载 |
| `python -B docs/loop_linearno_latent_adapter_audit/laa3/run_checks.py previous` | **LAA1/LAA2 52 passed / 0 failed / 0 error / 0 skipped** | 7.084s含加载 |
| `python -B docs/loop_linearno_latent_adapter_audit/laa3/run_checks.py regression` | **旧定向72 passed / 0 failed / 0 error / 0 skipped** | 51.626s含加载 |

旧72项覆盖pure attention官方源码数值对照/结构、PointDepthAttnRes、v1 SR/RB/LB、LL9R、v2 core/latent。oneDNN对Intel GPU支持的提示保留，非测试失败。全仓689项未重跑；LAA0历史644通过/9既有失败/36跳过仍为此前全量证据，不能把本轮定向通过表述为全仓全绿。

CPU完整oracle矩阵为 **6variant×4消融×2轮×2精度=96行**；记录所有中间量及全部输入/参数VJP。CPU FP64最大绝对差4.440892098500626e-16；FP32为3.5762786865234375e-7。预设容差分别3e-12/3e-10和4e-6/5e-5（atol/rtol），均通过。

CUDA完整oracle矩阵 **144/144通过**；每行包含非零B/W2（启用时）、输入/参数VJP、AdamW一步、strict reload。另144个zero-init/dropout组合通过；其中feature-off公共梯度逐位一致，active零初始化公共梯度按明示容差比较，不能把后者说成跨精度逐位一致。

| CUDA精度 | 行数 | forward最大绝对差 | VJP最大绝对差 | atol/rtol | reload误差 |
|---|---:|---:|---:|---|---:|
| FP32 | 48 | 2.51420e-8 | 1.76348e-8 | 4e-6 / 5e-5 | 0 |
| FP16 AMP | 48 | 1.79527e-4 | 8.53413e-5 | .004 / .03 | 0 |
| BF16 AMP | 48 | .00170429 | .000975569 | .04 / .08 | 0 |

CUDA小输入B2/N6/h2/dh3/M5；ShapeNet独立M另有CPU H208/h8/dh26/M5和M32的实际前向/反传。没有使用GradScaler，未把这些测试描述成完整任务AMP或续训验收。耗时仅为测试命令运行时间，不是模型效率指标。

## F. 冻结与优先自审

`end-freeze.json`：LAA3起点4164文件中，仅允许独立状态文档追加，其余保持内容，无删除；LAA0起点3377文件全部不变。LAA1/LAA2的生产代码、测试、证据与用户原diff保持。`static-checks.json`保存新增Python内存compile、既有130个shell的`bash -n`和`git diff --check`实际结果。新增代码与状态diff及公式位置见 `source-diff.patch`、`source-map.json`。

1. **off/native兼容：PASS**。继承原构造器，直接委托原forward；active去除特性后AST相同；CPU/CUDA输出、off梯度和dropout RNG逐位一致。
2. **插入位置/温度/轴：PASS**。96行CPU逐中间量oracle、错误温度顺序辨识、BMM/softmax事件证据。
3. **所有权与初始化：PASS（attention范围）**。同实例两轮latent复用、不同实例独立；公共key/value与后续RNG四消融一致；未来core/wrapper调度仍未实现。
4. **AMP/state：PASS（小型attention范围）**。144个非零特性CUDA组合前反向/AdamW/reload通过；无持久激活；异常恢复正常。
5. **冻结与阶段边界：PASS**。旧attention/原语/残差/入口/配置未改；仅独立attention与测试，不组装loop。

没有需要用户裁决的数学冲突。NOT RUN：LAA4及后续、完整v3 loop/wrapper/metadata checkpoint集成、真实数据/完整epoch/三seed/精度/SOTA、真实延迟/显存/epoch效率、远端环境、完整任务AMP/scaler续训、compile/distributed。完整checkpoint版本隔离沿用LAA1合同，尚非本阶段原语state_dict reload的验收范围。

本 LAA3 阶段结束，未执行下一阶段。
