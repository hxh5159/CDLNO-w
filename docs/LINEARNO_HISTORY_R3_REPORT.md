# R3 — LinearNO latent-summary AttnRes

## A. 范围、来源及审计

**PASS（R3 范围）。** 只实现创新 A，未实现 K-conditioning，未接 benchmark parser/factory/launcher，未执行真实数据、GPU 或长训练。当前 checkout 的纯 LinearNO 是基座；新增机制是本项目的 AttnRes-inspired 适配，不是官方 LinearNO、CDPA 或 Kimi 官方模型。

根目录 `/home/hwz/CDLNO`，branch `main`，HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`，tree `d74a1b07aa405009992879847aca0c48c75c31f0`。适用指令为根 `AGENTS.md`，本轮用户 R3 授权优先于其历史阶段说明。执行前记录了完整 tracked/untracked/ignored 文件清单、SHA-256、status、tracked/staged diff，并复制了已有非 ignored 源文件：

`/home/hwz/CDLNO-artifacts/linearno-history-r3-before-vb2brm6s/`

实际重读：研究提示词 §3.2、§3.4–3.5、R3；R1 schema/fair-initialization fixture；R2 core/context、三种 wrapper、测试与状态；纯 attention 初始化及旧测试入口。固定官方来源沿用已审查的 R0/L0–L10 证据，未重新下载来源或 checkpoint。

关键只读命令为 `rg --files -g AGENTS.md -g CLAUDE.md`、`git status --short --branch --untracked-files=all`、`git rev-parse HEAD`、`git ls-files -z` 及其 others/ignored 分类。生产 exp/main 脚本没有被 import。

## B. 决定、公式与代码

`cdlno/linearno_history/attnres.py:LatentSummaryAttnRes` 唯一拥有共享 `to_k/to_v`，并在 `receivers[1..L-1]` 注册独立 Q/O、RMSNorm scale、w、gamma。第一层没有 receiver 参数。四个投影均为无 bias `Linear(d_h,d_h)`，跨 head 共享。

对每个历史 i 单独计算：

```text
q = Wq_l(C_raw_l)
k_i = Wk(C_raw_i), v_i = Wv(C_raw_i)
A_li = softmax_history_tokens(q @ k_i.T / sqrt(d_h))
R_i = Wo_l(A_li @ v_i)
s_i = sum(w_l * RMSNorm_l(R_i), dim=-1)
alpha = softmax_sources([masked(s_0), ..., masked(s_l-1), 0_null])
H_l = sum_i alpha_i * R_i
C_tilde_l = C_raw_l + gamma_l * H_l
readout = original_Q_l @ C_tilde_l
```

`SummaryRMSNorm` 严格沿最后一维计算 mean-square，`keepdim=True`、eps=1e-6，只有 scale，无 bias。原始当前 C 不参加 source-softmax；null 的向量和分数固定为零，没有参数。未加额外点域残差。

训练时固定 p=.1，Cross/评分全部完成后只采样一次 `[B,S_real]` mask，广播 head/current-slot/channel；S=1 不采样，eval 不采样。无 inverted scaling；全真实来源 mask 时 alpha_null=1、H=0。代码没有生产 dropout 超参；私有 `_evaluate(..., _drop_mask=...)` 仅是内部测试接口，全 False mask 对应测试中的确定性无 dropout，不是正式 p=0 选项。

投影按已有普通 Linear 初始化规则 `trunc_normal_(std=.02)`，仅对 A 子树 apply 一次，然后再次置 w/gamma=0、scale=1。构造使用显式 `feature_seed` 的 CPU fork_rng，不改变全局主干 RNG，不接触 CUDA RNG；feature_seed 保存在 A 对象属性中，未来接线阶段再纳入生产 metadata。本轮不生成研究生产 checkpoint spec。

## C. 文件及差异

新增：

- `cdlno/linearno_history/attnres.py`：上述 A 算子及可关闭的按调用诊断 observer。
- `tests/linearno/attnres_reference.py`：独立、显式 sample/head/source/slot 循环 oracle；不调用生产 forward/norm/mask helper。
- `tests/linearno/test_latent_attnres.py`：13 个 R3 测试方法。
- 本报告与 `docs/linearno_history_audit/r3/` 数值/运行/冻结证据。

增量修改已有研究文件：

- `core.py`：接受研究 wrapper 持有的 A 对象，raw 形成后融合，用当前原 Q 重建；raw 仍在 block/head 完成后才入库。默认不额外算 `Q C_raw`，仅显式 observer 需要该诊断时计算；无 A 的 R2 路径不变。
- Standard `model/LinearNO_History.py`、工业 `cdlno/linearno_history/models.py`：新增内部 `AttnResModel`、`AirfRANSAttnResModel`、`ShapeNetAttnResModel`；先完整构造原主干，再初始化独立的 `.latent_attnres`。原 `blocks.*` 权重名不变。
- 研究 `__init__.py` docstring、研究 STATUS/RESEARCH_MATRIX。

`.latent_attnres` 是内部模块依赖，不是另一个用户配置开关。R1 两个 bool 仍是未来唯一配置真值；R3 不解析开关、不更改 R1 schema、不把 A-only 类注册到任何 benchmark。原纯 Model、A0K0、checkpoint loader、任务数据/损失/优化节奏及 monitor 保持原样。

## D. 实际命令及结果

在仓库根运行，GPU 显式隐藏：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_latent_attnres -v
```

实际运行时还设置了 `LINEARNO_R3_REPORT=/home/hwz/CDLNO-artifacts/linearno-history-r3-before-vb2brm6s/attnres-results.json`，最终 **13/13 passed，9.513 s**。第一次 9 项检查也通过；之后增加手算、有限差分、执行 hook、新进程加载，未放宽容差。证据：[数值摘要](linearno_history_audit/r3/numerical-summary.json)、[完整误差](linearno_history_audit/r3/attnres-results.json)、[最终日志](linearno_history_audit/r3/attnres-final.txt)。

| 检查 | 真实执行结果 |
|---|---|
| 独立 oracle float64 | forward max abs 1.11e-16；所有 forward/梯度/step 指标最大 abs 4.44e-16；atol1e-12/rtol1e-10 |
| 独立 oracle float32 | forward max abs 5.96e-8；所有指标最大 abs 1.79e-7；atol1e-6/rtol1e-5 |
| 全 mask oracle | output 与 current 精确相等，H=0、alpha_null=1；输出/梯度有限 |
| 手算反例 | 历史独立均值 R=(3,7)，null=0，w=0 ⇒ H=10/3；C=5、gamma=.5 ⇒ output=20/3 |
| double gradcheck | current、历史和活跃 A 参数全部通过有限差分（eps1e-6/atol1e-6/rtol1e-4） |
| 六变体 × L4/L8 完整模型 | gamma=0 eval：逐层/final、合成 MSE、输入/主干梯度、主干 AdamW 更新全部 atol=rtol=0 |
| strict checkpoint | A 模块及完整内部模型 state-dict；三个子项目 cwd 的独立 Python 进程非零 gamma/w 加载输出精确相同 |
| 对称性/隔离 | 历史 token 同步置换、source+mask 置换不变；current slot 置换等变；B2/H3 无混合 |
| dropout | 采样次数、p=.1 阈值、[B,S] 广播、singleton/eval 不采样、无1/(1-p)全部通过 |
| 层间 raw | 缓存 tensor identity 为 raw 而非 fused；连续 backward/B变化/异常后无泄漏 |
| 结构 | L4 恰有6份跨层 token-softmax；4次 FFN、1次最终 head；无 N×N/同层 latent SA |

隔离梯度验证从 `gamma*H` 出发，历史使用独立 leaf 张量，未使用完整主干 total loss 冒充历史分支。gamma=.7、w=0 时，Wk/Wv/Q/O/w、三份 raw 历史均非零 finite，RMS scale 梯度 L1=0；w 非零后 scale 梯度 L1=0.9440957334。无关 future tensor 的梯度为 None。完整 norm 数字见 VJP JSON。

参数量核对为 `2*d_h² + (L-1)*(2*d_h² + 2*d_h + 1)`，例如 d_h4/L4 是155，d_h4/L8 是319。Wk/Wv 在所有边上是同一对象；Q/O/norm/w/gamma 按 receiver 独立。计算仍有 O(L²) 历史边，参数不按边增长；dropout 不节省 Cross FLOPs。

旧回归实际命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest linearno.test_history_core linearno.test_history_schema \
    linearno.test_schema linearno.test_profiles linearno.test_attention_parity \
    linearno.test_attention_structure linearno.test_standard_model \
    linearno.test_standard_structure linearno.test_airfrans_model \
    linearno.test_shapenet_model linearno.test_legacy linearno.test_rng \
    monitor.test_monitor -v
```

**77 个方法：72 passed、4 GPU skipped、1 个既有失败方法（3 个 subtest 断言）；0 errors，测试运行79.383 s（含导入总时长81.516 s）。** 失败仍为 R0 已记录的 README/path.sh/旧 reproduction-matrix 快照漂移，本轮未修改它们，也没有新失败。完整输出：[regression.txt](linearno_history_audit/r3/regression.txt)。这里按方法和子断言分别计数；R2 文档之前的“59 passed”把3个 subtest 当成3个方法扣除，正确的 R2 方法计数是61 passed/4 skipped/1失败方法；原日志保留。

环境为 Python3.13.9、Torch2.13.0+cu130、PyG2.3.1、timm1.0.28、einops0.8.2，实际只用 CPU，未安装/更换依赖。此结果不代表远端 Torch2.11/cu128 已验证。

## E. 兼容性与自审

1. **纯路径**：原纯类与参数初始化、所有1242个 tracked 文件字节不变；R2 no-op 全部11项复跑通过。未把创新参数放进 A0K0。
2. **数学边界**：独立循环 oracle、手算不等长历史示例、有限差分共同核对公式，排除把历史拼成一次 softmax 或误加 current source；原 Q 重建由执行前 hook 核对。
3. **初始化/梯度**：独立 seed 不推进主干 RNG，gamma/w 在 init 后归零；分两步隔离 VJP 证明正确的零/非零梯度，未以总 loss 代替分支证据。
4. **缓存/随机性**：权威 context 仍只有 raw tuple；mask不跨调用保存。gamma=0 的 train 模式仍会采样 history mask，不能据 eval parity 声称训练 RNG 与基线相同。
5. **范围**：参数精确清单及模块种类检查排除 relation scorer、MLP/depth bias、第二点残差、同层 SA、K-conditioning、额外 loss。纯 monitor 的三个既有测试通过；研究 A 的 observer 尚未接监测 launcher。

冻结/差异证据见 [freeze.json](linearno_history_audit/r3/freeze.json)。保留用户已有 tracked/untracked/ignored 内容；未 reset/clean/stash、未 commit/push。

## F. 未执行与限制

GPU/AMP/compile、真实数据、实际任务训练/指标/收敛、远端环境、A 的 benchmark CLI 与生产 metadata-first resume、研究监测器和性能未执行。所有数值均为合成输入；full-model loss 为合成 MSE。工业输入使用真实 PyG Data 合同，不代表实际 graph loader/采样/散射验收。

本轮没有已知新增数学失败或待裁定架构冲突。已有3个历史冻结断言继续如实保留。p=0 不是正式配置，K 完全未实现。R1 的20组合/八任务生产矩阵不因本轮内部 A-only 测试升级为完成。

## G. 状态

**PASS。** 本 R3 阶段结束，未执行 R4 或后续阶段。
