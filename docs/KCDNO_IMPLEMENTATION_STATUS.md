# KCDNO 实施状态

## K0完成，待审查（2026-09-15）

当前授权仅K0，未执行K1及以后阶段。当前旧仓库HEAD为`9f72946e0adbfe27ba0b75b1646fa697ae16d3df`（main）；起点工作树干净。KCDNO规格文件存在且已读，**尚无KCDNO生产配置/模型/factory/任务入口**。

交付：[参考审计](KCDNO_REFERENCE_AUDIT.md)、[回归夹具索引](kcdno_audit/fixture_index.json)、[冻结基线](kcdno_audit/baseline.json)、[实际命令](kcdno_audit/commands.txt)。本轮只新增这两个文档与`docs/kcdno_audit/`内审计材料；旧CDLNO/A/V阶段文档、AGENTS、memory和全部生产/测试文件保留。

### A. 范围与结论

已审查KCDNO规格、CDLNO v1.2与当前状态、front A0–A4、最新输出/V1边界；建立部件复用映射、两FFN插入点、八任务字段/损失/入口/目录/保存合同及K1–K10最小建议范围。K1–K10独立执行指南当前未找到，分工是建议，后续用户明确点名内容优先。

现有Down/Up、RMSNorm、PlainFFN、dense ConvFFN可供新类组合。新history必须位于FFN1之后/FFN2之前；不调用整个旧no_sa block后补history。旧CDLNO继续F/P、bridge、persistent rear、CDPA和HF readout；新KCDNO独立L个点域block，不修改旧family或类路径。

### B. 新增文件与差异

| 文件/产物 | 用途 |
|---|---|
| `docs/KCDNO_REFERENCE_AUDIT.md` | 完整A–F审计、版本/来源/许可、代码映射、八任务接口、未完成项和后续建议 |
| 本文件 | 独立K阶段状态，不覆盖旧CDLNO/A/V记录 |
| `docs/kcdno_audit/make_regression_fixtures.py` | 独立capture/replay脚本；实际wrapper、原模型、三原cwd新进程、同权重/梯度/加载 |
| `docs/kcdno_audit/audit_static.py` | parser AST、安全CLI/注册/预设/函数摘要和冻结hash核查 |
| `baseline/environment/presets/entry-contracts/cli-*/capture*/replay*/fixture_index/freeze`等JSON/log | 真实命令/数值/版本/源码证据与失败记录 |
| 外部`/home/hwz/CDLNO-artifacts/k0-before-zf3l4cve` | 302文本源码快照、完整入口AST、41份权重/输入/输出/诊断梯度与工业对象/list，约16.26MiB最终夹具；不进入git |

无生产修改，无新kernel/history数学实现，无训练/数据/依赖/既有测试修改。git可审查diff全部为新增审计材料。全328个原tracked文件保持SHA256一致，见[freeze.json](kcdno_audit/freeze.json)。

### C. 公式/轴与参数归属

新KCDNO待实现：每层`point RMS→Down→U=S+FFN1(RMS1(S))→Uhat=history(U,tuple(past))→T=Uhat+FFN2(RMS2(Uhat))→Up→点残差/PointModule`，最终LN/head。S/U/T均[B,M,d]。

源s拥有Wk_s，缓存`K_s^T T_s [B,r,d]`和`sum_token(K_s) [B,r]`，raw T作value；接收l拥有Wq_l，Q[B,M,r]一次读取全部先前来源；来源softmax权重[B,M,l]，w0、gamma.1，`Uhat=U+gamma*(Σalpha*rawR-U)`。无M×M token softmax/current self-read/Wv/Wo。L8主版应7写7Q28读。以上是后续验收合同，K0不标为实现通过。

### D. 实际验证

| 项目 | K0结果/范围 |
|---|---|
| 捕获/回放 | **41/41通过**：24任务×front模式、9核心front×CDPA组合、8原Transolver模型；严格同权重/输入，输出/诊断梯度atol=rtol=0 |
| CPU合成 | 33份CDLNO：实际wrapper/core、FP32输出+诊断backward；无optimizer/原loss训练 |
| GPU实测 | 8份原Transolver模型，直接原源码CUDA路径，FP32；这是旧模型基准，KCDNO/GPU新模型检查未运行 |
| 真PyG对象 | Car/Air分别三CDLNO+原模型，单图N11/19；真实Data/Batch、输出/诊断梯度/整对象及Air list；不含radius_graph/真实数据 |
| checkpoint | 24份CDLNO实际StaticRun/CarRun/AirRun；same mode、chunk0→1、eval省略mode恢复、显式冲突拒绝、sidecar字节不变，全通过 |
| CLI/元数据 | 3/3子项目进程，八任务默认/老注册、24组三模式train/eval解析，8 JSON+Air YAML留存 |
| 旧冻结回归 | 既有7项AST/hash检查**7/7通过，1.221s**；未修改其逻辑，未重跑无限扫描 |
| 导入方式 | 三cwd显式PYTHONPATH均通过；去掉PYTHONPATH均ModuleNotFoundError，当前未editable安装；如实记录为当前本地安装限制 |
| 全部现有文件 | 328份tracked文件字节相同；所有新增只在K0 docs区域 |

[环境](kcdno_audit/environment.json)：Python3.13.9、torch2.13.0+cu130/CUDA13.0、RTX5090 Laptop、PyG2.3.1；无torch_cluster/pyg-lib。SDPA=MATH，FP32、单CPU线程、TF32/AMP/compile关，cuDNN benchmark关/deterministic开；只在审计子进程设置，未修改生产默认。目标仍为远端Python3.10/Torch2.11/cu128（3.11候选），未远端验证或安装任何依赖。

实际主命令：

```bash
python -B docs/kcdno_audit/make_regression_fixtures.py capture --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/kcdno_audit/capture.json
python -B docs/kcdno_audit/make_regression_fixtures.py replay --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures --result docs/kcdno_audit/replay.json
python -B docs/kcdno_audit/audit_static.py --artifacts /home/hwz/CDLNO-artifacts/k0-before-zf3l4cve/fixtures
python -B docs/kcdno_audit/audit_static.py --freeze
git diff --check
```

capture已完成，现有目录拒绝覆盖。后续只能重放同一份旧权重/输入；三进程完整命令和7项既有冻结测试命令见commands.txt。初版审计脚本错误已在本轮自身修正，最初22份不完整捕获不算最终验收；记录见execution-notes.json。

### E. 冻结及旧缺项

本轮未修改原Transolver/CDLNO核心、三mode/CDPA历史/初始化、八任务数据/采样/normalizer/loss/时间循环/optimizer/scheduler/指标、checkpoint实现、旧脚本或依赖。7项既有审查恢复原Transolver基线AST/字节合同；K0的328文件hash作后续新family接入的直接前基准。

实际A3/A4已存在且可用，不能照用户粘贴AGENTS旧段误写未实现。另行V1完整归档/绘图基础已存在，V2–V5任务resume/周期图未接入；普通model.pt/整模型文件不含完整optimizer/RNG续训状态。K0不自动补这些旧缺项。Car日志变量名/阻力路径、Air旧MAE分支及旧parser别名问题分别记录，未归为KCDNO回归或擅自修复。

### F. 已自审和未验证

已自行复核：两FFN与Up norm位置、writer/reader参数归属和旧CDPA隔离、真实git/阶段/目录状态、同权重回放与pickle稳定路径、源码冻结与验证边界。当前没有需要先重写旧baseline的实质冲突。

未验证：KCDNO模型与数学（尚未实现）、KCDNO八任务入口、远端Torch2.11/cu128、新模型AMP/性能；真实文件完整读取、采样/物理指标、训练、收敛、准确率和epoch效率。K0诊断loss不是原任务训练链路；当前合成夹具不证明全部历史checkpoint兼容，保留pre-A1真实旧夹具供后续。

| K阶段 | 状态 |
|---|---|
| K0 | 审计/独立夹具/CLI与冻结证据完成，待审查 |
| K1–K10 | 未实施；后续必须按用户单独点名的阶段执行 |

本K阶段结束，未执行下一阶段。
