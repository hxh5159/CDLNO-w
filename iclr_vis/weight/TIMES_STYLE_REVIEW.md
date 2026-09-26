# 无编号与 Times 风格绘图复核

2026-09-26；基线 HEAD：`da92c87c6deb0feac51c93051c6fc33ae6d32156`。

## A. 范围与状态

**PASS：本地绘图修改和合成验证完成。** 用户要求去掉状态图编号、重新参考论文、修正字体和字号，尤其正式图注。
Airfoil、Darcy、Elasticity、NS、Pipe 共用更新后的样式。NS 按实际 M=32 输出，不伪造64个状态。
本报告不把合成验收等同于远端真实权重绘图或完整论文终稿验收。

## B. 修改文件

| 文件 | 修改及原因 |
|---|---|
| `airfoil_states.py` | 共享字体解析/字号参数；去掉每格编号及编号留白；图内说明9pt、刻度8pt；实际字体记录；修正上边距；正式图注继续使用原生 LaTeX caption。 |
| `task_states.py` | 四任务复用相同样式；单独导出的状态图也去掉编号；更新 metadata 与 caption 描述。 |
| `README.md`、`TASKS.md` | 说明无编号布局、字体选择、图内字号与正式图注的区别、远端用法。 |
| `REFERENCE_NOTES.md` | 追加实际重读、字体提取与模板核对结果；保留先前阅读记录。 |
| 本文件、`evidence/*times*` | 保存本轮验收及最初失败，不覆盖旧证据。 |

没有改测试逻辑、模型、数据读取、checkpoint 或训练脚本。

## C. 参考与实现映射

再次查看 Transolver PDF p19 Figure10、Transolver++ 官方 Figure10 PNG、LinearNO PDF p17 Figure6，并读取相关章节/图注以及 LinearNO p18–19。
三个参考的状态网格都未显示逐格编号。字体提取原始结果见 [times-reference-review.json](evidence/times-reference-review.json)。

| 位置 | 实际读取结果 | 本次处理 |
|---|---|---|
| Transolver Figure10 正式图注 | NimbusRomNo9L-Regu；PDF 8.9664pt，对应9 TeX pt | 使用 Times 系参考；不冒称是微软 Times New Roman。 |
| LinearNO Figure6 正式图注 | NimbusRomNo9L-Regu；PDF 9.9626pt，对应10 TeX pt | 区分不同模板图注字号。图内分组标题实际为 Consolas-Bold，不能声称所有顶会图内文字必须是 Times。 |
| Transolver++ Figure10 | PNG/HTML 确认64个最后层状态及无编号布局 | PDF 部分下载不可用，字体身份 **NOT VERIFIED**，不凭截图推断。 |
| ICLR 2026 官方模板 | 示例使用 `iclr2026_conference,times`，默认图注10 TeX pt，正文宽5.5in | 默认 `--paper iclr` 与原生 caption。 |
| ICML 2026 官方模板 | 样式加载 Times，图注9 TeX pt，通栏宽6.75in | `--paper icml` 与 `figure*`；不自行覆盖 caption 字体。 |

资源链接、SHA-256、模板行号及下载失败保留在 [REFERENCE_NOTES.md](REFERENCE_NOTES.md)。TeX pt=1/72.27in，PDF点=1/72in，不能把换算差异当成字号错误。

- `airfoil_states.py::resolve_font/typography/style`：优先真实存在的 Times New Roman、TeX Gyre Termes、Nimbus Roman、Nimbus Roman No9 L、Liberation Serif；最后使用 Matplotlib 自带 Times 风格 STIXGeneral。显式指定不存在的字体时报错。控制台和 metadata 记录实际字体，PDF 嵌入字体。
- `atlas`：删除编号及其独立留白；原始状态顺序不变。`export_plots`：单图的状态号仅保留在文件名/NPZ，不画进图片。
- `write_latex` 与 `task_states.py::write_caption`：生成 `paper_figure.tex`，使用官方模板的 `\caption{}`；正式图注不栅格化进图片，不加载覆盖 caption 的字体/字号设置。
- 图内色条说明/场标签9pt、刻度8pt，属于本工具在最终宽度下的版式选择，**不是会议对所有图内字号的统一规定**。若改变论文插图宽度，应按目标宽度重新导出，不能缩图后仍声称原字号。
- 本轮没有改变 `Q=softmax_M(Lq/tau_q)`、`K=softmax_N(Lk/tau_k)`、shared/peak 显示数值或状态排序。

## D. 命令、环境与结果

本地 Python3.13.9、Torch2.13.0+cu130、Matplotlib3.10.6、NumPy2.2.6、SciPy1.16.3；CUDA可用。字体/PDF复核使用本机 PyMuPDF；未安装依赖。
本机实际字体为 **STIXGeneral**，不是微软 Times New Roman。字体文件SHA-256：
`167378031e2dddc6216d67819c9260e9a06ffc4c478e4e23cb98a6fd44b183c2`。

实际回归命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s iclr_vis/weight -p 'test_*.py' -v
git diff --check
git diff --exit-code -- cdlno linearno_loop PDE-Solving-StandardBenchmark Car-Design-ShapeNetCar Airfoil-Design-AirfRANS
```

| 检查 | 结果与证据 |
|---|---|
| 第一轮原有测试 | **14 passed**，31.767s；[日志](evidence/tests-times-no-numbers.txt)。 |
| 第一次独立 PDF 布局检查 | **FAIL，后已修复**：合成图标题字形边界略超出页面顶部（场图约0.354pt、atlas约0.834pt）。保留[原始失败](evidence/render-times-no-numbers.txt)。通过增加实际上边距修复，未放宽边界断言。 |
| 修复后的最终原有测试 | **14 passed，0 failed，0 skipped**，33.199s；[最终日志](evidence/tests-times-no-numbers-final.txt)。 |
| 五任务实际合成 checkpoint 严格加载、推理、渲染 | **PASS**；Airfoil9份PDF、其余任务各5份，共29份。没有真实训练。[渲染日志](evidence/render-times-no-numbers-final.txt)。 |
| PDF独立读取 | **29/29 PASS**：5.5/6.75in请求宽度、嵌入STIXGeneral、8/9pt文字、无状态编号、文字边界未出页面。[机器可读结果](evidence/times-no-numbers-pdf-check.json)。 |
| 字体与字号负例 | **PASS**：不存在字体明确拒绝；非法/非有限字号拒绝；明确字体与合法字号接受。[记录](evidence/times-font-controls.txt)。 |
| 静态检查 | **PASS**：目录内 Python `compile()`、全部 shell `bash -n`、`git diff --check`。 |

实际目视检查包括 Airfoil 全域Q、近景K和NS场图。合成图明确标注 `SYNTHETIC CHECK`；随机模型图案不作为真实物理状态结果。

## E. 冻结证据与自审

1. **呈现不改变数值**：五任务所有新旧 NPZ 数组逐位相同，包括Q/K、预测、坐标及NS rollout；见PDF检查JSON的 `old_new_npz`。
2. **提取/推理代码不变**：对 HEAD 比较 AST，Airfoil 的 `capture_last_attention/display_values/load_sample/sha256/structured_mesh`，四任务的 `infer/load_sample/point_paint/relative_l2/saved_normalizers/transform/verify_files` 均相同。
3. **无假字体声明**：实际字体必须存在；本机STIXGeneral及其嵌入已核实；远端有TNR才会使用TNR。
4. **正式图注不混用图内字号**：caption交由目标模板，不把8pt刻度设置应用到caption。用户应按目标年份模板编译。
5. **范围冻结**：Git修改仅限 `iclr_vis/weight/`，既有测试与容差未改，模型/任务生产目录 diff 为空。旧输出、旧证据保留。

## F. 交付与未验证边界

同步整个 `iclr_vis/weight/` 到远端 `looplin-vis`；既有绝对路径命令继续使用原 `looplin-v5-final` 下的 run。
默认已应用本次修改，无需新参数。跨机器固定字体可追加 `--font-family STIXGeneral`。
若远端已安装微软字体，可追加 `--font-family "Times New Roman"`；未安装则明确报错，不能改名冒充。

**NOT RUN**：用户远端真实权重/数据的最终出图、远端字体清单、完整论文 LaTeX 编译及缩放后的终稿目视检查。
现有证据支持实现与本地合成版式验收，不保证未经上述检查的终稿所有细节。

本阶段结束，未执行下一阶段。
