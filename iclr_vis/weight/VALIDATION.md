# 验收记录

2026-09-26，HEAD `96e3cac165f48ee66d25f547c1e069b5255f675c`。
状态：**PASS（脚本实现和本地合成验收）**。远端用户权重/真实数据出图 **NOT RUN**。

## A. 范围

新增独立V5 Airfoil最后一层Q/K可视化工具，默认定位用户给出的
P1C3R2S1/E4/M64/seed0 run。读取成对checkpoint和数据，eval前向提取，导出64状态空间图。
没有训练、optimizer、权重写回、安装依赖或改变任何模型/任务语义。

## B. 新文件

- `airfoil.sh`：可从任意cwd启动、支持指定Python解释器的薄入口。
- `airfoil_states.py`：保存metadata驱动的严格加载、只读数据校验、真实forward临时hook、科学排版和原始数组导出。
- `test_airfoil_states.py`：独立标量oracle、输入/文件负例和新进程端到端合成验证。
- `README.md`：指定run的远端命令、字段含义、文件列表、覆盖参数。
- `REFERENCE_NOTES.md`：实际阅读图注/方法及原图的依据，论文原图与本工具的语义区别。
- 本报告与 `evidence/`：原始执行结果。

全部修改均为 `iclr_vis/weight/` 中新增文件。开始时的未跟踪
`partial_share_feature_gate_v5_latest.tar.gz`、`pic_reference/` 保留。

## C. 数学到源码

- `capture_last_attention`：选择 `model.loop.suffix[-1].Attn` 的唯一visit，
  hook原始to_q/to_k/to_v输出及to_out输入。Q softmax轴=-1，K=-2；温度和clamp来自该真实模块。
- 独立验证 `Q(K^T V)` 与原to_out输入；不构造N×N/M×M注意力，不改变模型调用。
- 同样本无hook和有hook预测逐位一致；检查参数版本、Torch CPU/CUDA RNG、finite和概率归一。
- `load_sample`：保存checksum核对后只读mmap，测试index0→原index1000，目标channel4，float32和原网格点序。
- `structured_mesh`：仅每个原四边形单元分成两三角形，排除退化面；没有全局三角剖分填孔。
- `display_values`：shared Q原值 / shared N*K；peak W/max_i(W)，不用每格min-max。
  同一图共用色标；full/near先对完整N归一，后设置视窗。
- `atlas`：full默认4×16，near默认8×8；latent0–63保持原序；每head独立。

## D. 实际验证

环境：Python3.13.9、Torch2.13.0+cu130、NVIDIA GeForce RTX5090 Laptop GPU。
原始日志分别保存，并没有把分开执行的命令包装成一次全仓回归。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s iclr_vis/weight -p 'test_*.py' -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider \
  tests/loop_linearno_v5/test_v5_contract.py::test_visit_operator_matches_native_linearno \
  tests/loop_linearno_v5/test_v5_visit_norms.py

bash -n iclr_vis/weight/airfoil.sh
git diff --check
git diff --exit-code -- cdlno linearno_loop PDE-Solving-StandardBenchmark \
  Airfoil-Design-AirfRANS Car-Design-ShapeNetCar
```

- **6 passed / 15.463s**，0 failed / 0 skipped，见 `evidence/tests-first.txt`。
  包含4种Standard variant的独立Python点积/exp/分母oracle及非均匀温度/clamp；
  CUDA真实前向；原网格孔洞检查；样本/通道/哈希负例；新进程strict checkpoint→绘图。
  新进程测试使用带空格run/data/output路径，导出8组PNG/PDF总图及场图，检查NPZ概率和head维度。
  预览在新进程证明没有torch导入、不需要数据目录；已有输出拒绝覆盖；篡改weights被strict pair检查拒绝。
  所有原run文件前后SHA-256一致。
- **22 passed / 3.76s**，见 `evidence/v5-regression.txt`：既有V5 native parity和visit-independent norm合同。
- **完整尺寸GPU合成前向PASS**，见 `evidence/canonical-gpu.txt`：
  P1C3R2S1、E4/F128、H128、heads8、M64、221×51、1,623,713 trainable参数。
  两次eval，捕获shape `[8,11271,64]`；hook前后预测逐位相同；原生readout最大误差 **0**；
  Q归一最大误差 **2.384185791015625e-7**，K为 **4.172325134277344e-7**。
  该配置是本机构造的同形随机初始化模型，不是用户远端checkpoint，不声称config hash相同。
- Python源文件用内置 `compile(source,path,'exec')` 检查通过；shell语法、diff检查通过；新增源码/文档无行尾空白。
- 生成并实际打开65×17合成环形网格的full/near/PDF配套PNG，检查标题、64编号、共享色条、孔洞、留白。
  合成图标明 `SYNTHETIC CHECK`，仅用于排版验收，不是模型真实物理状态证据。
  临时审图目录 `/tmp/v5-weight-visual-review-ofkxm3hw/figures/`，不依赖它运行脚本。

失败/限制：新测试与针对性回归没有失败。参考下载中Transolver++整本PDF超时，
并发整本下载停滞后终止；已通过既有HTML与本次成功获取的官方图像完成相关参考核对。
未安装任何依赖，没有用下载失败代替已完成阅读的声明。

NOT RUN：远端Python3.10/Torch2.11栈、用户训练权重实际加载、真实Airfoil NPY读取与最终图片质量、
真实完整测试集指标、真实训练/收敛。没有运行全仓回归；本次无生产代码修改。

## E. 冻结证据

开始与结束tracked `git diff`均为空。模型、config、训练入口、旧checkpoint代码均未修改。
没有reset/clean/commit/push，没有修改用户给出的参考PNG或旧证据。
脚本运行不写用户run/data目录，输出唯一新目录；pytest使用临时synthetic checkpoint。

## F. 已完成自审与剩余边界

已复核5项：

1. 最后一层是最终suffix attention，唯一visit命中一次，而非core第二轮或最终Linear head。
2. Q/K轴和各自温度来自真实forward，标量oracle与native readout双重核对。
3. 样本/点序/channel4固定为任务实际协议，网格连接不跨越翼型孔洞。
4. 图示缩放没有变成模型归一化；peak图不能支持跨latent绝对强度结论，原值和统计完整保留。
5. 指定run不会退回别的模型，sidecar及数据校验严格，源run字节未改；输出路径含空格和重复运行可处理。

没有发现待用户决定的实现问题。远端文件是否完整、训练所得状态的实际纹理，需用户在提供的路径运行后确认。

## 后续字体/终稿尺寸复核（2026-09-26）

**状态：PASS（本地脚本与合成排版检查）；最终投稿页面/真实训练权重图 NOT RUN。**
上文初版功能验收不证明4.3–6.3pt小字适合论文最终印刷。用户提出字体要求后实际复查发现这一
问题，已修正。不能把“顶会水平”当成可以由脚本单独保证的认证。

### A–C：修改范围与对应

仅修改本目录 `airfoil_states.py`、README、REFERENCE_NOTES、本报告并新增纯文本证据；
`test_airfoil_states.py`、shell入口和所有生产源码未改。

- parser/validate：`--paper iclr|icml`解析为官方2026模板的5.5/6.75英寸正文宽度；
  `--width`仍可显式指定最终宽度；`--annotate`用于浏览。
- style/atlas/field_reference：全部图内文字统一9pt DejaVu Sans，字体嵌入，线宽0.5pt，
  atlas编号放在独立白色行中，避免遮挡分布；真实图默认去总标题。
- save_figure：取消tight裁切，保留确切物理页宽，避免在LaTeX中意外缩小字体。
- write_latex：新增输出 `paper_figure.tex`，ICLR用figure、ICML用figure*，
  由官方模板的原生caption控制正式图注字体/字号/间距。没有安装或依赖TeX进行绘图。
- main：metadata记录字体、页宽、模板选择等；新增的标签字符串不改变任何Q/K/数据计算。

### D：命令与结果

最终再次执行原6项脚本测试：**6 passed / 11.070s，0 failed / 0 skipped**，
原始日志 `evidence/tests-typography-final.txt`。此前一次字体修订验证为6 passed / 11.413s，
保留在 `evidence/tests-typography.txt`；两次是独立运行，没有合并为全仓回归。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s iclr_vis/weight -p 'test_*.py' -v

# 复用初版合成fixture，另建输出，旧图不覆盖。
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash iclr_vis/weight/airfoil.sh \
  --device cpu \
  --run-dir '/tmp/v5-weight-visual-review-ofkxm3hw/run with spaces' \
  --data-path '/tmp/v5-weight-visual-review-ofkxm3hw/data with spaces' \
  --output-dir '/tmp/v5-weight-visual-review-ofkxm3hw/figures-typography-iclr-final' \
  --dpi 250

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash iclr_vis/weight/airfoil.sh \
  --device cpu \
  --run-dir '/tmp/v5-weight-visual-review-ofkxm3hw/run with spaces' \
  --data-path '/tmp/v5-weight-visual-review-ofkxm3hw/data with spaces' \
  --output-dir '/tmp/v5-weight-visual-review-ofkxm3hw/figures-typography-icml-final' \
  --dpi 250 --paper icml
```

在同一本地Python/Torch环境中，用已有PyMuPDF读取两套共18份PDF，验证：

1. 页宽分别严格为5.5/6.75英寸（误差阈值1e-5in）；场图也遵循选定宽度。
2. 所有可见文字span均为9pt（阈值1e-4pt），字体为嵌入TrueType；字体数据确实存在。
3. 所有文本bbox位于页内，每张atlas的00–63编号全部可检索。
4. 两种生成tex均使用对应figure环境和实际宽度；caption未自行覆盖字体。
5. 与初版输出逐项比较NPZ中所有数组，**全部逐位相同**。

详见 `evidence/typography-pdf-check.json`。实际打开ICLR full和ICML near PNG，确认编号留白、
颜色条、孔洞和边界布局；此前也打开场图。图中显著标记SYNTHETIC CHECK，不能当作真实结果。

失败/限制：脚本测试、字体/PDF检查无失败。参考链接最初404、raw GitHub TLS/read超时均如实记录在
REFERENCE_NOTES，最终通过官方ZIP和GitHub API读取正确模板。
**NOT RUN**：本机无pdflatex/tectonic，未在官方TeX环境编译完整论文；远端真实数据/已训练权重、
真实纹理以及缩排进用户论文后的最终页面审查未运行。不新增“真实图已达顶会水平”的结论。

### E–F：冻结与自审

tracked git diff仍为空；仅独立绘图目录内文件变化。模型、数据协议、checkpoint实现均未动，
原始Q/K/预测数组逐位相同。自审重点：模板要求与自选9pt区分；实际页宽/嵌入字体；
64个编号不遮挡分布；正式caption继承模板；没有以合成图冒充真实权重结果。
