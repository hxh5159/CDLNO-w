# V5 FFN 可视化交付记录

2026-09-26；基线 main@`1191dc67d9ba512483649b78c78f50c659d280f8`；起点工作树干净。

## A. 授权范围与完成状态

**PASS（实现和本地合成验收）**。新增Airfoil、Darcy、Elasticity、NS、Pipe五个只读绘图入口，
读取既有V5 checkpoint及对应任务数据；未训练真实数据、未修改模型。
远端脚本目录为 `looplin-ffn`，输入实验仍在 `looplin-v5-final`。
NS仅提供了实验父目录，必须先 `--list-runs` 再选择明确的run；不自动猜最新/最好。

## B. 新增文件与原因

| 文件 | 职责 |
|---|---|
| `airfoil.sh/darcy.sh/elasticity.sh/ns.sh/pipe.sh` | 五个薄入口，完整转发参数与退出码，从任意工作目录定位脚本。 |
| `ffn_states.py` | 单一CLI、metadata-first、真实任务读取/normalizer复用、strict pair加载、NS预测反馈、输出和元数据。 |
| `capture.py` | 临时hook观察所有逻辑访问的router/专家；逐block验证残差、预测/RNG/参数不变；异常清理。 |
| `render.py` | 各专家空间权重、轮间差值、加权输出范数、熵、均值热图和正式LaTeX caption。 |
| `test_ffn_states.py` | 独立NumPy数学检查、错误注入、CPU/CUDA、E1/E8、R3、任务推理、预览边界。 |
| `verify_synthetic.py` | 可复跑的五任务新进程合成checkpoint/绘图/PDF验收，不依赖真实数据。 |
| `README.md`、本文件、`evidence/` | 绝对路径命令、输出语义、失败与验收记录。 |

全部新增位于 `iclr_vis/ffn/`；复用已有 `iclr_vis/weight`，不修改它。

## C. 公式、形状与原实现对应

生产真值：`cdlno/linearno_loop/v5/core.py::V5PhysicalBlock.forward`：

```text
z = x + Attn(LN1_visit(x))
u = LN2_visit(z)
pi = softmax(router_visit(u), dim=-1)             # [B,N,E]
result = z + scale * sum_e(pi[...,e] * Expert_e(u))
```

core的scale=1/R，prefix/suffix为1。所有专家执行，expert bank只在同一core跨轮共享，
router按visit独立。观察器分别hook每个visit的router、对应LN2的输入和共享专家输出，
从真实执行次序恢复[L,N,E]（L为executed depth），不按attention head拆分。
末尾block通过final LN的输入校验head之前的残差，不拿物理输出通道去比较hidden状态。

`capture.py::summarize`：raw entropy=-sum(pi log pi)，E>1时才除log(E)。
E1权重严格为1、raw entropy为0，归一化熵不定义；数组/图不输出该指标，JSON为null。
贡献范数为 `||scale*pi_e*Expert_e(u)||_2`，包含core的1/R；保存未加权专家范数和混合更新范数。
这不是因果归因，不假设不同方向向量不会抵消。

`render.py`：gate固定[0,1]；later−first visit固定[-1,1]；贡献图共用所有捕获值的最大值。
不逐专家peak拉伸、不混合独立block的专家身份。平均值按原节点等权，未冒充面积/体积平均。
图中只保留Expert A/B等列身份和Visit行身份，不加子图序号。

Times风格、标签9pt/刻度8pt、ICLR5.5in/ICML6.75in、0.5pt线宽、嵌入TrueType，
复用[前次论文/模板核对](../weight/TIMES_STYLE_REVIEW.md)。正式caption由官方模板排版。

## D. 实际命令、环境、结果

Python3.13.9、Torch2.13.0+cu130、CUDA可用、NumPy2.2.6、SciPy1.16.3、Matplotlib3.10.6。
PyMuPDF用于本地检查PDF；没有新增安装依赖。远端无需PyMuPDF即可运行绘图入口。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s iclr_vis/ffn -p 'test_*.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s iclr_vis/weight -p 'test_*.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B iclr_vis/ffn/verify_synthetic.py --report iclr_vis/ffn/evidence/render-delivery.json
```

| 验收 | 实际结果 |
|---|---|
| 新观察测试 | **8 passed，0 failed/errors/skipped**，8.381s；[日志](evidence/tests-second.txt)。涵盖四Standard attention变体、非零router、E1/E4/E8、custom R3/shared norms、原生四任务解码/NS十步反馈、CPU及CUDA FP32。 |
| 旧Q/K绘图回归 | **14 passed，0 failed/errors/skipped**，70.188s；[日志](evidence/weight-regression.txt)。这是独立的一次旧回归，不称作合并的单次22项执行。 |
| 五任务新进程真实CLI合成档案 | **5/5 PASS**，路径含空格、/tmp工作目录、strict加载、同一样本、原normalizer、原实验所有文件hash不变；已有输出目录被拒绝覆盖。 |
| 最终PDF审查 | **113/113 PASS**，Airfoil38、Darcy15、Elasticity/NS/Pipe各20；真实STIXGeneral嵌入，字形字号只有8/9pt、请求页面宽度、文字未越界；[日志](evidence/render-delivery.txt)、[逐文件JSON](evidence/render-delivery.json)。 |
| 前向透明性 | 五任务无hook/有hook预测逐位相同；Torch/NumPy/Python RNG保持，hook清理；逐block残差最大误差0。专家概率行和最大误差1.7881393432617188e-7，检查阈值5e-6。 |
| 独立数学 | NumPy矩阵乘、显式exp归一化及erf形式GELU对照生产权重和输入；不调用被测专家forward充当expected。贡献范数atol1e-7/rtol2e-5；概率atol1e-7/rtol1e-6。 |
| 判别性 | 人为给真实block输出加0.1，残差校验确实失败；router注入NaN明确拒绝；异常后临时hook不残留。不是mock掉数学来通过验收。 |
| E8版式 | 专家网格每页最多4列，均值热图为旋转标签预留底部空间；[单独检查](evidence/eight-expert-layout.txt)通过。 |
| 静态/冻结 | Python compile、五个shell的bash -n和真实--help、无旧文件diff、新增文件whitespace检查通过；见[最终记录](evidence/static-delivery.txt)。 |

实际打开查看了Airfoil近景、Pipe空间图、Elasticity真实节点图以及平均权重热图。
合成渲染验收用100dpi以控制产物大小；PDF字体/印刷宽度独立于DPI。用户CLI默认400dpi，
未把合成随机纹理当作真实物理专门化结果，也未声称已验证真实权重的画面。

失败没有删除：

1. [首次新测试](evidence/tests-first.txt)：7个方法中一个方法的四个variant子例因测试调用漏传必需的 `fx=None` 报TypeError；修正新测试调用，未改生产接口/容差。最终8方法通过（追加R3/E8覆盖）。
2. [首次PDF检查](evidence/render-first.txt)：均值热图色条说明略超页面下边界；提高色条位置修复，未放宽边界断言。修复后的[113份检查](evidence/render-final.json)通过；补充E8留白后又完成最终交付版113份全检查。
3. [首次静态检查](evidence/static-freeze.txt)：README末尾多一个空行；删除额外空行后，[第二次检查器](evidence/static-final.txt)又把 `git diff --no-index` 正常表示存在新增内容的退出码1误当检查失败。最终按“0/1且无whitespace诊断”为成功判定；错误诊断退出码3仍拒绝。Python/shell/冻结检查各次均通过。
4. E8单独版式探针出现本机Qt wayland插件提示，但正常完成PDF检查；正式绘图入口明确用Agg，无GUI要求。

## E. 冻结与优先自审

1. **归一轴及专家语义**：观察真实router输出、只在E上softmax；独立oracle覆盖，未把Q/K或attention head误当专家。
2. **共享/轮次**：首尾单visit，core多visit；代码次序与事件逐项匹配，R2/R3以及共享/独立norm均有实例证据。物理bank之间不做专家语义对齐。
3. **推理协议**：复用旧真实读取/校验/normalizer；NS用预测反馈并以独立完整rollout逐位比较，没有未来真值回填。
4. **记录透明性**：原模型输出/参数/RNG不变、临时hook异常也移除；5个输入run所有文件hash不变。只存detached数组，不持久保存图。
5. **视觉诚实性**：0–1固定gate色标、E1不造对比、全域归一后才裁视窗、真实节点/原网格连接、字号不随小图数缩小，正式caption不被脚本覆盖。

`git diff --exit-code HEAD -- .` 确认所有已跟踪文件与基线一致；只有新增 `iclr_vis/ffn/`。
模型/纯LinearNO/V1–V5/训练入口/checkpoint/旧绘图/旧测试/依赖完全未改。

## F. 未验证边界

**NOT RUN**：远端目标环境、用户真实checkpoint/数据出图、训练中在线采集、AMP/半精度观察、
真实训练/收敛、多个样本或整个测试集的专家统计、完整论文LaTeX编译与最终页面目视验收。
本工具是选定checkpoint/样本的离线FP32分析，不自动声称专家重要性、物理分工或提升。
没有发现需要用户决定的实现问题；NS具体run由用户选择。

本阶段结束，未执行下一阶段。
