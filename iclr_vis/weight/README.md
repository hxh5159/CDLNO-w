# V5 Airfoil：最后一层 64 个 latent 的空间权重图

入口：[airfoil.sh](airfoil.sh)。只新增独立可视化工具，不修改模型、训练入口或 checkpoint。
图形参考及公式区别见 [REFERENCE_NOTES.md](REFERENCE_NOTES.md)；本地验收见
[VALIDATION.md](VALIDATION.md)。

新增 **Darcy / Elasticity / NS / Pipe** 入口及“脚本在looplin-vis、实验在looplin-v5-final”的
完整命令见 [TASKS.md](TASKS.md)。Airfoil也可以用显式 `--run-dir` 读取另一checkout的实验。

## 远端直接运行

把本目录的 `airfoil.sh` 和 `airfoil_states.py` 一起同步到远端对应目录即可。
使用训练 V5 的现有 Python 环境，无需安装额外包；需要该环境已有 NumPy、PyTorch、
Matplotlib 以及 V5 原有模型依赖。不需要 LaTeX 或 PyMuPDF（后者只用于本地阅读论文）。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final

# GPU0，测试样本0，head0；训练权重不更新。
bash iclr_vis/weight/airfoil.sh --gpu 0
```

默认 run 已指向用户指定的这次实验，基于**脚本所在 checkout 根目录**解析：

```text
output/airfoil/partial_share_feature_gate_v5/airfoil__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E4F128__M64__seed0__cfgff64bba3179b__20260925T062002840662Z_52f49688
```

这次是 **P1-C3-R2-S1 / 8次执行 / E4 / M64**，并非之前的 depth 实验。
模型形状、专家数、norm 和所有有效设置从这个 run 的保存配置恢复；脚本不按名字猜模型。
末层选择 `loop.suffix[-1].Attn`，visit=0，即输出头之前的最后一个 suffix attention。

默认数据目录为：

```text
/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno/airfoil/naca
```

尊重 `CDLNO_AIRFOIL_ROOT`，其次为 `CDLNO_DATA_ROOT/fno/airfoil/naca`。
若终端继承了过期数据环境变量，可显式指定：

```bash
bash iclr_vis/weight/airfoil.sh --gpu 0 \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno/airfoil/naca
```

脚本将核对 X/Y/Q 三个 NPY 的 SHA-256 与训练 metadata 一致，读取所选测试样本。
核验哈希会顺序读取完整文件；实际数组用只读 mmap，仅抽取一个样本。
测试索引0对应原数组索引1000；目标取原入口的 `NACA_Cylinder_Q.npy[:,4]`。
保留原点序和实际网格；不对不匹配的网格进行隐式重采样。

## 常用选择

```bash
# 仅检查保存配置；不导入Torch、不读数据/权重tensor、不生成输出。
bash iclr_vis/weight/airfoil.sh --preview

# GPU1，测试样本5，head3。
bash iclr_vis/weight/airfoil.sh --gpu 1 --sample-index 5 --head 3

# 分别画全部8个heads；每个head各64个状态，不平均heads。
bash iclr_vis/weight/airfoil.sh --gpu 0 --head all

# 模仿Transolver++ Figure10的蓝-白-红配色。
bash iclr_vis/weight/airfoil.sh --gpu 0 --cmap coolwarm

# 仅生成8x8翼型近景，600dpi，并逐状态另存PNG/PDF。
bash iclr_vis/weight/airfoil.sh --gpu 0 --views near --dpi 600 --individual

# ICML双栏通栏图，按6.75英寸直接生成；默认是ICLR的5.5英寸。
bash iclr_vis/weight/airfoil.sh --gpu 0 --paper icml

# CPU也支持，通常比GPU慢。
bash iclr_vis/weight/airfoil.sh --device cpu

# 换另一次已保存的M64 V5 Airfoil实验；支持含空格路径。
bash iclr_vis/weight/airfoil.sh --gpu 0 \
  --run-dir '/absolute/path/to/another/run' \
  --output-dir '/absolute/path/to/new/figure directory'
```

`--gpu` 是当前进程可见设备的索引。未设置 CUDA_VISIBLE_DEVICES 时，GPU1 用 `--gpu 1`；
如果调度器/终端已设 `CUDA_VISIBLE_DEVICES=1`，该进程中使用 `--gpu 0`。
不自动改写 CUDA_VISIBLE_DEVICES；设备不可用时明确报错。
可用 `CDLNO_PYTHON=/absolute/path/to/python bash ...` 指定训练环境解释器。

默认 `--checkpoint final`；也支持 `latest` 和明确的 `epoch_XXXX`，不猜最新 run。
要求保留原 `architecture.json` 与 checkpoints/weights 的 manifest、metadata、成对PT。
只剩孤立 `model.pt` 或损坏配对时会报错，不使用 strict=False 或回退其他模型。

## 输出和选图

每次新建独立目录，控制台末尾打印绝对路径：

```text
iclr_vis/weight/outputs/airfoil_<config hash>_<UTC>_<unique>/
  q_head00_full_shared.png / .pdf    # 4x16全域，Q原始概率，共享色标
  q_head00_full_peak.png / .pdf      # 4x16全域，逐状态除空间最大值
  k_head00_full_shared.png / .pdf    # 4x16全域，N*K，共享色标
  k_head00_full_peak.png / .pdf      # 4x16全域，逐状态除空间最大值
  q_head00_near_shared.png / .pdf    # 8x8翼型近景
  q_head00_near_peak.png / .pdf
  k_head00_near_shared.png / .pdf
  k_head00_near_peak.png / .pdf
  field_reference.png / .pdf        # Mach真值、预测、绝对误差
  routing_weights.npz               # 原始Q/K，不是着色/归一化后的数组
  metadata.json                     # 配置/样本/head/epoch/哈希/公式/数值核对/色标
  caption.txt                       # 可调整的英文图注
  paper_figure.tex                   # 官方模板内的figure/figure*及正式caption
```

默认得到8张64状态总图和1张场图，每张同时导出PNG/PDF。
`--head all` 则每个head分别导出这8张总图，场图只保存一次。
`--individual` 另输出每head的64张Q和64张K近景图，统一使用明确标注的peak归一化。
已有输出目录拒绝覆盖；训练run内文件不写入。

**论文选择建议**：使用 `q_head00_full_peak.pdf` 展示与Transolver Figure10类似的全域空间模式，
并同时检查 `q_head00_full_shared.pdf`，避免把低权重latent误述为同等重要。
用K图补充信息来源；近景适合展示翼型前缘、尾缘与尾迹附近的路由差异。
不自动挑选“最漂亮”的head、样本或latent；所有图保留latent索引0–63。

### 定量含义

- `Q[B,heads,N,64] = softmax_M(Lq/tau_q)`：每点64个latent的权重之和为1。
- `K[B,heads,N,64] = softmax_N(Lk/tau_k)`：每个latent在所有点上的权重之和为1。
- `shared`：Q直接显示原概率；K显示 `N*K`，均匀空间权重对应1。
  同一总图64个子图共享同一线性色标，范围为0至全图实际最大值，不分位裁剪、不对每格自动拉伸。
- `peak`：每个latent的空间权重除以其在**全网格**上的最大值，色标统一0–1。
  **只用于比较空间形状，不比较不同latent的绝对强度**。不减最小值；均匀状态仍然是均匀色块。
  每个latent的原始max/mean/std写入metadata，原始数据保留在NPZ。
- `Q/K` 均使用真实forward的温度clamp；Airfoil conv_temp是 `[0.01,1]`。
  不添加Transolver++的自适应温度或Gumbel噪声。
- 全图先按完整N归一化，之后才裁视窗；裁近景不会重新归一化。

`routing_weights.npz`：Q/K形状为 `[selected_heads,N,64]`；head_indices记录实际head编号；
x/y/target/prediction为 `[grid_height,grid_width]`；点序与原模型reshape一致。
例如画head3时，NPZ的第0个head轴元素对应真实head3。
控制台的relative L2仅属于所选样本，不是200个测试样本的平均结果。

### 版式

默认viridis（Transolver/LinearNO参考风格）、白底、统一比例和色标、去坐标框、紧凑留白。
4x16保留原论文的全域布局；8x8默认视窗 x∈[-0.25,1.25], y∈[-0.5,0.5]，可用
`--near-xlim MIN MAX --near-ylim MIN MAX` 覆盖。真实数据的翼型形状不替换为手画模板。
只在原网格相邻单元内分三角形进行Gouraud颜色插值；不做全局Delaunay，不跨翼型孔洞连边，
不做场平滑/锐化/噪声增强。默认400dpi，PDF保留可检索的嵌入TrueType文字，密集色块栅格化以控制文件大小。
真实训练是否学到有区分度的空间模式由权重决定。

### 字体、图注和最终印刷尺寸

当前版本已按用户要求去掉状态图编号，并在重新核对三篇论文原图及模板后改为Times风格：

- 默认 `--paper iclr`：按ICLR 2026模板正文宽度 **5.5英寸** 导出。
  `--paper icml`：按ICML 2026双栏通栏宽度 **6.75英寸** 导出，LaTeX使用 `figure*`。
  64格总图应使用正文通栏宽度，不建议挤入ICML单栏。
- **不显示逐状态编号**，也删除编号所占的行间空白；单张导出也不在图内写状态号。
  原始latent顺序不变，可由NPZ、metadata和单张文件名追溯。
- 图内默认使用可用的 **Times New Roman → TeX Gyre Termes → Nimbus Roman →
  Nimbus Roman No9 L → Liberation Serif → STIXGeneral**。最后一项是Matplotlib自带的
  Times风格衬线字体，不是微软Times New Roman，也不会默默回退到DejaVu Sans。
  控制台打印实际字体，metadata记录字体家族、文件和SHA-256，PDF嵌入实际字体。
- **色条刻度8pt、色条说明/场图标签9pt**，按最终印刷宽度计算；避免把所有元素都设为一样大。
  这属于本工具的版式选择，不冒称是会议对图内字号的统一强制规定。
- 可见色条框/刻度线至少 **0.5pt**。
  数据图保留等比例、共享色标、viridis默认配色；不用彩虹色或视觉增强掩盖均匀权重。
- 默认取消图内总标题，由正式图注说明模型、样本和head；`--annotate`仅用于浏览。
  合成检查图始终保留 `SYNTHETIC CHECK` 标记，不应作为实验结果提交。
- 导出不再使用 `bbox_inches="tight"`，PDF页面宽度与请求宽度严格一致，避免二次缩放改变字号。
  `--width`用于**最终放进论文的宽度**，不要导出大图后再缩小。
  若把5.5英寸PDF缩至3.25英寸，9pt也会缩成约5.32pt，提高DPI不能解决小字问题。

正式**图注**不画进PNG/PDF。`paper_figure.tex`用原生 `\caption{...}`：
ICML 2026模板按其**9 TeX pt、Times系**规则排版；ICLR 2026官方示例使用
`\usepackage{iclr2026_conference,times}`，图注沿用其**默认10 TeX pt、Times系**。
不得把ICML的9pt硬套到ICLR，也不能把图内8pt刻度或STIX字体强加给正式caption。
不加载覆盖caption样式的自定义设置。图注位于图下，编号、间距由会议模板处理。
`paper_figure.tex`默认选择本次第一个head、第一个view的Q peak图；插入其他图时须同步修改
Q/K与shared/peak图注，不能混用。

所有五任务脚本可在原命令后追加这些选项：

```bash
# 若远端确实安装了Times New Roman，显式指定；不存在时明确报错，不换别的字体冒充。
--font-family "Times New Roman"

# 使用Matplotlib自带的Times风格字体，各机器保持一致，无需安装字体。
--font-family STIXGeneral

# 默认图内字号，可显式调整；不控制LaTeX正式图注。
--font-size 9 --tick-font-size 8
```

例如将输出PDF及tex复制到论文的 `figures/airfoil/` 后，在官方模板中使用：

```latex
% 导言区（模板通常已有graphicx）
\usepackage{graphicx}
\graphicspath{{figures/airfoil/}}
% 正文；保留模板原有caption样式
\input{figures/airfoil/paper_figure.tex}
```

规范依据及年份见 [REFERENCE_NOTES.md](REFERENCE_NOTES.md)。会议年份/版式变化时请重新核对。
本地检查覆盖PDF物理尺寸、文字字号/嵌入、合成图裁切与布局；用户远端真实权重的最终纹理和
插入完整论文后的页面效果尚未检查，不能据此保证“所有细节均已达投稿终稿水平”。
最新无编号/Times字体核对见 [TIMES_STYLE_REVIEW.md](TIMES_STYLE_REVIEW.md)；旧验收日志保留原样。

## 运行核对

工具执行同一样本的两次eval/inference_mode前向（未挂hook、挂hook），要求预测逐位一致。
临时hook只clone实际Q/K logits、V和to_out输入；不替换forward或注册新参数。
用提取的Q/K/V验证重建结果与原attention的readout一致，同时检查概率归一、finite、
参数版本及Torch RNG不变；成功或异常均移除hook。没有optimizer/backward，也不会修改训练日志。

若需要诊断报错，请保留完整终端输出。哈希、形状和模型版本冲突应修正文件/路径，不能关闭strict检查。
