# V5 并行FFN的空间分配与跨轮变化

五个入口：`airfoil.sh`、`darcy.sh`、`elasticity.sh`、`ns.sh`、`pipe.sh`。
脚本项目为 **looplin-ffn**，旧训练输出仍在 **looplin-v5-final**。
这些脚本只执行严格checkpoint加载、选定样本推理、离线绘图；不训练、不改模型或训练记录。

## 同步与依赖

请同步本目录 **iclr_vis/ffn/** 和现有 **iclr_vis/weight/** 到同一个兼容V5的checkout。
FFN脚本复用weight目录中已核对的数据读取、normalizer、网格、字体和图片导出；不要只复制五个sh。
使用原训练环境的Python/Torch/NumPy/SciPy/Matplotlib，不安装新包。
`CDLNO_PYTHON`可指定解释器；PyMuPDF仅用于本地PDF验收，不是远端绘图依赖。

## 远端命令：全部使用绝对路径

先激活原训练环境；可以在任意目录执行。默认final、测试样本0、GPU0。

### airfoil

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/airfoil.sh \
  --run-dir /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/airfoil/partial_share_feature_gate_v5/airfoil__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E4F128__M64__seed0__cfgff64bba3179b__20260925T062002840662Z_52f49688 \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno/airfoil/naca \
  --checkpoint final --sample-index 0 --gpu 0 \
  --paper iclr --font-family STIXGeneral --font-size 9 --tick-font-size 8
```

### darcy

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/darcy.sh \
  --run-dir /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/darcy/partial_share_feature_gate_v5/darcy__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E1F128__M64__seed0__cfg67e4606b658e__20260925T062136114615Z_c51723b1 \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno \
  --checkpoint final --sample-index 0 --gpu 0 \
  --paper iclr --font-family STIXGeneral --font-size 9 --tick-font-size 8
```

### elasticity

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/elasticity.sh \
  --run-dir /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/elasticity/partial_share_feature_gate_v5/elasticity__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E4F128__M64__seed0__cfg75d646149afe__20260925T062403549643Z_16e14cb9 \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno \
  --checkpoint final --sample-index 0 --gpu 0 \
  --paper iclr --font-family STIXGeneral --font-size 9 --tick-font-size 8
```

### pipe

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/pipe.sh \
  --run-dir /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/pipe/partial_share_feature_gate_v5/pipe__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E3F128__M64__seed0__cfg6fda889e763a__20260925T154525869976Z_c345becb \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno/pipe \
  --checkpoint final --sample-index 0 --gpu 0 \
  --paper iclr --font-family STIXGeneral --font-size 9 --tick-font-size 8
```

### NS：先选择一个明确实验

用户此前给的是NS父目录，不能据此猜最新或最好的模型。

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/ns.sh \
  --run-dir /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/ns/partial_share_feature_gate_v5 \
  --list-runs
```

将下例占位目录名替换为列表中选定的真实目录名；本命令在替换前不是可运行的完整实验路径。

```bash
bash /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/ns.sh \
  --run-dir "/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output/ns/partial_share_feature_gate_v5/替换为具体实验目录名" \
  --data-path /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno \
  --checkpoint final --sample-index 0 --forecast-step 10 --gpu 0 \
  --paper iclr --font-family STIXGeneral --font-size 9 --tick-font-size 8
```

## 输出及含义

每次生成唯一目录，并打印完整路径：

```text
/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-ffn/iclr_vis/ffn/outputs/<task>_<hash>_<UTC>_<unique>/
  prefix_00_gate_full.png / .pdf
  core_00_gate_full.png / .pdf
  core_00_gate_change_full.png / .pdf
  core_00_update_norm_full.png / .pdf
  core_00_entropy_full.png / .pdf
  ... 其他core、prefix、suffix，以及Airfoil的near视图
  mean_gate_weights.png / .pdf
  field_reference.png / .pdf
  expert_weights.npz
  expert_summary.json
  expert_summary.csv
  metadata.json
  caption.txt
  paper_figure.tex
```

| 输出 | 数学含义与解读 |
|---|---|
| `gate` | 每个点的softmax专家概率，所有图固定0–1色标；行=visit，列=专家。完全没有逐专家peak归一化。 |
| `gate_change` | 同一core同一专家，后续visit减第一visit；固定[-1,1]。R3会分别给出visit2−1和visit3−1，不是相邻轮之差。 |
| `update_norm` | 每点 `||s*pi_e*F_e(u)||_2`；core的s=1/R，首尾s=1。一个run的全部捕获位置/专家/visit共用同一个最大值。包括真实输出幅度，不能等同于最终预测的因果重要性，向量可能相互抵消。 |
| `entropy` | `-sum(pi*log(pi))/log(E)`：0偏向单专家，1均匀。E1数学上无定义，**不输出该图**，原始熵0仍保存。 |
| `mean_gate_weights` | 每次逻辑访问的各专家平均权重；所有原始节点等权平均，**不是面积/体积加权**，不是整个测试集统计。跨不同block的同名列仅是局部编号，不能视作同一个专家。 |
| `field_reference` | 相同样本的真值、预测、逐点绝对误差；真值/预测同色标，误差独立色标。 |
| `expert_weights.npz` | logits/probabilities/专家原始输出范数/加权输出范数为[L,N,E]；mixed_update_norm/entropy为[L,N]。还含坐标、预测/真值、logical_indices、groups、positions、visits、expert_scales。所有索引零基，图中Visit从1开始。 |
| `expert_summary.json/.csv` | 同一份逐visit/逐expert均值；E1归一化熵为null/空单元，不造数。 |
| `metadata.json` | checkpoint/数据/脚本哈希、epoch、预测一致性、RNG、残差检查误差、字体文件/哈希、绘图范围和样本协议。 |

最新P1C3R2S1：8次逻辑访问、5套物理专家bank；每个core的专家共享两轮，但router和默认LN按visit独立。
专家数超过4时按固定原序每页最多4列；没有挑选或重新排序。图上保留必要的列身份
**Expert A/B/...**与行身份**Visit 1/2/...**，不加子图序号或64状态编号。
专家身份只在同一个物理core内跨轮对齐，不能跨block/head/run直接对齐。

你提供的实验中，Airfoil/Elasticity为E4，Pipe为E3，Darcy为E1；
Darcy权重恒1、轮间权重差恒0是正确结果，FFN输出范数仍可变化。NS专家数从选定checkpoint恢复。
P1C3R2S1、E<=4时，Airfoil默认38组PNG/PDF，其他E>1任务20组，Darcy E1为15组。
默认全部prefix/core/suffix都观察，不只最后一层。

## 公式与数据边界

```text
z = x + Operator_visit(LN1_visit(x))
u = LN2_visit(z)
pi = softmax_expert(router_visit(u))         # [B,N,E]，无attention head轴
x_next = z + s * sum_e(pi_e * Expert_e(u))
```

这些是**点域FFN门控权重**，不同于Q/K空间路由图，也不是64个状态或每专家物理场预测。
全部专家执行；平均门控权重不表示节省了相同份额的计算。

直接hook真实router与专家输出，再用原softmax维度计算detached诊断；不替换forward、不改变参数，
不保留跨forward状态。对全部访问检查专家调用数与加权残差重建（包括最终head前状态），
同一输入无hook/有hook预测必须逐位相同，公共Torch/NumPy/Python RNG不变；异常时移除所有临时hook。

数据读取复用weight脚本：Airfoil通道4、Pipe通道0/原物理坐标、Darcy stride5及已保存normalizer、
Elasticity原节点。NS预测反馈推进十步，仅在所选future step捕获专家数据，不用未来真值回填；
NPZ仍保存完整10步预测与selected_input_history。NS默认step10，可改1–10。

## 样式、图注与可调整项

沿用[weight字体审核](../weight/TIMES_STYLE_REVIEW.md)：Times系字体优先，
显式STIXGeneral在Matplotlib环境中可复现；如确有安装，可换成
`--font-family "Times New Roman"`，不存在则报错。不会默默用别的字体冒充。

- 图内标签9pt、刻度8pt；默认ICLR5.5in，`--paper icml`为6.75in。
- 官方模板的正式caption保持原生排版（ICLR2026常规10 TeX pt，ICML2026为9 TeX pt），不绘入PNG。
- 保留原始几何、共享色标、白底、0.5pt线、可检索嵌入字体；没有数据平滑、按专家拉伸或虚构孔洞连边。
- `--width`是最终印刷宽度；不应导出后再缩小导致字号变化。`--dpi`默认400。
- `--preview`只读JSON，`--list-runs`列出明确候选；不读取数据或权重tensor，不创建图片。
- `--gpu 1`换设备（相对于CUDA_VISIBLE_DEVICES），`--device cpu`显式使用CPU。
- `--output-dir /绝对路径/新目录`拒绝覆盖；禁止输出到训练run内部。
- 没有`--head`参数：FFN门控不分head。
- 所有合成检查图强制显示SYNTHETIC CHECK，不能作为真实实验结果。

只有最终checkpoint可以分析最终模型；没有历史checkpoint时无法还原每个epoch的门控演化。
不把漂亮分区解释为已证明物理专门化、性能提升或因果贡献。

本地验证结果见[VALIDATION.md](VALIDATION.md)。远端真实权重出图和完整论文最终排版尚未验收。
