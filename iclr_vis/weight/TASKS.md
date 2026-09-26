# Darcy、Elasticity、NS、Pipe：V5 最后一层空间路由图

本目录按数据集提供 `darcy.sh`、`elasticity.sh`、`ns.sh`、`pipe.sh`，共用
`task_states.py` 与已有 `airfoil_states.py` 的透明hook、色标和排版工具。
请同步**整个 `iclr_vis/weight/` 目录**，尤其包括更新后的 `airfoil_states.py`。
这些脚本进行已训练模型的只读推理与绘图，不启动训练，不修改原实验的输出或权重。

## 脚本项目与实验项目分开时的命令

用户最新路径：脚本在 **looplin-vis**，实验输出仍在 **looplin-v5-final**。
`--run-dir`传完整实验目录；无须复制权重或修改训练目录。
模型代码从脚本所在checkout导入，因此looplin-vis应包含与这些V5档案兼容的代码。

先执行一次：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-vis

VIS_RUNS=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v5-final/output
VIS_DATA=/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno
```

Darcy（用户指定E1/M64实验，示例GPU1）：

```bash
bash iclr_vis/weight/darcy.sh \
  --run-dir "$VIS_RUNS/darcy/partial_share_feature_gate_v5/darcy__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E1F128__M64__seed0__cfg67e4606b658e__20260925T062136114615Z_c51723b1" \
  --data-path "$VIS_DATA" --gpu 1
```

Elasticity（用户指定E4/M64实验，示例GPU0）：

```bash
bash iclr_vis/weight/elasticity.sh \
  --run-dir "$VIS_RUNS/elasticity/partial_share_feature_gate_v5/elasticity__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E4F128__M64__seed0__cfg75d646149afe__20260925T062403549643Z_16e14cb9" \
  --data-path "$VIS_DATA" --gpu 0
```

Pipe（用户指定E3/M64实验）：

```bash
bash iclr_vis/weight/pipe.sh \
  --run-dir "$VIS_RUNS/pipe/partial_share_feature_gate_v5/pipe__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E3F128__M64__seed0__cfg6fda889e763a__20260925T154525869976Z_c345becb" \
  --data-path "$VIS_DATA/pipe" --gpu 0
```

NS：用户提供的是实验父目录，不是一个实验。先只读列出保存配置：

```bash
bash iclr_vis/weight/ns.sh \
  --run-dir "$VIS_RUNS/ns/partial_share_feature_gate_v5" --list-runs
```

列表包含每个实验的完整 `run_dir`、M、专家数、拓扑、config hash及是否有final指针。
此列表不读取权重tensor，也不将final指针的存在当成权重校验成功。不会自动选“最新”或“最好”。
将下例 `NS_RUN` 替换成列表中希望绘图的**完整run_dir**，然后执行：

```bash
NS_RUN='/将列表中选定实验的完整run_dir填在这里'
bash iclr_vis/weight/ns.sh \
  --run-dir "$NS_RUN" --data-path "$VIS_DATA" \
  --gpu 0 --forecast-step 10
```

已有Airfoil脚本同样支持从新项目读取旧项目实验：

```bash
bash iclr_vis/weight/airfoil.sh \
  --run-dir "$VIS_RUNS/airfoil/partial_share_feature_gate_v5/airfoil__partial_share_feature_gate_v5__paper_table8_on_release_model__P1-C3-R2-S1__operator_1_expert_1_over_r__norm-visit_independent__E4F128__M64__seed0__cfgff64bba3179b__20260925T062002840662Z_52f49688" \
  --data-path "$VIS_DATA/airfoil/naca" --gpu 0
```

以上命令默认测试样本0、head0、final checkpoint。`--gpu`为当前CUDA可见设备编号；若
`CUDA_VISIBLE_DEVICES=1`，进程中对应设备应使用 `--gpu 0`。不能用图像好看与否选择训练checkpoint。

## 数据和推理合同

| 任务 | `--data-path`含义 | 选样本和模型输入 | 实际绘图 |
|---|---|---|---|
| Darcy | fno根目录，含smooth1/2 MAT | smooth2前200；stride5采样；恢复保存的input统计编码coeff；output解码 | 原[0,1]空间网格；solution真值/预测/误差 |
| Elasticity | fno根目录，含elasticity/Meshes | XY为[N,2,samples]，sigma为[N,samples]；完整文件最后200；fx=None；output解码 | 原始散点坐标；stress真值/预测/误差，不虚构孔洞上的三角连边 |
| NS | fno根目录，含NavierStokes_V1e-5_N1200_T20子目录 | 完整文件最后200；前10帧输入；预测反馈推进10步，无normalizer | 指定future step的路由图和vorticity场；同时保存完整10步预测 |
| Pipe | fno/pipe，含Pipe_X/Y/Q.npy | **先取前1200，再取最后200**；Q第0通道；保存input统计编码坐标；output解码 | 原始物理坐标绘图，不能把编码后坐标当作真实几何；velocity场 |

所有数据文件SHA-256与训练metadata逐项核验。Darcy连训练文件也校验，但不重新读取训练
field数组或拟合normalizer；输入和输出统计均恢复checkpoint中保存的 `mean/std`。
原std已含epsilon，解码不额外增减epsilon。MAT格式用任务原有SciPy读取方式，NPY用只读mmap。
不自动猜MAT/HDF5布局或重采样到另一个网格；不兼容时明确报错。

NS `--forecast-step 1..10`指未来序列中的步数：step1对应原数组时间索引10，step10对应索引19。
默认step10，以完整10步预测反馈的输入提取末步attention；每步都重新计算attention，没有历史缓存。
为证明hook透明性，所选步做一次无hook基线和一次有hook前向，**输入相同，只推进一次时间窗口**。
不将未来真值反馈给模型。不是teacher-forced可视化，也不改变原评估协议。

NS默认M32，因此默认是**32个状态、4×8排版**；若实际checkpoint是M64，会自动画64个。
Darcy/Elasticity/Pipe指定实验均为M64，默认8×8。状态数以保存配置为准，不填充或复制状态。
每个head单独展示Q和K，Q在M上softmax，K在N上softmax；不平均不同heads的同号状态。

Elasticity没有可靠网格连通文件可从当前任务输入直接获得，因此使用**真实点上的散点着色**，
不通过Delaunay填充孔洞，不把人工推断的曲面当成原网格。散点图的白色间隙可能是点间空白，
不能全部解释为物理孔洞。其他三任务只使用相邻结构单元的原始连接，颜色作单元内Gouraud插值。

## 输出、排版和常用参数

每次新建：

```text
looplin-vis/iclr_vis/weight/outputs/<task>_<hash>_<UTC>_<unique>/
  q_head00_full_shared.png / .pdf
  q_head00_full_peak.png / .pdf
  k_head00_full_shared.png / .pdf
  k_head00_full_peak.png / .pdf
  field_reference.png / .pdf
  routing_weights.npz
  metadata.json
  caption.txt
  paper_figure.tex
```

每个head为4张路由总图，另有1张真值/预测/绝对误差场图。shared Q为原概率；shared K显示N*K，
均匀分布对应1。peak每个状态仅除其全域最大值，展示空间形状，不比较绝对强度；不减最小值。
NPZ保存**原始Q/K**、原坐标、解码后目标/预测、选样本/head/latent索引、实际模型输入。
NS另外保存initial_history、selected_input_history、predicted_rollout、target_rollout等。

metadata保存模型/脚本校验、checkpoint身份、数据哈希、epoch、归一化/绘图协议及单样本误差。
NS另有每步rL2、十步平均rL2和十步拼接full rL2。**这些只是选定的一个测试样本，不是整个测试集结果。**
数据目录和原训练run保持只读。`--output-dir`可指定另一个全新目录，已有路径会拒绝覆盖。

可在上述任意四任务命令末尾追加：

```bash
# 只检查保存的JSON配置；不读取数据或权重tensor、不创建输出
--preview
# 全部heads分别出图，不平均
--head all
# 测试样本5、第3个head（索引2）
--sample-index 5 --head 2
# ICML两栏通栏宽度；默认ICLR
--paper icml
# 600dpi栅格输出及密集色块；PDF文字仍保持嵌入
--dpi 600
# 每个状态另存单张Q/K peak图片
--individual
```

所有状态格均**不标编号**；Times系/Times风格字体、8pt刻度/9pt说明、
ICLR5.5英寸/ICML6.75英寸、0.5pt色条线宽、无总标题默认、PDF固定宽度、
正式caption继承模板等与Airfoil完全共用。`paper_figure.tex`默认对应**第一个选定head的Q peak总图**。
换成K或shared图时必须同步修改caption。实际像素模式由训练权重决定，不用色标制造虚假的物理分区。
可用 `--font-family "Times New Roman"` 显式选择已安装字体，或用 `--font-family STIXGeneral`
保证使用Matplotlib自带的Times风格字体。图内字体/字号不会覆盖论文模板的正式图注。
详见 [README排版说明](README.md#字体图注和最终印刷尺寸) 和 [参考依据](REFERENCE_NOTES.md)。

安装/验证边界：使用原训练环境已有Torch、NumPy、SciPy、Matplotlib及V5模型依赖，无需另装包。
本地合成验证不是远端真实数据验收；完整论文TeX编译与真实权重最终纹理仍须检查。
