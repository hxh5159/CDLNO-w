# 八任务每50轮可视化交付（2026-09-16）

本次实现的是训练过程的输出场诊断：CDLNO（full/no_sa/identity及原CDPA模式）、KCDNO（all/off）和lrsa_matched在八个已接入任务中，每完成50个epoch生成图片；实际最后一轮额外生成一次，与周期重合时只生成一次。例如AirfRANS默认398轮为50/100/150/200/250/300/350/398。模型结构、参数、forward、loss、optimizer/scheduler、采样和真实时间训练协议不变。原Transolver分支及其现有评价绘图保持原样，没有增加这套自动观察器。

最新请求只要求可视化。本次不改变权重保存频率、checkpoint格式，也不把既有普通权重宣称为完整断点续训档案。旧V1独立归档基础保留；其余任务resume接入尚未完成。本报告更新早期“任务周期绘图未接入”的状态，不把历史V2–V5整体标为完成。

## 1. 使用与输出

同步本次修改后，继续使用现有脚本即可，不需要新的模型参数。例如：

```bash
# 在实际远端仓库根目录运行；这两条均会启动真实训练，此次交付未执行。
bash tran_evaluate/kcdlno/darcy.sh train_eval all --gpu 0
bash tran_evaluate/kcdlno/run_seed.sh 0 --gpu 1

# CDLNO继续使用原薄脚本和可选消融参数。
bash tran_evaluate/train_eval.sh darcy --gpu 0 --front-latent-mode no_sa
```

八任务脚本仍是 `tran_evaluate/kcdlno/{darcy,elasticity,airfoil,pipe,ns,plasticity,car,airfrans}.sh`；各任务路径参数、训练/评价入口及数据要求见该目录README。显式run路径仍有效。已在运行的旧Python进程不会自动获得新逻辑，不能仅同步脚本就宣称远端代码已生效。

图位于本次实验run目录内；例如KCDNO的默认位置：

```text
output/<dataset>/kcdno/<run>/
  train_results.json                 # visualization_events列出完成/失败
  visualizations/
    member_000/                     # AirfRANS各独立模型按member区分
      events.jsonl
      scales/case_000_<view>.json    # 首次建立，此后只读
      epoch_0050/case_000/<view>/
        fields.pdf                  # 嵌入TrueType字体、矢量文字/色标，场层栅格化
        fields.png                  # 600 dpi
        fields.npz                  # 坐标、真值、预测、绝对/有符号误差、mask
        metadata.json               # 通道、视角、色标、超色标比例、单例误差
        caption.txt
        caption.tex
      epoch_0050/case_001/...
      epoch_0100/...
```

CDLNO使用其原本的 `output/<dataset>/<run>/`；自定义run目录下子结构相同。每次只展示前两个held-out案例（不足两个则取现有案例），绝不按预测误差挑选案例。重复调用同epoch不会覆盖已存在的案例目录；失败会记录warning/event，恢复训练状态后继续原循环。图生成成功不代表正式数据集评价通过。

## 2. 论文依据与排版

重新阅读五篇论文可视化相关正文/附录和图注，并实际查看选定场图。阅读范围不是全文数学复审。固定版本URL、图注、HTML及已查看图片hash见[来源证据](periodic_visualization_audit/paper_sources.json)，未把论文图片复制进交付示例。

| 论文 | 阅读/查看内容 | 本次借鉴 |
|---|---|---|
| [Transolver](https://arxiv.org/html/2402.02366v2#S4.F7) Fig.7/18 | 真值、预测、误差；Car表面与体积 | 三列对照、按真实几何显示、Car压力/速度分开 |
| [Transolver++](https://arxiv.org/html/2502.02414v2#A3.F16) Fig.6/16 | Elasticity孔洞、翼型激波附近场误差 | 保留几何空洞、翼型固定物理窗口、低误差浅色 |
| [Transolver-3](https://arxiv.org/html/2602.04940v2#S4.F7) Fig.7/13/15 | 表面压力和体积流场侧视/底视 | Car固定三维视角及中心体积薄层 |
| [LRSA](https://arxiv.org/html/2604.03582v1#S3.F3) Fig.3 | NS/Elasticity/Airfoil/Plasticity，共用误差尺度 | 稳定色标、非方形网格、时间索引、相对L2诊断 |
| [LinearNO](https://arxiv.org/html/2511.06294v3#Sx14.F9) Fig.4/9/10 | AirfRANS及各任务误差场 | 多通道及近翼型场图，不引入slice/attention解释图 |

论文图注没有给出可核验的精确字体与字号。因此采用适合双栏论文的统一规范，**不声称精确复刻它们的字体参数**：STIXGeneral/STIX数学字体（Matplotlib自带），标称双栏宽7.16in≈181.9mm；标题9pt、轴标签8pt、刻度7pt；PDF嵌入TrueType，PNG600dpi。导出使用tight边界，实际文件宽略随文字边界变化。英文图注包含任务、模型、epoch、seed（入口已提供时）、显示区域和数值处理，可复制LaTeX caption；最终投稿仍按目标期刊栏宽排版。

每个通道的真值与预测共用色标；跨epoch、同一案例/视图/通道的色标固定。场颜色范围来自真值min/max，误差范围默认0到真值range的20%，不根据模型误差调整。常量真值使用明确的微小正范围避免退化。不同案例/通道/时间帧可有不同范围；跨模型比较须使用相同案例和相同limits，不能将不同limits的图直接比较。超限颜色会饱和，但原数组不裁剪，metadata记录全部及显示区域的饱和比例。

误差图使用 `abs(prediction-truth)`，NPZ同时保留signed error。显示坐标、标签和量纲严格按源码合同，不凭字段名猜单位。结构网格使用原H/W及点序的pcolormesh/Gouraud显示插值；这是对原相邻节点的颜色插值，不是重新计算物理解。点云只scatter，不虚构三角面或把孔洞填上。PDF中密集场层栅格化以控制文件体积，文字/色标仍为矢量；不是全矢量的网格场。

已实际查看[非方形弯曲网格示意](periodic_visualization_audit/synthetic_previews/curved_grid/fields.pdf)、[带孔点云示意](periodic_visualization_audit/synthetic_previews/point_cloud_hole/fields.pdf)、[三维表面示意](periodic_visualization_audit/synthetic_previews/surface_3d/fields.pdf)。这些是明确标注的解析合成场加扰动，仅用于检查排版，**不是训练结果，也不能作为准确率证据**。没有伪造其他方法的预测对照。

## 3. 八任务合同

| 任务 | 模型输入/绘图处理 | 每案例输出 |
|---|---|---|
| Darcy | 原坐标、fx1；预测经原y_normalizer.decode；held-out真值已经是原尺度，不重复decode | solution三列场图 |
| Elasticity | 原点云坐标、fx=None；原y_normalizer.decode预测；不补网格 | stress三列点云图，孔洞保留 |
| Airfoil | 原弯曲H/W网格、fx=None、Mach输出 | 全场；固定x∈[-.25,1.5],y∈[-.5,.5]近翼型点云图 |
| Pipe | 原归一化坐标供模型；显示副本经原x_normalizer.decode，预测经y_normalizer.decode | 原弯曲网格速度分量图 |
| NS | 原10帧输入，单次输出1帧，预测回填连续10次，每次独立forward | 预测第1/5/10帧；10帧全部原数组；单例逐帧relative-L2曲线 |
| Plasticity | 空间101×31、fx1、T[B,1]；20个T逐次独立前向，无反馈，不将时间并入N | 第1/10/20个时间点：通道0:2的变形坐标，通道2:4的模长；四通道全部轨迹和逐时L2 |
| ShapeNet-Car | 原(cfd,geom)单图；用原coef_norm解码[vx,vy,vz,p]；完整图输入不按显示mask裁剪 | surf压力；volume中y方向中央薄层x/z视图的vx/vy/vz/速度模长；保存完整节点场和mask |
| AirfRANS | 原x7/pos；按原subsampling数量无放回抽样并用原r/loop/max_neighbors建radius_graph，复制固定案例；原coef_norm解码[vx,vy,p,nut] | 四通道图、近翼型压力；保存抽样idx和原N |

Car薄层half-width为体积y范围的2%，若该薄层无点则扩至最近的原体积节点平面；只显示原点，不宣称等同VTK三角曲面或插值CFD切片。当前没有真实网格数据，因此最终表面细节、遮挡和视角仍需在真实案例上确认。

AirfRANS固定案例图是**训练诊断**，不是原正式反复采样/idx scatter平均评价。这里显示解码后的原始预测，没有在诊断中强行置零壁面速度/nut，原正式评价的边界后处理保持在原评价器中。原每epoch训练抽样和反复验证没有删改；可视化额外执行一次固定诊断采样和真实图构造。图构造在模型所在device执行，其余缓存保存在CPU。各任务固定案例准备在RNG隔离区内完成，诊断seed0不改变训练seed。

Plasticity误差图在真值坐标上按原节点对应显示，预测面板使用预测变形坐标。逐时relative-L2包含全部原输出通道（Plasticity包括坐标输出），是单案例诊断，不替代原评价聚合。零真值范数明确保存defined=false，并从曲线剔除。

## 4. 实现位置与冻结证据

| 文件/符号 | 改动 |
|---|---|
| `cdlno/periodic_visualization.py:PeriodicFields` | 频率、固定案例、任务适配、解码、rollout、mask、日志与CPU缓存 |
| `cdlno/visualization.py:render_fields/render_curves` | 复用V1绘图基础，增加统一出版排版、模型标注、图注、曲线、变形坐标和显示mask指标 |
| `cdlno/experiment.py:Experiment.visualize` | 仅训练record触发，按Air成员隔离，结果event入原记录文件 |
| 六个`exp_*.py`、Car/Air`train.py` | 既有record_epoch之后加一次观察调用 |
| Air`main.py` | 仅记录型新模型kwargs转发原coef_norm；旧Transolver不接收新参数 |
| 新`tests/test_periodic_visualization.py`与`visualization_projection.py` | 真实wrapper/PyG与可视化隔离检查、完整AST投影 |
| 原记录/KCDNO冻结测试 | 仅认可精确新增观察调用；Car旧测试pos2修为实际合同pos3，以真正完成绘图而非只验证失败恢复 |

预修改448文件快照：`/home/hwz/CDLNO-artifacts/visualization-before-133lyhbo/source`。161个受保护的生产模型/配置/数据/工具/脚本文件逐字节不变；9个完整entry/train AST剥离**精确的新visualize调用及Air的可选norm传参**后与预修改相同。loss、时间循环、采样/指标、原模型分支均包含在完整AST比较中。细节见[freeze.json](periodic_visualization_audit/freeze.json)及[本次代码patch](periodic_visualization_audit/implementation.patch)。既有seed修复等用户工作不计成本次改动，未覆盖或回滚。

沿用V1 `isolated_evaluation`：eval/no_grad包住案例准备、前向、绘图；退出及异常时恢复Python/NumPy/Torch CPU/CUDA RNG、所有子模块train/eval标志和buffers。不访问optimizer，不改变已有gradient。固定案例只保存CPU副本，原数据不原地修改。绘图增加墙钟开销，原工业`time_elapsed`自然包含该开销，不能据此与未绘图训练比较纯模型效率。

## 5. 实际验证

本机Python3.13.9，torch2.13.0+cu130，PyG2.3.1，RTX5090 Laptop GPU；并非远端目标Python3.10/torch2.11/cu128验收。Matplotlib3.10.6沿用本机已安装版本；没有升级/安装依赖。

```bash
PYTHONPATH=tests:. python -B -m unittest discover -s tests -v
PYTHONPATH=tests:. python -B -m unittest test_periodic_visualization test_visualization -v
PYTHONPATH=tests:. python -B -m unittest test_experiment_records.ExperimentRecords.test_industrial_real_epoch_observer_preserves_weights_rng_and_original_loss_values -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator PYTHONPATH=tests:. python -B -m unittest test_lrsa_reference -v
python -B -m docs.periodic_visualization_audit.gpu_isolation
```

- 完整回归：287 tests，319.519s，0 failures/errors，3个skip标记：Air完整抽样epoch、Air新诊断建图缺torch_cluster，以及未设置LRSA外部目录。随后提供真实LRSA目录单独补跑2/2通过；点/卷积前向、T和梯度差均0。
- 最终绘图定向：11 tests，17.178s，0 failures/errors，仅Air真实抽样建图skip。检查原解码、N/H/W及原始点序、NS预测回填10次、Plasticity20次T、真实PyG图输入/4通道/mask、固定色标、PDF/PNG/NPZ/caption、字体全局配置恢复和渲染失败恢复。
- 完整Car合成训练epoch：使用原安全train.main、真实PyG pos3、原mask loss；带/不带观察器权重及Python/NumPy/Torch RNG逐值一致，真实图片导出完成。该测试文件的Air子项因缺torch_cluster跳过。最初旧pos2合成夹具触发绘图失败，只证明失败隔离，不能冒称成功绘图；修正夹具并增加completed断言后32.537s通过。
- CPU KCDNO合成AdamW步：观察前后weights/已有grads/RNG/训练标志一致，下一步weights/optimizer/scheduler逐值一致；渲染抛异常时也恢复。三个CDLNO front模式和matched LRSA均经过真实观察前向且state_dict不变。
- 新GPU有限检查：FP32、math SDPA、TF32off、compileoff、AMPoff、小5×7 Darcy KCDNO。观察前后权重/RNG一致；测试启用确定性算法/cuDNN和CUBLAS工作区后，后续更新逐值一致。首次未固定确定性后端时，零容差后续权重比较最大差约1.9e-9，原始失败日志保留；这不通过放松生产精度或修改模型规避。生产训练未启用新的确定性设置。
- 实际查看3种几何的合成导出；PDF字节含FontFile2/CIDFontType2，不含Type3；`pdffonts`工具不存在，没有安装。每张600dpi PNG和配套PDF/NPZ在示例目录。

完整日志、初次GPU严格比较失败及最终证据在[审计目录](periodic_visualization_audit/)。完整回归之后仅补了时间曲线图注、测试夹具与报告，相关定向检查重跑；没有重复扩大测试矩阵。

## 6. 剩余限制与远端核查

没有真实数据读取验收/真实训练/收敛/准确率检查，没有真实案例最终视觉质量或完整AirfRANS采样评价验收。真实PyG对象前向和渲染通过不等于原始网格数据完整链路通过。Air真实radius_graph依赖缺失项必须在远端已有环境核查，禁止伪造模块或为出图跳过图构造。新可视化的AMP/compile和目标远端torch2.11尚未实测。

在远端根目录可先运行以下无数据命令；静态快照不在远端时该项会明确skip，也可将原快照同步后用 `CDLNO_VISUALIZATION_BASELINE` 指定，不能修改后重造“旧基准”：

```bash
PYTHONPATH=tests:. python -B -m unittest test_periodic_visualization test_visualization -v
python -B -m docs.periodic_visualization_audit.gpu_isolation
```

已自审：模型/数据冻结、两时间协议、单图及表面mask、归一化/原标签顺序、RNG与异常恢复均有源码或可执行证据；没有需要用户裁定的架构变化。可直接使用导出的PDF和caption进行排版，但最终论文必须使用真实训练结果并如实注明指标/案例选择，示意图不能替代实验。

本阶段结束，未执行下一阶段。
