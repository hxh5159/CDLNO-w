# CDLNO — Cross-Depth Latent Neural Operator

基于 [Transolver（ICML 2024）](https://arxiv.org/abs/2402.02366) 的八任务模型扩展。共享包/核心为 `cdlno` / `CDLNO`，跨深度机制仍称 **CDPA**。默认计算图为：

```text
原任务 stem → H0[B,N,d]
→ 2 × 完整 LRSA block（down → FFN1 → latent SA → FFN2 → up → 点FFN）
→ HF[B,N,d] → IPOT bridge → Z0[B,M,d]
→ entry CDPA(Z0, [T1,T2]) → 6 × 独立 latent SA+GEGLU
→ HF 查询最终 latent、保留 HF 点残差 → 点FFN → LN/head → 原任务输出
```

上图是默认`front_latent_mode=full`，T在FFN2后、Up专属norm之前取得。八任务也支持`no_sa`（保留两次latent FFN、跳过完整SA子层）及`identity`（T直接等于Down输出），三者之后都保留Up和点更新。规则网格点FFN为普通`groups=1`的3×3 ConvFFN；潜空间无卷积。完整规格及最终核对见 [实施报告](docs/CDLNO_IMPLEMENTATION_REPORT.md)、[需求矩阵](docs/CDLNO_REQUIREMENTS_MATRIX.md)、[八任务清单](docs/CDLNO_TASK_LAUNCHERS.md) 和 [来源/许可](docs/CDLNO_THIRD_PARTY_NOTICES.md)。旧 Transolver 模型与脚本保留，下方原论文结果仅属于原 Transolver。

交付通过静态、合成张量、原损失连接、真实 PyG 接口、checkpoint 和本地 GPU 检查。**尚未运行真实数据训练或完整抽样评价，未验证收敛、预测精度、真实 epoch 效率或远端目标环境。**

新实验现在默认写入 `output/<数据集>/<UTC时间戳>/`，包含启动配置、实际参数量、训练日志/结果和各次评估结果。`bash tran_evaluate/train_eval.sh darcy --gpu 0` 可训练成功后自动评估；八任务命令及旧结果加载见 [统一实验记录](docs/CDLNO_EXPERIMENT_OUTPUTS.md)。

## CDLNO 环境与共享包

目标为用户已工作的 **Python 3.10、torch 2.11 / CUDA 12.8**，Python 3.11 是候选。原 ShapeNet-Car 在该远端栈的训练成功由用户报告；尚无新 CDLNO 远端验收。保留现有 torch/PyG/pyg-lib，不执行旧 requirements 中的 torch 1.10.1 降级。核心只用 PyTorch SDPA，不依赖自定义 CUDA/Triton、xFormers 或 LRSA/IPOT 的训练框架。

下面是供远端执行的安装/只读预检命令，本次未执行安装：

```bash
# 在仓库根，用已有 Python 3.10/3.11 环境；不安装/替换依赖。
python -m pip install -e . --no-deps --no-build-isolation
python -B tools/cdlno_environment_preflight.py
```

安装使用已有 setuptools>=68。`pyproject.toml` 限定 Python>=3.10,<3.12，并不自动解析任务依赖。模型核心需要 torch；任务侧继续需要原 numpy/scipy/einops/timm/matplotlib，工业任务还需 PyG、PyYAML，以及原读图/VTK/指标路径的依赖；具体路径见 [环境审查](docs/CDLNO_REFERENCE_AUDIT.md)。预检只检查基础导入/SDPA，不代表 VTK、邻居图或八任务端到端可用。依赖不足时应针对实际错误处理，不整套重装。

无需安装的源码导入可在仓库根使用 `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"` 后进入原工作目录。最终本地检查使用 Python3.13.9/torch2.13+cu130/PyG2.3.1 与此导入方式，**没有**把该环境当成目标 editable 安装验证。

## 八任务训练与评价示例

使用现有`tran_evaluate/train_eval.sh TASK --front-latent-mode full|no_sa|identity`可以在一次命令中先训练、成功后评估。八任务逐条命令、模式目录和参数表见[三模式命令页](docs/CDLNO_FRONT_ABLATION_A2_COMMANDS.md)，最新结构/成本/实测范围见[A4交付](docs/CDLNO_FRONT_ABLATION_A4.md)。以下未传新字段的旧CDLNO命令仍使用full。所有训练由用户自行启动，本次不执行。

以下为用户准备好真实数据后手动运行的命令，交付过程未执行这些入口。数据路径按原项目 README 的文件布局准备，不更改字段/划分/采样。所有新脚本显式列出模型与训练初值，末尾用户参数优先；打印的新 run 路径须保留。评价需要已有 run，并重复训练时非默认的架构、fold/task/nmodel、预算等参数。

六标准任务从 `PDE-Solving-StandardBenchmark` 目录运行（`--data_path` 保持原入口语义；Plasticity 为 MAT 文件）：

```bash
cd PDE-Solving-StandardBenchmark
# Darcy
bash scripts/CDLNO_Darcy.sh --data_path /data/fno
bash scripts/CDLNO_Darcy_Eval.sh --data_path /data/fno --cdlno-run-dir /runs/darcy
# Elasticity
bash scripts/CDLNO_Elas.sh --data_path /data/fno
bash scripts/CDLNO_Elas_Eval.sh --data_path /data/fno --cdlno-run-dir /runs/elasticity
# Airfoil
bash scripts/CDLNO_Airfoil.sh --data_path /data/fno/airfoil/naca
bash scripts/CDLNO_Airfoil_Eval.sh --data_path /data/fno/airfoil/naca --cdlno-run-dir /runs/airfoil
# Pipe
bash scripts/CDLNO_Pipe.sh --data_path /data/fno/pipe
bash scripts/CDLNO_Pipe_Eval.sh --data_path /data/fno/pipe --cdlno-run-dir /runs/pipe
# Navier–Stokes：保持10→10，训练真值回填、评价预测回填。
bash scripts/CDLNO_NS.sh --data_path /data/fno
bash scripts/CDLNO_NS_Eval.sh --data_path /data/fno --cdlno-run-dir /runs/ns
# Plasticity：空间101×31，每个时间点独立输出4通道并按原节奏更新。
bash scripts/CDLNO_Plasticity.sh --data_path /data/fno/plas_N987_T20.mat
bash scripts/CDLNO_Plasticity_Eval.sh --data_path /data/fno/plas_N987_T20.mat --cdlno-run-dir /runs/plasticity
```

ShapeNet-Car 从 `Car-Design-ShapeNetCar` 目录运行；一次 run 只对应一个 fold。`--data_dir/--save_dir` 继续分别沿用原始数据/预处理数据路径语义，`--run_dir` 是新增的模型运行目录。

```bash
cd Car-Design-ShapeNetCar  # 相对于仓库根
bash scripts/CDLNO.sh --data_dir /data/car/training_data --save_dir /data/car/preprocessed --fold_id 0
bash scripts/CDLNO_Evaluation.sh --data_dir /data/car/training_data --save_dir /data/car/preprocessed --fold_id 0 --run_dir /runs/car
```

AirfRANS 从 `Airfoil-Design-AirfRANS` 目录运行。**训练 `--my_path` 是 Dataset 本身；评价 `--my_path` 是它的父目录**。`--task full|scarce|reynolds|aoa` 与 `--nmodel` 保持原协议；默认398 epochs。原图构造、每 epoch/重复验证抽样、scatter/平均与边界后处理仍执行。

```bash
cd Airfoil-Design-AirfRANS  # 相对于仓库根
bash scripts/CDLNO.sh --my_path /data/naca/Dataset --task full --nmodel 1
bash scripts/CDLNO_Evaluation.sh --my_path /data/naca --task full --nmodel 1 --run_dir /runs/airfrans
```

工业任务只支持单图（原 batch=1、可变N），多图明确报错。AirfRANS 默认训练脚本 `--score 0`；如用户显式选择 `--score 1`，会进入原完整评价流程。新任务的所有16个脚本和8个 JSON 初值见 [清单](docs/CDLNO_TASK_LAUNCHERS.md)。

## 配置、消融与 checkpoint

默认 `L=8,F=2,P=L-F=6,cdpa_mode=entry`。Pipe M32，其余 M64；d/h/epoch/batch 以任务清单为准。旧 parser 的 L3/M32/宽度默认仅属旧分支；CDLNO 专用默认和显式脚本覆盖新分支，未改写旧模型默认。

| 含义 | 标准任务参数 | 工业任务参数 |
|---|---|---|
| 总深度L / 前段F | `--n-layers` / `--front-blocks` | `--n_layers` / `--front_blocks` |
| d / heads / M | `--n-hidden` / `--n-heads` / `--slice_num` | `--n_hidden` / `--n_heads` / `--slice_num` |
| 模式 | `--cdpa-mode off|entry|every_block` | `--cdpa_mode off|entry|every_block` |
| 前段latent processor | `--front-latent-mode full|no_sa|identity` | `--front-latent-mode full|no_sa|identity` |
| 来源分块 | `--cdpa-source-chunk-size 0` | `--cdpa_source_chunk_size 0` |
| 前段/点FFN ratio | `--mlp_ratio 2` | `--mlp_ratio 2` |
| 后段GEGLU ratio | `--latent-ffn-ratio 2` | `--latent_ffn_ratio 2` |
| 已有评价/新的训练目录 | `--cdlno-run-dir` | `--run_dir` |

`0≤F<L`；P仅派生，不接受第二个 rear-depth。`L12/F2` 得到2+10，支持 F>6。M在全阶段一致。`off` 不收集历史；`entry` 只融合前段T；`every_block` 逐后段访问前段T和更早完整Z（包含raw bridge Z0，但不重复当前）。F0/entry不建CDPA参数，计算路径与同权重off一致；配置模式仍是严格比较字段。所有历史每次 forward 重建，不能跨真实时间复用。

chunk0把来源折叠进batch，chunk1逐来源，正整数k分块；每份历史内部仍在M个token上softmax，最后一次统一来源融合。它仅改变执行组织，不减少来源份数或数学MAC。默认entry是2份来源/1次历史SDPA；every是27份/6次（chunk0）。不同chunk可加载相同架构权重。

每个新训练使用独立目录并保存 `architecture.json`。标准任务保存严格 state_dict（weights_only）；Car保留 `model_<epochs>.pth` 整模型；Air保留 `member_000/model` 等整模型及run根 `CDLNO` 模型列表。工业整模型仅按原受信任本地checkpoint协议使用局部 `weights_only=False`，不接受来源不明文件。

评价**先读既有sidecar，再核对架构、wrapper语义及相应任务条件，最后加载**；不覆盖sidecar或训练权重，结果写入独立子目录。省略front模式时从明确指定run的sidecar恢复该字段；其它自定义参数仍须重复提供。仅能明确识别的历史完整full配置允许缺front字段，旧工业对象缺该属性也按历史full路径运行。显式mode冲突、错误权重键/shape等均严格拒绝，F0也不放宽架构校验；chunk/device/dtype属于运行字段。**no_sa和identity默认从头训练，不支持full到消融的权重转换。**

| 默认L8/F2/P6 | full | no_sa | identity |
|---|---:|---:|---:|
| 前段SA / latent FFN | 2 / 4 | 0 / 4 | 0 / 0 |
| 后段SA / GEGLU | 6 / 6 | 6 / 6 | 6 / 6 |
| Down/Bridge、Up/Readout、规则ConvFFN | 各3 | 各3 | 各3 |

固定CDPA比较三模式检验前段SA/FFN必要性，不能单独证明CDPA替代了它们；后段SA仍存在，不能称为“无层内注意力”。另一个可视化/续训计划目前仅V1公共组件完成，任务入口未接入新的resume/周期保存，本A4不推进该计划。普通训练仍拒绝已有目录。

## 无数据验收与性能工具

在仓库根运行已约定套件；直接LRSA对照需要用户自己的指定快照（缺少则明确skip）：

```bash
CDLNO_LRSA_ROOT=/path/to/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
# 无数据CPU成本检查
python -B tools/cdlno_benchmark.py --task airfoil --grid 5 7 --B 2 --d 16 --h 4 --M 4 --audit-only --output /tmp/cdlno-cost.json
# 远端GPU有限代表配置；输出文件必须不存在
python -B tools/cdlno_benchmark.py --task elasticity --comparison matched --device cuda:0 --chunks 0 1 2 --warmup 5 --iterations 20 --output /tmp/cdlno-elasticity.json
# 只改CDLNO前段；matched LRSA始终full。
python -B tools/cdlno_benchmark.py --task elasticity --models cdlno_entry --front-latent-mode identity --chunks 0 --device cuda:0 --precision fp32 --backend math --output /tmp/cdlno-identity.json
```

工具比较原 Transolver、连续L个完整LRSA的 `lrsa_matched` 和 CDLNO off/entry/every_block。`--comparison task` 使用原任务各自配置；`matched` 使用同一d/h/M/L/点FFN设置。matched LRSA仅是结构性能对照，没有注册进训练器，也不是LRSA论文复现。

矩阵成本采用1 MAC=1乘加、2 FLOPs/MAC；额外norm/softmax/depth/临时stack与历史/点特征存储单列，SDPA不会按0计。实测报告forward及合成MSE/AdamW步的median/p90、GPU同步、峰值allocated显存、初始化optimizer state和精度/backend/warmup。所有模型用相同AMP/TF32/compile条件。详细口径、原始数据和限制见 [工具说明](docs/CDLNO_PERFORMANCE_TOOLS.md) 与 [阶段9报告](docs/CDLNO_PHASE9_PERFORMANCE.md)。模型计时不能换算真实epoch；已有结果也不构成普遍加速保证。

---

以下保留原 Transolver README、论文结果与引用：

# Transolver (ICML 2024 Spotlight)

:triangular_flag_on_post:**News** (2026.02) We present a new member of the Transolver Family, named [Transolver-3](https://arxiv.org/pdf/2602.04940), which can handle **100-million-scale** geometries with SOTA results in full-size DrivAerML.

:triangular_flag_on_post:**News** (2025.07) We have released the code of [Transolver++](https://arxiv.org/abs/2502.02414v1). Please check this [GitHub Repository](https://github.com/thuml/Transolver_plus).

:triangular_flag_on_post:**News** (2025.04) We have released [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library) as a simple and neat code base for PDE solving. It contains 17 well-reproduced neural solvers. Welcome to try this library and join the research in solving PDEs.

:triangular_flag_on_post:**News** (2025.02) We present an upgraded version of Transolver, named [Transolver++](https://arxiv.org/abs/2502.02414v1), which can handle million-scale geometries in one GPU with more accurate results.

:triangular_flag_on_post:**News** (2024.10) Transolver has been integrated into [NVIDIA physicsnemo](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/darcy_transolver).

Transolver: A Fast Transformer Solver for PDEs on General Geometries [[Paper]](https://arxiv.org/abs/2402.02366) [[Slides]](https://wuhaixu2016.github.io/pdf/ICML2024_Transolver.pdf) [[Poster]](https://wuhaixu2016.github.io/pdf/poster_ICML2024_Transolver.pdf)

In real-world applications, PDEs are typically discretized into large-scale meshes with complex geometries. To capture intricate physical correlations hidden under multifarious meshes, we propose the Transolver with the following features:

- Going beyond previous work, Transolver **calculates attention among learned physical states** instead of mesh points, which empowers the model with **endogenetic geometry-general capability**.
- Transolver achieves **22% error reduction over previous SOTA in six standard benchmarks** and excels in **large-scale industrial simulations**, including car and airfoil designs.
- Transolver presents favorable **efficiency, scalability and out-of-distrbution generalizability**.

<p align="center">
<img src=".\pic\Transolver.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of Transolver.
</p>


## Transolver v.s. Previous Transformer Operators

**All of the previous Transformer-based neural operators directly apply attention to mesh points.** However, the massive mesh points in practical applications will cause challenges in both computation cost and capturing physical correlations.

Transolver is based on a more foundational idea, that is **learning intrinsic physical states under complex geometrics**. This design frees our model from superficial and unwieldy meshes and focuses more on physics modeling.

As shown below, **Transolver can precisely capture miscellaneous physical states of PDEs**, such as (a) various fluid-structure interactions in a Darcy flow, (b) different extrusion regions of elastic materials, (c) shock wave and wake flow around the airfoil, (d) front-back surfaces and up-bottom spaces of driving cars.

<p align="center">
<img src=".\pic\physical_states.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of learned physical states.
</p>

## Get Started

1. Please refer to different folders for detailed experiment instructions.

2. List of experiments:

- Core code: see [./Physics_Attention.py](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
- Standard benchmarks: see [./PDE-Solving-StandardBenchmark](https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark)
- Car design task: see [./Car-Design-ShapeNetCar](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar)
- Airfoil design task: see [./Airfoil-Design-AirfRANS](https://github.com/thuml/Transolver/tree/main/Airfoil-Design-AirfRANS)

## Results

Transolver achieves consistent state-of-the-art in **six standard benchmarks and two practical design tasks**. **More than 20 baselines are compared.**

<p align="center">
<img src=".\PDE-Solving-StandardBenchmark\fig\standard_benchmark.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Results on six standard benchmarks.
</p>

<p align="center">
<img src=".\Airfoil-Design-AirfRANS\fig\results.png" height = "300" alt="" align=center />
<br><br>
<b>Table 2.</b> Results on two design tasks: Car and Airfoild design.
</p>

## Showcases

<p align="center">
<img src=".\pic\showcases.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 3.</b> Comparison of Transolver and other models.
</p>

- Application of Transolver for crash dynamics modeling: [https://arxiv.org/pdf/2510.15201](https://arxiv.org/pdf/2510.15201)

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [wuhx23@mails.tsinghua.edu.cn](mailto:wuhx23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Latent-Spectral-Models

https://github.com/Extrality/AirfRANS
