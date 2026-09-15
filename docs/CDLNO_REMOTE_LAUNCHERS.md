# 远端八任务启动脚本交付

本轮仅执行用户点名的远端路径配置、各数据集训练/评估脚本和只读数据格式检查工具；未启动前段latent processor消融补充阶段，未运行真实训练或评价。

## A. 范围与依据

起点HEAD：`769fa333742f73c132868cf560bce5ec21529362`（main，用户此前提交的现有实现）；起点唯一未跟踪文件为用户的空 `path.sh`（0字节）。本輪不执行commit/push。与阶段0上游commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` 区分，保护当前所有已提交实现。

用户提供远端仓库 `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w` 和数据根 `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data`。仓库根自动从path.sh位置定位；公共六PDE根为data/fno。逐个阅读真实exp/main的路径拼接、parser、JSON和已确认launchers，保留v1.2及随后NS无clip修正。

核心发现：Elasticity参数是fno父目录，入口追加elasticity/Meshes；NS参数是fno父目录并追加完整子目录/MAT；Plasticity参数是单个MAT。AirfRANS train/eval的my_path分别是Dataset和其父目录。不能统一给所有入口追加任务文件夹。

## B. 文件与diff摘要

| 文件 | 原因/改动 |
|---|---|
| [path.sh](../path.sh) | 填充用户空文件，配置远端data/fno、可移植repo根、每任务路径、run根/tag、解释器；各项可环境覆盖；不修改shell系统变量 |
| [tran_evaluate/_common.sh](../tran_evaluate/_common.sh) | 接入path.sh，增加Air的my_path/save_path相对路径处理；保留dry-run及Car原路径guard |
| [tran_evaluate/_standard.sh](../tran_evaluate/_standard.sh) | 六任务共享启动参数分派，显式任务d/h/M/batch/位置/采样/clip；仅shell，不复制模型或训练器 |
| [darcy.sh](../tran_evaluate/darcy.sh)、[elasticity.sh](../tran_evaluate/elasticity.sh)、[airfoil.sh](../tran_evaluate/airfoil.sh)、[pipe.sh](../tran_evaluate/pipe.sh)、[ns.sh](../tran_evaluate/ns.sh)、[plasticity.sh](../tran_evaluate/plasticity.sh) | 每任务train/eval/help入口；NS原显式设置移至共享分派，行为与JSON一致 |
| [car.sh](../tran_evaluate/car.sh) | 原设置不变，增加远端raw/cache和独立run默认；完整阻力评价仍受原guard限制 |
| [airfrans.sh](../tran_evaluate/airfrans.sh) | 新增两模式入口，398epochs/batch1，正确转换Dataset/parent；原入口无gpu参数，使用CUDA_VISIBLE_DEVICES |
| [inspect_data.sh](../tran_evaluate/inspect_data.sh)、[inspect_data.py](../tran_evaluate/inspect_data.py) | 只读目录/NPY头/MAT字段/manifest/有限VTK头；候选查找有深度和数量限制，缺失缓存计数；不读取task loader或torch |
| [README](../tran_evaluate/README.md)、STATUS、AGENTS、memory、八任务清单、本报告 | 更新远端命令和证据；明确工业目录是假设候选、前段消融尚未授权实施 |
| [remote_launch_audit/](remote_launch_audit/) | 起点SHA/文本、实际参数检查代码/日志/JSON、冻结清单及可审查patch |

本轮可审查差异：[remote-launch-changes.patch](remote_launch_audit/remote-launch-changes.patch)。不修改三个原任务项目内的任何源码、模型、配置或原脚本。

## C. 配置、形状与原调用对应

统一默认L8/F2/P6、完整LRSA前段、CDPA entry/chunk0、前段/点FFN与后段GEGLU ratio2、dropout0。未新增front_latent_mode参数；no_sa/identity属于已记录但未实施的补充计划。

| 任务 | d/h/M | epochs/batch | 单次模型接口/位置条件 | 脚本数据参数 |
|---|---|---|---|---|
| Darcy | 128/8/64 | 500/4 | x2+fx1；reference64替换xy，stem65；[B,7225,1] | fno |
| Elasticity | 128/8/64 | 500/1 | x2/placeholder，972点；[B,972,1] | fno，非fno/elasticity |
| Airfoil | 128/4/64 | 500/4 | 原xy/placeholder；[B,221×51,1] | fno/airfoil/naca |
| Pipe | 128/4/32 | 500/8 | 原归一化xy/placeholder；[B,129×129,1] | fno/pipe |
| NS | 256/8/64 | 500/2 | reference64+fx10，stem74；每次[B,4096,1] | fno，由原entry追加MAT路径 |
| Plasticity | 128/8/64 | 500/8 | x2/fx1/T[B,1]；每次[B,3131,4] | fno/plas_N987_T20.mat |
| Car | 256/8/64 | 200/1 | (cfd,geom)，x7→[N,velocity3+pressure1] | 候选mlcfd_data/{training_data,preprocessed_data} |
| AirfRANS | 256/8/64 | 398/1 | x7+reference64→[N,vx/vy/p/nut] | 候选AirfRANS/Dataset；eval用其parent |

六PDE lr1e-3/weight_decay1e-5，NS无裁剪，其他五任务clip0.1；与原任务clip配置及当前CDLNO preset一致。NS10次真值回填后一次backward/step、测试预测回填；Plasticity20次独立时间条件更新；Elasticity CosineAnnealing按epoch；其余原scheduler不变。Car Adam/reg0.5/fold0、Air Adam/398及原图抽样参数保持。F/P、CDPA/bridge/readout公式没有变化。

## D. 实际命令、结果、未验证项

实际执行：

```bash
python -B docs/remote_launch_audit/verify_remote_launchers.py
git diff --check
```

验证工具实际对path.sh和tran_evaluate各shell执行bash -n，对Python脚本按Python3.10语法解析；16条train/eval默认命令执行dry-run，并在三个原cwd新进程只执行AST提取的真实parser定义及安全helper（不import exp/main），逐字段比较既有JSON/YAML。再经shell解析8组架构/chunk/相对路径/含空格/flag=value覆盖；核对train/eval同目录、八任务互不覆盖、环境data/run/tag覆盖。

| 项目 | 本轮结果 |
|---|---|
| 8任务×train/eval=16条默认命令，JSON/YAML/真实parser比较 | **通过** |
| 8组用户覆盖参数、环境覆盖、相对/空格路径、dry-run无写入 | **通过** |
| 8个独立run默认路径、train/eval路径一致 | **通过** |
| Shell语法、Python3.10语法、git diff --check | **通过** |
| 数据检查器：空临时目录/不存在目录，缺文件如实报告、不创建文件 | **通过** |
| NPY v1/v2/v3 header解析、错误magic拒绝 | **通过**；仅内存BytesIO测试，没有创建假数据文件 |
| 远端真实NPY/MAT/HDF5/manifest/VTK读取 | **未运行**；工具已提供，无远端挂载/SSH连接证据 |
| 模型前反向、PyG集成、GPU/合成训练步 | **本轮未重跑**；没有改模型，已有120项通过是上一轮证据 |
| 远端torch2.11/cu128新模型兼容、真实训练、收敛/精度/epoch效率 | **未运行/未验证** |

### 远端旧版 `path.sh` 兼容补充

用户在远端执行旧版 `path.sh` 时，原文件只导出了 `REPO`、`DATA`、`FNO_DIR`、`CAR_DATA_DIR`、`CAR_SAVE_DIR`、`AIRFRANS_DATASET` 和 `CONDA_ENV`，没有 `CDLNO_PYTHON`，因此 `inspect_data.sh` 在 `set -u` 下于第5行退出。现已在 `inspect_data.sh` 和 `_common.sh` 增加兼容回退：解释器默认当前 `python`，由 `DATA/FNO_DIR` 派生 PDE 路径、Plasticity MAT、run 根/tag；当前 checkout 路径仍由脚本自身定位，不受旧 `REPO=/.../Transolver_re` 劫持。

在临时旧版 `path.sh`（无任何 `CDLNO_*` 变量）下实际执行：

```bash
bash tran_evaluate/inspect_data.sh --no-search
bash tran_evaluate/{darcy,elasticity,airfoil,pipe,ns,plasticity,car,airfrans}.sh train --dry-run
bash tran_evaluate/{darcy,elasticity,airfoil,pipe,ns,plasticity,car,airfrans}.sh eval --dry-run
bash -n tran_evaluate/inspect_data.sh tran_evaluate/_common.sh tran_evaluate/_standard.sh
```

结果为 **inspect 通过（空临时根目录返回正常 JSON）**、**16/16 train/eval dry-run 通过**、**Shell语法通过**；没有调用 Python 任务入口、读取真实数据或创建训练目录。当前本地 `path.sh` 指向的远端目录未挂载，直接检查返回 `root_exists=false`/状态码2，这是数据不可见的明确报告，不是变量错误。

日志与实参：[launcher-checks.log](remote_launch_audit/launcher-checks.log)、[launcher-checks.json](remote_launch_audit/launcher-checks.json)。实际验证使用当前已有本地解释器；不安装依赖，不把本地Python3.13的parser执行当作目标Python3.10/torch2.11训练验收。Python3.10语法检查也不是远端实际执行证据。

## E. 冻结证据

[start.json](remote_launch_audit/start.json) 保存起点187个文件的SHA（含空path.sh）；本轮仅改变path/tran_evaluate及本报告表列状态文档，具体数量见 [freeze-check.json](remote_launch_audit/freeze-check.json)。所有`cdlno/`核心、三个子项目的源码/配置/launchers、既有tests及依赖逐文件字节相同。没有下载、解压、转换、重采样或创建假数据，没有导入会读数据的实验模块，没有自动commit/push。

## F. 自审与需注意的剩余事项

1. **路径语义已自审通过**：Elasticity/NS父目录、Plasticity文件、AirfRANS两入口差异均由实际源码与最终argv验证；自定义my_path可覆盖脚本默认。
2. **模型/训练默认已自审通过**：16条命令逐项匹配已接受preset；Pipe32/batch8、Airfoil4heads、Air398、NS无clip没有被原parser旧默认覆盖。前段消融参数未混入。
3. **目录和覆盖已自审通过**：八任务分目录、train/eval默认相同；已有训练目录仍由原run helper拒绝复用，未创建latest自动选择或隐式resume；tag不设置随机种子。
4. **远端文件格式仍未知**：用户只提供根路径和Elasticity子目录，不能声称所有细节已观察。应在远端运行inspect_data.sh，依据报告修改path.sh中的实际子目录；工业两个默认候选尤其需要确认。
5. **Car原阻力评价路径限制仍未解决**：原函数硬编码`/data/PDE_data/mlcfd_data/training_data/param0`，与新`/inspire/...`默认不同。现有同数据目录路径映射可兼容；没有映射时新eval仍明确报错。单改脚本不能改变被冻结的metric内部常量，本轮不改指标、不自动建链接；这是原有问题，不是新脚本支持任意root/fold的证据。训练加载本身可使用新raw/cache路径。原loader会跳过缺样本，工具报告缺失计数但不改原行为。

远端可执行命令和完整说明见 [tran_evaluate/README.md](../tran_evaluate/README.md)。

**本补充阶段结束，未执行下一阶段。**
