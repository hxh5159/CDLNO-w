# CDLNO 阶段7：ShapeNet-Car

完成日期：2026-09-14。用户已审查通过阶段6；本阶段仅授权 ShapeNet-Car。AirfRANS 未接入。

## A. 完成范围

完成稳定 wrapper、模型选择及参数转发、JSON 预设、独立训练/评估脚本、运行目录和可信整模型 checkpoint/sidecar 接入。复用共享 `cdlno.core.CDLNO`；没有复制或修改数学核心。

默认 d=256、h=8、M=64、L=8、F=2、P=6，point FFN、entry CDPA、chunk=0。保留原 Adam、lr=0.001、batch=1、200 epochs、reg=0.5、OneCycle 和 val_iter=10。一个 run 只对应一个 fold，fold_id=0..8。

## B. 文件、理由与有限 diff

| 文件 | 变更和理由 |
|---|---|
| [models/CDLNO.py](../Car-Design-ShapeNetCar/models/CDLNO.py) | 稳定 `models.CDLNO.Model` 类；7 通道输入提升、原 placeholder、共享核心、单图检查；不注册 time_fc |
| [models/cdlno_run.py](../Car-Design-ShapeNetCar/models/cdlno_run.py) | 仅处理 CDLNO 参数、sidecar、run/eval 目录和可信整模型加载；旧模型路径不加载共享模型代码 |
| [main.py](../Car-Design-ShapeNetCar/main.py) | 新增 CDLNO 构造与运行路径分支；原 hparams、loader 和 train.main 调用保留 |
| [main_evaluation.py](../Car-Design-ShapeNetCar/main_evaluation.py) | CDLNO sidecar/整模型加载和独立结果路径；原指标数组/反归一化/mask/drag 处理保留 |
| [shapenet_car.json](../Car-Design-ShapeNetCar/configs/CDLNO/shapenet_car.json) | 显式架构预设与原训练默认记录 |
| [CDLNO.sh](../Car-Design-ShapeNetCar/scripts/CDLNO.sh)、[CDLNO_Evaluation.sh](../Car-Design-ShapeNetCar/scripts/CDLNO_Evaluation.sh) | 显式模型配置；用户参数放在最后，可以覆盖；不自动执行多个 fold |
| [test_shapenet_car.py](../tests/test_shapenet_car.py) | 13 项接口/损失/梯度/新进程/sidecar/脚本/冻结验证；只在真实 PyG 缺失时跳过对应集成项，不伪造模块 |
| AGENTS、STATUS、memory、本报告 | 记录本阶段交付、已自审结果和实际未运行项 |

原 tracked 文件的本阶段 diff 合计 **26 行新增、10 行删除**：main.py 为 +8/−2，main_evaluation.py 为 +18/−8；上述新文件另外列入交付，不混入先前六标准任务的未提交 diff。

两项明确授权的兼容例外：

1. 原评估 `--nb_epochs` 为 `type=float`；显式 `--nb_epochs 200` 会构造 `model_200.0.pth`，训练保存的是 `model_200.pth`。仅改这一处 parser 类型为 int；默认不传参数时的 200 路径保持，Transolver 路径修复也有测试。
2. 原训练保存完整模型对象，不改为 state_dict。本项目自己的可信 checkpoint 在 CarRun 和原评估加载位置显式使用 `weights_only=False, map_location=device`；没有全局 monkeypatch、环境变量或 allowlist 放宽，也未改其他任务加载器。

默认新训练目录带 `fold/L/F/M/mode/时间戳/UUID`；显式 `--run_dir` 表示最终目录，已存在则报错。评估必须指向已有目录；每次结果写到该目录下新的 `eval_<时间戳_UUID>/`，不覆盖其他运行或评估。

从原 Car 工作目录使用（仅示例，本阶段没有启动这些真实入口）：

```bash
# 根级 cdlno 需已在当前环境可导入；本阶段未安装或替换任何依赖。
cd Car-Design-ShapeNetCar
bash scripts/CDLNO.sh --data_dir /path/to/training_data --save_dir /path/to/preprocessed_data --fold_id 0
bash scripts/CDLNO_Evaluation.sh --run_dir /path/to/the/printed/run --data_dir /path/to/training_data --save_dir /path/to/preprocessed_data --fold_id 0
```

自定义 L/F/M/d/h、epoch 或 fold 的评估，需要传入与训练一致的值；默认值不会自动覆盖 sidecar。chunk 可独立改变。原 `--cfd_mesh`/`--r` 含义不变，评估与训练需匹配。

## C. 公式、张量与源码对应

```text
(cfd_data, geom_data)
cfd_data.x: [N,7] = [xyz3, sdf1, normal3]（保持原归一化后的字段）
H0 = Linear_2(GELU(Linear_1(x[None,...]))) + placeholder[None,None,:]
     [1,N,7] -> [1,N,2d] -> [1,N,d]
H0 -> F个完整前段 -> HF[1,N,d], T_i[1,M,d]
HF -> bridge -> Z0[1,M,d] -> CDPA/后段 -> ZP[1,M,d]
readout(HF,ZP): [1,N,4] -> [N,4] = [velocity3, pressure1]
```

stem 的同权重输出与原 Transolver `preprocess + placeholder` 精确一致。placeholder 为 `(1/d)*rand(d)`，只在本任务实际 `fx=None` 路径注册；不添加时间投影、reference 编码或 geom encoder。普通线性层局部初始化；wrapper 不递归 apply，不覆盖核心 queries 或零初始化 w。

wrapper 只读取 x 和 batch/ptr 分图元数据；不读取 y、surf、pos、edges 或 geom_data，不修改输入。输出节点与输入逐点对应，未重排/采样；节点置换等变也有合成检查。

普通 Data 无 batch/ptr 可以使用；全零 `[N]` batch 和 `[0,N]` ptr 合法。含多个图的 batch 或 ptr（包括全零 batch 但 ptr 表示第二个空图）都明确报错；畸形分图元数据也拒绝。N 不写死，测试包含 7、29 及合成 32186 点。

原训练损失保持：

```text
loss_press = MSE(out[surf,-1], y[surf,-1]).mean()
loss_velo  = MSE(out[:,:-1], y[:,:-1]).mean(dim=0).mean()
loss       = loss_velo + reg * loss_press
```

训练速度损失使用全部节点；原完整评估的体积速度指标使用 `~surf`。两者的区别保留，未统一 mask。原 train.py 的 loss 返回值命名/日志聚合也未顺便修改。

checkpoint 顺序：读取 sidecar → 比较请求架构、wrapper、fold/epoch/原图设置 → 局部加载可信完整模型 → 校验模型类和内外 config → 对同配置临时实例执行 strict state_dict 结构核验 → 应用请求的运行 chunk。返回加载的模型；不重新初始化其权重，不改原整模型保存语句或频率。

## D. 实际验证与证据边界

命令：

```bash
python -B -m unittest discover -s tests -p test_shapenet_car.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator \
  python -B -m unittest discover -s tests -p 'test_*.py' -v
```

阶段定向测试最终 **13/13 通过**。全回归最终 **94/94 通过，0失败/错误/跳过，57.568秒**，日志 `/tmp/cdlno-phase7-final-tests.log`。测试执行环境为 Python3.13.9、torch2.13.0+cu130、PyG2.3.1，非用户远端目标环境；没有安装/替换依赖。

| 实际覆盖 | 证据 |
|---|---|
| 默认 d/h/M/L/F、off/entry/every 和活跃路径 | 默认宽度小 N 前反向，所有注册参数梯度存在且有限；w严格零，允许 score-only norm 首步梯度为零 |
| 原 placeholder/输入提升、初始化 | 同权重原 Transolver stem 精确对照；front query 正交初始化和无 time_fc/Conv2d 检查 |
| 真 PyG 接口、可变 N、无原地修改 | Data/单图 Batch、字段值及 storage、改/删 y、替换 geom/pos、节点置换、32186 点减小 d/M 前反向 |
| 多图负向 | 真实双图 Batch、仅 ptr、多图 ptr+全零 batch、畸形元数据拒绝 |
| 原 mask loss/更新节奏 | 安全导入只定义函数的原 train.py；真 PyG DataLoader 合成一批，执行原 train.train/test；对照 loss 和所有梯度，并核对一次 Adam/OneCycle 更新 |
| 独立进程整模型保存/加载 | 从 Car cwd 启动两个独立 Python 进程：第一个执行原 train.py 提取的完整保存表达式，第二个读 sidecar/load；输出一致、类路径稳定、chunk0→1、sidecar 字节不变 |
| sidecar/目录隔离 | 错误 M/mode/fold/epoch 在 torch.load 前拒绝；错误 wrapper/缺失 sidecar 不写回；错误模型类及缺失参数拒绝；原文件字节保持，重复训练目录拒绝、各 eval 路径独立 |
| 原入口冻结与启动脚本 | AST 仅投影新分支、结果路径别名和两项授权兼容改动后，两个完整入口均等于基线；bash语法、捕获命令参数覆盖、3.10语法解析通过 |

另行在实际 RTX5090 Laptop 上执行真 PyG 单图 Batch 的 CUDA FP32 off/entry/every 三模式检查（N31、d16、h4、M4、L8/F2）：输出形状与所有参数梯度有限检查通过；不计入上述 unittest 数量，不代表 Car FP16/BF16 或远端 GPU 验收。该 probe 从 Car cwd 以 `PYTHONPATH=/home/hwz/CDLNO python -B -` 执行合成输入及原 pressure/velocity mask 公式；没有运行 main 入口。

独立进程检查仅设置 `PYTHONPATH` 为仓库根，使共享包可导入；本阶段没有执行 editable 安装，不能把此结果描述成已在远端安装成功。没有 import/执行顶层数据读取的 main.py/main_evaluation.py；测试 parser 仅提取其 argparse 定义。

中间结果：初版 3 项测试/84 项回归没有覆盖 placeholder、ptr、独立结果目录和先读 sidecar 等缺口，不能据此作为最终验收。续做自审补齐实现后，新增测试曾因把 w 写成不存在的 `depth_score` 报 1 个 AttributeError；修正测试名后 13 项全部通过。早期手工多图样例字段不齐导致 PyG 构造阶段 `KeyError: pos`，不计作多图拒绝证据；最终使用字段一致的真实双图 Batch 验证。

未运行：远端 Python3.10/3.11、torch2.11、CUDA12.8；真实 VTK/预处理数据、完整 loader/radius_graph/drag 后处理；真实 ShapeNet-Car 训练/评估、九折、收敛/准确率/性能对比。已有用户远端 Transolver 训练成功证据保持，但不替代 CDLNO 验收。

## E. 冻结区域及已有修改保护

Git 仍为 `main`，基线 `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。六标准任务已有 diff 和共享包属于先前阶段，未重置或改动。

原 Car/AirfRANS 共 **37 个非入口文件逐字节等于 Git 基线**；包括 Car train.py、dataset、原 Transolver、原脚本及全部 AirfRANS。两个 Car 入口执行完整 AST 投影核验，非授权片段均等于原文件。数据、fold、normalizer、surf、训练/测试循环、optimizer/scheduler、指标和节点顺序无修改；变化仅限新模型/参数/路径/checkpoint及 eval epoch 类型。

续做快照 `/tmp/cdlno-phase7-continuation-baseline/hashes.json` 共135文件；11个本阶段目标被更新，其余 **124个非目标文件哈希全部不变**，包括阶段0–6代码/测试及已有依赖配置。清单SHA256为 `b664de9942e5f9ce4683ed086e7fd152de3b86d037ab510f36af921005101612`。新增JSON预设另列入交付。未下载数据、创建伪造数据集、安装依赖、真实训练、commit/push、PR 或 reset。

## F. 已完成自审与待验证项

已自行检查五项：原7通道+placeholder及初始化；batch/ptr/可变N和节点顺序；训练全节点速度/表面压力与评估mask区别；sidecar读取顺序/整模型strict核验/独立目录；旧分支及AirfRANS冻结。这些通过项不再作为未经检查的清单交给用户。

本阶段范围内未发现剩余待修复实现问题。待验证的是远端目标环境、真实 loader/几何后处理及实际训练效果；不声称单fold等于九折。AirfRANS等待下一次明确授权。

**本阶段结束，未执行下一阶段**
