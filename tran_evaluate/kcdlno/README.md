# KCDLNO/KCDNO 八任务启动脚本

此目录提供每个数据集一个薄包装。仓库中实际注册的模型 family 是
`kcdno`（KCDNO）；目录名 `kcdlno` 只是本组启动脚本的名称，不改变模型注册。
包装复用 `tran_evaluate/kcdno/` 的已验证入口、profile、checkpoint 和原任务协议，
不复制模型或训练循环。

从仓库根目录运行：

```bash
bash tran_evaluate/kcdlno/darcy.sh train_eval --gpu 0
bash tran_evaluate/kcdlno/elasticity.sh train_eval --gpu 0
bash tran_evaluate/kcdlno/airfoil.sh train_eval --gpu 0
bash tran_evaluate/kcdlno/pipe.sh train_eval --gpu 0
bash tran_evaluate/kcdlno/ns.sh train_eval --gpu 0
bash tran_evaluate/kcdlno/plasticity.sh train_eval --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdlno/car.sh train_eval
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdlno/airfrans.sh train_eval
```

每条命令对应一个独立实验，按需选择执行。无参数或 `--help` 只显示用法。
追加 `--dry-run` 只预览命令；直接传 `--dry-run` 也会预览训练和评估两步。

`train_eval` 先训练，只有训练成功才评价同一 run。可用 `train` 或 `eval`
分别执行，也可把 `all`、`off` 或 `lrsa_matched` 作为第二个参数选择
KCDNO 主版、无历史对照或完整 LRSA 对照：

```bash
bash tran_evaluate/kcdlno/darcy.sh train_eval off --gpu 0
bash tran_evaluate/kcdlno/darcy.sh train --gpu 0 --kcdno-run-dir "$PWD/output/darcy/kcdno/run1"
bash tran_evaluate/kcdlno/darcy.sh eval --kcdno-run-dir "$PWD/output/darcy/kcdno/run1" --gpu 0
bash tran_evaluate/kcdlno/darcy.sh train_eval all --dry-run
```

默认结构由 `kcdno_v1` profile 提供（L=8、kernel rank=16、history=all，
两个 latent FFN 均为 hidden=2d 的 GELU FFN）。任务预设如下，脚本从现有配置读取，
不把这些尺寸硬写成覆盖 profile 的 CLI 参数：

| 数据集 / 脚本 | d | h | M | 点模块 | epochs / batch |
|---|---:|---:|---:|---|---|
| Darcy / darcy.sh | 128 | 8 | 64 | ConvFFN | 500 / 4 |
| Elasticity / elasticity.sh | 128 | 8 | 64 | PointFFN | 500 / 1 |
| Airfoil / airfoil.sh | 128 | 4 | 64 | ConvFFN | 500 / 4 |
| Pipe / pipe.sh | 128 | 4 | 32 | ConvFFN | 500 / 8 |
| Navier–Stokes / ns.sh | 256 | 8 | 64 | ConvFFN | 500 / 2 |
| Plasticity / plasticity.sh | 128 | 8 | 64 | ConvFFN | 500 / 8 |
| ShapeNet-Car / car.sh | 256 | 8 | 64 | PointFFN | 200 / 1 |
| AirfRANS / airfrans.sh | 256 | 8 | 64 | PointFFN | 当前 YAML 398 / 1 |

显式参数仍放在命令末尾覆盖默认值，例如 `--profile transolver_shape_match`、
`--kernel-rank 32`。KCDNO 不使用 CDLNO 的 front-blocks/front-latent-mode/CDPA 选项。
独立 `eval` 不注入 history 默认值，会从保存的 sidecar 恢复。

连续运行默认创建 `output/<数据集>/kcdno/<UTC时间戳>_all_<进程号>/`，
训练与评估共享该目录；配置、实际参数量、权重和结果沿用已有记录机制。
`--kcdno-run-dir` 用于指定新的训练目录或已有评价目录。训练目录必须不存在；评价
先读取 sidecar 并严格校验架构。不要提前 mkdir 该实验目录。

路径仍由仓库根 `path.sh` 及 `CDLNO_*` 环境变量提供，脚本可从任意工作目录调用，
也可迁移到远端 checkout。工业数据默认路径目前只是既有候选，须按实际位置设置。
Car 的 `--data_dir` 是原始数据、`--save_dir` 是预处理数据，均不是权重目录；
原完整 drag 评价仍要求 fold0 及 `/data/PDE_data/mlcfd_data/training_data/param0`
原始路径可用，已有 preflight 会核查。本包装不改变此限制。

AirfRANS 沿用原项目的路径语义：训练 `--my_path`
指向包含 manifest 的 `Dataset`，评价指向其父目录；因此自定义 AirfRANS 路径时建议
用环境变量让已有脚本分别处理两步，也可以分别运行 train 和 eval：

```bash
CDLNO_AIRFRANS_DATASET=/actual/path/Dataset CUDA_VISIBLE_DEVICES=0 \
  bash tran_evaluate/kcdlno/airfrans.sh train_eval
```

不要给 AirfRANS 顺序包装传同一个 `--my_path` 覆盖两步。选已有 Python 可设置
`CDLNO_PYTHON=/absolute/path/to/python`。模型保存沿用标准任务 state_dict、
Car 整对象、AirfRANS 列表/整对象；这些任务文件尚不含完整 optimizer/scheduler/RNG
续训状态，不能把此脚本当成断点续训入口。

更完整的配置和加载说明见 [KCDNO 命令文档](../../docs/KCDNO_COMMANDS.md)。

本次最终核查：9 个 shell 语法检查、120 个受保护的分发/帮助/负向检查通过，
共预览 112 条底层命令；最终检查用会报错的 Python 替身确认未调用数据入口。
参数覆盖、含空格路径、同一 run、Air 两种路径语义通过；173 个已有生产/脚本文件
hash 不变。明细见 [verification.json](verification.json)。

初次帮助检查曾发现并修复 `--help` 被消费后误入训练入口的问题；该调用因本地
缺失远端 Darcy 文件而在数据读取处失败，未进行模型训练，其失败目录已清理。
最终版本的帮助、无参数和选项独立调用已分别复查。此次没有真实数据训练/评价或
新增模型/GPU验收；沿用此前已交付的模型验证范围。

本阶段结束，未执行下一阶段。
