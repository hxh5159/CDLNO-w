# A2：八任务前段消融训练与评估命令

当前输出规则（2026-09-15）：默认改为 `output/<dataset>/<timestamp>`。以下显式旧路径仍能加载；历史按模式追加目录后缀的规则已由时间戳规则替代。选择固定 `CDLNO_RUN_TAG` 时，不同模式必须使用不同tag；优先使用不设tag的 `train_eval.sh TASK --front-latent-mode MODE`，详见 [统一实验记录](CDLNO_EXPERIMENT_OUTPUTS.md)。

A4交付时重新核对本页实际脚本/参数和预览命令，仍适用于full/no_sa/identity；见[A4报告](CDLNO_FRONT_ABLATION_A4.md)。两个消融默认从头训练，不提供跨模式权重迁移。本页是训练/评估命令，不包含尚未接入任务入口的V1续训公共组件。

从远端实际仓库根目录运行，沿用 `path.sh` 和已有环境。本页命令未由代理执行真实训练。八任务均支持 `--front-latent-mode full|no_sa|identity`（也接受下划线写法）；省略时新训练选择full。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
mode=no_sa                 # 可改为 full 或 identity
export CDLNO_RUN_TAG="trial1_${mode}" # 每次实验选择新标识，禁止覆盖；不设tag则自动时间戳
```

下面每行是独立任务，选择需要的一行运行。训练成功后才评估，同一个mode和run贯穿两步。模式属于共享参数，应放在 `--train-args/--eval-args` 之前；其余用户覆盖仍放在默认值后面。

```bash
bash tran_evaluate/train_eval.sh darcy      --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh elasticity --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh airfoil    --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh pipe       --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh ns         --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh plasticity --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh car        --front-latent-mode "$mode" --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans --front-latent-mode "$mode"
```

末尾加 `--dry-run` 只预览两条命令，不读取sidecar/数据、不创建run、不调用Python入口：

```bash
bash tran_evaluate/train_eval.sh darcy --front-latent-mode identity --gpu 0 --dry-run
```

也可分开执行；下面八对命令适用于三个mode，训练与评估使用同一个 `CDLNO_RUN_TAG`：

| 任务 | 训练 | 评估 |
|---|---|---|
| Darcy | `bash tran_evaluate/darcy.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/darcy.sh eval --front-latent-mode "$mode"` |
| Elasticity | `bash tran_evaluate/elasticity.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/elasticity.sh eval --front-latent-mode "$mode"` |
| Airfoil | `bash tran_evaluate/airfoil.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/airfoil.sh eval --front-latent-mode "$mode"` |
| Pipe | `bash tran_evaluate/pipe.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/pipe.sh eval --front-latent-mode "$mode"` |
| Navier–Stokes | `bash tran_evaluate/ns.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/ns.sh eval --front-latent-mode "$mode"` |
| Plasticity | `bash tran_evaluate/plasticity.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/plasticity.sh eval --front-latent-mode "$mode"` |
| ShapeNet-Car | `bash tran_evaluate/car.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/car.sh eval --front-latent-mode "$mode"` |
| AirfRANS | `bash tran_evaluate/airfrans.sh train --front-latent-mode "$mode"` | `bash tran_evaluate/airfrans.sh eval --front-latent-mode "$mode"` |

标准任务和Car默认GPU0；AirfRANS沿用 `CUDA_VISIBLE_DEVICES`，没有 `--gpu` 选项。现有子项目 `scripts/CDLNO*.sh` 也可直接追加mode和原run选项；它们最后的 `"$@"` 已支持转发，不必复制三份脚本。

## 默认配置和目录

各模式保持F2/L8/P6、两种FFN ratio2、entry CDPA、chunk0。

| 任务 | d/h/M | epochs / batch | 学习率 |
|---|---|---|---|
| Darcy | 128/8/64 | 500 / 4 | .001 |
| Elasticity | 128/8/64 | 500 / 1 | .001 |
| Airfoil | 128/4/64 | 500 / 4 | .001 |
| Pipe | 128/4/32 | 500 / 8 | .001 |
| NS | 256/8/64 | 500 / 2 | .001 |
| Plasticity | 128/8/64 | 500 / 8 | .001 |
| Car | 256/8/64 | 200 / 1，fold0 | .001 |
| AirfRANS | 256/8/64 | 398 / 1，full/nmodel1 | .001 |

NS仍不裁剪梯度；其余标准任务clip0.1。原optimizer/scheduler、损失、时间语义均不变。NS保持10→10及训练真值/测试预测回填；Plasticity仍按20个时间点分别更新，原时间嵌入不变。

当前根任务脚本默认run路径，R=`CDLNO_RUNS_ROOT`（默认仓库根/output），tag为明确设置的 `CDLNO_RUN_TAG` 或每次训练自动生成的UTC时间戳：

| 类别 | 三模式通用目录 |
|---|---|
| 六标准任务 | `R/<task>/<timestamp或tag>` |
| Car | `R/car/<timestamp或tag>` |
| AirfRANS | `R/airfrans/<timestamp或tag>` |

每种模式每次训练获得独立时间戳。手动固定tag时必须为新实验换tag；不能只改变mode并复用原tag。
显式 `--cdlno-run-dir`（标准）或 `--run_dir`（工业）保持用户路径，不追加模式；请选择不同的新目录。已有目录拒绝再次训练，不清空、不resume。直接使用Python入口/子项目脚本且省略run目录时，也使用统一output根和新的UTC时间戳。标准任务显式 `--save_name` 原样保留，现有唯一目录及覆盖防护继续生效。

## 评估已有checkpoint

评估先读取已有sidecar。省略mode时从sidecar恢复；显式mode必须相同。例如训练沿用上面的trial1及默认架构：

```bash
bash tran_evaluate/darcy.sh eval \
  --cdlno-run-dir "$PWD/runs/CDLNO/darcy/trial1_no_sa" --gpu 0
bash tran_evaluate/car.sh eval \
  --run_dir "$PWD/runs/CDLNO/car/fold0_trial1_identity" --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/airfrans.sh eval \
  --run_dir "$PWD/runs/CDLNO/airfrans/full_trial1_no_sa"
```

这里只自动恢复新增mode；其他自定义d/M/F/L、fold、epoch、task/nmodel等仍按原协议在评估时重复提供，并未实现全部配置自动恢复。根脚本未指定mode或run时仍定位旧full路径；要省略mode评估消融，请显式指定消融run目录。错误mode在加载权重前报架构冲突，sidecar不改写。不支持跨mode权重迁移，两消融按原新训练流程从头训练。

Car保持完整阻力评价原限制：fold0，原指标要求同一份raw数据在 `/data/PDE_data/mlcfd_data/training_data/param0` 可访问；`--save_dir` 是预处理数据目录。AirfRANS训练 `--my_path` 指Dataset，评估指其父目录，组合脚本保留两种语义。详见 [原启动说明](../tran_evaluate/README.md)。远端路径和数据是否满足未由本次合成验证确认。

本页提供可执行命令，不代表已经验证真实训练、收敛、精度或性能。实现和实际证据见 [A2报告](CDLNO_FRONT_ABLATION_A2.md)。
