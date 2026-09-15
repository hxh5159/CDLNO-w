# Darcy 前段 latent processor 消融

两个目录分别提供 `train.sh`、`eval.sh`、`train_eval.sh`，复用现有 Darcy 入口。组合脚本训练成功后才评估，两步共用同一配置与实验目录。

| 目录 / 模式 | Down 与 Up 之间的计算 |
|---|---|
| `no_sa` | `A=S+FFN1(N1(S)); T=A+FFN2(N2(A))`，跳过 latent SA 及其 norm |
| `identity` | `T=S`，跳过两个 latent FFN、latent SA 及其专属 norms |

两者均保留 Down/Up Cross attention、点域 dense ConvFFN、bridge、后段 SA/GEGLU、CDPA 和最终读出。因此 identity 不是将整个 block 改成线性映射，也不是全模型无 attention。

默认配置继承现有 Darcy 脚本：L=8、F=2、P=6、d=128、heads=8、M=64，CDPA=entry、chunk=0；FFN ratios=2、dropout=0、reference=8、downsample=5、ntrain=1000；500 epochs、batch=4、lr=0.001、weight decay=1e-5、clip=0.1，原 AdamW/OneCycleLR 与 normalizer、相对 L2 + 0.1 梯度项训练损失保持。

## 训练后自动评估

在当前仓库根目录运行，以下两条分别是两个独立实验：

```bash
# 去掉前段 latent SA，保留两个 latent FFN
bash tran_evaluate/ablation/no_sa/train_eval.sh --gpu 0

# 去掉前段两个 latent FFN 和 latent SA
bash tran_evaluate/ablation/identity/train_eval.sh --gpu 0
```

先预览可在任一命令末尾加 `--dry-run`，只打印训练/评估命令，不读取数据、创建实验目录或训练。脚本按自身位置定位仓库，可在远端原样使用，也可以从任意工作目录传脚本绝对路径调用。

默认每次新训练使用 `output/darcy/<UTC时间戳>/`，记录配置（含模式与实际参数量）、训练结果及评估结果，沿用现有输出管理。组合脚本生成同一个时间戳供两步使用；`CDLNO_RUNS_ROOT`、`CDLNO_RUN_TAG` 和显式 `--cdlno-run-dir` 继续有效。新训练目录必须尚不存在；不要用组合脚本重新训练到已有实验目录。

## 分别训练和评估

将下面路径替换成一个新实验目录；训练结束后评估使用同一路径：

```bash
bash tran_evaluate/ablation/no_sa/train.sh --gpu 0 \
    --cdlno-run-dir "$PWD/output/darcy/no_sa_run1"
bash tran_evaluate/ablation/no_sa/eval.sh --gpu 0 \
    --cdlno-run-dir "$PWD/output/darcy/no_sa_run1"

bash tran_evaluate/ablation/identity/train.sh --gpu 0 \
    --cdlno-run-dir "$PWD/output/darcy/identity_run1"
bash tran_evaluate/ablation/identity/eval.sh --gpu 0 \
    --cdlno-run-dir "$PWD/output/darcy/identity_run1"
```

评估仍先读已有 sidecar，模式冲突拒绝，严格加载当前模式权重。这两个消融默认从头训练；不做 full 到消融的权重迁移。沿用现有 checkpoint 保存协议，不新增断点续训参数。

路径由根目录 `path.sh` 提供，Darcy 的 `--data_path` 指向同时包含 `piececonst_r421_N1024_smooth1.mat` 和 `piececonst_r421_N1024_smooth2.mat` 的目录。保留当前已工作的 Python/torch 环境，可通过 `CDLNO_PYTHON` 选择解释器。

用户余参最后转发，可以覆盖既有默认值；修改架构后单独评估也要传同样的架构参数。组合脚本的共同参数须放在 `--train-args` / `--eval-args` 之前。例如，明确选择 F=3 并关闭 CDPA：

```bash
bash tran_evaluate/ablation/no_sa/train_eval.sh --gpu 0 \
    --n-layers 8 --front-blocks 3 --cdpa-mode off --dry-run
```

本次只新增启动包装与说明；没有修改模型、训练/评估入口、数据或依赖，没有运行真实训练。验证结果见 [启动核查记录](verification.json)。

