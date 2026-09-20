# LL8：八任务 Looped LinearNO 命令与公平比较

以下是真实训练的**未来运行命令**。LL8只运行CPU合成检查和dry-run，没有读取真实数据、执行远端/GPU训练或产生准确率结果。当前仅支持已批准的三种点域残差模式，不与旧latent A/K flags混用。

## 远端准备与统一入口

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
export CDLNO_REPO_ROOT="$PWD" CDLNO_RUNS_ROOT="$PWD/output"
source ./path.sh
# 使用远端已可工作的环境；此处不安装/升级任何依赖。
```

仓库目录如果另有改名，只改第一行cd，其余相对路径不变。请勿预先mkdir最终RUN目录。每题脚本统一为：

```bash
bash tran_evaluate/linearno_loop/<task>.sh train  [显式loop/topology/residual参数]
bash tran_evaluate/linearno_loop/<task>.sh resume --gpu 0 --experiment-dir "$RUN"
bash tran_evaluate/linearno_loop/<task>.sh eval   --gpu 0 --experiment-dir "$RUN"
```

task是`airfoil/darcy/elasticity/pipe/ns/plasticity/airfrans/car`。新train需要显式`--linearno-loop 1`、topology、residual；rank multiplier缺省2，允许1；不能和actual `--linearno-rank`同时给。旧`--n-layers`、`--slice_num`和A/K/fair-run参数不能传入。profile的原数据、目标、metric与训练节奏不变；Darcy official_release仍拒绝非500 epochs。

- `--dry-run`：调用原launcher生成argv，再执行真实parser及metadata校验；不构造模型、不读取数据、不反序列化权重、不建运行目录。eval/resume会读取JSON并核验成对文件hash，因此必须有真实RUN；resume会拒绝已完成运行。
- `--print-run-dir`：只输出本次解析出的准确RUN，可给变量赋值；配置变更后需重新生成。目录含task/profile/P-C-R-S/residual/M/seed/config hash/时间戳，路径允许空格。
- `--plan-json`：输出可机器读取的命令与resolved config，不执行任务。
- `--then-eval`：用于train或resume，前一命令成功才评估同一个RUN；失败退出码原样传递。eval只用刚保存的metadata，不复制训练结构默认值。

### preset A：P1-C3-R2-S1

Airfoil完整训练→评估示例（先生成准确RUN，再启动；不是字面路径`RUN`）：

```bash
FLAGS=(--linearno-loop 1
  --linearno-loop-topology p1_c3_r2_s1
  --linearno-loop-residual-mode sr_1_over_r
  --linearno-loop-rank-multiplier 2
  --linearno-profile paper_table8_on_release_model
  --seed 17 --gpu 0)
RUN="$(bash tran_evaluate/linearno_loop/airfoil.sh train "${FLAGS[@]}" --print-run-dir)"
bash tran_evaluate/linearno_loop/airfoil.sh train "${FLAGS[@]}" \
  --experiment-dir "$RUN" --then-eval
```

先预览可把最后一行加上`--dry-run`。不显式传RUN时，parser同样会自动生成唯一目录，控制台打印准确路径。

### preset B：P2-C2-R2-S2

```bash
bash tran_evaluate/linearno_loop/darcy.sh train --then-eval \
  --linearno-loop 1 --linearno-loop-topology p2_c2_r2_s2 \
  --linearno-loop-residual-mode rb_attnres \
  --linearno-loop-rank-multiplier 2 \
  --linearno-profile paper_table8_on_release_model --seed 17 --gpu 1
```

### custom：P0-C2-R3-S1 与 M×1 控制

```bash
bash tran_evaluate/linearno_loop/elasticity.sh train --then-eval \
  --linearno-loop 1 --linearno-loop-topology custom \
  --linearno-loop-prefix-blocks 0 --linearno-loop-core-blocks 2 \
  --linearno-loop-repeats 3 --linearno-loop-suffix-blocks 1 \
  --linearno-loop-residual-mode lb_attnres_1_over_r \
  --linearno-loop-rank-multiplier 1 \
  --linearno-profile paper_table8_on_release_model --seed 17 --gpu 0
```

custom四字段必须齐全，不能与命名preset混填。若控制actual M，把multiplier整对参数移除后改成`--linearno-rank M`；Car的M必须是head_dim的正整数倍。

### 续训与独立评估

```bash
RUN='/absolute/path/copied/from/the/training/output'
# 仅对未完成的运行；成功后评估final。
bash tran_evaluate/linearno_loop/airfoil.sh resume \
  --gpu 0 --experiment-dir "$RUN" --then-eval
# 对已有完整运行独立复评；每次生成新的evaluations子目录。
bash tran_evaluate/linearno_loop/airfoil.sh eval --gpu 0 --experiment-dir "$RUN"
```

其余七任务仅替换脚本名。无需重复profile/seed/rank/PCRS/mode/fold/Air task或nmodel；若显式重复，只作一致性断言，冲突在模型构造与torch.load前拒绝。数据根使用path.sh；数据搬家时显式传相应路径，但样本checksum、normalizer等仍严格核对。不要只复制便捷`model.pt`：恢复需要architecture、成对checkpoint/weights及manifest；Air还需根目录与成员目录。

## 八任务与数据路径

| task脚本 | path.sh变量 | 入口参数与含义 | 默认base M → loop M×2 |
|---|---|---|---|
| airfoil.sh | CDLNO_AIRFOIL_ROOT | `--data_path`，`fno/airfoil/naca`，包含原NACA数组 | 64→128 |
| darcy.sh | CDLNO_DARCY_ROOT | `--data_path`，原Darcy两个mat文件所在的fno目录 | 64→128 |
| elasticity.sh | CDLNO_ELASTICITY_ROOT | `--data_path`，原Elasticity子目录所依附的fno根目录 | 64→128 |
| pipe.sh | CDLNO_PIPE_ROOT | `--data_path`，`fno/pipe`数组目录，129×129 | 64→128 |
| ns.sh | CDLNO_NS_ROOT | `--data_path`，fno根目录；入口追加`NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat` | 32→64 |
| plasticity.sh | CDLNO_PLASTICITY_FILE | `--data_path`，`plas_N987_T20.mat`文件本身 | 64→128 |
| airfrans.sh | CDLNO_AIRFRANS_DATASET | `--my_path`，包含manifest/data的Dataset目录；原适配也接受其父目录，train/eval沿用同一配置路径 | 32→64 |
| car.sh | CDLNO_CAR_RAW_ROOT / CDLNO_CAR_CACHE_ROOT | `--data_dir` 原param/fold几何目录；`--save_dir`预处理缓存目录，不是实验输出目录 | 32→64 |

本表是paper_table8_on_release_model与official_release的基础rank；始终以所选profile解析值为准，不把倍增M硬编码成所有任务相同。工业目录在path.sh里是未在远端观测的候选，请按真实存放处覆盖环境变量。原`tran_evaluate/inspect_data.sh`可只读检查文件存在情况。Loop沿用已批准的Car sample路径/非零fold evaluator，不套用旧CDLNO特有的固定param0检查；这不代表真实VTK指标已验收。

## GPU映射

未设置CUDA_VISIBLE_DEVICES时，`--gpu 1`选择物理GPU1。调度器已设置掩码时，`--gpu`是掩码内索引，例如：

```bash
CUDA_VISIBLE_DEVICES=4,7 bash tran_evaluate/linearno_loop/darcy.sh train \
  --linearno-loop 1 --linearno-loop-topology p1_c3_r2_s1 \
  --linearno-loop-residual-mode sr_1_over_r --seed 0 --gpu 1 --dry-run
```

上述选择物理GPU7。Standard原入口会重写CUDA_VISIBLE_DEVICES，故桥接后传原`--gpu 7`；工业入口在单卡掩码下使用`--gpu 0`。控制台记录请求索引/继承掩码/最终掩码。Standard原parser只接受数字GPU ID，UUID掩码会明确拒绝；不要用此方式启动CPU Standard训练。显式空掩码用于本地无GPU的dry-run与合成验证，不是远端训练设置。

## 三个paired seeds的完整未来矩阵

以下Bash循环包含8任务×2拓扑×3残差×3 seeds，每次训练成功后再评估同一运行，共144组真实训练计划。**这是未来命令，LL8没有执行它。**先保持`PREVIEW=1`只做预览；确认数据/预算后由用户自行改为0。

```bash
(
set -euo pipefail
PREVIEW=1
for seed in 0 1 2; do
  for task in airfoil darcy elasticity pipe ns plasticity airfrans car; do
    case "$task" in darcy|pipe|plasticity) gpu=1 ;; *) gpu=0 ;; esac
    for topology in p1_c3_r2_s1 p2_c2_r2_s2; do
      for mode in sr_1_over_r rb_attnres lb_attnres_1_over_r; do
        flags=(--linearno-loop 1 --linearno-loop-topology "$topology"
          --linearno-loop-residual-mode "$mode" --linearno-loop-rank-multiplier 2
          --linearno-profile paper_table8_on_release_model --seed "$seed" --gpu "$gpu")
        if [[ "$PREVIEW" == 1 ]]; then
          bash "tran_evaluate/linearno_loop/${task}.sh" train "${flags[@]}" --then-eval --dry-run
        else
          run="$(bash "tran_evaluate/linearno_loop/${task}.sh" train "${flags[@]}" --print-run-dir)"
          bash "tran_evaluate/linearno_loop/${task}.sh" train "${flags[@]}" \
            --experiment-dir "$run" --then-eval
        fi
      done
    done
  done
done
)
```

原profile的训练epochs不改变。训练用原train split，保持原validation/test过程；最终比较固定这三个seed和预声明的final checkpoint，逐seed报告并聚合均值/离散程度，不能用test结果选“最好seed”或“最好checkpoint”。rank×1另开相同paired矩阵，修改multiplier即可；不可将它混入M×2结果。

公平比较限定相同task/profile/topology/actual rank/seed的三残差：公共stem、物理block和head由相同seed生成相同键/值；receiver query=0、norm=1，不消耗初始化RNG；不要分别重设不同feature seed改变主干。DataLoader使用已批准schema的独立train/test generator，Plasticity时间collate使用原独立generator封装，Air/Car的Python采样状态沿用原恢复协议。每个Air成员独立初始化；成员编号相同的三残差应比较相同hash与数据序列，不能共享同一个core对象。manifest中的loader seed是整次运行的起始seed；Air沿原协议让generator状态跨成员连续推进，不在每个成员重置数据序列。

两拓扑**不参数匹配**：A的unique_depth=5，B=6，均executed_depth=8。SR无router；A/B的RB分别13/9 receiver、26H/18H参数；两者LB均2 receiver、4H参数。总参数以实际模型统计为准，不能把独立参数减少当成FLOPs或epoch加速。

## 输出与manifest

```text
output/<task>/linearno_loop/<task>__linearno_loop__<profile>__P?-C?-R?-S?__<mode>__M?__seed?__cfg<hash>__<timestamp>/
  config.json / architecture.json / train.log / train_history.jsonl / train_results.json
  loop_run_manifest.json
  checkpoints/epoch_*.pt / epoch_*.metadata.json / epoch_*.json / latest.json / final.json
  weights/epoch_*.pt
  model.pt                         # Standard/Car便捷权重，非独立续训包
  member_000/, member_001/, ...     # Air成员各自architecture/checkpoints/weights/model
  ensemble.json / ensemble_state_dict.pth  # Air最终安全成员列表
  visualizations/...              # 原任务记录器已有图/数组
  evaluations/<unique_eval_id>/... / eval_results.json
```

具体任务仍保留原输出差异，例如Car另有`model_<epochs>.pth`。新manifest只由本目录的新launcher启用，记录config hash、公平seed、每成员初始公共backbone hash、完整初值hash、实测总/主干/router参数量、unique/executed depth、预期与**首次成功forward实际**operator/MLP/router/final LN/head调用序列。首forward前actual为null并标注pending，不能把计划写成已执行。观察不改模型/state_dict、optimizer或RNG，首次成功后移除全部观察hook；无张量历史缓存。

manifest独立于既有checkpoint schema。resume/eval保持原训练manifest字节；Air续训开始尚未训练的新成员时才追加该成员初始记录。LL6/LL7或直接旧launcher生成的运行没有这个文件时，仍可严格恢复，但会明确提示“没有初始记录”，不会从已训练权重伪造初始hash。原先pure/history的输出位置、launcher、checkpoint与monitor均不改。逻辑visit相似度monitor属于LL9，不在本命令中宣称已接入。

实施和验证证据见[LL8报告](LOOP_LINEARNO_LL8_LAUNCHERS.md)。
