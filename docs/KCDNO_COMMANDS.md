# KCDNO 与 matched LRSA 命令

从仓库根目录执行，沿用 `path.sh` 和当前 Python 环境。设置 `CDLNO_PYTHON=/absolute/python` 可选择已有环境；不需要重装依赖。默认路径为 `output/<task>/<family>/<profile>/<UTC时间与结构标识>`；明确 `--kcdno-run-dir` 可指定新训练目录或既有评价目录。训练目录排他创建，不能用已有目录重新启动训练。

| TASK / 脚本 | 主profile L/d/h/M/r | 点模块 | 训练预设 epochs/batch |
|---|---|---|---|
| darcy / `tran_evaluate/kcdno/darcy.sh` | 8/128/8/64/16 | ConvFFN | 500/4 |
| elasticity / `tran_evaluate/kcdno/elasticity.sh` | 8/128/8/64/16 | PointFFN | 500/1 |
| airfoil / `tran_evaluate/kcdno/airfoil.sh` | 8/128/4/64/16 | ConvFFN | 500/4 |
| pipe / `tran_evaluate/kcdno/pipe.sh` | 8/128/4/32/16 | ConvFFN | 500/8 |
| ns / `tran_evaluate/kcdno/ns.sh` | 8/256/8/64/16 | ConvFFN | 500/2 |
| plasticity / `tran_evaluate/kcdno/plasticity.sh` | 8/128/8/64/16 | ConvFFN | 500/8 |
| car / `tran_evaluate/kcdno/car.sh` | 8/256/8/64/16 | PointFFN | 200/1 |
| airfrans / `tran_evaluate/kcdno/airfrans.sh` | 8/256/8/64/16 | PointFFN | 当前YAML398/1 |

三种计算图：`--model kcdno --history-mode all` 为主版，`--history-mode off` 为matched无SA对照，`--model lrsa_matched` 为完整LRSA对照。Car的原模型选择选项是 `--cfd_model`，其余为 `--model`。matched不接收kernel-rank/history-mode/front/CDPA参数，固定full；没有从旧模型跨架构迁移权重。两个profile的公共尺寸应用于三种计算图，显式CLI最后覆盖profile。

八任务均可用同一个薄包装按顺序训练、成功后评价；以下是**待用户运行**示例，没有自动启动：

```bash
# TASK 可替换为上表八个名称；all 可换成 off 或 lrsa_matched。
bash tran_evaluate/kcdno/train_eval.sh darcy all --gpu 0
bash tran_evaluate/kcdno/train_eval.sh elasticity off --gpu 0
bash tran_evaluate/kcdno/train_eval.sh airfoil lrsa_matched --gpu 0
bash tran_evaluate/kcdno/train_eval.sh pipe all --gpu 0
bash tran_evaluate/kcdno/train_eval.sh ns all --gpu 0
bash tran_evaluate/kcdno/train_eval.sh plasticity all --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdno/train_eval.sh car all
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/kcdno/train_eval.sh airfrans all
# 只预览两个命令，无Python入口或数据读取：
bash tran_evaluate/kcdno/train_eval.sh darcy lrsa_matched --dry-run
```

单独训练/评价（推荐保存明确的run变量；训练不允许已存在目录）：

```bash
run="$PWD/output/darcy/kcdno/manual_$(date -u +%Y%m%dT%H%M%S)_$$"
bash tran_evaluate/kcdno/darcy.sh train --history-mode all --gpu 0 --kcdno-run-dir "$run"
bash tran_evaluate/kcdno/darcy.sh eval --gpu 0 --kcdno-run-dir "$run"
# 对照：两个命令都追加 --model lrsa_matched（Car为 --cfd_model）。
# off评价可省略history-mode，按已有sidecar恢复；显式冲突会拒绝。
```

`--profile transolver_shape_match` 将Airfoil h改8、Pipe h8/M64、NS/Car/AirfRANS M32，其余与主profile相同。仅匹配L/d/h/M，并非等参数/等FLOPs或官方LRSA论文复现。新脚本只显式指定模型/profile，未将主profile尺寸硬写为CLI覆盖。KCDNO r默认16；可用 `--kernel-rank` 覆盖。共同维度沿原选项，如标准任务 `--n-hidden --n-heads --n-layers --slice_num`，工业任务支持underscore及对应hyphen别名。

路径合同：六标准任务沿 `--data_path`；Car `--data_dir`原始数据、`--save_dir`预处理数据，一run一fold，原拖曳评价仍要求fold0与其硬编码param0原始路径（脚本会检查）。Air训练 `--my_path` 为包含manifest.json的Dataset目录，评价为Dataset的父目录。顺序包装默认从path.sh为两步选择各自正确路径；如果显式覆盖Air的my_path，应分开运行train/eval并分别给路径，不把同一个路径传给顺序包装。

保存协议不改变：六标准任务 `model.pt` strict state_dict；Car `model_<nb_epochs>.pth`可信整对象；Air root `<family>`模型列表、member_000/model整对象。新run有architecture.json（完整core/初始化/运行记录）和task.json（wrapper/协议/实参），加现有训练配置及结果日志。eval先读再核对显式结构，禁止覆盖sidecar；错误family、r、history、L/d/M、adapter或权重严格拒绝。旧CDLNO/Transolver使用原命令与旧加载路径。

这些权重沿用现有模型保存协议，**不包含完整optimizer/scheduler/RNG续训状态**。旧V1归档公共能力存在，但任务级resume后续阶段尚未接入；本次K阶段没有新增resume，也不能把模型权重加载称为精确断点续训。
