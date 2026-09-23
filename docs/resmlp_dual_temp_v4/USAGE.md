# 八任务使用说明

当前脚本只选择 v4；旧纯 LinearNO/V1/V2/V3 继续使用各自脚本。三模式模型结构固定八个逻辑 block，不能用旧 D12/profile/adapter/topology 参数覆盖。

从仓库根目录执行：

```bash
export CDLNO_REPO_ROOT="$PWD"
source ./path.sh
export CDLNO_RUNS_ROOT="$PWD/output"
```

`train_eval` 或 `train --then-eval` 在训练成功后评估同一 run；失败时不会执行 eval。新训练默认创建含 task/profile/mode/seed/hash/UTC/UUID 的唯一目录。终端会打印 `Run:`，恢复和评估须使用该确切目录，不能猜最新 run。

`--dry-run`、`--print-config` 均经过真实任务 parser；读取已保存配置时仅读 metadata/检查 pair hash，不执行 torch.load、数据读取或训练。`--print-run-dir` 只打印预留名称而不创建目录；若将它保存到变量，启动时传回 `--experiment-dir "$RUN"`，避免两次规划生成不同唯一名称。

## 各任务训练与自动评估

### airfoil

```bash
bash tran_evaluate/linearno_loop_v4/airfoil.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/airfoil.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### darcy

```bash
bash tran_evaluate/linearno_loop_v4/darcy.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/darcy.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### elasticity

```bash
bash tran_evaluate/linearno_loop_v4/elasticity.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/elasticity.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### pipe

```bash
bash tran_evaluate/linearno_loop_v4/pipe.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/pipe.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### ns

```bash
bash tran_evaluate/linearno_loop_v4/ns.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/ns.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### plasticity

```bash
bash tran_evaluate/linearno_loop_v4/plasticity.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/plasticity.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### airfrans

```bash
bash tran_evaluate/linearno_loop_v4/airfrans.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/airfrans.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```
### car

```bash
bash tran_evaluate/linearno_loop_v4/car.sh train_eval --temperature-mode latent_k_point_q --seed 0 --gpu 0
bash tran_evaluate/linearno_loop_v4/car.sh train_eval --temperature-mode point_k_point_q --seed 0 --gpu 0
```

## 恢复与评估矩阵

以下每行分别对应一种训练模式的已保存 run；`RUN_...` 必须填写终端打印的实际目录。省略模式/profile/seed 时从 metadata 恢复；显式不一致在加载 tensor 前拒绝。

```bash
# airfoil: latent-K 实验
bash tran_evaluate/linearno_loop_v4/airfoil.sh resume --experiment-dir "$RUN_airfoil_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/airfoil.sh eval --experiment-dir "$RUN_airfoil_latent" --gpu 0
# airfoil: point-K 实验
bash tran_evaluate/linearno_loop_v4/airfoil.sh resume --experiment-dir "$RUN_airfoil_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/airfoil.sh eval --experiment-dir "$RUN_airfoil_point" --gpu 0
```
```bash
# darcy: latent-K 实验
bash tran_evaluate/linearno_loop_v4/darcy.sh resume --experiment-dir "$RUN_darcy_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/darcy.sh eval --experiment-dir "$RUN_darcy_latent" --gpu 0
# darcy: point-K 实验
bash tran_evaluate/linearno_loop_v4/darcy.sh resume --experiment-dir "$RUN_darcy_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/darcy.sh eval --experiment-dir "$RUN_darcy_point" --gpu 0
```
```bash
# elasticity: latent-K 实验
bash tran_evaluate/linearno_loop_v4/elasticity.sh resume --experiment-dir "$RUN_elasticity_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/elasticity.sh eval --experiment-dir "$RUN_elasticity_latent" --gpu 0
# elasticity: point-K 实验
bash tran_evaluate/linearno_loop_v4/elasticity.sh resume --experiment-dir "$RUN_elasticity_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/elasticity.sh eval --experiment-dir "$RUN_elasticity_point" --gpu 0
```
```bash
# pipe: latent-K 实验
bash tran_evaluate/linearno_loop_v4/pipe.sh resume --experiment-dir "$RUN_pipe_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/pipe.sh eval --experiment-dir "$RUN_pipe_latent" --gpu 0
# pipe: point-K 实验
bash tran_evaluate/linearno_loop_v4/pipe.sh resume --experiment-dir "$RUN_pipe_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/pipe.sh eval --experiment-dir "$RUN_pipe_point" --gpu 0
```
```bash
# ns: latent-K 实验
bash tran_evaluate/linearno_loop_v4/ns.sh resume --experiment-dir "$RUN_ns_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/ns.sh eval --experiment-dir "$RUN_ns_latent" --gpu 0
# ns: point-K 实验
bash tran_evaluate/linearno_loop_v4/ns.sh resume --experiment-dir "$RUN_ns_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/ns.sh eval --experiment-dir "$RUN_ns_point" --gpu 0
```
```bash
# plasticity: latent-K 实验
bash tran_evaluate/linearno_loop_v4/plasticity.sh resume --experiment-dir "$RUN_plasticity_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/plasticity.sh eval --experiment-dir "$RUN_plasticity_latent" --gpu 0
# plasticity: point-K 实验
bash tran_evaluate/linearno_loop_v4/plasticity.sh resume --experiment-dir "$RUN_plasticity_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/plasticity.sh eval --experiment-dir "$RUN_plasticity_point" --gpu 0
```
```bash
# airfrans: latent-K 实验
bash tran_evaluate/linearno_loop_v4/airfrans.sh resume --experiment-dir "$RUN_airfrans_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/airfrans.sh eval --experiment-dir "$RUN_airfrans_latent" --gpu 0
# airfrans: point-K 实验
bash tran_evaluate/linearno_loop_v4/airfrans.sh resume --experiment-dir "$RUN_airfrans_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/airfrans.sh eval --experiment-dir "$RUN_airfrans_point" --gpu 0
```
```bash
# car: latent-K 实验
bash tran_evaluate/linearno_loop_v4/car.sh resume --experiment-dir "$RUN_car_latent" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/car.sh eval --experiment-dir "$RUN_car_latent" --gpu 0
# car: point-K 实验
bash tran_evaluate/linearno_loop_v4/car.sh resume --experiment-dir "$RUN_car_point" --gpu 0 --then-eval
bash tran_evaluate/linearno_loop_v4/car.sh eval --experiment-dir "$RUN_car_point" --gpu 0
```

## 消融、配对种子与路径

```bash
# 静态温度消融，仍使用 v4 八块和五个 ResMLP owner
bash tran_evaluate/linearno_loop_v4/elasticity.sh train_eval --temperature-mode base --seed 0 --gpu 0

# 只生成/查看命令，不启动矩阵
bash tran_evaluate/linearno_loop_v4/plasticity.sh train --temperature-mode latent_k_point_q --seed 1 --gpu 1 --dry-run

# 实际运行前由用户自行选择任务；三个 seed 顺序执行
for seed in 0 1 2; do
  bash tran_evaluate/linearno_loop_v4/elasticity.sh train_eval --temperature-mode latent_k_point_q --seed "$seed" --gpu 0 || break
done
```

默认保持各任务纯 LinearNO 的 H/Hd/M/ratio：

| 任务 | H | heads | M | FFN ratio |
|---|---:|---:|---:|---:|
| airfoil | 128 | 8 | 64 | 1 |
| darcy | 128 | 8 | 64 | 1 |
| elasticity | 128 | 8 | 64 | 1 |
| pipe | 128 | 8 | 64 | 1 |
| ns | 256 | 8 | 32 | 2 |
| plasticity | 128 | 8 | 64 | 1 |
| airfrans | 256 | 8 | 32 | 2 |
| car | 256 | 8 | 32 | 2 |

路径可含空格。使用 `--data-root /远端数据根目录` 或现有 path.sh 环境变量。Standard 五任务默认 `<root>/fno`，Plasticity 默认 `<root>/fno/plas_N987_T20.mat`，AirfRANS 默认 `<root>/AirfRANS/Dataset`，Car 默认 `<root>/mlcfd_data/training_data` 和 `<root>/mlcfd_data/preprocessed_data`。这些是路径合同，未验证远端文件是否存在。优先使用你已能运行旧模型的真实数据路径。

```bash
# 在已同步本次代码的远端仓库根目录
bash tran_evaluate/linearno_loop_v4/plasticity.sh train_eval \
  --temperature-mode latent_k_point_q --seed 0 --gpu 0 \
  --data-root /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data \
  --output-root "$PWD/output"

# 可逐文件覆盖，必须传正确的原任务类型
bash tran_evaluate/linearno_loop_v4/plasticity.sh train_eval \
  --data_path "/data with spaces/plas_N987_T20.mat" --gpu 0
```

旧 shell 允许的训练参数放在最后，配置层检查冲突。工业任务固定 batch1 和原 objective；不支持通过 v4 结构字段改 H/M/ratio。生产训练未启用 AMP；本地 AMP 是合成数值检查。

## 输出与边界

复用现有 config.json、train/status/log、epoch JSONL、field visualization、weights、完整 optimizer/scheduler/RNG archive、每次独立 evaluations 和 eval_results 索引。`loop_run_manifest.json` 专门记录 v4 ownership、初始化哈希和首次实际 forward 顺序。参数量在模型实际构造后记录。它不会自动增加可靠同步推理延迟、每 epoch 纯训练耗时或峰值显存日志；这些仅在单独 benchmark 报告中测量。

NS 保留十步 teacher-forced 训练、预测反馈评估；Plasticity 保留二十次时间条件更新。AirfRANS 保留原 weighted MSE 与原字段/力指标；Car 保留 normalized MSE、surface weight0.5、fold/单图与 drag 接口。Car 原拖曳后处理的 canonical raw path/fold 限制仍存在，不等于任意路径的全指标已经验收。

真实数据、生产 epoch、收敛/精度/SOTA、远端栈、distributed/compile 均 NOT RUN。建议在你确认配置与成本表后，按 base→latent-K→point-K 顺序，使用相同 seed/profile/预算开展真实实验。本次未启动这些实验。
