# LinearNO history：远端训练、续训和评估

以下命令交付供之后授权的真实运行使用；本次仅执行了合成测试。远端环境和真实数据尚未验收。保留原 `tran_evaluate/linearno/` 命令；研究 launcher 与它并行，复用 `path.sh`、原数据入口与输出可视化。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
source ./path.sh
TASK=airfoil       # R6: airfoil / darcy / elasticity / pipe
PROFILE=paper_table8_on_release_model  # 也可 official_release / transolver_matched
L=8                # 4 / 5 / 6 / 7 / 8
A=1; K=1           # 0 0 纯模型；1 0 仅 A；0 1 仅 K；1 1 联合
SEED=17            # 预声明 paired seeds: 17 / 29 / 43
GPU=0              # 换为 1 可使用物理 GPU1
RUN="$PWD/output/$TASK/linearno_history/${TASK}__${PROFILE}__L${L}__A${A}K${K}__seed${SEED}"
bash "tran_evaluate/linearno_history/$TASK.sh" train \
  --gpu "$GPU" --seed "$SEED" --linearno-profile "$PROFILE" --n-layers "$L" \
  --linearno_latent_attnres "$A" --linearno_history_k_conditioning "$K" \
  --experiment-dir "$RUN"
bash "tran_evaluate/linearno_history/$TASK.sh" eval --gpu "$GPU" --experiment-dir "$RUN"
```

中断后恢复同一 RUN；模型开关、层数和训练配置从该 run metadata 恢复：

```bash
bash "tran_evaluate/linearno_history/$TASK.sh" resume --gpu "$GPU" --experiment-dir "$RUN"
```

同一 RUN 不可重新 train 覆盖。已经完成的 final run 不能 resume；直接 eval。若同时用 `CUDA_VISIBLE_DEVICES=1`，进程只见一张卡，此时传 `--gpu 0`，不要再传物理编号 1。

公平 launcher 的 train 显式开启 `--linearno-fair-run 1`：四组合独立 DataLoader generator、公共主干权重与隔离 feature seed，并在外部 `linearno_run_manifest.json` 记录。A0K0 仍使用纯类和旧 checkpoint schema，不增加 innovation_spec。旧命令省略此运行选项或仅显式 A0K0，仍维持旧 RNG 和 loader 行为。

训练保留 config/log/status、epoch 曲线、原周期场图、严格 checkpoint+weights 对与最新/最终指针；eval 保存新的独立评估目录，不覆写训练 metadata。数据、normalizer、objective、scheduler 仍按所选旧 LinearNO profile。

研究源码或任务协议改变后，resume 的 source hash 会拒绝继续旧运行；请在正式实验前冻结最终代码。严格匹配的 eval 仍先检查构造/profile/data metadata；不支持把纯权重当作研究 resume 或在 A/K 组合间互载。

R7 的 `TASK=ns`、`TASK=plasticity` 使用上面同一组命令（`--n-layers 4..8`、四组合、train/resume/eval）。NS保持10帧输入、每次预测1帧、10步teacher forcing训练与10步预测回填评估；Plasticity保持20个查询、每query一次optimizer、每batch一次scheduler。研究raw history每次forward重新建立，与真实时间无关。公平运行中Plasticity原per-sample Torch时间排列使用独立DataLoader generator；恢复该generator及完整模型RNG，不靠重新seed。

R8 工业任务的层数字段是 `--linearno-layers`，并使用其原生GPU参数：

```bash
TASK=airfrans      # 或 car
PROFILE=paper_table8_on_release_model
L=8; A=1; K=1; SEED=17; GPU=0
RUN="$PWD/output/$TASK/linearno_history/${TASK}__${PROFILE}__L${L}__A${A}K${K}__seed${SEED}"
bash "tran_evaluate/linearno_history/$TASK.sh" train \
  --gpu "$GPU" --seed "$SEED" --linearno-profile "$PROFILE" --linearno-layers "$L" \
  --linearno_latent_attnres "$A" --linearno_history_k_conditioning "$K" \
  --experiment-dir "$RUN"
bash "tran_evaluate/linearno_history/$TASK.sh" eval --gpu "$GPU" --experiment-dir "$RUN"
# 中断后（不是已完成final）
# bash "tran_evaluate/linearno_history/$TASK.sh" resume --gpu "$GPU" --experiment-dir "$RUN"
```

四组合和L4–8均支持。AirfRANS可在train显式附加原`--task full|scarce|reynolds|aoa --nmodel 1`；每个ensemble成员有独立权重、metadata严格检查、forward局部history和初始主干hash，成员顺序保存在`ensemble.json`。成员0公共seed等于运行seed，成员i为seed+i；feature seed按既定任务命名空间派生并记录，成员之间不共享模型参数或history。Car保留`--fold_id 0..8`、`--preprocessed 0|1`和原数据路径字段；eval从运行恢复fold/协议。`path.sh`中的工业数据目录仍需在远端确认。

AirfRANS实际训练MSE权重1，Car实际训练MSE权重0.5；二者不是统一的rL2训练解释。force评估需要完整VTK/原始文件，本地合成闭环只验证模型、字段指标及严格加载，不能替代真实force结果。

## 八任务统一调用示例

在以上远端项目根目录执行下面的函数定义，然后选择一行任务调用。函数只将明确字段转发给已有launcher；不会替用户挑选最新运行。

```bash
history_job() {
  local task="$1" action="$2" layers="$3" a="$4" k="$5" seed="$6" gpu="$7"
  local profile="${8:-paper_table8_on_release_model}"
  local run="$PWD/output/$task/linearno_history/${task}__${profile}__L${layers}__A${a}K${k}__seed${seed}"
  local layer_flag=--n-layers
  [[ "$task" == airfrans || "$task" == car ]] && layer_flag=--linearno-layers
  local -a args=(--gpu "$gpu" --experiment-dir "$run")
  if [[ "$action" == train ]]; then
    args+=(--seed "$seed" --linearno-profile "$profile" "$layer_flag" "$layers"
           --linearno_latent_attnres "$a" --linearno_history_k_conditioning "$k")
  fi
  bash "tran_evaluate/linearno_history/$task.sh" "$action" "${args[@]}"
}
# 下列是供日后真实实验选择执行的命令；本轮未执行：
history_job airfoil    train 8 1 1 17 0
history_job darcy      train 8 1 1 17 0
history_job elasticity train 8 1 1 17 0
history_job pipe       train 8 1 1 17 0
history_job ns         train 8 1 1 17 0
history_job plasticity train 8 1 1 17 0
history_job airfrans   train 8 1 1 17 0
history_job car        train 8 1 1 17 0
# 例如另一张卡、另一个预声明seed、仅K、L6：
history_job elasticity train 6 0 1 29 1
history_job elasticity eval  6 0 1 29 1
# 仅对尚未完成但已有完整epoch archive的运行续训：
# history_job elasticity resume 6 0 1 29 1
```

其余七题的eval/resume用同一函数，将train替换为eval/resume，并保持task/L/A/K/seed/profile以定位同一目录。L允许4..8；A/K均允许0/1；profile可显式第八参数official_release或transolver_matched。三个paired seeds为17/29/43；不选择test最优checkpoint、seed或深度。原纯profile文本中的历史seed建议不替代本研究外部manifest的实际运行seed与paired实验声明。

[诊断与性能用法](LINEARNO_HISTORY_DIAGNOSTICS.md)提供四组合兼容的可选监测。完整480项命令数组（八任务×五深度×四组合×train/resume/eval）在`linearno_history_audit/r9/commands.json`，每项只执行过`--dry-run`预览，不能将预览误称远端数据/训练验收。预期训练产物、epoch pairs、独立eval目录及external fair manifest沿用此前说明。
