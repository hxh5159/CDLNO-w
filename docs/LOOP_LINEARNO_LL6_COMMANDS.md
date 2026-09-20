# LL6 六个 Standard 任务远端命令

这些命令用于后续用户自行运行；LL6 只验证命令解析与 CPU 合成闭环，**未运行真实 loader、500 epoch 或远端 GPU**。本阶段复用现有 `tran_evaluate/linearno/` 的透传脚本，未新增 launcher。只有显式 `--linearno-loop 1` 才开启新 family；eval/resume 从指定运行的 metadata 恢复。

在远端当前 checkout 执行一次以下设置和函数定义。它使用纯 schema 生成准确目录名，训练成功后才评估同一目录；每次调用生成独立时间戳，避免覆盖。不要预先创建最终 `run` 目录。

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/LinearNO-monitor
export CDLNO_REPO_ROOT="$PWD" CDLNO_RUNS_ROOT="$PWD/output"
source ./path.sh

loop_train_eval() (
    set -euo pipefail
    task="${1:?task}"; gpu="${2:?gpu}"
    seed="${LOOP_SEED:-0}"
    profile="${LOOP_PROFILE:-paper_table8_on_release_model}"
    topology="${LOOP_TOPOLOGY:-p1_c3_r2_s1}"
    mode="${LOOP_MODE:-lb_attnres_1_over_r}"
    multiplier="${LOOP_RANK_MULTIPLIER:-2}"
    cd "$CDLNO_REPO_ROOT"
    id="$("$CDLNO_PYTHON" -B - "$task" "$profile" "$topology" "$mode" "$multiplier" "$seed" <<'PY'
import sys
from linearno_loop.config import resolve_config, run_directory_id
task, profile, topology, mode, multiplier, seed = sys.argv[1:]
config = resolve_config(task, profile,
    options=dict(topology_preset=topology, residual_mode=mode, rank_multiplier=int(multiplier)),
    profile_overrides={'runtime.seed': int(seed)})
print(run_directory_id(config))
PY
    )"
    run="$CDLNO_RUNS_ROOT/$task/linearno_loop/${id}__$(date -u +%Y%m%dT%H%M%S%NZ)"
    printf 'Run: %s\n' "$run"
    bash "tran_evaluate/linearno/${task}_train.sh" \
        --gpu "$gpu" --seed "$seed" --experiment-dir "$run" \
        --linearno-profile "$profile" --linearno-loop 1 \
        --linearno-loop-topology "$topology" \
        --linearno-loop-residual-mode "$mode" \
        --linearno-loop-rank-multiplier "$multiplier" \
    && bash "tran_evaluate/linearno/${task}_eval.sh" \
        --gpu "$gpu" --experiment-dir "$run"
)
```

各任务分别训练后评估（默认 seed0/P1/LB/M×2，GPU 为示例）：

```bash
loop_train_eval airfoil 0
```

```bash
loop_train_eval darcy 1
```

```bash
loop_train_eval elasticity 0
```

```bash
loop_train_eval pipe 1
```

```bash
loop_train_eval ns 0
```

```bash
loop_train_eval plasticity 1
```

同一函数覆盖另一 topology、三个互斥 residual、rank×1 控制与另一 profile。例如以下各自都是新的 Darcy 运行：

```bash
LOOP_TOPOLOGY=p2_c2_r2_s2 LOOP_MODE=sr_1_over_r LOOP_SEED=1 loop_train_eval darcy 1
LOOP_TOPOLOGY=p1_c3_r2_s1 LOOP_MODE=rb_attnres loop_train_eval darcy 1
LOOP_RANK_MULTIPLIER=1 loop_train_eval darcy 1
LOOP_PROFILE=official_release loop_train_eval darcy 1
```

两个 profile 仍使用原始任务协议。Darcy `official_release` 必须保持 500 epochs，显式非 500 拒绝。不要附加旧 `--n-layers`、`--slice_num` 或 A/K flags；实际 M 可以用 `--linearno-rank` 控制，但不能与 multiplier 同时传。若自定义更多训练字段，需要同步修改纯 schema 的 `profile_overrides` 以匹配目录 hash，或省略 `--experiment-dir` 让 parser 自动生成并打印准确路径。

恢复或重新评估已存在运行时，复制训练输出中的完整路径；**不要重新计算一个新时间戳，也不要传猜测的 topology/mode**：

```bash
# 把下面的路径替换成先前打印的真实运行目录；训练尚未完成才可 resume。
RUN_DIR='/absolute/path/to/the/existing/darcy__linearno_loop__...'
bash tran_evaluate/linearno/darcy_train.sh --gpu 1 --resume --experiment-dir "$RUN_DIR" \
&& bash tran_evaluate/linearno/darcy_eval.sh --gpu 1 --experiment-dir "$RUN_DIR"
```

独立 eval：

```bash
bash tran_evaluate/linearno/darcy_eval.sh --gpu 1 --experiment-dir "$RUN_DIR"
```

输出沿用已有记录器：`architecture.json` 保存独立 loop schema；`config.json` 同时包含原 profile 与 `loop_config`；`checkpoints/epoch_*.{pt,json,metadata.json}` 与 `weights/epoch_*.pt` 成对提交，`latest.json`/`final.json` 指向已提交 manifest；另有 `model.pt` 便捷权重。训练日志、epoch JSONL、结果、`visualizations/member_000/epoch_*/` 与每次独立的 `evaluations/<id>/` 保留原格式。eval 不覆盖训练 sidecar、不重新拟合 normalizer。恢复使用已提交 epoch，不能从仅含 `model.pt` 的副本恢复 optimizer/RNG。

Loop 的逻辑执行层相似度 monitor 接线属于后续 LL9；这些命令没有声称已支持该功能。AirfRANS/Car 不在 LL6 生产接线范围。
