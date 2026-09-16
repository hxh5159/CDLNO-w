# 六标准任务单 seed 顺序训练/评估

本次按用户要求新增一个可直接调用的入口
[run_seed.sh](../tran_evaluate/kcdlno/run_seed.sh)，每次选 seed=0/1/2 中一个。
执行顺序为 **Darcy → Airfoil → Plasticity(plas) → Elasticity(elas) → NS → Pipe**。
每个任务独立训练、立即评估并显示带 seed 的结果，完成后才运行下一任务。
本次仅实现和无数据验证，没有启动实际数据集训练。

## 使用

### 远端报 `unrecognized arguments: --seed 0`

该错误说明实际加载的标准任务parser没有seed选项；顺序脚本已经传了seed但Python接入
代码仍旧。它发生在参数解析阶段，尚未进行训练，timm的FutureWarning不是退出原因。
本地用修改前parser复现了同样的exit2，当前parser已验证接受`--seed 0 --gpu 1`。

除了`run_seed.sh`和`_seed_suite.py`，种子功能还需要同步三个现有文件：

1. `PDE-Solving-StandardBenchmark/cdlno_entry.py`：接收seed。
2. `cdlno/kcdno/entry.py`：实际设置RNG、记录初始化seed、eval恢复。
3. `cdlno/experiment.py`：把seed写入训练和评估结果，供逐任务摘要核对。

不能只添加parser选项或删除脚本的seed参数：前者可能没有实际设置RNG，后者会使
指定seed实验失效。已提供仅涉及这三个文件的[最小补丁](kcdno_audit/seed_suite/seed-support.patch)。
补丁在独立旧文件副本完成`git apply --check`和apply，得到的三个文件与本地实现
字节相同，未触及模型和数据循环。核查记录见
[remote-seed-diagnosis.json](kcdno_audit/seed_suite/remote-seed-diagnosis.json)。

把补丁上传到远端KCDLNO仓库根目录、文件名保留`seed-support.patch`，在该目录执行：

```bash
git apply --check seed-support.patch && git apply seed-support.patch
```

这会保留补丁之外的内容；如远端已有不同修改而check失败，应先核对差异，不强行覆盖。
也可在对齐差异后同步上述三个文件的完整最新版。下面仅提取并执行真实argparse定义，
不会import exp或读取数据/构造模型/创建实验目录：

```bash
python - <<'PY'
import argparse, ast, pathlib, sys
root = pathlib.Path.cwd()
project = root / 'PDE-Solving-StandardBenchmark'
sys.path[:0] = [str(project), str(root)]
import cdlno_entry
tree = ast.parse((project / 'exp_darcy.py').read_text())
nodes = [n for n in tree.body if
         (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'parser') or
         (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
          ast.unparse(n.value.func) == 'parser.add_argument')]
scope = {'argparse': argparse}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<parser-only>', 'exec'), scope)
args = cdlno_entry.parse_args(scope['parser'], 'darcy',
    argv=['--model', 'kcdno', '--seed', '0', '--gpu', '1'])
print('parser:', cdlno_entry.__file__)
print('seed=', args.seed, 'gpu=', args.gpu)
PY
```

预期输出seed=0/gpu=1且parser路径在当前KCDLNO目录。再执行原命令即可创建全新的
队列标识，不需要删除此前失败的汇总：

```bash
bash tran_evaluate/kcdlno/run_seed.sh 0 --gpu 1
```

上述补丁和检查已在本地完成，远端文件尚需用户同步；没有访问远端或启动真实训练。

### 正常启动

```bash
# 每条是一次独立选择；保持该任务原预设。
bash tran_evaluate/kcdlno/run_seed.sh 0 --gpu 0
bash tran_evaluate/kcdlno/run_seed.sh 1 --gpu 0
bash tran_evaluate/kcdlno/run_seed.sh 2 --gpu 0

# 先预览（不读数据、不创建实验目录）：
bash tran_evaluate/kcdlno/run_seed.sh --seed 1 --gpu 0 --dry-run

# 终端直接运行会提示输入种子：
bash tran_evaluate/kcdlno/run_seed.sh
```

队列使用既有KCDNO主profile、L8/r16/history all。六任务的原d/h/M/FFN、500epochs、
batch、优化器、normalizer、loss、点序和NS/Plasticity时间协议保持。
脚本支持 seed、gpu、dry-run；其余任务专用参数保留在原单任务脚本。
数据路径沿用path.sh及环境变量，不为六个不同数据合同强塞一个data_path。

## 实际变更与seed生效位置

| 文件/符号 | 作用 |
|---|---|
| `tran_evaluate/kcdlno/run_seed.sh` | 公共入口、终端选seed、help与非交互缺参处理、复用现有Python/路径 |
| `tran_evaluate/kcdlno/_seed_suite.py:run_suite` | 固定六任务顺序，逐任务train→eval→report，传相同seed和独立run目录，失败停止 |
| 同文件 `task_summary` | 直接读原训练末轮/独立评估JSON，核对completed/task/seed，拒绝空结果，不重算指标 |
| `PDE-Solving-StandardBenchmark/cdlno_entry.py:parse_args` | 新家族可选 `--seed`；默认SUPPRESS使旧命令Namespace不新增此字段；旧CDLNO/Transolver显式seed清楚报错 |
| `cdlno/kcdno/entry.py:seed_process/resolve_args` | 在数据迭代/模型构造前设置Python random、NumPy、torch CPU/CUDA RNG；eval省略seed可从已有task记录恢复 |
| 同文件 `StandardRun` | 在已有初始化元数据的seed槽记录真实seed，保留strict权重加载；不改变架构字段 |
| `cdlno/experiment.py:Experiment.__init__` | 可选seed写入训练/评估结果并打印，配置已有resolved_arguments也记录seed |
| `tests/test_kcdno_seed_suite.py` | 初始化、配置/checkpoint、顺序、报告、失败及安全预览的定向检查 |

种子设置前按原args.gpu设置CUDA_VISIBLE_DEVICES，CUDA seed使用延迟初始化API。
队列的每个Python子进程在启动前还获得PYTHONHASHSEED；没有仅在父进程设置一次种子。
无seed的旧训练路径不调用seed_process。支持的新家族包含kcdno和已有lrsa_matched，
但本次六任务队列默认且仅启动KCDNO。
seed属于复现信息，不属于参数形状；单独eval显式seed可另行设置，省略时恢复原训练seed。
历史未记录seed的checkpoint不补造历史seed。eval不改写训练config/task/architecture。

## 输出与失败语义

每任务：`output/<task>/kcdno/<UTC时间戳>_seed<N>_<唯一标识>/`。
同一次队列用相同标识关联六个任务；时间戳和唯一标识使相同seed的多次实验也不覆盖。
入口在读数据前排他创建任务目录，调度程序不预创建该目录。

每次评估结束立即打印 `[seed=N] <task>: training + evaluation completed`，其后为
原末轮训练指标与独立评估指标，并写`seed_summary.json`。原`train_results.json`和
`evaluations/.../results.json`及`eval_results.json`保留且有seed标记。
对应训练配置和checkpoint架构文件分别有`resolved_arguments.seed`和`initialization.seed`。
模型权重格式不变，本次不新增resume。

队列汇总位于`output/_seed_suites/kcdno/<同一标识>/summary.json`，每次完成任务后
更新，包含已完成任务、seed、当前任务/阶段。训练失败不进入该任务eval；eval失败不
开始下个任务；记录缺失或seed错误也停止，不显示伪造成功。之前的任务结果保留。
Ctrl-C在可正常捕获时记录interrupted；强制kill/断电可能留下running，不等价于完成。

## 实际验证

起点为main分支commit `9c5569e059eca8d57f5321fdbf0c0dc4aeb40860`，起点git status干净。
环境Python3.13.9、torch2.13.0+cu130、NumPy2.2.6；没有安装或改变依赖。

```bash
PYTHONPATH=tests:. python -B -m unittest \
  test_kcdno_seed_suite \
  test_kcdno_tasks.StaticKCDNO.test_train_eval_roundtrips_and_conflicts \
  test_kcdno_delivery.DeliveryChecks.test_legacy_parser_imports_without_shared_package -v

PYTHONPATH=tests:. python -B -m unittest \
  test_experiment_records.ExperimentRecords.test_all_task_modes_early_config_actual_counts_epochs_repeat_eval_checkpoint \
  test_experiment_records.ExperimentRecords.test_old_runs_without_config_keep_loadable -v

bash -n tran_evaluate/kcdlno/run_seed.sh
git diff --check
```

第一组8/8通过，6.169s；第二组2/2通过，10.901s，合计10项，无失败或跳过。
第二组[实际日志](kcdno_audit/seed_suite/record-regression.log)。检查范围：

- 六任务×三个seed，用实际parser AST和真实小模型构造器各重复两次；同seed每个state_dict
  张量精确一致，不同seed确实产生不同权重。Python/NumPy/torch抽样一致；不传seed不重置RNG。
- KCDNO和matched两家族实际CPU forward、strict保存/加载输出精确一致；初始化/config/result
  中seed正确；eval恢复seed且原三个sidecar字节不变。这里的记录用显式synthetic_record_only，
  不称作原训练损失。
- 三种seed及重复seed的队列调度用**结果记录夹具**核对12次train/eval调用及每任务立即打印；
  不创建任何冒充真实数据集的文件。中途train/eval失败保留Darcy结果并停止。
- 五次真实shell串行预览（直接helper三个seed、公共入口两种写法），共60条底层命令。
  helper的三个seed测试以禁止调用的Python替身作保护；所有预览不进入exp、不创建输出目录。
  help、非交互缺参、非法seed检查通过。
- 复用原无seed静态任务all/off checkpoint及旧模型独立parser/24任务模式记录回归。
  [冻结检查](kcdno_audit/seed_suite/freeze.json)：195个已有源文件中192个hash未变，
  仅上表三个seed/记录helper发生授权修改；六个exp、所有模型数学、配置和原脚本均未改。

初次新测试有一个输入布局错误：把辅助构造器生成的5×5 Darcy模型配成35点，正确被
wrapper拒绝。测试改为读取实际H×W后通过，生产网格合同保持；不是放松N检查。

五项自审已完成：种子实际生效、任务/阶段顺序、逐任务即时结果、路径隔离与失败停止、
默认旧路径和数学/时间循环冻结。没有需要用户裁定的架构变化。
未验证真实数据读取、完整训练收敛/精度、远端Torch2.11/cu128、真实GPU训练重复性。
未改变SDPA/TF32/cuDNN/AMP或启用强制确定性，因此不保证不同GPU/backend逐位一致。

本阶段结束，未执行下一阶段。
