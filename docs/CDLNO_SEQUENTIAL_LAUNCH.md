# 八任务训练后自动评估启动补充

日期：2026-09-14。用户明确暂停A1，另行要求八任务训练/评估命令及可连续执行的脚本。本轮只完成启动编排，不继续实现前段消融，不实际训练。

- **A 范围**：新增 [tran_evaluate/train_eval.sh](../tran_evaluate/train_eval.sh)，一次指定一个任务，先调用原 `TASK.sh train`；仅退出码0时调用 `TASK.sh eval`。训练失败保留退出码并停止；eval失败亦保留退出码。
- **B 修改**：新增一个组合脚本及本报告/只读验收材料；[tran_evaluate/README.md](../tran_evaluate/README.md)增量补充8条组合命令、覆盖参数及路径说明。AGENTS/STATUS/memory增量记录暂停和完成边界。已有8任务脚本、path.sh、模型、配置、训练/评估入口、数据、loss/optimizer/scheduler/metrics及依赖均未改。相对本轮起点的文件证据见 [freeze.json](sequential_launch_audit/freeze.json)，保留用户先前远端修改和A0/A1产物。
- **C 模型/配置**：无数学或张量变化，仍使用现有完整full前段、L8/F2/P6、entry/chunk0、两FFN ratios2。Pipe M32，其余M64；六标准500epochs，Car200，AirfRANS398；其余d/h/batch/位置/时间/loss/循环沿用各任务现有脚本。`front_latent_mode`尚未实现，没有加入相关CLI。shared参数传给两步；`--train-args`和`--eval-args`分别只传给一端。
- **D 实测**：`python -B docs/sequential_launch_audit/verify.py`通过8组默认+8组覆盖，共32条实际parser命令；不import/运行exp/main，通过AST只提取parser定义，再调用已有CDLNO parser helper。`bash -n tran_evaluate/train_eval.sh`及`git diff --check`通过。纯shell替身验证train→eval顺序、train退出17不eval、eval退出19传递、两步tag一致、空格引用、专属参数隔离、5种错误调用及dry-run不创建目录。结果在 [results.json](sequential_launch_audit/results.json)。未执行模型前反向、PyG/GPU集成、真实数据训练/评估；旧环境不安装/不更换。
- **E 输出/冻结**：未设置CDLNO_RUN_TAG时只生成一次时间+进程号tag，两步共用；已设置则沿用。显式run目录必须放shared区且仍由已有Run helper拒绝覆盖；不自动找latest/best或resume。AirfRANS从同一Dataset环境变量分别派生训练Dataset自身/评估父目录，禁止共享同一`--my_path`。Car保留原固定raw路径/param0和fold0的完整阻力评价限制，训练成功不保证远端该路径已就绪。
- **F 自审/限制**：已自行核对成功/失败顺序、实际parser默认和覆盖、同run/同模型参数、两工业不同路径语义及冻结文件。无剩余脚本缺陷被本轮检查发现；远端数据格式/路径、依赖完整性和真实训练/评估仍未实测。需要改Car原metric固定路径时应另行明确处理，不在本次顺便改数据或指标。

使用示例（选一个任务运行）：

```bash
bash tran_evaluate/train_eval.sh darcy --gpu 0 --dry-run
bash tran_evaluate/train_eval.sh darcy --gpu 0
bash tran_evaluate/train_eval.sh elasticity --gpu 0
bash tran_evaluate/train_eval.sh airfoil --gpu 0
bash tran_evaluate/train_eval.sh pipe --gpu 0
bash tran_evaluate/train_eval.sh ns --gpu 0
bash tran_evaluate/train_eval.sh plasticity --gpu 0
bash tran_evaluate/train_eval.sh car --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans
```

原单步 `TASK.sh train` / `TASK.sh eval` 继续可用。完整配置、远端工作目录、共享与独有参数范例见 [README](../tran_evaluate/README.md)。

## A1 暂停点（不算A1验收）

A0已获用户通过。A1在首次生产编辑之前被用户暂停，**尚未修改cdlno/config.py/modules.py/core.py或八任务实现**。

已保存修改前工作树至 `/home/hwz/CDLNO-artifacts/front-a1-before-_okzkms0/source`；起点哈希在同目录父级 `start.json`。捕获文本为 [capture_before.py](front_ablation_audit/a1/capture_before.py)。CPU FP32/math SDPA、eval/dropout0、threads1、确定性设置下，12个front/core基准和2个AirfRANS旧对象/list基准成功，两次输出/梯度逐元素一致，固定atol=rtol=0。Car进程因脚本执行路径未将原工作目录加入sys.path而报 `ModuleNotFoundError: models`，未生成成功基准；这是捕获工具调用路径问题，不能记为模型失败或基准通过。恢复A1时先修正该独立进程PYTHONPATH并使用新子目录/保留失败记录，核对原source哈希，不能改后伪造旧对象。

上述基准环境为本地Python3.13.9/torch2.13.0+cu130；不是用户远端torch2.11/cu128验收。A1后续实现和针对性测试仍未执行。本轮启动脚本交付不表示A1完成，也不授权继续A2。
