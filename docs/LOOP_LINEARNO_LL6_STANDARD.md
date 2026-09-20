# LL6 — 六 Standard 任务接线与原生合成闭环

唯一阶段状态：**PASS**。仅 Airfoil、Darcy、Elasticity、Pipe、NS、Plasticity 接入 `linearno_loop`；LL7–LL10 未执行。真实数据、500 epoch、GPU、远端验收均 **NOT RUN**。

## A. 范围与真实调用链

仓库 `/home/hwz/CDLNO`，`main`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`，tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`，remote `https://github.com/hxh5159/CDLNO-w.git`。保留起点全部 tracked/untracked/ignored 文件；没有 reset/clean/stash/checkout/rebase/commit/push。起点 [2285 文件清单](loop_linearno_audit/ll6/start-manifest.json)、[status](loop_linearno_audit/ll6/start-status.txt)、原 diff 和修改前源码均已保存。

实际链：原 `tran_evaluate/linearno/<task>_{train,eval}.sh` → `_static.sh` → 对应 `exp_*.py` 原 parser → `cdlno_entry.parse_args` 的新显式 loop guard → `cdlno.linearno_loop.standard_entry` → 原 `model_dict.get_model` 的 loop guard → LL3–LL5 `build_from_config` → `cdlno.linearno_loop.standard.Model` → 原 exp 的完整训练/评估主干 → `linearno_entry.StandardRun` 的 loop dispatcher → `LoopStandardRun` / 独立 loop checkpoint。

| task | 原 exp / 有效 factory key | 原 variant | base M → 默认实际 M | 网格/接口 |
|---|---|---|---|---|
| Airfoil | exp_airfoil / LinearNO_Structured_Mesh_2D | conv_temp | 64 → 128 | 221×51，fx=None |
| Darcy | exp_darcy / LinearNO_Structured_Mesh_2D | conv_temp | 64 → 128 | 85×85，fx1，unified off |
| Elasticity | exp_elas / LinearNO_Irregular_Mesh | temp | 64 → 128 | 972点，fx=None |
| Pipe | exp_pipe / LinearNO_Structured_Mesh_2D | conv_temp | 64 → 128 | 129×129，原点序 |
| NS | exp_ns / LinearNO_Structured_Mesh_2D | plain | 32 → 64 | 64×64，输入10帧，输出1帧 |
| Plasticity | exp_plas / LinearNO_Structured_Mesh_2D | conv | 64 → 128 | 101×31，T[B,1]，输出4通道 |

表为 `paper_table8_on_release_model` / `official_release`；旧 profile 保持原值，包括 NS d256/ratio2/ref10。Loop actual M 与 U/E 单独保存在 loop_spec，不能把原 profile 的 layers8/base M 误读为实际模型构造参数。P1 存5套/执行8次，P2 存6套/执行8次；直接构造 U 个物理 block，无先构造8层再丢弃。模型仍是 `Model(x,fx,T=None)`。

## B. 文件与关键增量

新增三个生产适配模块：

- `cdlno/linearno_loop/standard_entry.py`：显式参数来源、metadata恢复、factory构造、独立目录、原 Run 接口、loader generator 和续训状态适配。
- `cdlno/linearno_loop/checkpoint.py`：独立 loop 格式的 JSON先读、strict pair/manifest、optimizer校验与原子提交。
- `cdlno/linearno_loop/provenance.py`：loop真实源码hash，以及旧 pure/history 的逐字节精确路由投影。

已有生产代码只改六文件：三个 Standard 公共路由、`cdlno/experiment.py` 的 family/记录字段识别、两个 history provenance/checkpoint 的来源指纹兼容点。后两者仅调整源码 hash 输入，未改变模型、权重加载、checkpoint 格式或 RNG。新增片段必须精确匹配才能剥离；未知变更失败，绝非忽略源码不一致。

新增 `tests/loop_linearno/test_standard_entry.py`、`native_worker.py`、`tests/loop_entry_projection.py`；增量更新旧 AST projection helper 和 loop isolation 测试。隔离测试仍检查原始 hash，只剥离已授权的新 guard；不豁免整个路由文件。文档/执行证据在本报告、[远端命令](LOOP_LINEARNO_LL6_COMMANDS.md) 和 [ll6](loop_linearno_audit/ll6/)。

**六个 exp 文件、原模型、loop core/wrapper/AttnRes/body、纯 schema/profile、industrial入口、launcher、monitor、依赖全部未改。** 详见 [增量 diff](loop_linearno_audit/ll6/source.diff)、[最终冻结](loop_linearno_audit/ll6/end-freeze.json)。

## C. 配置、数学边界与 checkpoint

新 train 必须显式 `--linearno-loop 1`、topology 和 residual，并使用有效的原 LinearNO factory key。eval/resume 必须给准确 run 路径，先检查 `architecture.json` 的 family，再从完整 loop config 恢复模型；省略 `--model` 也可恢复正确 key。省略所有 loop 字段的旧命令仍走旧分支。

公共字段采用 `argparse.SUPPRESS` 来源追踪。拒绝 preset/custom混用、actual rank/multiplier混用、未知模式、旧 `--n-layers`/`--slice_num`、loop与任何A/K/history-dropout/fair-run flags混用（含显式false）、错误model/task/结构/profile。显式 eval/resume 值仅作一致性断言，冲突在构造和 `torch.load` 前拒绝。Darcy official_release 非500epochs明确拒绝，避免原固定500的 OneCycle 和训练循环错步。

原科学计算保持：Darcy decode/边界置零/0.1导数项/zero-padding/dx=1/85；四静态任务 sample-wise flatten/no-epsilon rL2；NS训练10步真值回填→一次backward/update，评估10步预测回填；Plasticity每batch20个独立时间查询/更新→一次scheduler。现有 Plasticity 实际由 collate 每样本 `torch.randperm(20)`，没有替换成历史文字所述的 NumPy 排列。NumPy状态仍完整保存恢复。卷积原点序、位置替换/拼接、normalizer、数据split、loss、optimizer、scheduler和可视化都通过原 exp 执行。

共享 core 每次 visit 重算 Q/K/V/C/O；每次模型 forward 新建局部历史。NS每个滚动时间调用和Plasticity每个时间查询均验证router来源数重新从首receiver开始。没有跨时间缓存或复用输出头。

Checkpoint family=`linearno_loop`、extension=`loop_linearno_v1`、class=`cdlno.linearno_loop.standard.Model`、pair format=`linearno-loop-epoch-pair-v1`，完整 loop_spec/构造kwargs/profile/data/objective/evaluation/provenance/normalizer/resume各自记录。pair内full archive与独立state_dict和metadata先写入，再提交带SHA-256的epoch manifest，最后更新latest/final；共享core权重仅一份。纯权重便利文件 `model.pt` 不代替完整恢复档案。

读档先验证schema、config、sidecar不变字段、路径和文件hash，再 `weights_only=True`，逐键/shape/dtype/finite检查并 `strict=True`。Resume另核验optimizer命名参数组/数量/ID/矩moment shape/dtype、scheduler构造和实际步数、拓扑/router键、sampler/generator。RNG在构造、权重、optimizer、scheduler全部恢复后最后恢复。旧epoch回滚、同epoch改权重覆盖、跨mode恢复、已有目录新train都拒绝；独立锁避免并发提交。

沿用 LL1 paired seeds：三mode公共backbone初始化一致，DataLoader独立generator；Plasticity仅用已有 `GeneratorCollate` 将原collate的Torch排列纳入该generator，不引入旧A/K模块。eval/resume通过保存的命名normalizer数值恢复，禁止重新fit；train sidecar不重写。

## D. 实际验证与数值证据

环境：本地 Python3.13.9、torch2.13+cu130、PyG2.3.1；所有本轮执行显式 `CUDA_VISIBLE_DEVICES=''`、CPU单线程、Agg、禁pyc。这不是远端Python3.10/torch2.11+cu128验收。

1. **完整 loop suite：78方法，76pass、2CUDA skip、0failure/error，121.052s。** 新4方法覆盖108真实parser profile配置、custom/rank1、非法字段/错配、真实小档案strict/optimizer损坏/防覆盖、旧parser/factory/RNG及原来源hash。LL2–LL5七份数学/参数/MAC报告与LL5完全相等，未放宽容差。
2. **六任务×两preset×三mode：36/36原生合成闭环，144个新进程。** 每例连续3epoch，对照1epoch中断→新进程续训2epoch→再新进程eval。最终权重、完整optimizer/scheduler/Python/NumPy/Torch/generator状态、batch序列和预测hash完全一致（max_abs=0）。所有checkpoint strict。真实空间N，d8/h2/M4、B2、4train/2test、attention dropout0.1；不是全宽或真实数据训练。
3. NS每例连续训练60forward/6backward/6optimizer/6scheduler；Plasticity每例120forward/120backward/120optimizer/6scheduler。真值/预测回填、每次forward router来源计数/8次attention/单head、时间序列和无跨调用状态都在真实main AST中检查。[逐例全部结果](loop_linearno_audit/ll6/native-matrix.json)、[汇总](loop_linearno_audit/ll6/native-summary.json)。
4. **修改前真实 pure/history 档案**重放：在生产修改前生成连续/中断基线，改后新进程恢复/评估，两个family的final权重/optimizer/scheduler/RNG/批次/预测均精确相同。保留外部原始档案 `/home/hwz/CDLNO-artifacts/loop-ll6-before-tzkiikht`，不以改后自生成fixture冒充兼容。[重放](loop_linearno_audit/ll6/pre-checkpoint-replay.json)。
5. 旧LL0的153方法仍为 **147pass、2历史失败方法（4断言）、4CUDA skip、0error**；24模块结果及历史失败文本与已批准LL0完全一致。历史失败来自README/path.sh/旧复现矩阵冻结断言，未改golden使其通过；不能声称全仓测试全绿。history59/59、monitor10/10通过。[回归汇总](loop_linearno_audit/ll6/regression-summary.json)。
6. **72训练命令预览 +36已有run评估预览**，进一步把launcher打印的实际argv送入真实parser，通过全部配置比较；无数据读取/任务import。[命令证据](loop_linearno_audit/ll6/command-previews.json)。

原训练记录器实际生成最终field PNG/PDF/NPZ、日志/指标及独立eval记录。测试沿用已批准的AST worker：只替换真实loader为合成张量、CUDA搬运为CPU，并关闭独立eval的showcase绘图片段；没有替换train/eval科学主干。训练最终field绘图实际执行，独立eval showcase图不在本轮验收范围。

初次验证记录保留：pilot数值/续训成功，但测试错误地只在 `training_artifacts` 查找图片，后改为包含实际 `visualizations`；第一轮suite旧LL1 isolation产生1方法/7子断言失败，改为精确剥离新路由后校验旧字节，最终78方法通过。二者都是测试范围/路径修正，没有据此更改科学代码或数值容差。日志 `native-pilot.log`、`loop-first-tests.log` 保留。

可重跑命令（仓库根，均为合成测试）：

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests:$PWD" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -B docs/loop_linearno_audit/ll6/run_native_matrix.py
python -B docs/loop_linearno_audit/ll6/run_existing_regressions.py
python -B docs/loop_linearno_audit/ll6/check_commands.py
```

完整loop suite将report变量分别指向本阶段证据目录后运行（避免写旧报告）：

```bash
LOOP_LL2_ATTNRES_REPORT=docs/loop_linearno_audit/ll6/attnres-report.json \
LOOP_LL2_BODY_REPORT=docs/loop_linearno_audit/ll6/body-report.json \
LOOP_LL3_PARITY_REPORT=docs/loop_linearno_audit/ll6/sr-parity.json \
LOOP_LL3_ACCOUNTING_REPORT=docs/loop_linearno_audit/ll6/accounting.json \
LOOP_LL4_REPORT=docs/loop_linearno_audit/ll6/rb-report.json \
LOOP_LL5_REPORT=docs/loop_linearno_audit/ll6/lb-report.json \
LOOP_LL5_MODES_REPORT=docs/loop_linearno_audit/ll6/modes-report.json \
python -B -m unittest discover -s tests/loop_linearno -p 'test_*.py' -v
python -B docs/loop_linearno_audit/ll6/finalize_evidence.py
```

## E. 冻结与交付自审

起点2285文件：仅六生产路由/provenance、两个测试helper/isolation和本研究STATUS九文件发生授权变化，其余2276内容/分类一致。新ignored仅本阶段日志，合成运行/权重放仓库外，不提交模型二进制。HEAD/tree/branch未变、staged diff未变，`git diff --check`通过。六exp完整字节相等，比去除guard后的AST检查更强；六公共生产修改文件精确投影到修改前全部字节。[投影](loop_linearno_audit/ll6/routing-projection.json)、[freeze](loop_linearno_audit/ll6/end-freeze.json)。

五项优先自审均已闭环：旧路由不变且新flags fail-fast；actual M/U/PCRS仅取loop真值；真实时间循环调用/步数不变；metadata先读与strict完整恢复；可视化/输出路径及真实数据边界说明准确。[机器可读审查](loop_linearno_audit/ll6/delivery-review.json)。没有新增待用户裁定的架构冲突。

## F. 未运行与限制

真实loader、500epoch、远端GPU、AMP/compile、全宽任务训练、精度/收敛/SOTA/epoch性能、独立eval showcase图均NOT RUN。合成证据不代表这些项目通过。Loop的AirfRANS/Car生产接线属于LL7；专门loop launcher和逻辑visit monitor属于后续阶段，本轮不实施。现有构造模块的早期“synthetic-only”说明是LL3历史描述，LL6只通过独立adapter调用同一已验证构造函数，未改其数学。

本 LL6 阶段结束，未执行下一阶段。
