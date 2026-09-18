# R6：四个静态 Standard 任务生产接入

## A. 范围与审计

本轮连续 R6–R10 授权优先于历史单阶段停顿条款；每阶段独立验收再继续。R6 只接 Airfoil、Darcy、Elasticity、Pipe。读取对应阶段原文、现有 R0–R5 状态、pure/research 模型、factory、parser、StandardRun、experiment、任务 exp、launcher 和测试。原文保存在 `linearno_history_audit/r6/prompt.txt`。

修改前完整 tracked/untracked/ignored 快照在 `/home/hwz/CDLNO-artifacts/linearno-history-r6-before-hd6eqt71`；HEAD `d5abe014ed05ec9286200d677b039bbd68697f96`。六个 exp 和工业入口保持字节相同，没有运行真实数据/GPU、训练、commit/push 或清理工作树。

## B. 决定及依据

两个架构真值仍为 A/K bool，p=.1 固定。任务 profile/model/grid/loss/data/optimizer/scheduler 原样复用。研究保存独立 family/schema，eval/resume 先 metadata 后构造，再完整 strict load。原参数 key 和 baseline checkpoint 不增加研究 metadata。

R1 公平协议落实为：研究构造时从当前原初始化 RNG 生成公共 pure 主干，在隔离 fork_rng 中生成后逐键复制；实际构造只推进一次原主干 RNG；新增参数 feature seed 隔离。显式公平 launcher 的 A0K0 保持原类，使用相同公共种子得到逐值相同主干。DataLoader train/test 独立 generator，使 A dropout 不改变样本顺序。该运行协议记录在外部 manifest；原命令缺省或仅 A0K0 不启用公平运行，旧行为不变。

共享接线必然改变生产源码 fingerprint。`provenance.baseline_source` 仅将清单内精确路由片段恢复成原字节再作旧 fingerprint，未知变更拒绝，不忽略任何原源码字节；研究 fingerprint 另含实际路由和全部研究源码。验证旧两个 hash 精确相等。后续阶段改变研究源码会按原严格规则拒绝旧研究训练继续；正式实验须冻结最终版本，不绕过 hash。

## C. 文件与计算映射

- 新研究 `cli.py/standard_entry.py/fair_run.py/provenance.py`：独立解析、run/signature、原生 Run 子类、严格 metadata-first checkpoint、公平初值和 generator。
- 研究 `factory.py`：新增显式公共主干逐键复制入口；不修改 A/K/core/context 或三套 wrapper 数学。
- Standard `cdlno_entry.py/model_dict.py/linearno_entry.py`：仅 dispatch 到新 family；已有 key/default 不变。
- `cdlno/experiment.py`：研究 family 使用原输出和可视化路径；`cdlno/linearno/standard_entry.py` 仅旧源码 fingerprint 投影。
- `tran_evaluate/linearno_history/{airfoil,darcy,elasticity,pipe}.sh`：平行 train/resume/eval launcher，复用原实际 factory key、路径和命令。
- 新测试 `history_static_worker.py/test_history_static.py`；精确的测试 AST 投影附加 `history_entry_projection.py`，保留旧断言和模型/数据 AST。

实际链为 launcher → 原 `_static.sh` → exp parser → cdlno_entry/intercept → model_dict/research factory → 已验证 Standard research class → 原 exp main → HistoryStandardRun → 独立研究 checkpoint；eval 相同入口先读 metadata。A0K0 返回原 Model/原 StandardRun（显式公平运行仅 Run 的 generator/外部 manifest wrapper）。

四任务默认保持 Airfoil conv_temp/M64/H221/W51；Darcy conv_temp/M64/H85/W85/unified off；Elasticity temp/M64、fx=None；Pipe conv_temp/M64/ratio1/batch4/H129/W129。Darcy decode、.1导数、zero-padding、dx=1/85及边界操作未动；official_release 非500 epochs仍拒绝。纯 plain/conv 由既有六变体 parity 测试覆盖，未提前启用时间任务。

新增理论张量：raw 每层 `[B,h,M,d_h]`；A 每真实来源 `[B,h,M,M]` 的跨深度 Cross；K 的 `[B,h,M,l*M]` 历史读取和 `[B,h,N,M]` 修正；无 N×N。真实参数统计见 `final-results.json` 的80行（四任务×L4–8×四组合），不是论文借用值。

## D. 验证命令与结果

统一环境：CPU、Python3.13.9 / torch2.13 cu130 build；`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。没有安装依赖。

- `python -B -m unittest linearno.test_history_static -v`：80 preset 实际构造/参数统计、原 RNG/主干精确匹配、16原生合成 train→checkpoint→resume→新进程eval闭环。合成使用真实 N，d8/h2/M4/L4，3 epochs、4train/2test；不是真实数据训练。对同一任务四组合的公共初始 hash 和 batch顺序相等；每组合连续/恢复权重、optimizer/scheduler/RNG和预测 hash完全一致。12 launcher preview、非法参数和历史 source hash精确匹配。
- 旧 pure 四任务 `linearno.test_static_integration` 单独运行：原数据/损失/空间点序/新进程续训检查；保留旧 path.sh hash 漂移失败，已证明该文件本轮未改。
- R1–R5、pure官方parity、Transolver八任务、随机流和 monitor：`regression-final.txt`。原始并发运行中，在进程已加载旧 factory 后新增函数导致 stale import；最终固定源码后重跑，不将原始 errors冒充生产缺陷或成功。日志完整保留。
- 其他模型/记录/可视化：`other-models.txt`。CDLNO/KCDNO/MSAR数学与合成记录检查执行。两个历史 AST 方法中的5条Air相关失败，在真实R6前快照重跑得到同样失败（`preexisting-output-ast.txt/preexisting-periodic-ast.txt`）；本阶段未动其工业文件。

实际最终数字见本报告末尾验收记录，不把历史失败汇总成全绿。

## E. 兼容性

`freeze.json` 逐文件包含原tracked/untracked/ignored内容和分类变化。旧pure/Transolver/其他模型数学、六exp、数据/指标、LINEARNO/monitor和原launcher原样保留。生产变更仅上述路由/观察调用；旧源码指纹有独立实际快照逐字验证。A0K0旧checkpoint仍由旧helper读取，研究checkpoint不 fallback/strict=False。

## F. 自审及边界

已核对4项：metadata先于构造；四题原损失/点序；A0K0类/schema/RNG；公平初始化和续训generator。NS/Plasticity未启用，工业未接入。真实loader/mini-run/完整训练、GPU/AMP/远端兼容、精度/收敛/epoch时长均NOT RUN。远端可执行命令见 `LINEARNO_HISTORY_COMMANDS.md`；本阶段没有执行那些真实命令。

## G. 最终验收

最终固定源码旧回归114方法：109通过、4 GPU跳过、1历史失败方法含3断言，0 errors，123.248s。pure静态7方法：6通过，1历史path.sh断言失败，84.879s。其他模型/记录69方法：61通过、6资源跳过、2历史AST失败方法含5断言，80.409s；失败均在本轮前快照复现。R6新增4/4方法通过，356.901s；80构造计数、16原生完整闭环均PASS，最大精确比较误差0。唯一状态 **PASS**。本R6已独立审查完成，依据连续授权进入R7。
