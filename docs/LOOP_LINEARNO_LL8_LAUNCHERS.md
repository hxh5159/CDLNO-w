# LL8：统一launcher、公平记录与命令验收

唯一阶段状态：**PASS**。本阶段没有执行真实数据/GPU训练，未进入LL9。

## A. 实际范围与读取

根目录`/home/hwz/CDLNO`，branch main，remote `https://github.com/hxh5159/CDLNO-w.git`，HEAD `5b991226c5354af3332b2f7306b370aef0950c79`、tree `a661e0a53d367e09dfe9b5afaabcab29ee8aec63`。当前事实是LL7已批准，AGENTS中的旧CDLNO阶段状态保持历史记录。读取根AGENTS及既有outputs使用/报告、LL6/7状态/报告/命令、分阶段提示词LL8、path.sh、原16个LinearNO launcher与_common、工业Car旧eval检查、八任务parser/路由及loop配置/构造/core/checkpoint/独立generator协议、相关回归与native workers。

修改前[2524文件manifest](loop_linearno_audit/ll8/start-manifest.json)包括1549 tracked、387 untracked、588 ignored；400行原status及tracked/staged diff全部保存。没有reset/clean/stash/checkout/rebase/commit/push、依赖变更、真实数据或GPU调用。

## B. 文件、插入点与兼容决定

新增`tran_evaluate/linearno_loop/<task>.sh`共8个，统一train/resume/eval。它们source原_common/path.sh，调用新增`launch.py`。planner先调用当前纯LinearNO launcher的dry-run获得真实factory key/路径/argv，再只执行原入口AST中的parser构造与公共parse_args；不import exp/main数据脚本、不加载torch权重。有效配置才执行`entry.py`，它从原工作目录以原argv/runpy进入原任务入口。

新增`recording.py`是明确由新launcher启用的观察层：包装loop专用实际构造入口，保存初值hash/实测参数，挂载一次前向的无张量调用计数hook；首成功forward后移除。没有改任何`cdlno/`生产文件、纯schema、任务入口、原launcher、checkpoint来源hash、模型数学或monitor。记录不加入旧state_dict或checkpoint，写独立`loop_run_manifest.json`。

新启动桥不调用旧`check_car_eval.py`：该旧辅助脚本检查的是CDLNO whole-object/fold0/硬编码raw路径，对LL7已经验证的loop safe pair与非零fold不适用。新桥执行真实loop parser与saved metadata检查，再进入原main_evaluation的LinearNO分支，继续复用已批准的Car evaluator。旧helper字节不变。

新增测试`test_launchers.py`和仅用于CPU合成的`launch_record_worker.py`；旧`test_artifacts.py`仅把LL1历史“新launcher目录不存在”断言更新为LL8八脚本存在，保留原LL1 fixture和schema/catalog断言。没有放宽数值容差或修改旧golden。另新增本报告、[完整命令](LOOP_LINEARNO_COMMANDS.md)、本阶段audit脚本/证据，更新研究STATUS。

## C. 公平初始化、数据流和运行记录

沿用LL1/LL3–LL7合同：相同task/topology/rank/profile/seed的三模式公共主干键/值一致，query0/norm1初始化不推进backbone RNG，construct在原fork_rng隔离范围执行。LL8测量真实已构造模型的初值，不新建随机模型来代替实际记录。

运行manifest逐member保存：config hash、unique/executed depth、actual M、公平pair ID、public/member seed、独立train/test generator seed、公共初值SHA256、完整初值SHA256、实测总/主干/router参数量、launcher源码hash。初值hash明确按排序后的tensor键、dtype、shape与原字节计算；不包含router的公共hash用作三模式比较。

实际调用序列由各物理block的Attn/MLP、各点域receiver、末suffix的LN/head hook记录。初始化只存expected，actual为null且标记pending，首成功forward才比较并保存actual；失败forward不产生“通过”记录，下一forward清空临时字符串列表。没有Q/K/C或计算图历史缓存。

拓扑A独立5块、B独立6块，均执行8块；绝不称参数匹配。三mode只按同拓扑作paired比较；RB新增26H/18H，LB两preset均4H，SR0。manifest实际参数计数必须等于schema router count。数据协议沿用独立generator及完整恢复状态，Air每member模型/router独立，generator沿原协议连续推进而不重置；Python随机采样和Plasticity时间查询保持。

`--then-eval`只在train/resume返回0后评估确切RUN；eval结构由metadata恢复，数据路径原样保留，记录与checkpoint均不按test最佳选择。三个seeds=0/1/2和final checkpoint应在实验前固定；完整未来循环包含8×2×3×3=144运行。两种profile的科学含义保持。

## D. 实际验证与结果

本机Python3.13.9、torch2.13/cu130、PyG2.3.1；CPU、单线程、CUDA_VISIBLE_DEVICES空，未改环境。Python3.10仅语法兼容核验，不能冒充远端torch2.11/cu128验收。

- 完整loop suite **88方法，86pass、2CUDA skip、0failure/error，54.950s**；加入bridge委托检查后的最终定向suite **6/6，5.429s**，不是另一次全套重跑。LL2–LL7七份oracle/梯度/optimizer/参数/MAC数值报告与LL7逐项相同。
- **184个真实shell/原launcher/公共parser预览**：正式8×2×3的48train，实际LL7小型档案的48eval、48resume，另8题×custom/rank×1/actual M/official_release/transolver_matched的40train。独立新run路径无碰撞，dry-run零目录创建、零权重加载。resume预览使用隔离复制的真实epoch1 pair，不谎称可继续已完成run。[矩阵](loop_linearno_audit/ll8/dry-run-matrix.json)。
- 负例包括缺字段/错误enum/bool/NaN/rank冲突、A/K混用、PCRS边界、旧depth、hidden/head整除及saved profile/rank/seed等240项；额外saved hidden/heads/variant/每个PCRS字段/loopfalse/Aflag的144项，均在torch.load前拒绝。[补充冲突](loop_linearno_audit/ll8/additional-conflicts.json)。
- **144小模型**：8题×2preset×3seeds×3mode，真实公共初值hash、DataLoader序列与generator状态跨三mode全等；逐模型forward/backward与actual call schedule检查通过。[公平报告](loop_linearno_audit/ll8/fair-report.json)。
- **24原生CPU合成运行**：8题×preset A×3mode，原LL6/LL7 worker只加新记录器；真实Standard空间shape/小宽度，Air真实PyG双member，Car fold3，均调用原科学训练/metric/保存代码。与LL7无观察的同权重训练相比，最终权重、optimizer/scheduler/RNG、批次和输出逐位相等；跨三mode同任务公共初值hash与batch/采样序列相同。[证据](loop_linearno_audit/ll8/recorded-native.json)。这是3epoch内存合成验证，不是真实训练、全宽工业训练或VTK数值验收。
- 三mode单步dropout/AdamW对照，开启记录与关闭记录的输出、全部state_dict、optimizer state、Torch RNG全0差；异常后下一forward可正常记录。native factory安装及独立Air成员通过；resume/eval不重写已有manifest。
- 八shell bash -n、路径带空格、GPU掩码4,7→gpu1选择7、错误GPU、train失败17不启动eval、eval失败23向上传递、成功顺序train→eval通过。顺序测试用明确的执行stub和真实parser/checkpoint副本，不称为真实训练。bridge委托测试覆盖8题×3action原argv/工作目录，禁止执行真实数据entry。

首轮dry-matrix发现测试选用的LL7 split运行已完成，真实parser报`run has completed its resolved epochs`；保留初次日志，改用真实epoch1隔离副本，未改变生产检查。旧LL7明确记录的3个历史回归失败方法未修订或伪称消失；本阶段不重跑11分钟旧history suite，其生产来源文件全部byte-equal，另重跑完整loop与24 native记录回归。

主要命令（仓库根）：

```bash
export PYTHONPATH="$PWD/tests:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -B docs/loop_linearno_audit/ll8/dry_matrix.py
python -B docs/loop_linearno_audit/ll8/additional_conflicts.py
LOOP_LL8_FAIR_REPORT=docs/loop_linearno_audit/ll8/fair-report.json \
  python -B -m unittest loop_linearno.test_launchers -v
python -B docs/loop_linearno_audit/ll8/run_recorded_native.py
python -B docs/loop_linearno_audit/ll8/finalize_evidence.py
```

完整套件的七个报告环境变量均指向ll8，原阶段证据不覆盖；完整实际命令保存在本阶段`test-command.txt`。

## E. 冻结与优先自审

复核五项：真实launcher/parser委托而非复制训练；三mode初始化/随机流公平；实际与预期schedule和参数计数区分；metadata-first/旧checkpoint及输出兼容；远端路径/GPU/失败顺序。生产代码、schema、全部旧launcher、path.sh、monitor及历史证据都不改；旧文件允许变动仅研究STATUS与上述一项过期测试断言。逐文件content/hash/classification、Git HEAD/tree/staged、Python3.10语法、文档Bash语法和diff检查见[最终冻结](loop_linearno_audit/ll8/end-freeze.json)。

## F. 未运行与风险

真实loader/VTK/drag/lift统计、真实训练/准确率/收敛、全宽工业任务训练、远端Python3.10/torch2.11+cu128、GPU/AMP/compile、正式144运行矩阵均NOT RUN。工业路径需远端按真实文件位置设置；本轮没下载数据。无性能或SOTA声明。专用loop相似度monitor/效率profiling仍属LL9，未实现。

本 LL8 阶段结束，未执行下一阶段。
