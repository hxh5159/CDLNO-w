# 四任务空间路由绘图交付验收

2026-09-26；本次开始HEAD为 `0acdf7778975d0ff225be9059331d30ebb586be0`，开始时工作树干净。
**PASS：脚本实现与本地合成验证。NOT RUN：远端真实checkpoint/数据及最终论文页面。**

## A. 授权与范围

为用户给出的Darcy E1、Elasticity E4、Pipe E3实验及NS实验父目录新增独立绘图脚本，复用
Airfoil标准。中途确认脚本放在远端looplin-vis、权重仍在looplin-v5-final，因此支持并记录
通过绝对 `--run-dir` 读取另一checkout输出的完整命令。NS不猜实验，提供 `--list-runs`。
只推理/导出，不训练、不安装依赖，不复制/更改模型，不改训练输出目录。

## B. 文件和diff

- 新增 `darcy.sh`、`elasticity.sh`、`ns.sh`、`pipe.sh`：一个任务一个可从任意cwd启动的薄入口。
- 新增 `task_states.py`：真实数据协议、保存normalizer恢复、NS rollout、通用绘图和记录。
- 最小扩展原 `airfoil_states.py`：capture可显式传fx；atlas/field_reference可传散点painter；
  field_reference可传物理场名称。所有原Airfoil参数默认行为保留，数学和排版未另写一套。
- 新增 `test_task_states.py`：8项新测试；原6项Airfoil测试文件未改。
- 新增 `TASKS.md`、本报告和三份 `evidence/` 原始记录；README增加四任务导航。

## C. 数据、数学和源码

| 合同 | 实际源码位置 | 判别性验证 |
|---|---|---|
| Q=softmax_M(Lq/tau_q)，K=softmax_N(Lk/tau_k)，Y=Q(K^T V) | airfoil_states.capture_last_attention | 既有独立标量oracle、native readout及透明hook；新fx路径与native forward对比 |
| 最后suffix attention唯一visit | capture_last_attention；task_states.main记录路径 | 四任务实际hook次数、strict配对模型 |
| Darcy smooth2前200、stride5、coeff输入编码和sol输出解码 | exp_darcy.py main数据段；task_states.load_sample/saved_normalizers | 非恒定格点值及指定mean/std，逐点/点序断言 |
| Elasticity整个文件最后200、原坐标、fx=None、stress解码 | exp_elas.py main；load_sample/infer | fixture长度1207，确认选1007而非1000，验证sigma轴 |
| Pipe先截前1200再取后200，Q第0通道、坐标编码但原坐标绘图 | exp_pipe.py main；load_sample | fixture长度1207、多通道sentinel，区分1000/1007与错误通道 |
| NS前10输入、后10目标、十次预测反馈 | exp_ns.py eval；task_states.infer | 独立native十步rollout，step1/6/10、输入窗口和预测逐位相同；改未来真值不影响预测/Q/K |
| mean/std只恢复、不拟合、不再加epsilon | standard_entry.normalizer；task_states.saved_normalizers/transform | 解码与直接算式逐位相同；缺normalizer拒绝 |
| 结构任务只按相邻网格单元连边，Elasticity只画原点 | structured_mesh；task_states.point_paint | 旧孔洞连接测试；散点offset精确等于输入，不构造面 |
| 每head原序M个状态，Q/K分开，shared/peak明确 | task_states.export_plots；Airfoil atlas/display_values | NS M32、其他M64、head索引、归一化、颜色标度文件检查 |

NS的透明性检查在所选未来步额外做一次相同输入的前向，随后窗口只推进一次；两次预测逐位相同。
NS输出包含所选步rL2、逐步/平均rL2和十步拼接full rL2，但全都是**单个测试样本**。
没有把可视化样本结果称为整个测试集复现。

## D. 命令、环境、结果

Python3.13.9、Torch2.13.0+cu130、CUDA可用、NumPy2.2.6、SciPy1.16.3、Matplotlib3.10.6。
原始命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s iclr_vis/weight -p 'test_*.py' -v

for p in iclr_vis/weight/*.sh; do bash -n "$p" || exit; done
git diff --check
git diff --exit-code -- cdlno linearno_loop PDE-Solving-StandardBenchmark \
  Car-Design-ShapeNetCar Airfoil-Design-AirfRANS
```

**14 passed / 33.209s，0 failed / 0 skipped**。原始日志：
`evidence/tests-four-tasks-first.txt`。包含原6项Airfoil检查及8项新增检查：

1. 四种真实数据布局、测试索引、通道、坐标点序和保存normalizer。
2. 静态三个任务与直接native模型调用/解码逐位一致。
3. NS独立十步预测反馈、step1/6/10窗口、未来真值隔离、Torch RNG与临时hook清理。
4. 本地GPU Darcy含fx捕获及NS完整十步推理。
5. 错误head、sample、forecast-step、width/columns、缺normalizer、篡改数据负例。
6. Elasticity绘图只保留原点，不虚构网格连边。
7. 四任务各自从外部临时run在新进程中strict加载并完整导出；run/data/output路径均含空格；
   `--preview`无Torch导入、不需要数据；Q/K维度、head索引、全部文件数量正确；
   run和data全部文件前后SHA-256一致；已有输出拒绝覆盖。
8. NS父目录两实验的JSON列表无需Torch，不自动选择任何一个实验，不产生新文件。

Python用内置compile作语法检查通过；全部shell语法及git diff --check通过。
由于只扩展绘图目录，共享生产源码没有改动，没有额外运行全仓或长训练回归。

另用同一fixture构造器生成17×17合成网格及Elasticity散点，逐任务执行：

```text
task_states.py TASK --run-dir <external synthetic run> --data-path <synthetic data>
  --device cpu --output-dir <new directory> --dpi 220
NS额外：--paper icml --forecast-step 6
```

四任务共20份PDF通过PyMuPDF核验：实际宽5.5in/6.75in、所有可见文字9pt、字体实际嵌入、
所有文字bbox在页内、M个状态编号齐全。结果保存
`evidence/four-task-pdf-check.json`，绘制过程原始日志 `evidence/render-four-tasks.txt`。
临时出图目录 `/tmp/v5-four-task-vis-7bm9ak0i/`；已实际打开Elasticity atlas、Pipe atlas和NS场图，
另查看Darcy atlas。均有SYNTHETIC CHECK标记，仅验证布局与执行，不代表训练物理规律。

FAIL：上述检查无失败。NOT RUN：远端Python3.10/Torch2.11环境，用户真实MAT/NPY、
用户指定训练权重、真实完整测试集指标、真实训练/收敛、完整官方模板LaTeX编译。
真实权重能否形成清晰物理分区和最终论文页面效果不由合成检查保证。

## E. 冻结区

模型、配置、schema、checkpoint实现、训练/评估入口、原始数据协议全部未改。
现有Airfoil独立标量oracle/CPU/GPU/新进程strict绘图测试仍通过；旧test文件未改断言或容差。
源run/data只读；新进程校验其文件哈希不变。无reset/clean/commit/push，没有删除旧证据。

## F. 自审与边界

已完成五项重点复核：跨checkout绝对路径与无自动选run；四任务normalizer和split；
NS未来真值不可进入预测反馈；Elasticity不假造孔洞连接；9pt字体/色标语义与原始Q/K存档。
没有待用户批准的模型设计变更。NS还需用户从列表选定一个完整实验目录；这是未提供的实验身份，
不能由父目录或mtime推断。实际远端文件完整性由运行时strict校验进一步验证。
