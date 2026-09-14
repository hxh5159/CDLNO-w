# CDLNO 阶段6：Navier–Stokes 与 Plasticity

日期：2026-09-13。阶段5已审查通过。本阶段只接入NS和Plasticity；四静态任务保持不变，ShapeNet-Car/AirfRANS仍未接入。未修改原时间数据协议或训练语义。

## A. 完成范围

新增时间结构wrapper、模型注册、任务配置、独立训练/评估脚本和有限checkpoint分支。共享核心仍由`cdlno.core.CDLNO`提供，每次wrapper `forward`都重新执行完整核心；不存在跨真实时间步的latent/history缓存。

| 任务 | 外部调用 | stem | 网格/输出 | 默认d/h/M/L/F | 训练协议 |
|---|---|---:|---|---|---|
| NS | `forward(x[B,N,2], fx[B,N,10], T=None)` | 74 | 64×64，N4096，`[B,N,1]` | 256/8/64/8/2 | 10步真值回填，累计loss后一次backward/step |
| Plasticity | `forward(x[B,N,2], fx[B,N,1], T[B,1])` | 3 | 101×31，N3131，`[B,N,4]` | 128/8/64/8/2 | 20个T逐点forward/backward/step |

新脚本：`scripts/CDLNO_NS.sh`、`CDLNO_NS_Eval.sh`、`CDLNO_Plasticity.sh`、`CDLNO_Plasticity_Eval.sh`。用户参数置于显式默认值之后，可覆盖。NS没有改成10→20/40；Plasticity不做预测反馈。

## B. 修改文件与原因

- [cdlno/standard.py](../cdlno/standard.py)：新增`TEMPORAL_TASKS`和`TemporalStandardModel`。NS固定10个历史通道；Plasticity只在`Time_Input=True`时建立time_fc。
- [PDE-Solving-StandardBenchmark/model/CDLNO_Temporal_Structured_Mesh_2D.py](../PDE-Solving-StandardBenchmark/model/CDLNO_Temporal_Structured_Mesh_2D.py)：稳定wrapper导出`Model`。
- `model_dict.py`：仅为`cdlno_task=ns/plasticity`选择时间wrapper；未开放工业任务。
- `configs/CDLNO/ns.json`、`plasticity.json`及4个脚本：显式记录任务结构和训练默认。
- `exp_ns.py`、`exp_plas.py`：只增加新模型构造、StaticRun sidecar/checkpoint和独立结果路径分支；旧分支不接收新kwargs。
- `tests/test_temporal_standard.py`：B>1时间回填、T梯度、20步更新、AST合同、strict checkpoint和脚本检查。
- `cdlno_entry.py`、AGENTS、STATUS、memory及本报告：复用已有有限入口协议并更新阶段记录。

## C. 公式、输入提升与时间流

NS沿原结构模型的固定index-grid reference规则生成64个距离通道，替换xy后拼接10步`fx`：`[B,4096,64]+[B,4096,10]→[B,4096,74]→H_0[B,N,256]`。一次`model(x,fx)`只消费当前10步窗口；核心内部从H0重新构建前段T、Z0和后段latent。

NS训练仍是原循环：

```text
for t in range(0,T,step):
    y = yy[...,t:t+step]
    im = model(x,fx)
    loss += TestLoss(im,y)
    fx = cat(fx[...,step:], y)       # truth forcing
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

测试仍将`im`回填到`fx`，再进入下一步；没有把预测写入训练路径。10步结束后按原`pred`与`yy`计算full loss。每个循环迭代都调用完整core，不能把bridge或latent跨时间复用。

Plasticity保留`output.transpose(-2,-1)`后的标签轴序，再reshape为`[B,3131,4,20]`。每个时间`t`取`T[:,t:t+1]`，wrapper生成原sin/cos timestep embedding并通过time_fc加到每点stem结果：`[B,3131,3]→H0[B,3131,128]`，输出`[B,3131,4]`。原循环每个T单独计算loss、清零梯度、backward、clip、optimizer.step并按原OneCycle步进；无时间之间的预测反馈。标签空间没有展平到`3131×20`作为节点。

共享核心最终仍执行L8/F2/P6、entry CDPA和`H_F` readout；NS/Plasticity的每个真实时间调用都独立建立Python history。

## D. 实际验证

```bash
python -B -m unittest discover -s tests -p test_temporal_standard.py -v
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator \
  python -B -m unittest discover -s tests -p 'test_*.py' -v \
  > /tmp/cdlno-phase6-full-tests.log 2>&1
```

定向测试 **8/8通过**；最终全回归 **81/81通过，0失败/错误/跳过**。环境为Python3.13.9、torch2.13.0+cu130、CUDA13.0、RTX5090 Laptop GPU；未安装依赖。

覆盖：

- B=2的NS三步真值回填与预测回填；hook证明每步产生新的core调用和不同latent输入。
- Plasticity不同T产生不同输出，T的梯度非零有限；20个时间点分别完成optimizer update。
- NS 74维stem、Plasticity 3131点/4通道输出、time_fc仅Plasticity存在。
- 原`TestLoss`连接、strict state_dict/sidecar新进程往返、脚本bash语法和Python3.10 AST解析。
- 原`exp_ns.py`与`exp_plas.py`的新增分支投影后，完整AST分别等于基线commit中的原文件；数据读取、标签轴、时间循环、loss、optimizer/scheduler和旧保存路径未改变。

未运行：真实NS长轨迹、Plasticity真实数据、远端Python3.10/torch2.11/cu128、真实训练/评估、NS 10→20/40。测试使用合成张量，不能替代这些结果。

## E. 冻结区域与证据

本阶段复用阶段5基线`/tmp/cdlno-static-baseline-0a0j551e/hashes.json`。NS/Plasticity exp仅在模型构造、parser/helper、checkpoint和结果路径位置增加分支；其旧分支AST投影与原commit完全一致。未修改loader、normalizer、loss工具、数据字段/划分、时间循环语句、optimizer/scheduler或依赖配置。未下载数据、创建假数据、import顶层exp进行训练、commit/push、PR或reset。

## F. 交付前自审

已自行核对以下重点：

1. NS在训练/测试的反馈方向分别是真值/预测；每个时间调用完整core，未缓存bridge或latent。
2. Plasticity保留`T[B,1]`、标签`[B,3131,4,20]`及每时间optimizer step；time_fc只在Time_Input=True注册。
3. NS stem=74、Plasticity stem=3，实际N和输出通道与原入口合同一致；无空间时间展平。
4. 新旧模型kwargs隔离，sidecar先读后比较，strict加载和独立run/eval路径沿用阶段5协议。
5. 原时间exp投影AST与基线一致，未悄悄改动损失或调度器。

交付前复核还修正了 `model_dict.py` 中非法任务错误提示，使其反映当前已注册的六个标准任务；该修正不改变模型选择逻辑。修正后阶段6定向测试仍为 8/8 通过。

本阶段自审未发现新增实现缺陷。远端兼容性和真实轨迹效果仍未验证，属于运行证据缺口而非已发现的代码错误。

**本阶段结束，未执行下一阶段**
