# A4：前段三模式计数、有限性能验证与交付

日期：2026-09-15。用户已审查通过A3，本轮只执行A4。**full仍默认，三模式在八任务中的参数/接口/评估加载已完成；性能工具现已按实际模式计数并通过有限验证。没有新增模型结构或真实训练实验。** 另行可视化/续训V1公共组件保持，V2–V5没有推进。

## A. 完成范围与模块定义

唯一架构字段为`front_latent_mode ∈ {full,no_sa,identity}`，默认full，全部F个前段一致。S为完整Down输出；N1/Nsa/N2为各分支专属pre-norm：

```text
full:     A=S+FFN1(N1(S)); B=A+SA(Nsa(A)); T=B+FFN2(N2(B))
no_sa:    A=S+FFN1(N1(S)); T=A+FFN2(N2(A))
identity: T=S
```

S、A、B、T均为`[B,M,d]`。T在Up专属norm之前取得、保留梯度；三模式都继续执行完整Up、点残差和点FFN/ConvFFN。identity只表示latent processor恒等，**不是整个front block恒等**。实际公式实现仍是`cdlno/modules.py:LRSAFrontBlock.forward`，本轮未改该文件。

| 模块、norm与残差 | full | no_sa | identity |
|---|---|---|---|
| 点pre-norm、完整Down、queries及Q/K norm/K/V/O | 保留 | 保留 | 保留 |
| FFN1及N1、整个残差分支 | 保留 | 保留 | 不注册/不计算 |
| SA、Nsa、内部Q/K norm/Q/K/V/O、整个残差分支 | 保留 | 不注册/不计算 | 不注册/不计算 |
| FFN2及N2、整个残差分支 | 保留 | 保留 | 不注册/不计算 |
| 历史T | B+FFN2(N2(B)) | A+FFN2(N2(A)) | 原Down输出S本身 |
| Up专属norm、完整Up、点残差/点FFN及norm | 保留 | 保留 | 保留 |
| bridge、P个后段SA+GEGLU、CDPA、最终HF feature readout | 保留 | 保留 | 保留 |

L仍表示F+P个处理block，P=L−F；不再把L解释成所有模式的SA数量。F0没有front参数，三模式同权重计算等价；架构字段依然严格比较。模型无前段CDPA、无后段Kimi AttnRes、无新增缓存或替代机制。

## B. 修改文件与diff审查

起点分支main/HEAD `769fa333742f73c132868cf560bce5ec21529362`。284份真实工作区文件保存在`/home/hwz/CDLNO-artifacts/front-a4-before-ftvcu781/source`；[start.json](front_ablation_audit/a4/start.json)记录起点hash及未提交修改。原Transolver历史基线仍为`75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。本轮保护已接受A1–A3及独立V1成果。

| 文件 | A4实际修改 |
|---|---|
| `tools/cdlno_perf/models.py` | Case增加默认full字段/合法性；仅向CDLNO wrapper传模式；Transolver对照不接收新kwargs，matched LRSA实际config/每层显式锁full |
| `tools/cdlno_benchmark.py` | 复用既有CLI，增加`--front-latent-mode`和下划线别名；记录有效模式及full block数，默认命令不变 |
| `tools/cdlno_perf/costs.py` | 修正旧“每进入front就计一次SA”的错误统计；hook实际Down/Up/SA/FFN调用；补前/后段分项、实际参数分项和逐投影/卷积MAC账 |
| `tests/test_performance.py` | 增加2项测试，扩展既有边界/CLI检查；核对三模式×CDPA×chunk、完整成本公式、移除参数、matched锁full、F0/扩展L |
| README、PERFORMANCE_TOOLS、FRONT_ABLATION、八任务命令页、总报告/需求矩阵/阶段4与9/参考审查 | 增量交付模式、计数、兼容、命令、归因和实际证据；把历史“SA=L”限定为full，不重写原论文/原报告结果 |
| STATUS、AGENTS、memory、`front_ablation_audit/a4/` | 当前阶段状态、冻结/累计审查、命令、实际原始结果与可审查patch |

**没有修改`cdlno/`生产代码、任务entry/wrapper/配置预设、训练/评估脚本、依赖或计时实现`measure.py`。** [本轮增量patch](front_ablation_audit/a4/a4-changes.patch)覆盖A4代码/文档；[A1–A4累计运行代码patch](front_ablation_audit/a4/front-ablation-runtime.patch)覆盖相对真正pre-A1的25份已有运行文件，方便审查模式/必要接入/兼容/性能增量。该累计patch刻意区分另行V1新增的3个公共组件：它们不属于前段消融，本轮均字节不变。不能把工作区总git diff都算作A4。

## C. 计数、实际参数与完整运算账

### C1. 实际调用计数

成本审查用`_SelfAttention`、`PlainFFN`、`GEGLUFFN`、`_DownAttention`、`_UpAttention`的真实forward hook；CDPA历史Cross另计。所有hook仅在工具的未计时审查期间注册并移除，不进入生产模型或正式计时。

| 项目 | full | no_sa | identity |
|---|---:|---:|---:|
| 前段SA | F | 0 | 0 |
| 前段latent FFN | 2F | 2F | 0 |
| 后段SA | P | P | P |
| 后段GEGLU | P | P | P |
| Down+Bridge | F+1 | F+1 | F+1 |
| Up+FinalReadout | F+1 | F+1 | F+1 |
| 规则点ConvFFN | F+1 | F+1 | F+1 |
| 总latent SA（L8/F2/P6） | 8 | 6 | 6 |
| 前段latent FFN（L8/F2） | 4 | 4 | 0 |

默认entry逻辑来源总F=2，chunk0历史SDPA=1；every逻辑总`PF+P(P−1)/2=27`，chunk0历史SDPA=6。三种前段模式历史位置/份数相同，内容按各自真实T改变。每位置s份来源：chunk0一次，chunkk共ceil(s/k)次；各位置统一源softmax保持。

| L8/F2，chunk0 | full总SDPA | no_sa/identity总SDPA | 历史来源 / 历史SDPA |
|---|---:|---:|---:|
| off | 14 | 12 | 0 / 0 |
| entry | 15 | 13 | 2 / 1 |
| every_block | 20 | 18 | 27 / 6 |

33行CPU成本/梯度结果含三front模式、原Transolver、matched LRSA、CDLNO三CDPA模式及chunk0/1/2；同模式不同chunk严格恢复同份权重，输出/全部参与参数梯度按原FP32容差atol2e-5/rtol5e-4比较。chunk改变不减少来源或MAC。另有单元测试覆盖chunk99，以及L1/F0、L8/F0、L12/F2、L16/F6、L10/F8；原核心21格F0…6×CDPA及扩展边界继续回归。

### C2. 实际正式预设参数量

以下通过八个真实wrapper的正式d/h/M/L/F/ratio构造，计算`sum(p.numel() for p in model.parameters())`。没有用局部删除比例代替整网计数；全部为实际可训练参数，不含buffer。统一entry/F2/L8/ratio2，**非小张量测试预设**；仅构造模型，不训练。数据见[summary.json](front_ablation_audit/a4/summary.json)。

| 任务 | d/h/M | full | no_sa | identity |
|---|---|---:|---:|---:|
| Darcy | 128/8/64 | 2,531,297 | 2,399,649 | 2,135,457 |
| Elasticity | 128/8/64 | 2,072,545 | 1,940,897 | 1,676,705 |
| Airfoil | 128/4/64 | 2,515,521 | 2,383,809 | 2,119,617 |
| Pipe | 128/4/32 | 2,503,233 | 2,371,521 | 2,107,329 |
| NS | 256/8/64 | 10,015,169 | 9,489,729 | 8,437,057 |
| Plasticity | 128/8/64 | 2,548,836 | 2,417,188 | 2,152,996 |
| ShapeNet-Car | 256/8/64 | 8,211,652 | 7,686,212 | 6,633,540 |
| AirfRANS | 256/8/64 | 8,244,420 | 7,718,980 | 6,666,308 |

独立核对公式（d_h=d/h，r为整数FFN ratio，本次正式r2）：每前段删除SA+Nsa减少`4d²+2d+2d_h`；再删除两次FFN+norm减少`4rd²+2(r+2)d`。整网乘F，F0为0。实际参数差额与公式相符；各移除参数确实不存在，不把闲置参数标成“不参与成本”。不同模式初始化消耗随机数不同，不承诺相同seed使所有保留权重相等；数值等价测试显式复制保留权重，性能对照只固定配置和初始化方法。

### C3. 全模型成本口径

沿用1 MAC=一次乘加，矩阵FLOPs=2×MAC；`matrix_macs`只计稠密矩阵/卷积乘加，**不是所有标量FLOPs**。bias、norm/QK norm、GELU/GEGLU、softmax、depth FP32归约/累加及拷贝/stack均有单独ATen及payload记录，不能当0。SDPA QK/AV由实际Q/K/V形状补计，不依赖profiler的FLOPs支持。

令q=L（full）或P（no_sa/identity），S为跨所有位置的历史来源总数，A为活跃CDPA位置数：

```text
CDLNO QK+AV全部矩阵MAC = 2Bd[2(F+1)NM + (q+S)M²]
每个front Down+Up投影 = Bd²(4N+3M)
每个front SA投影（仅full） = 4BMd²
每个保留的front latent FFN = 2rBMd²
Bridge+最终Up投影 = Bd²(4N+4M)
后段SA投影 = 4PBMd²
后段GEGLU = 3r_rear PBMd²
点FFN线性 = 2r(F+1)BNd²
规则dense3×3卷积 = 9(F+1)BNd²
CDPA Q/K/V/O投影 = BMd²(A+3S)
再加实际任务stem、时间投影、最终head的矩阵成本
```

这里N规模Down K/V、Up Q/O均保留；消融不删除点FFN/卷积、bridge/rear/readout的成本。matched LRSA保持L个完整full block、stem和LN/head，最后一层已Up，不加bridge/CDPA/final-up。它不是LRSA论文复现训练。

独立闭式全成本测试先核对full，再核对删去的完整SA投影/QK/AV及FFN线性差额，并与实际每层账合计互证。参数、norm、bias、临时stack、历史/点特征与autograd保存storage也保留。payload累计有别名和生命周期，不能相加冒充峰值；实际峰值另测CUDA allocated。

## D. 实际有限性能、命令和验证结果

本机Python3.13.9、torch2.13.0+cu130、CUDA13.0、PyG2.3.1、cuDNN92000，RTX5090 Laptop GPU/driver591.86。**不是用户远端Python3.10/torch2.11/cu128验收**；未安装或替换任何依赖。

各模式相同输入/目标/配置，FP32参数及输入、math SDPA、TF32关、compile关、cuDNN benchmark关、CPU线程1、seed20260914。每行5次warmup、20次测量，逐次CUDA同步，报告median/p90；实际SDPA探针确认math路径。forward为eval/no_grad；训练步为合成FP32 MSE+backward+AdamW(lr.001,weight_decay1e-5,foreach=False)，不是任务loss、时间循环或scheduler。训练测量前state已初始化，每个活跃参数25次成功更新；额外未计时诊断1次更新在测量之后。详见每行原始samples、state bytes与profiler记录。

### D1. 原任务配置：Elasticity

N972/B1/d128/h8/M64/L8，CDLNO F2/P6，point FFN。`--comparison task`读取原/新各自任务配置；本任务恰好d/h/M相同。所有输入为合成点和标签，**不是实际Elasticity样本**。

| 模型/模式 | 实际参数 | 完整矩阵GMAC | forward median/p90 ms | 合成step median/p90 ms | forward/step峰值MiB |
|---|---:|---:|---:|---:|---:|
| 原Transolver | 976,833 | 1.126924 | 4.870 / 5.278 | 26.795 / 28.964 | 73.439 / 156.843 |
| matched LRSA，full | 3,133,569 | 1.440710 | 13.774 / 15.054 | 66.820 / 69.707 | 82.795 / 213.611 |
| CDLNO off，full | 2,006,113 | .617185 | 7.468 / 8.721 | 36.636 / 40.313 | 78.480 / 134.838 |
| CDLNO entry，full | 2,072,545 | .626622 | 8.117 / 11.432 | 37.595 / 41.798 | 78.765 / 136.571 |
| CDLNO every，full | 2,404,705 | .736722 | 10.735 / 11.751 | 50.745 / 54.066 | 80.032 / 151.344 |
| CDLNO entry，no_sa | 1,940,897 | .616136 | 6.557 / 7.971 | 36.774 / 42.678 | 78.261 / 134.113 |
| CDLNO entry，identity | 1,676,705 | .599359 | 6.118 / 8.130 | 29.464 / 33.006 | 77.222 / 130.213 |

原始文件为`gpu-elasticity-task-{full,no_sa,identity}.json`，均在[证据目录](front_ablation_audit/a4/)中。

### D2. 同配置结构比较：缩小Airfoil网格

明确使用合成17×23、N391/B2、d128/h4/M64/L8/F2/P6、entry/chunk0及相同dense ConvFFN。**不是原Airfoil221×51/B4任务规模**；不改变正式任务JSON。

| 前段模式 | 实际参数 | 完整矩阵GMAC | forward median/p90 ms | 合成step median/p90 ms | forward/step峰值MiB |
|---|---:|---:|---:|---:|---:|
| full | 2,515,521 | .989209 | 7.646 / 9.498 | 39.859 / 47.590 | 77.999 / 140.643 |
| no_sa | 2,383,809 | .968238 | 6.656 / 8.956 | 34.265 / 39.620 | 77.495 / 137.498 |
| identity | 2,119,617 | .934683 | 6.613 / 8.188 | 31.071 / 32.967 | 76.425 / 132.722 |

原始文件为`gpu-airfoil-matched-{full,no_sa,identity}.json`。这些有限样本有可见p90波动，没有多次独立运行/置信区间；不得承诺普遍加速比例。删除前段子层并未移除N规模处理，MAC降幅有限。

值得保留的结果：Elasticity中entry full甚至identity虽MAC低于Transolver，实测仍较慢。未计时训练诊断ATen调用：Transolver7234、entry full8718/no_sa7818/identity7134；本工具显式归一化、多投影、小算子及逐tensor AdamW发射会使低MAC不等于低延迟。已有profiler用于定位这些来源，但事件数不是严格因果分解，也不能把原Transolver手写attention和CDLNO SDPA视为相同kernel。未为追求速度改变CDPA、共享参数、detach历史、缓存K/V或启用不同精度。

### D3. 实际执行命令

以下均已运行，未导入exp/main/loader或执行真实数据训练：

```bash
python -B -m unittest discover -s tests -p test_performance.py -v
python -B docs/front_ablation_audit/a4/run_performance.py
python -B docs/front_ablation_audit/a4/summarize.py
python -B docs/front_ablation_audit/a4/verify_commands.py
CDLNO_LRSA_ROOT=/home/hwz/LRSA-Operator python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B docs/front_ablation_audit/a4/audit_scope.py
git diff --check
```

修改前还实际执行同小配置full成本CLI，保存`full-before.json`。修改后所有五种原性能模型的初始化权重hash、参数、完整MAC、历史数、原有调用计数与修改前完全一致。计数新增分项不影响full旧值。

| 验证 | 实际结果 |
|---|---|
| 性能定向 | 13/13通过，14.496s；[日志](front_ablation_audit/a4/performance-targeted.txt) |
| 完整回归 | **203/203通过，126.687s；0失败/错误/跳过**；[日志](front_ablation_audit/a4/regression-final.txt) |
| CPU成本/同权重chunk | 33行全部通过；三front模式×既有5模型/适用chunk，完整矩阵账及梯度 |
| GPU有限性能 | 上述10行全部完成；无GPU大网格/超参扫描 |
| 原full性能基准 | 五种模型前后初始化hash/原计数/参数/MAC保持，未重生成所谓“旧基准” |
| 八任务正式参数 | 24个实际wrapper构造并统计；全部requires_grad参数计入 |
| 训练/评价命令 | 24组顺序dry-run、48次真实parser通过；实际脚本/模式/run/预设一致，[命令记录](front_ablation_audit/a4/commands.json) |
| 加载/旧full回归 | 复用实际八任务三模式strict state/整对象/list、sidecar不覆盖、显式冲突与真实pre-A1旧对象/16份数值fixture、三原cwd新进程检查，均通过；未新增跨模式转换 |
| 冻结/语法 | 128份本轮生产/任务/脚本等冻结文件字节一致；40份shell语法通过；新增修改Python3.10 AST语法通过（非3.10运行证明） |

本阶段没有失败的验收测试或失败的性能行。缺少外部LRSA可选xformers/liger打印的是其fallback提示，实际参考测试通过，未安装这些框架。203项包含独立V1已有测试；它们通过不意味着本阶段实现了八任务续训。

### D4. 远端可运行的有限无数据命令

在已有远端环境/源码中执行，无需安装新框架；输出文件必须不存在。以下未在远端执行：

```bash
cd /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/CDLNO-w
# CPU完整成本/非方格/来源分块，一次选择一个front模式
python -B tools/cdlno_benchmark.py --task airfoil --grid 5 7 \
  --B 2 --d 16 --h 4 --M 4 --front-latent-mode identity \
  --chunks 0 1 2 --audit-only --output /tmp/a4-identity-cost.json

# 仅固定entry的三种前段模式；同规模、精度、backend
for mode in full no_sa identity; do
  python -B tools/cdlno_benchmark.py --task elasticity --comparison matched \
    --models cdlno_entry --front-latent-mode "$mode" --chunks 0 \
    --device cuda:0 --precision fp32 --backend math --warmup 5 --iterations 20 \
    --output "/tmp/a4-elasticity-${mode}.json"
done
```

已有benchmark旧命令不传模式仍full；原Transolver脚本仍保留，例如六标准任务的`scripts/Transolver_Darcy.sh`、工业两个`scripts/Transolver.sh`，从各自原工作目录运行，参数与原数据路径语义不变。没有为了本轮新增LRSA真实训练系统。

## E. 八任务命令、checkpoint兼容与冻结证据

完整三模式的八任务训练、评估、顺序命令、默认配置及目录规则见[命令页](CDLNO_FRONT_ABLATION_A2_COMMANDS.md)。最常用形式如下：在仓库根设置`mode=full`、`no_sa`或`identity`，选择一行执行；使用未占用的`CDLNO_RUN_TAG`，不要自动循环启动全部任务：

```bash
mode=no_sa
export CDLNO_RUN_TAG=front_trial1
bash tran_evaluate/train_eval.sh darcy      --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh elasticity --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh airfoil    --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh pipe       --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh ns         --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh plasticity --front-latent-mode "$mode" --gpu 0
bash tran_evaluate/train_eval.sh car        --front-latent-mode "$mode" --gpu 0
CUDA_VISIBLE_DEVICES=0 bash tran_evaluate/train_eval.sh airfrans --front-latent-mode "$mode"
```

上述整块含八条独立任务命令，复制时只选择所需行；本轮只执行了带`--dry-run`的预览。训练失败不启动eval。单独训练/评价用同名`TASK.sh train|eval`；余参最后覆盖。正式默认仍F2/L8/P6、entry/chunk0，Pipe M32其余M64，PDE500/Car200/Air398、batch/lr/optimizer/scheduler不改。Car一次run一个fold、完整drag原路径/fold0限制保留；Air训练`--my_path`指Dataset、评价指父目录。

| 任务 | full/no_sa/identity train/eval | checkpoint形式 | A4实际证据 |
|---|---|---|---|
| Darcy/Elasticity/Airfoil/Pipe | 全部已接入 | strict state_dict `model.pt` | parser预览、现有合成训练/损失/加载回归 |
| NS/Plasticity | 全部已接入 | strict state_dict `model.pt` | 现有10→10窗口/20次时间更新及加载回归；性能不是时间rollout |
| ShapeNet-Car | 全部已接入 | 稳定模块整对象 `model_<epochs>.pth` | 真实PyG/现有mask损失、原cwd新进程加载 |
| AirfRANS | 全部已接入 | 每成员整对象+根模型list | 真实PyG/现有工业loss/原cwd对象及list加载；正式weighted覆盖边界见F |

eval先读已有sidecar：省略front模式只恢复这个字段；显式不匹配先报架构冲突，不能用CLI默认full覆盖no_sa。其它自定义d/M/F/L/fold/task/epochs等继续按既有协议提供。旧配置只有被识别为完整历史CDLNO full、且仅缺mode才内存兼容；不随意补其它字段。旧整对象没有新属性时运行历史full路径，保留类路径/可信局部加载；不扩大torch.load权限。无论F0是否参数相同，显式模式冲突仍拒绝。

**两个消融默认从头训练；不承诺旧full权重直接变成消融权重，不实现跨模式迁移。** 同模式strict往返、chunk改变可载入，错误键/shape/模式拒绝。sidecar不改写；隐式消融目录含模式，full旧路径不变；显式run/save_name保留已有拒绝覆盖防护。

[冻结及累计审计](front_ablation_audit/a4/freeze.json)进一步重查pre-A1：除LRSAFrontBlock外的原共享数学类/函数AST相同；CDPA字节相同；core.forward及bridge/latent_blocks/readout/cdpa_at构造AST相同；三个wrapper仅去除新增mode参数/转发后AST相同；八份JSON只新增full字段，训练字段相同；原12个exp/main/train、数据/normalizer/采样/指标/原脚本/YAML/依赖保持。没有改变CDPA/rear/decoder/data来制造性能收益。

## F. 归因边界、未验证项与自审结论

1. 固定CDPA比较full/no_sa/identity检验前段SA与FFN是否必要；不能单独证明CDPA替代了它们。已有off可用于用户后续同模式控制，但本轮没有开启额外真实训练矩阵。后段P次SA始终存在，不能称“全模型无层内注意力”。
2. 当前GPU实测仅两个代表性合成配置/10行，统一FP32/math；没有完整八任务新模式GPU性能矩阵、Flash/AMP/compile/TF32不同精度性能对照或多机重复。GPU chunk0/1/2性能矩阵本轮没有重跑，CPU的同权重成本/梯度等价已运行；沿用已有其它精度测试并不等于新性能验收。
3. 本轮不运行真实数据读取/训练/轨迹/工业采样评价，未验证数据完整性、收敛、准确率或真实epoch效率。不能从参数量/MAC比例或forward推导训练加速。现有Car日志压力/速度名字交换及drag固定路径/fold限制、Air旧MAE条件问题不属于本次回归，未顺手修复。
4. Air实际main选择MSE_weighted；A3三模式测试的函数默认MSE覆盖不能冒充完整三模式正式weighted入口链路验收。该历史表述已在V1计划纠正，本次保留真实原loss代码及既有weighted测试，不因A4成本验证宣称这个新链路边界已补齐。
5. 另行V1只提供完整归档/恢复/绘图公共组件；八任务新增周期保存/可视化/`--resume`尚未接入，本轮没有推进它们。现有纯权重/整模型评估兼容不能当作完整断点续训。

交付前已自行审查：实际子层计数与闭式完整成本相符；matched full锁定且旧full数值/成本基准未变；mode×chunk权重与梯度及source账保持；累计模型/数据协议AST与hash冻结；命令/sidecar/旧对象兼容回归通过。**未发现本轮遗留实现缺陷需要裁定**。用户审查重点为上述有限实测适用范围和归因限制，以及是否接受A1–A4消融交付；没有未经自审的技术问题留给用户。

**本补充阶段结束，未执行下一阶段。**
