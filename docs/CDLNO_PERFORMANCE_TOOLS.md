# CDLNO 无数据性能工具

A4补充：`--front-latent-mode full|no_sa|identity`（下划线别名也可）只控制CDLNO前段，默认full。`lrsa_matched`显式固定完整full；原Transolver不接收新kwargs。实际子模块hook分别报告`front_sa/front_latent_ffn/rear_sa/rear_ffn`，不把进入一个前段block当成必然执行SA。A4实际结果与范围见[交付报告](CDLNO_FRONT_ABLATION_A4.md)。

阶段9工具只生成合成输入，计时一次模型调用或一次合成 MSE/AdamW 更新。它不导入 exp/main、loader，不构造数据图，不替代真实任务训练/指标；NS 的10次调用和 Plasticity 的20次更新不在单步计时中。基础模型出处为 [Transolver, ICML 2024](https://arxiv.org/abs/2402.02366)。LRSA/IPOT 源码差异及许可沿用 [参考审查](CDLNO_REFERENCE_AUDIT.md)。

入口：[tools/cdlno_benchmark.py](../tools/cdlno_benchmark.py)。无需安装新依赖；在已有可导入 torch、numpy、einops、timm 的环境运行；工业合成输入使用真实 PyG Data，缺失 PyG 明确失败，不伪造依赖。脚本自动定位仓库根，不需要导入任何训练入口。

```bash
# CPU功能/成本核对：一个小的5×7结构网格，默认五种模型和chunk0/1/2。
python -B tools/cdlno_benchmark.py --task airfoil --grid 5 7 \
  --B 2 --d 16 --h 4 --M 4 --audit-only --output /tmp/cdlno-cost.json

# 有GPU的远端代表性检查：Elasticity原N972，同配置结构对照。
python -B tools/cdlno_benchmark.py --task elasticity --comparison matched \
  --device cuda:0 --warmup 5 --iterations 20 --output /tmp/cdlno-elasticity.json

# 原任务配置比较：Airfoil原221×51/B4，原Transolver h8，CDLNO/matched LRSA h4。
# 这个命令显式选择一个任务，不会自动展开八任务矩阵；会比小配置消耗更多资源。
python -B tools/cdlno_benchmark.py --task airfoil --comparison task \
  --device cuda:0 --chunks 0 --warmup 5 --iterations 20 --output /tmp/cdlno-airfoil-task.json

# 同配置结构比较，小型非方形Airfoil索引网格；这是合成缩小网格，不是原任务规模。
python -B tools/cdlno_benchmark.py --task airfoil --comparison matched --grid 17 23 \
  --B 2 --device cuda:0 --warmup 5 --iterations 20 --output /tmp/cdlno-airfoil-matched.json

# 固定CDPA entry、同配置/精度的三前段模式；只执行有限合成性能检查。
# 输出须不存在；不读取数据、不训练实际数据集。
for mode in full no_sa identity; do
  python -B tools/cdlno_benchmark.py --task elasticity --comparison matched \
    --models cdlno_entry --front-latent-mode "$mode" --chunks 0 \
    --device cuda:0 --precision fp32 --backend math --warmup 5 --iterations 20 \
    --output "/tmp/cdlno-front-${mode}.json"
done
```

`--task` 支持八任务；工业保持B1，NS/Plasticity保持原空间网格。`--comparison task` 采用原启动脚本/工业main构造与新JSON的各自架构，拒绝 d/h/M/L/F/ratio 覆盖。B/N/grid 如被显式缩小，JSON会标记不是原任务几何/batch；不能将其称为原任务规模实测。原任务标准脚本实际L8，不能拿旧parser默认L3当作论文主配置。

`--comparison matched` 使用新任务d/h/M/L/ratio作为共同配置，可显式覆盖；实际执行结构另在 `effective_structure` 中记录。Transolver 的 slice/deslice、每头共享小投影和前置卷积仍按原模型运行，不会变成LRSA；相同超参数也不意味着相同参数量或归一化。`lrsa_matched` 与CDLNO使用同一个任务wrapper的stem、位置/时间条件、placeholder、输出LN/head及相同point FFN/ConvFFN；连续运行L个现有完整LRSA block，末层后只接LN/head。没有bridge、CDPA、额外final-up，也没有放宽CDLNO的P≥1配置检查。这个仅用于benchmark的组合没有注册到训练器，不是LRSA论文复现实验。

每次新构造固定seed；chunk比较会从同一份CPU state_dict严格恢复，报告SHA256。不同chunk在计时前比较输出及所有参与参数梯度，FP32容差atol=2e-5/rtol=5e-4；AMP另用atol=4e-3/rtol=3e-2并报告实际最大绝对差，不能声称AMP与FP32同精度。各chunk优化器均新建，起始权重和输入一致。计时中的优化器更新属于合成工作量，不保存训练模型。

## MAC、标量运算及内存口径

`matrix_macs`只计forward稠密Linear、Conv、QK/AV及原Transolver的slice/deslice乘法；一个乘加记1 MAC，`matrix_flops_2_per_mac=2*MAC`。Conv按包含padding位置的dense核计算，不把边缘零乘法当作结构稀疏。参数量为实际注册/可训练参数，不根据论文估算；原模型没有梯度的参数单列。

完整矩阵账包括stem/head、down的N规模K/V、up的N规模Q/O、实际保留的前段latent FFN、点FFN/dense3×3卷积、后段GEGLU，以及每个CDPA位置的一次Q和每份来源各自K/V/O。基于实际张量形状统计SDPA的QK+AV，通用profiler未支持SDPA时也不会遗漏该项。通用profiler的数字专门命名为 `profiler_partial_flops`，不能当作总FLOPs。`linear_conv_macs_by_module`给出实际每个投影/卷积的账目；参数总量和`parameters.by_component`从实际注册对象统计，移除分支没有参数或成本。

设单次配置B/N/d/M/L/F，P=L−F，各位置历史数为s，S为所有s之和，A为活跃位置数；q为latent SA数（full时L，其余P），f_ff为前段FFN个数（full/no_sa时2F，identity时0）。独立公式测试核对：

```text
LRSA attention MAC = 2Bd(2LNM + LM²)
CDLNO attention MAC = 2Bd(2(F+1)NM + (q+S)M²)
CDPA projection MAC = BMd²(A+3S)
front保留的latent FFN总MAC = f_ff × 2rBMd²
rear GEGLU / block = 3rBMd²
point FFN线性 / 次 = 2rBNd²
dense 3×3 Conv / 次 = 9BNd²
```

前段Down+Up投影每块为`Bd²(4N+3M)`；其SA投影仅full时加`4BMd²`。Bridge+最终Up为`Bd²(4N+4M)`；后段SA投影合计`4PBMd²`。再加上述前段FFN、后段GEGLU、F+1次点处理、stem/head和CDPA即为整网矩阵账（实际stem维度与输出通道按任务）。前段FFN与点FFN使用前段ratio，后段GEGLU使用后段ratio；工具的共同`--ratio`同时设二者，与当前正式ratio2一致。

默认2+6的SA总数为8/6/6，前段FFN为4/4/0，后段SA/GEGLU均6次；Down/Bridge=3、Up/FinalReadout=3、规则ConvFFN=3。entry仍2逻辑来源/chunk0一次历史SDPA，every仍27/6；CDPA Cross不计作前段SA。固定CDPA比较三模式只能检验前段SA/FFN必要性，不能单独证明CDPA替代这些子层；后段SA始终存在。阶段9旧结果与旧`q=L`表仅适用于full。

矩阵MAC**不是完整标量FLOPs**。bias加法、LN/RMS/QK norm的元素数/宽度/dtype、GELU/SiLU、乘除、归约、softmax等另有实际ATen操作清单；源内softmax即使融合在SDPA中也有由Q/K形状恢复的逻辑元素数。depth单独列出FP32的RMS、评分、source softmax和RAW融合标量工作量。这些操作的硬件指令数/exp/rsqrt代价依backend而变，不编造一个“精确全算子FLOPs”。输入reference距离和time embedding运行时工作也进入操作清单；构造时预计算固定位置buffer的成本不属于forward。

`temporary_materializations`记录实际stack/cat/clone/repeat/类型转换输出payload，包括history stack、可能物化的Q expand/reshape副本和FP32 depth中间量；不是零开销。depth同时列出候选raw、norm keys、weighted values、scores/weights的理论payload。`storage`列H_F、当前latent、最大历史列表payload及autograd保存的激活 backing-storage 去重总量。history只是引用，不额外clone；保存同一storage的多个view不会重复计入unique项。

这些payload/累计保存量有别名、生命期及allocator复用，**不能相加冒充峰值显存**。实测峰值另取CUDA `max_memory_allocated`；不含allocator reserved/驱动外部内存。CPU峰值标null。显存已经计入生产公式的全部临时张量和训练图，没有detach、共享跨层参数、K/V缓存或省略来源。

## 计时及精度

每个样本用 `cuda.synchronize → perf_counter → 工作 → cuda.synchronize`，包括Python/host发射及wrapper开销。默认5次warmup、20次正式采样，报告中位数、线性插值p90和全部原始样本。没有用forward速度换算epoch耗时。

forward使用eval/no_grad；开始时没有参数梯度和优化器state。合成训练步包括zero_grad(set_to_none)、一次forward、FP32 MSE、backward、AdamW.step；AdamW lr=.001、weight_decay=1e-5、foreach=False，所有模型统一使用。warmup至少一次，测量前state已初始化；记录state实际bytes、活跃参数tensor数和成功更新次数，AMP溢出导致跳过step则失败。训练峰值包含已初始化state和梯度；CPU step标量的state不冒充CUDA显存。

untimed forward/backward/optimizer profiler仅用于定位来源，独立于正式计时。训练诊断额外做一次已注明的优化器step，发生在所有计时/峰值采集后；其事件耗时有instrumentation，不能替换正式延迟。forward区域记录与ATen/kernel分开，不把标注区间和子kernel相加。

默认所有模型FP32、TF32关闭、cuDNN benchmark关闭、compile关闭、一个CPU线程。`--precision amp-fp16|amp-bf16`、`--tf32`、`--backend auto|math|flash|efficient|cudnn`、`--compile`都是整次比较共享设置，不允许仅给CDLNO开启。原Transolver使用手写attention，所以其SDPA调用数为0，attention矩阵成本仍明确非0。auto选择的实际SDPA算子由未计时profiler报告；不把auto误称为flash。JSON记录torch/CUDA/PyG/driver、设备、dtype、TF32、reduction设置、warmup/采样和代码指纹。

可选 `--compile` 只编译模型，loss/optimizer仍eager；先计编译首调用，再warmup，编译首forward/backward耗时与稳态分开，编译输出/梯度与eager核对，额外记录编译路径的backend探针。编译后ATen可能融合，原始数学MAC仍来自eager结构审查；不能把该清单视为编译后逐kernel指令计数。未实际运行的compile/精度/backend组合必须保留未验证标记，不能由CLI支持推断已通过。

输出文件必须不存在，已有报告不覆盖。单配置失败会保留失败原因，进程以非0退出；不得只挑选成功行宣称整个配置通过。远端Python3.10/3.11、torch2.11/cu128仍须在目标机器运行上述命令；本地GPU不替代远端验证。

## KCDNO独立扩展（2026-09-16）

现有工具可显式选择 `kcdno_all kcdno_off lrsa_matched_trainable`，旧默认列表保持。新模型完整成本与有限GPU证据见 [K9报告](KCDNO_K9_PERFORMANCE.md)，`--kernel-rank`仅适用核模型；lrsa_matched_trainable为生产对照，旧lrsa_matched临时性能类仍保留。
