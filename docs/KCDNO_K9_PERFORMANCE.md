# K9 完整成本与有限性能（2026-09-16）

## 实现与口径

复用 `tools/cdlno_benchmark.py`、`cdlno_perf/{models,costs,measure}.py`。旧默认模型列表与旧CDLNO/Transolver/matched临时类行为保持；新增选择 `kcdno_all`、`kcdno_off`、`lrsa_matched_trainable`（实际生产family=lrsa_matched）。新模型的F/front/CDPA/chunk字段不适用，不参与构造；计数不把核读取算作SA。

新 `kernel_costs.py` 用规格独立闭式公式核对live MAC。参数来自实际model/core；N规模Down/Up投影、两latent FFN、point FFN/dense3×3卷积、任务lift/time/head及全部SDPA QK/AV均计入。K2用F.linear调用Q/K而绕过nn.Linear hook，因此明确按真实Writer/Reader输入形状补计，writer每份写一次、Reader Q一次；分母收缩也计入MAC。1MAC=2矩阵FLOPs；bias、norm、ELU/clamp、mass sum、除法、depth dot/softmax/raw融合、gate与copy另列，不宣称完整标量FLOPs。depth dot的等效MAC单列，不计入规格主矩阵式。真实aten操作、临时stack/cat载荷及autograd storage另列，不能相加冒充峰值显存。

默认L8实际Down8/Up8/SA0/latentFFN16/PointModule8；all Writer7/Q7/28逻辑读取、7次Reader调用，分子/分母各7个batched einsum；off全部历史计数0；matched full SA8。没有改核公式、gamma/精度/decoder或数据协议来优化。

## 公式与参数

主项history_MAC=BMdr(L−1)(L+6)/2，removed_SA=BL(4Md²+2M²d)，matched_point=BL(8Nd²+15Md²+4NMd+2M²d)，conv另加9BLNd²。核分母另有BMrL(L−1)/2 MAC，已在live总数计入。

L8/d128/h8/M64/r16：主要参数差495616；实际差494329，差额−1287来自被删SA专属norm/bias/QK norm与新增key/query/depth norm、scorer、gamma：2Ld+2L(d/h)−4(L−1)d−(L−1)。实际point core：matched 3,099,392 / off 2,572,800 / all 2,605,063；conv core：matched 4,280,064 / off 3,753,472 / all 3,785,735。主项减少16.09%/11.63%，实际参数与主项近似严格区分。

Darcy N7225，Conv核心主要MAC从18,127,126,528降至18,091,606,016（0.195952%），加分母28,672后仍只约0.196%；不可将局部SA→history的84.69%下降写成整网提速。FP32持久摘要B(L−1)(rd+r)=14,448数=57,792字节≈0.0551MiB。训练还要保留原计算图与中间候选，不能以摘要大小推断训练显存。

## 实际验证和计时

15/15 CPU功能/成本测试通过15.760s（新2+原性能13）；旧完整MAC/核心相关边界与chunk相同权重检查继续通过。测试曾把0.196%的近似常量写得过细，修正测试舍入到文档实际3位小数后通过；公式和实现没有为常量改动。所有失败/成功日志保留。

本机Python3.13.9、Torch2.13+cu130、RTX5090 Laptop/driver591.86；非远端2.11/cu128。仅两个代表性合成配置，均L8/d128/h8/M64/r16/B1，point N972，conv N7225。Elasticity按task preset；Darcy按同尺寸结构比较、B减至1（原训练B4）。FP32、TF32关、AMP关、compile关、auto SDPA，实测为efficient attention；原Transolver使用原手工attention。每模型warmup3/测10次，CUDA同步perf_counter，包含主机launch；AdamW .001/wd1e-5/foreachFalse，优化器state已热身，合成FP32 MSE+backward+step。无数据、scheduler、NS rollout或真实epoch。

| 配置 | 模型 | 整网参数 | 整网GMAC | forward median/p90 ms | step median/p90 ms | forward/step峰值allocated MiB |
|---|---|---:|---:|---:|---:|---:|
| Elasticity/task | transolver | 976,833 | 1.126924 | 4.572/6.225 | 27.720/39.921 | 73.44/156.84 |
| Elasticity/task | lrsa_matched_trainable | 3,133,569 | 1.440710 | 9.682/11.497 | 63.221/76.887 | 80.83/181.37 |
| Elasticity/task | kcdno_off | 2,606,977 | 1.398767 | 10.200/16.959 | 48.471/57.897 | 78.34/172.53 |
| Elasticity/task | kcdno_all | 2,639,240 | 1.405218 | 13.814/17.984 | 75.640/82.364 | 78.55/178.79 |
| Darcy/matched B1 | transolver | 3,090,113 | 23.570814 | 20.657/20.978 | 56.574/58.353 | 372.14/890.87 |
| Darcy/matched B1 | lrsa_matched_trainable | 4,330,241 | 18.485024 | 14.796/15.065 | 82.047/84.131 | 121.88/729.27 |
| Darcy/matched B1 | kcdno_off | 3,803,649 | 18.443081 | 13.521/16.559 | 72.404/72.757 | 116.34/720.42 |
| Darcy/matched B1 | kcdno_all | 3,835,912 | 18.449532 | 15.287/15.935 | 85.640/86.878 | 116.56/726.69 |

仅为本机有限样本，不能外推其他GPU或真实epoch。首轮point测量与CPU测试进程重叠，保留为初始记录但不作表中计时证据；随后同一配置独占本任务计算复测（gpu-point-final.json），Darcy随后单独执行。未自动扩展全任务、AMP/compile或rank扫描。

新all未呈现预期提速：point all较matched的forward和训练步更慢。untimed profiler记录7个reader/7个writer，矩阵计算之外还有FP32 RMS/ELU/stack/除法/融合和大量小算子launch；point all训练诊断约5037次cudaLaunchKernel，CPU launch开销显著。保留完整逐算子和region数据，instrumented region耗时不与正式无hook中位数混用。Down/Up/point工作仍占大头，少算SA MAC不保证更快。

## 可运行命令与边界

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_performance test_performance -v
python -B tools/cdlno_benchmark.py --task elasticity --comparison task --models transolver lrsa_matched_trainable kcdno_off kcdno_all --chunks 0 --device cuda:0 --precision fp32 --backend auto --warmup 3 --iterations 10 --output /absolute/new/point.json
python -B tools/cdlno_benchmark.py --task darcy --comparison matched --B 1 --models transolver lrsa_matched_trainable kcdno_off kcdno_all --chunks 0 --device cuda:0 --precision fp32 --backend auto --warmup 3 --iterations 10 --output /absolute/new/darcy.json
# 无GPU，有限CPU成本核查：
python -B tools/cdlno_benchmark.py --task elasticity --comparison matched --N 35 --d 16 --h 4 --M 4 --kernel-rank 5 --models lrsa_matched_trainable kcdno_off kcdno_all --device cpu --audit-only --output /absolute/new/cpu.json
```

输出JSON拒绝覆盖。普通same-size比较可显式传共同d/h/M；task比较保留新旧各自主预设，不能用旧parser值覆盖。命令只作为交付示例，除本报告列出的有限合成工具外未执行真实训练。自审核对MAC全项、Q/K hook遗漏补计、逻辑来源vsAPI、参数/缓存/峰值口径、计时精度一致性及退化原因；继续已授权K10。
