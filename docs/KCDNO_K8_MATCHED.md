# K8 可训练 matched LRSA-full（2026-09-16）

旧 `tools/cdlno_perf/models.py:LRSAMatched` 确实是L个完整block且只有一次最终head，但此前只在性能工具里，未注册生产训练。现在新 `cdlno/kcdno/matched.py:MatchedLRSA` 直接组合原LRSAFrontBlock，固定front_latent_mode=full，返回点特征，无Bridge/persistent/额外Up。新独立family=lrsa_matched、MatchedLRSAConfig/MatchedMetadata记录full结构，不包含闲置kernel/history字段。旧性能类和pickle路径保持。

新家族选择分支接受kcdno或lrsa_matched，共享K4–K7的任务lift/时间/reference/head、loss/训练/加载；共同尺寸由profile解析，显式旧front/CDPA参数与matched的kernel/history参数均拒绝。KCDNO off仍是同core关闭history，保留2L latent FFN，无需第三套core。三种模型正常各自从头训练，strict checkpoint只接受同family/结构；权重白名单复制仅在数学测试。

| 计算图 | Down/Up | 两latent FFN | SA | 历史 |
|---|---|---|---|---|
| matched full | 各L | 2L | L | 无 |
| KCDNO off | 各L | 2L | 0 | 无参数/计算 |
| KCDNO all | 各L | 2L | 0 | L−1写/Q，L(L−1)/2读取 |

八任务训练/评价/配置/strict保存加载均接入，共用原wrapper类但由完整架构family选择独立core。详见 [八任务命令](KCDNO_COMMANDS.md)。新增顺序薄脚本 `tran_evaluate/kcdno/train_eval.sh TASK all|off|lrsa_matched`；只在用户调用时执行训练，并在成功后评价同run。原Transolver/CDLNO入口与默认保留。

```bash
PYTHONPATH=tests:. python -B -m unittest test_kcdno_matched test_kcdno_tasks test_kcdno_temporal test_kcdno_car test_kcdno_airfrans -v
bash tran_evaluate/kcdno/train_eval.sh darcy lrsa_matched --gpu 0 --dry-run
```

18/18通过19.517s：新4检查涵盖point/5×7conv，full SA输出置零后与显式复制公共权重的KCDNO off完全一致；模块/storage独立；六标准+两工业matched合成优化器步/strict往返/family冲突，工业两原cwd新进程整对象/列表加载。K4–K7 all/off原loss/时间窗口/PyG回归继续通过。新matched步的诊断loss明确不是新的原loss链路证明；复用之前相同wrapper/原循环的证据。

自审发现并修复K7新增Air parser实际argv=None路径未转sys.argv（此前测试给了显式argv），现在按Car相同方式处理，有生产调用方式的针对性测试；不改原数据或结构。初次测试误把SA输出投影路径写为latent_sa.to_out，实际为latent_sa.attn.to_out；已修正测试，未改公共模块。失败日志保留。

参数与完整MAC/计时在K9用实际模型核查，本阶段不以SA删除宣称提速。固定公共训练协议与尺寸只支持结构对照，不是LRSA论文复现。未运行真实数据训练/指标、远端环境。K8自审完成，继续K9。
