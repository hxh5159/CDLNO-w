# CDLNO 来源、归属与许可核对

最终审查：2026-09-14。以下为实际快照事实，不按论文名推断代码或许可。

| 来源 | 固定版本与链接 | 实际归属/许可 | 本工程使用范围 |
|---|---|---|---|
| thuml/Transolver | [75e0f67643806a81cd1d3f6adc88dd8c02416fe7](https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7) | 根 [LICENSE](../LICENSE)：MIT，Copyright 2024 THUML @ Tsinghua University | 当前基仓库，保留原模型、数据/训练/指标、输入提升语义与原README引用 |
| Adversarr/LRSA-Operator | [47b03f8c8c8da30bbcc0737b008dc4548f9cb98e](https://github.com/Adversarr/LRSA-Operator/tree/47b03f8c8c8da30bbcc0737b008dc4548f9cb98e) | 作者信息 Zherui Yang/Adversarr；审查快照未找到LICENSE/COPYING，pyproject无license声明；不推断MIT | 按确认公式重新实现完整block/ConvFFN；未vendor上游源码或训练框架。外部checkout仅供可选同权重对照测试 |
| 7tl7qns7ch/IPOT | [18c177846267505ee9503445a146dfd7dee34c41](https://github.com/7tl7qns7ch/IPOT/tree/18c177846267505ee9503445a146dfd7dee34c41) | MIT，Copyright2023 Seungjun Lee；原声明全文保留在本文件末尾 | bridge/query残差、pre-LN processor与GEGLU参考；使用本工程独立SDPA实现，不复制原数据/训练框架 |
| 原AirfRANS数据/派生项目 | [已有LICENSE](../Airfoil-Design-AirfRANS/LICENSE) 与原子项目README | Open Database License（ODbL），是数据库条款；未改写为全仓代码MIT | 保留原数据归属、读取、指标、边界处理及引用；本次未下载或再分发数据 |

Transolver当前origin使用SSH `git@github.com:thuml/Transolver.git`，与用户给的HTTPS地址指向同一仓库。分支main/HEAD与阶段0一致，无reset/commit/push。LRSA本地 `/home/hwz/LRSA-Operator` HEAD吻合且工作区干净。历史IPOT临时checkout已不可用，本次通过固定commit的GitHub raw只读重取LICENSE及3个相关源码文件，记录URL/字节数/SHA256于 [ipot-source.json](final_audit/ipot-source.json)；没有引入运行依赖。LRSA文件指纹见 [lrsa-source.json](final_audit/lrsa-source.json)。

LRSA相关源码为 `modeling/layers/attn.py`、`layers/mlp.py`、`perceiver.py`、`perceiver_structured.py`。实际顺序是down→latent FFN1→latent SA→latent FFN2→up，外层点残差/点FFN也保留。`dwconv`虽然命名如此，实际构造未传groups，故groups=1。本实现未采用其RoPE/gate、不同latent width、环境/训练presets及自定义核；参考测试显式固定同配置后比较输出、T及参数梯度。

IPOT `ipot_encoder.py` 在forward使用learned query residual，但未调用所声明encoder_ff；本实现只保留实际约定bridge。`ipot_processor.py` 重复追加同一attention/FFN对象，本实现改为每层独立实例；其 `layers.PreNorm` 可能只归一化query而保留传入context，本实现按确认公式令Q/K/V来自同一LN(Z)。这些是有依据的移植差异，不声称与原processor逐行或默认preset相同。

对应生产代码：[modules.py](../cdlno/modules.py)、[core.py](../cdlno/core.py)、[cdpa.py](../cdlno/cdpa.py)。最终原LRSA同配置对照2项通过，bridge/rear用显式数学reference核对；具体测试映射见 [需求矩阵](CDLNO_REQUIREMENTS_MATRIX.md)。

论文归属保留：[Transolver, ICML2024](https://arxiv.org/abs/2402.02366)、[LRSA](https://arxiv.org/abs/2604.03582)、[IPOT](https://arxiv.org/abs/2312.10975)。CDPA的depth评分借鉴所提供Kimi/AttnRes文献，但加入逐历史Cross对齐，未照搬其residual replacement。理论来源不是新增gate、quadrature、正交约束、损失或坐标decoder的授权。

## IPOT原MIT声明

```text
MIT License

Copyright (c) 2023 Seungjun Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
