# 绘图参考阅读记录

2026-09-26。本次先查阅下列方法/可视化章节、图注并实际查看原图，再实现脚本。
只声称阅读与本任务有关的内容，不声称遍读顶会论文或复现作者未公开的绘图代码。

| 来源 | 实际查看位置 | 采用的设计 / 语义边界 |
|---|---|---|
| [Transolver, ICML 2024](https://arxiv.org/abs/2402.02366) | Appendix D.1，PDF p19，Figure 10；原版Physics_Attention_Structured_Mesh_2D | 64个slice的**4×16**全域小多图，白色无数据区域、等比例几何、viridis。画slice weights，不是64张物理预测场。原文未给出足以确定其head与色标归一化的完整绘图协议，本工具将自己的选择明确记录。 |
| [Transolver++](https://arxiv.org/abs/2502.02414v1) | 第3–4节；Figure 2；Appendix C.1，Figure 8–10；实际下载查看eidetic_state.png及airfoil_transolver_plus.png | Figure10明确画最后一层64个eidetic states；4×16，冷暖色和清晰翼型留白。权重均匀化是需要如实呈现的现象，不能靠每格减最小值制造分区。它的自适应温度/Gumbel属于模型方法，**不复制进V5绘图**。 |
| [LinearNO v3](https://arxiv.org/abs/2511.06294v3) | PDF p4 Figure3及Eq.(7)–(9)；Appendix G p16–19、Figure6–8；实际查看Darcy整页图 | phi(Q)与psi(K)归一轴不同，分别展示Query/Key weight maps。8×8排列、相同空间视窗、按原始latent编号展示。V5应分别画Q/K；不能冒充Transolver的一套双向共享slice权重。 |
| [FNO, ICLR 2021](https://arxiv.org/abs/2010.08895) | Figure1场比较、实验相关说明，实际查看PDF p2 | 并列场图使用一致几何/视窗，真值和预测采用相同色标，误差单独标度。没有照搬其彩虹色；本工具默认用viridis及magma。 |

## 从参考到脚本

1. 原论文4×16全域图与8×8翼型近景均导出，避免只放大局部而丢掉全域信息。
2. 某head内M=64；heads之间没有天然一一对应的latent编号，因此不平均heads。
3. 单一连续正值路由权重默认采用感知较均匀的viridis；可显式选coolwarm模仿Transolver++的风格。
4. 共享色标图供定量观察；逐状态peak图供形状观察。两者同时交付并明确命名，不静默缩放。
5. 同样的色值必须有明确含义。峰值缩放仅除最大值，不减最小值，不排序/选优latent。
6. 白色区域来自原网格孔洞，采用原相邻单元连接；不以规则矩形imshow替代曲线网格。
7. PDF嵌入TrueType文字，无外部TeX要求；密集色块栅格化，不声称整份PDF是全矢量。
8. 图片美观不构成“学到某种物理机制”的证明。caption描述routing weights及样本，不把颜色归因于具体物理因果。

## 参考资源身份

| 资源 | SHA-256 |
|---|---|
| Transolver PDF（本会话缓存） | `00110155267aa51ba7434449521006c49ccdcdeeb4ce345183e80e7fbf6066c6` |
| LinearNO v3 PDF | `422d9381c03aa909e88e4c443dc2f64308f86422ebc001f8e178610a79866c00` |
| FNO PDF | `6cd2645d4a50d1a61a261529b33747c674d78851bdbe6993b0961f81a17dc6f2` |
| Transolver++ v1 HTML，本仓库既有stage0参考文件 | `d186bbf716a45141a32bbf5d594f9b46556b74c79641ad523345803b606a8935` |
| [Transolver++ Airfoil图](https://arxiv.org/html/2502.02414v1/airfoil_transolver_plus.png) | `cf81dde97be2e1a86322482a8070a0497826cd642390ace91c373edbf21b239e` |
| [Transolver++ Figure2](https://arxiv.org/html/2502.02414v1/eidetic_state.png) | `3dd61fd6b0a1fd593714dbf6907d55b406a3fab9128a799c03ce456371004726` |

网络限制：Transolver++整本PDF下载超时，另一次并发整本下载停滞后终止；改为阅读已有完整HTML和直接查看官方PNG。
没有把失败下载称为整本PDF阅读成功。参考PDF/PNG仅放本地临时目录；脚本运行不联网，不需要论文文件。

## 2026-09-26 字体和会议模板复核

初版图内4.3–6.3pt小字不宜作为已通过最终印刷验收的结果；本次明确纠正，而非把此前合成
功能测试等同于投稿终稿质量。以下均为实际成功读取的官方资源：

- [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide) 指向官方ICLR模板。
  [iclr2026_conference.tex](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex)
  第179–194行要求图像整洁、清晰可读；图号和图注位于图后，黑白打印仍应可解释。
  同目录 `iclr2026_conference.sty` 第49行设置 `textwidth=5.5 true in`。
  没有发现统一强制图内使用9pt DejaVu Sans的条款；这是本工具的可读性选择。
- [ICML 2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions) 指向
  [icml2026.zip](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip)。
  实际读取 `example_paper.tex` 第258–263行：正文通栏宽6.75英寸；第403–418行：
  artwork应清晰可读，线宽至少0.5pt，图内不加总标题，图注在图下且为9pt。
  `icml2026.sty` 第262行确认6.75in；第668–670、679–687行用 `small`/9pt控制caption。
- 字体嵌入的细节应按目标年份的官方说明核对。ICML 2026 Author Instructions 明说本年
  “there is no Type 3 font check”，并鼓励适合的图使用矢量图、复杂可视化可用位图；
  example_paper仍有旧Type-1表述。本工具使用嵌入TrueType的PDF文字与栅格化密集色块，
  不声称整份PDF都是矢量，也不把“只能Type-1”或“全图必须Times”当成所有会议的统一规则。

实现对应：默认5.5in、`--paper icml`用6.75in；编号/色条/场标签统一9pt，编号移至每格上方；
色条框/刻度0.5pt；去掉真实数据图的总标题；PDF不tight-crop导致宽度变化。
`paper_figure.tex`通过原生 `\caption`继承官方模板字体、大小、位置、间距，不自行重设图注样式。
原始提取、Q/K归一化、色标数据与模型数学不变。

参考下载记录：最初猜测的根目录 `iclr2026_conference.tex` 和ICML直接tex链接404；
随后官方目录/author guide定位了正确资源。raw GitHub直连ZIP及tex/sty发生TLS/read超时，
最终经GitHub contents API取得完整2026 tex/sty；ICML官方ZIP成功下载。
ICML ZIP SHA-256：`8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`。
以上参考只用于核对，不是脚本远端运行依赖。
