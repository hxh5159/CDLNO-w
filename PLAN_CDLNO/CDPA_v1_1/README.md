# CDPA 修改计划 v1.1

本包用于下一步在 Transolver 仓库实现已商定的新模型。当前只有完整计划、依赖候选与无数据环境检查脚本；没有修改 Transolver/LRSA/IPOT 模型，也没有下载数据或运行真实训练。

## 文件

| 文件 | 用途 |
|---|---|
| `CDPA_Transolver_Implementation_Plan_v1_1.md` | 唯一主计划：架构、CDPA公式、八任务数据合同、修改地图、验收、效率与延期范围 |
| `requirements-cdpa-cu128.txt` | 新环境的14项Python直接依赖候选版本 |
| `constraints-cdpa-cu128.txt` | 固定CUDA核心及相关直接依赖，防止pip替换torch组合 |
| `check_cdpa_environment.py` | 不需数据集的依赖/API功能检查；不安装依赖、不训练 |

## 已有远端环境：先检查，保留已经成功的配置

用户已用 CUDA12.8 / torch2.11 + PyG/pyg-lib 完成 ShapeNet-Car 训练。不要为了匹配候选清单直接重装或降级这个环境。先激活该环境，在本包目录运行；把仓库路径换成实际值：

```bash
python -m pip freeze > environment-before-cdpa.txt
python -m pip check
python check_cdpa_environment.py --device cuda --repo /absolute/path/to/Transolver --json-path environment-report.json
```

根据报告只补充真正缺失或不兼容的依赖。暂时没有仓库路径时可以省略 `--repo`，此时不会完成仓库安全模块导入检查。`--device cpu` 明确不验证GPU；脚本仍检查目标torch为2.11/cu128构建，不把其他CPU构建称为目标环境通过。

通过后可保存该工作环境的完整 freeze 作为复现记录。不要运行原标准目录的旧 requirements 把torch降到1.10.1，也不要无约束 `pip install -U`。

## 需要新环境时的具体候选

目标：CPython3.11，Linux x86_64，glibc>=2.28，官方cu128 wheel。现有Python版本兼容并已成功训练时无需更换。

在另外创建并激活的环境中、从本目录依次执行：

```bash
python -m pip install --only-binary=:all: "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-index --only-binary=:all: --no-deps "pyg-lib==0.6.0+pt211cu128" -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
python -m pip install --only-binary=:all: -r requirements-cdpa-cu128.txt -c constraints-cdpa-cu128.txt
python -m pip check
python check_cdpa_environment.py --device cuda --repo /absolute/path/to/Transolver --json-path environment-report.json
```

仅在检查通过后保存：

```bash
python -m pip freeze > requirements-validated-cu128.lock.txt
```

新环境候选固定 PyG2.8.0.post1；该版本的 radius_graph 通过 pyg-lib>=0.6 工作，当前审查的八任务不要求额外安装 torch-cluster。PyTorch/torchvision 配对与 PyG 二进制有官方依据：[PyTorch安装配对](https://pytorch.org/get-started/previous-versions/#v2110)、[PyG对应wheel](https://data.pyg.org/whl/torch-2.11.0+cu128.html)、[PyG2.8 radius_graph源码](https://pytorch-geometric.readthedocs.io/en/2.8.0/_modules/torch_geometric/nn/pool.html)。

已核对候选包发布版本、Python3.11 wheel与直接依赖约束；尚未完成远端联合GPU验收。requirements/constraints也不是完整传递依赖lock，最终以检查通过的freeze为准。

## 本次计划复核结果

默认仍为2个完整LRSA block、一次bridge/入口CDPA、6个persistent latent block、LRSA特征条件最终读出。每任务统一M、后置dense ConvFFN、F/L可调、off/entry/every_block和严格无数据验收均保持。稀疏Darcy、递增M、CDPA-Slice仍只记录、不实施。

效率实现补强为：Q一次投影、同次融合共享历史投影参数、来源折叠batch/分块SDPA、只保留必要历史、禁止跨独立CDPA层缓存投影KV，并提供matched LRSA合成性能对照。默认2+6下entry为2份历史对齐，every为27份；全来源批处理分别需1次/6次历史SDPA API调用，但不减少对应MAC，API调用也不等于CUDA kernel数。完整成本、参数、延迟与显存口径见计划第9节。

## 已执行检查与边界

- 计划与关键公式、依赖版本、参考源码接口已审阅。
- 环境脚本语法及`--help`通过；本地缺torch/PyG等依赖时正确报告FAIL并退出1，没有误报成功。
- 本地已可用的NumPy/SciPy、scikit-learn功能和pip check检查通过。
- 未安装候选GPU依赖、未验证远端GPU、未修改模型仓库、未执行模型单测或真实训练。

给Codex执行时使用主计划第11节指令；不要再引用旧v1文件。模型实现以后再执行主计划第8节的数学/adapter/checkpoint测试，环境预检不能代替这些模型验收。
