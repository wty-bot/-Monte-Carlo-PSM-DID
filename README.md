# 因果推断与机器学习课程作业

本仓库是《因果推断与机器学习》课程作业项目，主题是：**用 Monte Carlo 模拟比较 `PSM`、`DID`、`DID-X` 和 `PSM-DID` 在不同识别环境下的表现，并结合一篇真实 PSM-DID 论文评价其适用条件与可信度。**

项目已完成模拟设计、代码实现、结果诊断、研究报告、论文评价和 AI 工具使用记录。仓库中的代码可以从头生成模拟数据，运行估计，并输出报告中使用的主要表格和图形。

## 交付材料

| 作业要求 | 仓库位置 | 说明 |
|:---|:---|:---|
| 研究报告 | [reports/研究报告.md](reports/研究报告.md) | Markdown 正式报告，包含问题背景、模拟设计、结果展示、理论解释、文献评价和结论 |
| 可复现代码 | [scripts/](scripts/) 与 [src/](src/) | `scripts/` 是运行入口，`src/` 是 DGP、估计方法和汇总代码 |
| 结果图表 | [outputs/figures/](outputs/figures/) 与 [outputs/checks/](outputs/checks/) | 报告内图表均由脚本生成，GitHub 可直接查看 |
| 结果表格 | [outputs/tables/](outputs/tables/) 与 [data/processed/](data/processed/) | 提交轻量汇总表，逐次模拟明细可运行脚本重生成 |
| 论文材料 | [data/external/论文材料.md](data/external/论文材料.md) | 单独列出被评价论文的题名、作者、期刊、年份、DOI 和稳定链接 |
| AI 使用记录 | [reports/AI工具使用记录.md](reports/AI工具使用记录.md) | 记录关键 prompt、主要输出、人工判断和最终采纳内容 |

## 研究设计简述

模拟背景设定为：某地区在第 4 年初实施更严格的环境监管政策，部分企业进入重点监管名单，研究者希望估计政策对企业绿色技术投资额的影响。项目构造 1000 家企业、6 期观测的平衡面板，并重复 500 次 Monte Carlo。

四个场景分别对应不同识别条件：

- `A`：近似随机分配，处理组和控制组满足无条件平行趋势。
- `B`：可观测协变量 `X` 同时影响处理分配和无政策趋势，基础 DID 有偏，PSM-DID 和 DID-X 应能修正。
- `C`：不可观测变量 `U` 同时影响处理分配和无政策趋势，仅平衡 `X` 不能恢复识别。
- `D`：在 `B` 的识别结构上加入异质处理效应，用于讨论匹配后估计对象更接近 `ATT` 还是 `ATE`。

比较方法包括 `PSM`、基础 `DID`、控制 `X` 与年份交互的 `DID-X`，以及先匹配再双重差分的 `PSM-DID`。

## 仓库结构

```text
.
├── README.md
├── PRD.md
├── 课程作业.md
├── 项目实时进度.md
├── requirements.txt
├── src/                      # DGP、估计方法、汇总与敏感性分析
├── scripts/                  # 可直接运行的复现脚本
├── data/
│   ├── external/             # 论文材料索引
│   ├── processed/            # 轻量汇总结果
│   ├── interim/              # 可重生成的中间结果，默认不提交
│   └── raw/                  # 可重生成的模拟原始数据，默认不提交
├── outputs/
│   ├── checks/               # 自检报告与检查图
│   ├── figures/              # 报告主图
│   └── tables/               # 报告主表
├── docs/                     # 规划、审计、研究过程记录
└── reports/                  # 研究报告与 AI 使用记录
```

`data/raw/` 和 `data/interim/` 的逐次模拟数据体积较大，按 GitHub 提交习惯默认不纳入版本控制；它们均可通过下方脚本从零生成。仓库保留报告所需的轻量图表、检查报告和汇总表，方便老师直接浏览结果。

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_environment.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
```

## 代码复现路径

### 快速烟雾测试

如果只想确认环境和主流程能跑通，可以先运行小规模版本：

```bash
python scripts/run_task1_generate.py --n-sims 10
python scripts/run_task2_estimate.py --smoke
python scripts/run_sensitivity.py
python scripts/run_event_study.py
```

注意：后两个扩展脚本会读取当前 `data/raw/` 中已有的模拟数据。若前一步只生成 10 次模拟，它们也会基于 10 次数据运行，结果只适合作为流程检查。

### 完整复现报告结果

从空数据目录开始，按以下顺序运行即可重生成主要数据、估计结果、图表和检查报告：

```bash
python scripts/run_task1_generate.py --n-sims 500
python scripts/run_task2_estimate.py --n-sims 500
python scripts/run_sensitivity.py
python scripts/run_event_study.py
```

运行完成后，重点检查这些输出：

- `data/raw/scenario_*/task1_scenario_*.parquet`：任务一生成的正式模拟面板数据
- [outputs/checks/task1_self_check_report.md](outputs/checks/task1_self_check_report.md)：任务一处理率、趋势和共同支持自检
- [outputs/checks/task2_self_check_report.md](outputs/checks/task2_self_check_report.md)：任务二估计结果与识别诊断自检
- [outputs/tables/task2_summary.csv](outputs/tables/task2_summary.csv)：报告表 1 的主要来源
- [outputs/tables/task2_identification_summary.csv](outputs/tables/task2_identification_summary.csv)：报告表 2 的主要来源
- [outputs/figures/estimate_distributions.png](outputs/figures/estimate_distributions.png)：估计值分布图
- [outputs/figures/common_support.png](outputs/figures/common_support.png)：共同支持图
- [outputs/figures/covariate_balance.png](outputs/figures/covariate_balance.png)：协变量平衡图
- [outputs/figures/preperiod_balance.png](outputs/figures/preperiod_balance.png)：处理前结果与趋势诊断图
- [outputs/figures/event_study.png](outputs/figures/event_study.png)：事件研究图

完整复现的任务二和事件研究会运行较多回归，耗时明显长于烟雾测试。已有一次完整运行结果已保存在 `outputs/` 和轻量汇总 CSV 中。

## 主要结论

- 场景 `A` 中，基础 `DID` 已经接近无偏，匹配不是识别所必需。
- 场景 `B` 中，基础 `DID` 明显上偏，`DID-X` 和 `PSM-DID` 能基本修复由可观测 `X` 驱动的趋势差异。
- 场景 `C` 中，`PSM-DID` 仍然明显有偏，说明平衡可观测变量不等于解决不可观测时间变化混淆。
- 场景 `D` 中，`PSM-DID` 的估计对象更接近匹配后处理组的 `ATT`，而不是总体 `ATE`。

详细解释见 [reports/研究报告.md](reports/研究报告.md)。

## 过程文档

- [项目实时进度.md](项目实时进度.md)：当前项目状态与交接说明
- [docs/process/研究过程.md](docs/process/研究过程.md)：研究推进过程说明
- [docs/audits/PSM匹配审计记录.md](docs/audits/PSM匹配审计记录.md)：PSM 与 PSM-DID 匹配管道审计
- [docs/audits/DID回归审计记录.md](docs/audits/DID回归审计记录.md)：DID、DID-X 与事件研究审计
- [reports/AI工具使用记录.md](reports/AI工具使用记录.md)：AI 协作与人工判断记录

## 备注

- 仓库包含中文文件名，建议在 UTF-8 环境下查看和运行。
- `.venv/`、`data/raw/`、`data/interim/`、运行日志和大体量逐次模拟结果不作为 GitHub 提交内容；需要时可用复现脚本重新生成。
