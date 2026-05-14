# scripts

本目录存放可直接执行的项目脚本，是源码层与结果层之间的操作入口。

## 当前脚本清单

- `check_environment.py`：检查 Python 版本与核心依赖是否安装齐全
- `run_task1_generate.py`：生成任务 1 的四个场景模拟数据，并输出任务 1 自检结果
- `run_task2_estimate.py`：运行任务 2 的方法估计、结果汇总与任务 2 自检
- `run_sensitivity.py`：运行敏感性分析，包括 Rosenbaum bounds 与 caliper 对比
- `run_event_study.py`：生成事件研究系数与图表
- `run_task2_background.ps1`：Windows 后台运行任务 2 的 PowerShell 包装脚本
- `run_task2_background.cmd`：Windows 后台运行任务 2 的命令行包装脚本

## 使用建议

- 正式结果优先看 `data/processed/`、`outputs/checks/`、`outputs/tables/`、`outputs/figures/`
- 如果只是了解项目结构，先读根目录的 `项目实时进度.md`
- 如果准备继续推进写作，优先结合 `docs/process/研究过程.md` 与 `reports/AI工具使用记录.md`
